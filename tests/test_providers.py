"""The Provider protocol, error kinds, and the Claude provider's side of the contract."""

import asyncio
from typing import Any

import anthropic
import httpx2
import pytest

from evalkit.pricing import PriceTable, TokenCounts
from evalkit.providers import (
    Completion,
    Embeddings,
    ErrorKind,
    Provider,
    ProviderError,
    kind_for_status,
)
from evalkit.providers.anthropic import ClaudeClient
from evalkit.targets import TargetError
from evalkit.targets.base import Usage
from tests.fake_anthropic import FakeAnthropic, json_reply, message


def run(coro: Any) -> Any:
    return asyncio.run(coro)


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (400, ErrorKind.INVALID_REQUEST),
        (401, ErrorKind.AUTH),
        (403, ErrorKind.AUTH),
        (404, ErrorKind.NOT_FOUND),
        (408, ErrorKind.TIMEOUT),
        (409, ErrorKind.CONFLICT),
        (413, ErrorKind.INVALID_REQUEST),
        (422, ErrorKind.INVALID_REQUEST),
        (429, ErrorKind.RATE_LIMITED),
        (500, ErrorKind.SERVER),
        (502, ErrorKind.SERVER),
        (503, ErrorKind.OVERLOADED),
        (504, ErrorKind.TIMEOUT),
        (529, ErrorKind.OVERLOADED),
        (402, ErrorKind.UNKNOWN),
        (501, ErrorKind.UNKNOWN),
        (302, ErrorKind.UNKNOWN),
    ],
)
def test_kind_for_status(status: int, kind: ErrorKind) -> None:
    assert kind_for_status(status) is kind


def test_retryable_statuses_are_unchanged_by_the_kinds() -> None:
    # The statuses the Claude client retried before error kinds existed.
    retried = {408, 409, 429, 500, 502, 503, 504, 529}
    for status in range(300, 600):
        assert kind_for_status(status).retryable is (status in retried), status


def test_only_transient_kinds_are_retryable() -> None:
    transient = {"rate_limited", "overloaded", "server", "conflict", "timeout", "connection"}
    assert {k.value for k in ErrorKind if k.retryable} == transient


def test_provider_error_is_a_target_error_retryable_by_kind() -> None:
    err = ProviderError(
        "slow down", kind=ErrorKind.RATE_LIMITED, provider="p", status=429, retry_after=3.0
    )
    assert isinstance(err, TargetError)
    assert (err.retryable, err.retry_after, err.status, err.provider) == (True, 3.0, 429, "p")
    assert str(err) == "slow down"
    blocked = ProviderError("blocked", kind=ErrorKind.SAFETY_BLOCK, provider="p")
    assert not blocked.retryable
    assert blocked.status is None
    assert blocked.retry_after is None


def test_completion_json() -> None:
    c = Completion(text='{"a": [1, 2]}', stop_reason="end_turn", usage=Usage(), model="m")
    assert c.json() == {"a": [1, 2]}
    assert c.refusal_category is None


def test_claude_client_implements_the_protocol() -> None:
    client = ClaudeClient(PriceTable())
    try:
        assert isinstance(client, Provider)
        assert client.name == "anthropic"
    finally:
        run(client.aclose())


def test_objects_missing_an_operation_are_not_providers() -> None:
    class NoEmbed:
        name = "x"
        prices = PriceTable()

        async def complete(self, **_: Any) -> Completion:
            raise NotImplementedError

        async def count_tokens(self, **_: Any) -> int:
            raise NotImplementedError

        def price(self, *_: Any, **__: Any) -> float:
            raise NotImplementedError

        def classify_error(self, exc: BaseException) -> ProviderError:
            raise NotImplementedError

        async def aclose(self) -> None:
            return

    assert not isinstance(NoEmbed(), Provider)


def test_price_matches_the_price_table() -> None:
    prices = PriceTable()
    client = FakeAnthropic().client(prices)
    tokens = TokenCounts(
        input_tokens=1000, output_tokens=200, cache_read_tokens=5000, cache_write_tokens=300
    )
    for model in ("claude-opus-5-5", "claude-haiku-5-5"):
        assert client.price(model, tokens) == prices.cost(model, tokens)
        assert client.price(model, tokens, batch=True) == prices.cost(model, tokens, batch=True)
    assert client.price("not-a-model", tokens) == 0.0


def test_completion_cost_goes_through_price() -> None:
    fake = FakeAnthropic(json_reply(message(usage={"input_tokens": 1000, "output_tokens": 100})))
    client = fake.client()
    result = run(client.complete(model="claude-opus-5-5", prompt="x"))
    expected = client.price("claude-opus-5-5", TokenCounts(input_tokens=1000, output_tokens=100))
    assert result.usage.cost_usd == pytest.approx(expected)
    assert expected == pytest.approx((1000 * 4 + 100 * 20) / 1_000_000)


def test_claude_has_no_embeddings_and_sends_nothing() -> None:
    fake = FakeAnthropic()
    with pytest.raises(ProviderError, match="no embeddings endpoint") as info:
        run(fake.client().embed(model="claude-opus-5-5", texts=["a", "b"]))
    assert info.value.kind is ErrorKind.UNSUPPORTED
    assert not info.value.retryable
    assert fake.requests == []


def _request() -> httpx2.Request:
    return httpx2.Request("POST", "http://fake-anthropic/v1/messages")


def test_classify_error_covers_every_exception_shape() -> None:
    client = ClaudeClient(PriceTable())
    try:
        own = ProviderError("x", kind=ErrorKind.SAFETY_BLOCK, provider="anthropic")
        assert client.classify_error(own) is own

        response = httpx2.Response(
            429, headers={"retry-after": "4", "request-id": "req_1"}, request=_request()
        )
        status = anthropic.RateLimitError("rate limited", response=response, body=None)
        err = client.classify_error(status)
        assert (err.kind, err.status, err.retry_after) == (ErrorKind.RATE_LIMITED, 429, 4.0)
        assert "request req_1" in err.message

        timeout = client.classify_error(anthropic.APITimeoutError(request=_request()))
        assert timeout.kind is ErrorKind.TIMEOUT
        conn = client.classify_error(anthropic.APIConnectionError(request=_request()))
        assert conn.kind is ErrorKind.CONNECTION

        for bad in (ValueError("bad json"), TypeError("None"), AttributeError("content")):
            malformed = client.classify_error(bad)
            assert malformed.kind is ErrorKind.MALFORMED_RESPONSE
            assert not malformed.retryable

        other = client.classify_error(RuntimeError("boom"))
        assert other.kind is ErrorKind.UNKNOWN
        assert other.message == "Claude API call failed: RuntimeError: boom"
        assert not other.retryable
        assert all(e.provider == "anthropic" for e in (err, timeout, conn, other))
    finally:
        run(client.aclose())


def test_cancellation_is_not_turned_into_an_error() -> None:
    def cancelled(_: dict[str, Any]) -> httpx2.Response:
        raise asyncio.CancelledError

    fake = FakeAnthropic(cancelled)
    with pytest.raises(asyncio.CancelledError):
        run(fake.client().complete(model="claude-opus-5-5", prompt="x"))


def test_embeddings_dataclass() -> None:
    e = Embeddings(vectors=[[0.0, 1.0]], usage=Usage(input_tokens=2), model="m")
    assert e.vectors[0][1] == 1.0
    assert e.usage.input_tokens == 2
