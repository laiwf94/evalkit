import asyncio
from pathlib import Path
from typing import Any

import pytest

from evalkit.config import ConfigError, Settings
from evalkit.datasets import Case
from evalkit.errors import InvalidInputError
from evalkit.pricing import ModelPrice, PriceTable, TokenCounts
from evalkit.providers import ErrorKind, ProviderError
from evalkit.providers.anthropic import FALLBACK_BETA, ClaudeClient
from evalkit.targets import PromptTargetSpec, TargetError
from evalkit.targets.prompt import PromptTarget
from tests.fake_anthropic import FakeAnthropic, error_reply, json_reply, message, sse_reply


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_request_shape_caching_effort_and_fallbacks() -> None:
    fake = FakeAnthropic(json_reply(message("Paris")))
    client = fake.client()
    result = run(
        client.complete(
            model="claude-opus-5-5", prompt="Capital?", system="Be brief.", effort="low"
        )
    )
    assert result.text == "Paris"
    assert result.stop_reason == "end_turn"
    body = fake.requests[0]
    assert fake.paths == ["/v1/messages"]
    assert body["model"] == "claude-opus-5-5"
    assert body["messages"] == [{"role": "user", "content": "Capital?"}]
    assert body["system"] == [
        {"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}
    ]
    assert body["output_config"] == {"effort": "low"}
    assert body["fallbacks"] == "default"
    assert FALLBACK_BETA in fake.headers[0]["anthropic-beta"]
    for removed in ("temperature", "top_p", "top_k", "thinking"):
        assert removed not in body
    assert fake.headers[0]["x-api-key"] == "sk-ant-test-0000000000"


def test_no_system_no_fallback_and_structured_output() -> None:
    fake = FakeAnthropic(json_reply(message('{"score": 4}', model="claude-haiku-5-5")))
    schema = {"type": "object", "properties": {"score": {"type": "integer"}}, "required": ["score"]}
    result = run(
        fake.client().complete(
            model="claude-haiku-5-5", prompt="Grade", output_schema=schema, fallbacks=True
        )
    )
    body = fake.requests[0]
    assert "system" not in body
    assert "fallbacks" not in body  # Haiku has no server-side fallback
    assert "anthropic-beta" not in fake.headers[0]
    assert body["output_config"]["format"] == {"type": "json_schema", "schema": schema}
    assert result.json() == {"score": 4}


def test_fallbacks_can_be_turned_off() -> None:
    fake = FakeAnthropic(json_reply(message()))
    run(fake.client().complete(model="claude-opus-5-5", prompt="x", fallbacks=False))
    assert "fallbacks" not in fake.requests[0]


def test_thinking_blocks_are_ignored_in_text() -> None:
    thinking = {"type": "thinking", "thinking": "", "signature": "sig"}
    fake = FakeAnthropic(json_reply(message("answer", extra_blocks=[thinking])))
    assert run(fake.client().complete(model="claude-opus-5-5", prompt="x")).text == "answer"


def test_refusal_drops_partial_text_and_keeps_category() -> None:
    fake = FakeAnthropic(
        json_reply(
            message(
                "partial",
                stop_reason="refusal",
                stop_details={"type": "refusal", "category": "cyber", "explanation": "no"},
            )
        )
    )
    result = run(fake.client().complete(model="claude-opus-5-5", prompt="x"))
    assert result.text == ""
    assert result.stop_reason == "refusal"
    assert result.refusal_category == "cyber"


def test_large_max_tokens_streams() -> None:
    fake = FakeAnthropic(sse_reply(message("long answer")))
    result = run(fake.client().complete(model="claude-opus-5-5", prompt="x", max_tokens=64_000))
    assert result.text == "long answer"
    assert fake.requests[0]["stream"] is True
    assert result.usage.output_tokens == 20


def test_cost_with_cache_reads_matches_hand_computation() -> None:
    usage = {
        "input_tokens": 1_000,
        "output_tokens": 500,
        "cache_read_input_tokens": 10_000,
        "cache_creation_input_tokens": 2_000,
    }
    fake = FakeAnthropic(json_reply(message(usage=usage)))
    result = run(fake.client().complete(model="claude-opus-5-5", prompt="x"))
    # $4/M input, $20/M output, $0.20/M cache read, 1.25 x $4/M cache write
    expected = (1_000 * 4 + 500 * 20 + 10_000 * 0.20 + 2_000 * 5.0) / 1_000_000
    assert result.usage.cost_usd == pytest.approx(expected)
    assert (result.usage.cache_read_tokens, result.usage.cache_write_tokens) == (10_000, 2_000)


def test_fallback_iterations_are_priced_by_the_model_that_ran() -> None:
    usage = {
        "input_tokens": 300,
        "output_tokens": 100,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "iterations": [
            {
                "type": "message",
                "model": "claude-opus-5-5",
                "input_tokens": 100,
                "output_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
            {
                "type": "fallback_message",
                "model": "claude-opus-4-8",
                "input_tokens": 200,
                "output_tokens": 100,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        ],
    }
    fake = FakeAnthropic(json_reply(message(model="claude-opus-4-8", usage=usage)))
    result = run(fake.client().complete(model="claude-opus-5-5", prompt="x"))
    assert result.model == "claude-opus-4-8"
    assert result.usage.cost_usd == pytest.approx((100 * 4 + 200 * 5 + 100 * 25) / 1_000_000)


@pytest.mark.parametrize(
    ("reply", "kind", "retryable", "retry_after"),
    [
        (
            error_reply(429, "rate_limit_error", {"retry-after": "7"}),
            ErrorKind.RATE_LIMITED,
            True,
            7.0,
        ),
        (error_reply(529, "overloaded_error"), ErrorKind.OVERLOADED, True, None),
        (error_reply(500), ErrorKind.SERVER, True, None),
        (error_reply(503, headers={"retry-after": "2"}), ErrorKind.OVERLOADED, True, 2.0),
        (error_reply(400, "invalid_request_error"), ErrorKind.INVALID_REQUEST, False, None),
        (error_reply(401, "authentication_error"), ErrorKind.AUTH, False, None),
        (error_reply(403, "permission_error"), ErrorKind.AUTH, False, None),
        (
            error_reply(404, "not_found_error", {"request-id": "req_9"}),
            ErrorKind.NOT_FOUND,
            False,
            None,
        ),
        (error_reply(413, "request_too_large"), ErrorKind.INVALID_REQUEST, False, None),
    ],
)
def test_errors_map_to_provider_errors(
    reply: Any, kind: ErrorKind, retryable: bool, retry_after: float | None
) -> None:
    fake = FakeAnthropic(reply)
    with pytest.raises(ProviderError) as info:
        run(fake.client().complete(model="claude-opus-5-5", prompt="x"))
    assert isinstance(info.value, TargetError)
    assert info.value.kind is kind
    assert info.value.provider == "anthropic"
    assert info.value.status == reply.status_code
    assert info.value.retryable is retryable
    assert info.value.retry_after == retry_after
    assert "sk-ant-test" not in info.value.message


def test_request_id_is_in_error_message() -> None:
    fake = FakeAnthropic(error_reply(404, "not_found_error", {"request-id": "req_9"}))
    with pytest.raises(TargetError, match="request req_9"):
        run(fake.client().complete(model="claude-opus-5-5", prompt="x"))


def test_connection_errors_and_timeouts_are_retryable() -> None:
    import httpx2

    def boom(_: dict[str, Any]) -> httpx2.Response:
        raise httpx2.ConnectError("refused")

    def slow(_: dict[str, Any]) -> httpx2.Response:
        raise httpx2.ReadTimeout("slow")

    cases = (
        (boom, "cannot reach", ErrorKind.CONNECTION),
        (slow, "timed out", ErrorKind.TIMEOUT),
    )
    for reply, message_part, kind in cases:
        fake = FakeAnthropic(reply)
        with pytest.raises(ProviderError, match=message_part) as info:
            run(fake.client().complete(model="claude-opus-5-5", prompt="x"))
        assert info.value.retryable
        assert info.value.kind is kind
        assert info.value.status is None


def test_unknown_model_is_rejected_before_any_request() -> None:
    fake = FakeAnthropic()
    with pytest.raises(InvalidInputError, match="unknown model 'claude-opus-9'"):
        run(fake.client().complete(model="claude-opus-9", prompt="x"))
    assert fake.requests == []


def test_count_tokens() -> None:
    fake = FakeAnthropic(json_reply({"input_tokens": 42}))
    n = run(fake.client().count_tokens(model="claude-opus-5-5", prompt="hello", system="sys"))
    assert n == 42
    assert fake.paths == ["/v1/messages/count_tokens"]
    assert fake.requests[0] == {
        "model": "claude-opus-5-5",
        "messages": [{"role": "user", "content": "hello"}],
        "system": "sys",
    }
    no_system = FakeAnthropic(json_reply({"input_tokens": 1}))
    run(no_system.client().count_tokens(model="claude-opus-5-5", prompt="x"))
    assert "system" not in no_system.requests[0]


def test_count_tokens_errors() -> None:
    import httpx2

    def boom(_: dict[str, Any]) -> httpx2.Response:
        raise httpx2.ConnectError("refused")

    with pytest.raises(TargetError) as info:
        run(
            FakeAnthropic(error_reply(429, "rate_limit_error"))
            .client()
            .count_tokens(model="claude-opus-5-5", prompt="x")
        )
    assert info.value.retryable
    with pytest.raises(TargetError, match="cannot reach"):
        run(FakeAnthropic(boom).client().count_tokens(model="claude-opus-5-5", prompt="x"))


def test_prompt_target_renders_and_returns_usage() -> None:
    fake = FakeAnthropic(json_reply(message("4")))
    spec = PromptTargetSpec(
        type="prompt", template="Q: {{question}}", system="Answer with a number.", max_tokens=50
    )
    target = PromptTarget(spec, fake.client())
    gen = run(target.generate(Case(id="1", input={"question": "2+2?"}, expected="4")))
    assert gen.text == "4"
    assert gen.usage.input_tokens == 100
    assert fake.requests[0]["messages"][0]["content"] == "Q: 2+2?"
    assert fake.requests[0]["max_tokens"] == 50
    assert target.provider == "anthropic"
    run(target.aclose())


def test_prompt_target_rejects_unknown_model() -> None:
    spec = PromptTargetSpec(type="prompt", template="x", model="gpt-4")
    with pytest.raises(InvalidInputError, match="unknown model"):
        PromptTarget(spec, FakeAnthropic().client())


def test_client_construction_without_key_and_close() -> None:
    client = ClaudeClient(PriceTable())
    assert client.sdk is not None
    run(client.aclose())


def test_price_table() -> None:
    prices = PriceTable({"custom-model": ModelPrice(input=1, output=2)})
    assert "custom-model" in prices
    assert "claude-opus-5-5" in prices.models()
    assert prices.cost("unknown", TokenCounts(input_tokens=10)) == 0.0
    tokens = TokenCounts(input_tokens=1_000_000, output_tokens=1_000_000)
    assert prices.cost("custom-model", tokens) == pytest.approx(3.0)
    assert prices.cost("custom-model", tokens, batch=True) == pytest.approx(1.5)
    # Cache reads default to a tenth of the input price.
    assert prices.cost("custom-model", TokenCounts(cache_read_tokens=1_000_000)) == pytest.approx(
        0.1
    )
    # Haiku 5.5 charges the long-context rate past 100K prompt tokens.
    short = prices.cost("claude-haiku-5-5", TokenCounts(input_tokens=100_000, output_tokens=1_000))
    long = prices.cost("claude-haiku-5-5", TokenCounts(input_tokens=100_001, output_tokens=1_000))
    assert short == pytest.approx((100_000 * 0.10 + 1_000 * 0.50) / 1e6)
    assert long == pytest.approx((100_001 * 0.50 + 1_000 * 2.50) / 1e6)
    partial = PriceTable({"m": ModelPrice(input=1, output=1, long_context_threshold=10)})
    assert partial.cost("m", TokenCounts(input_tokens=20)) == pytest.approx(20 / 1e6)


def test_prices_file(tmp_path: Path) -> None:
    f = tmp_path / "prices.toml"
    f.write_text("[prices.my-model]\ninput = 1.5\noutput = 3\n")
    s = Settings.from_env({"EVALKIT_PRICES_FILE": str(f)})
    assert s.prices["my-model"].input == 1.5
    f.write_text("[prices.my-model]\ninput = -1\noutput = 3\n")
    with pytest.raises(ConfigError, match="invalid configuration"):
        Settings.from_env({"EVALKIT_PRICES_FILE": str(f)})


def test_unparseable_responses_are_case_errors() -> None:
    import httpx2

    bad_json = httpx2.Response(
        200, content=b"{not json", headers={"content-type": "application/json"}
    )
    wrong_shape = httpx2.Response(200, json={"unexpected": True})
    for reply in (bad_json, wrong_shape):
        with pytest.raises(ProviderError, match="unexpected response") as info:
            run(FakeAnthropic(reply).client().complete(model="claude-opus-5-5", prompt="x"))
        assert not info.value.retryable
        assert info.value.kind is ErrorKind.MALFORMED_RESPONSE
