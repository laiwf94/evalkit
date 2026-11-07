"""The OpenAI-compatible provider over raw HTTP: request bodies, answers, usage and errors."""

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from pydantic import SecretStr

from evalkit.errors import InvalidInputError
from evalkit.pricing import ModelPrice, PriceTable
from evalkit.providers import ErrorKind, Provider, ProviderError
from evalkit.providers.openai_compat import OpenAICompatClient, strict_compatible
from evalkit.targets import TargetError

PRICES = PriceTable(
    {
        "gpt-test": ModelPrice(input=2.0, output=8.0, cache_read=0.5),
        "embed-test": ModelPrice(input=0.02, output=0.0),
    }
)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


class Server:
    def __init__(self, reply: httpx2.Response | Callable[[httpx2.Request], httpx2.Response]):
        self.reply = reply
        self.requests: list[httpx2.Request] = []

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.reply(request) if callable(self.reply) else self.reply

    def body(self, index: int = -1) -> Any:
        return json.loads(self.requests[index].content)


def client(server: Server, **kwargs: Any) -> OpenAICompatClient:
    kwargs.setdefault("api_key", SecretStr("sk-test"))
    return OpenAICompatClient(PRICES, transport=httpx2.MockTransport(server.handle), **kwargs)


def chat(
    content: Any = "hello",
    *,
    finish: str = "stop",
    usage: dict[str, Any] | None = None,
    refusal: str | None = None,
    model: str = "gpt-test-2026",
) -> httpx2.Response:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if refusal is not None:
        message["refusal"] = refusal
    return httpx2.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage
            or {
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "prompt_tokens_details": {"cached_tokens": 400},
            },
        },
    )


def complete(c: OpenAICompatClient, **kwargs: Any) -> Any:
    async def go() -> Any:
        try:
            return await c.complete(model="gpt-test", prompt="Say hi", **kwargs)
        finally:
            await c.aclose()

    return run(go())


def test_satisfies_the_provider_protocol() -> None:
    assert isinstance(client(Server(chat())), Provider)


def test_exact_request_body_and_headers() -> None:
    server = Server(chat())
    complete(client(server), system="Be brief.", max_tokens=300)
    request = server.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-test"
    assert request.headers["content-type"] == "application/json"
    assert server.body() == {
        "model": "gpt-test",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Say hi"},
        ],
        "max_completion_tokens": 300,
    }


def test_no_key_no_system_older_token_field_and_base_url() -> None:
    server = Server(chat())
    c = client(
        server,
        api_key=None,
        base_url="http://vllm.internal:8000/v1/",
        output_limit_field="max_tokens",
        name="vllm",
    )
    assert c.name == "vllm"
    complete(c, max_tokens=50)
    request = server.requests[0]
    assert str(request.url) == "http://vllm.internal:8000/v1/chat/completions"
    assert "authorization" not in request.headers
    assert server.body() == {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 50,
    }


@pytest.mark.parametrize(
    ("effort", "sent"),
    [("low", "low"), ("medium", "medium"), ("high", "high"), ("xhigh", "high"), ("max", "high")],
)
def test_reasoning_effort_only_when_enabled(effort: str, sent: str) -> None:
    server = Server(chat())
    complete(client(server, reasoning_effort=True), effort=effort)
    assert server.body()["reasoning_effort"] == sent
    complete(client(server), effort=effort)
    assert "reasoning_effort" not in server.body()


def test_strict_json_schema_output() -> None:
    schema = {
        "type": "object",
        "properties": {"score": {"type": "integer"}, "reason": {"type": "string"}},
        "required": ["score", "reason"],
        "additionalProperties": False,
    }
    server = Server(chat('{"score": 3, "reason": "fine"}'))
    completion = complete(client(server), output_schema=schema)
    assert server.body()["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "output", "schema": schema, "strict": True},
    }
    assert completion.json() == {"score": 3, "reason": "fine"}


def test_open_schema_is_sent_without_strict() -> None:
    schema = {"type": "object", "properties": {"score": {"type": "integer"}}}
    server = Server(chat("{}"))
    complete(client(server), output_schema=schema)
    assert server.body()["response_format"]["json_schema"]["strict"] is False


@pytest.mark.parametrize(
    ("schema", "strict"),
    [
        (True, True),
        ({"type": "string"}, True),
        ({"type": "object", "properties": {}, "additionalProperties": False}, True),
        ({"type": "object", "properties": {"a": {}}, "required": ["a"]}, False),
        (
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": [],
                "additionalProperties": False,
            },
            False,
        ),
        ({"properties": [], "additionalProperties": False}, False),
        (
            {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "object", "properties": {"x": {"type": "number"}}},
                    }
                },
                "required": ["items"],
                "additionalProperties": False,
            },
            False,
        ),
        (
            {
                "anyOf": [{"$ref": "#/$defs/a"}, {"type": "null"}],
                "$defs": {
                    "a": {
                        "type": "object",
                        "properties": {"type": {"type": "string"}},
                        "required": ["type"],
                        "additionalProperties": False,
                    }
                },
            },
            True,
        ),
        ({"anyOf": ["not a schema"]}, False),
    ],
)
def test_strict_compatible(schema: Any, strict: bool) -> None:
    assert strict_compatible(schema) is strict


def test_usage_and_cost_with_cached_prompt_tokens() -> None:
    completion = complete(client(Server(chat())))
    assert completion.text == "hello"
    assert completion.stop_reason == "stop"
    assert completion.model == "gpt-test-2026"
    usage = completion.usage
    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens) == (600, 200, 400)
    assert usage.cache_write_tokens == 0
    assert usage.cost_usd == pytest.approx((600 * 2.0 + 200 * 8.0 + 400 * 0.5) / 1e6)


def test_missing_usage_counts_zero_and_null_content_is_empty() -> None:
    response = httpx2.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": None}, "finish_reason": None}]
        },
    )
    completion = complete(client(Server(response)))
    assert completion.text == ""
    assert completion.stop_reason is None
    assert completion.model == "gpt-test"
    assert completion.usage.cost_usd == 0


def test_content_parts_are_joined() -> None:
    parts = [
        {"type": "text", "text": "a"},
        {"type": "image_url", "image_url": {}},
        {"type": "text", "text": "b"},
    ]
    assert complete(client(Server(chat(parts)))).text == "ab"


def test_length_finish_keeps_partial_text() -> None:
    completion = complete(client(Server(chat("partial", finish="length"))))
    assert (completion.text, completion.stop_reason) == ("partial", "length")


def test_refusal_is_an_empty_answer_with_stop_reason_refusal() -> None:
    completion = complete(client(Server(chat(None, refusal="I can't help with that."))))
    assert completion.text == ""
    assert completion.stop_reason == "refusal"
    assert completion.usage.output_tokens == 200


def test_content_filter_is_a_safety_block() -> None:
    with pytest.raises(ProviderError) as info:
        complete(client(Server(chat("", finish="content_filter"))))
    assert info.value.kind is ErrorKind.SAFETY_BLOCK
    assert not info.value.retryable
    assert info.value.provider == "openai"


def test_unknown_model_fails_before_any_request() -> None:
    server = Server(chat())
    c = client(server)
    with pytest.raises(InvalidInputError):
        run(c.complete(model="gpt-unknown", prompt="x"))
    with pytest.raises(InvalidInputError):
        run(c.embed(model="gpt-unknown", texts=["x"]))
    assert server.requests == []


@pytest.mark.parametrize(
    ("status", "body", "kind", "retry_after"),
    [
        (
            429,
            {"error": {"message": "slow down", "type": "requests", "code": "rate_limit_exceeded"}},
            ErrorKind.RATE_LIMITED,
            None,
        ),
        (
            429,
            {"error": {"message": "no credit", "code": "insufficient_quota"}},
            ErrorKind.AUTH,
            None,
        ),
        (
            400,
            {"error": {"message": "too long", "code": "context_length_exceeded"}},
            ErrorKind.INVALID_REQUEST,
            None,
        ),
        (
            400,
            {"error": {"message": "flagged", "code": "content_policy_violation"}},
            ErrorKind.SAFETY_BLOCK,
            None,
        ),
        (404, {"error": {"message": "no such model", "code": None}}, ErrorKind.NOT_FOUND, None),
        (401, {"error": {"message": "bad key", "code": "invalid_api_key"}}, ErrorKind.AUTH, None),
        (500, {"error": "boom"}, ErrorKind.SERVER, None),
        (503, {"message": "warming up"}, ErrorKind.OVERLOADED, None),
        (502, "<html>bad gateway</html>", ErrorKind.SERVER, None),
    ],
)
def test_error_statuses(status: int, body: Any, kind: ErrorKind, retry_after: Any) -> None:
    reply = (
        httpx2.Response(status, json=body)
        if isinstance(body, dict)
        else httpx2.Response(status, text=body)
    )
    with pytest.raises(ProviderError) as info:
        complete(client(Server(reply)))
    error = info.value
    assert isinstance(error, TargetError)
    assert error.kind is kind
    assert error.status == status
    assert error.retryable is kind.retryable
    assert error.retry_after == retry_after
    assert str(status) in error.message


def test_error_message_includes_server_message_and_code() -> None:
    reply = httpx2.Response(400, json={"error": {"message": "too long", "code": "ctx"}})
    with pytest.raises(ProviderError, match=r"returned 400: too long \(ctx\)"):
        complete(client(Server(reply)))
    reply = httpx2.Response(400, json={"error": {"code": "only_code"}})
    with pytest.raises(ProviderError, match=r"returned 400: only_code$"):
        complete(client(Server(reply)))
    reply = httpx2.Response(400, text="x" * 2000)
    with pytest.raises(ProviderError) as info:
        complete(client(Server(reply)))
    assert len(info.value.message) < 600


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after": "7"}, 7.0),
        ({"retry-after-ms": "1500", "retry-after": "7"}, 1.5),
        ({"retry-after-ms": "soon", "retry-after": "3"}, 3.0),
        ({"retry-after-ms": "inf"}, None),
        ({"retry-after-ms": "-5"}, None),
        ({}, None),
    ],
)
def test_retry_after_headers(headers: dict[str, str], expected: float | None) -> None:
    reply = httpx2.Response(429, json={"error": {"message": "slow"}}, headers=headers)
    with pytest.raises(ProviderError) as info:
        complete(client(Server(reply)))
    assert info.value.retry_after == expected


def test_redirect_is_not_followed() -> None:
    server = Server(httpx2.Response(307, headers={"location": "https://elsewhere.test/"}))
    with pytest.raises(ProviderError) as info:
        complete(client(server))
    assert info.value.kind is ErrorKind.INVALID_REQUEST
    assert len(server.requests) == 1


@pytest.mark.parametrize(
    "reply",
    [
        httpx2.Response(200, text="not json"),
        httpx2.Response(200, json=[1, 2]),
        httpx2.Response(200, json={"choices": []}),
        httpx2.Response(200, json={"choices": [{}]}),
        httpx2.Response(200, json={"choices": "x"}),
        httpx2.Response(200, json={"choices": [{"message": {"content": 5}}]}),
        httpx2.Response(
            200,
            json={
                "choices": [{"message": {"content": "x"}}],
                "usage": {"prompt_tokens": -1},
            },
        ),
        httpx2.Response(
            200,
            json={
                "choices": [{"message": {"content": "x"}}],
                "usage": {"completion_tokens": True},
            },
        ),
    ],
)
def test_malformed_answers(reply: httpx2.Response) -> None:
    with pytest.raises(ProviderError) as info:
        complete(client(Server(reply)))
    assert info.value.kind is ErrorKind.MALFORMED_RESPONSE
    assert not info.value.retryable


def test_oversized_response_is_cut_off() -> None:
    server = Server(chat("x" * 5000))
    with pytest.raises(ProviderError) as info:
        complete(client(server, max_response_bytes=1024))
    assert info.value.kind is ErrorKind.MALFORMED_RESPONSE
    assert "exceeds 1024 bytes" in info.value.message


def raises(exc: BaseException) -> Callable[[httpx2.Request], httpx2.Response]:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise exc

    return handler


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (httpx2.ReadTimeout("slow"), ErrorKind.TIMEOUT),
        (httpx2.ConnectTimeout("slow"), ErrorKind.TIMEOUT),
        (httpx2.ConnectError("refused"), ErrorKind.CONNECTION),
        (httpx2.RemoteProtocolError("hung up"), ErrorKind.CONNECTION),
        (httpx2.DecodingError("bad gzip"), ErrorKind.MALFORMED_RESPONSE),
    ],
)
def test_transport_failures(exc: Exception, kind: ErrorKind) -> None:
    with pytest.raises(ProviderError) as info:
        complete(client(Server(raises(exc))))
    assert info.value.kind is kind
    assert info.value.retryable is kind.retryable


def test_classify_error_passes_provider_errors_through_and_maps_the_rest() -> None:
    c = client(Server(chat()))
    original = ProviderError("x", kind=ErrorKind.AUTH, provider="openai")
    assert c.classify_error(original) is original
    assert c.classify_error(KeyError("choices")).kind is ErrorKind.MALFORMED_RESPONSE
    unknown = c.classify_error(RuntimeError("odd"))
    assert unknown.kind is ErrorKind.UNKNOWN
    assert "RuntimeError" in unknown.message


def test_cancellation_passes_through() -> None:
    async def go() -> None:
        c = client(Server(raises(asyncio.CancelledError())))
        await c.complete(model="gpt-test", prompt="x")

    with pytest.raises(asyncio.CancelledError):
        run(go())


def test_count_tokens_is_unsupported_without_a_request() -> None:
    server = Server(chat())
    with pytest.raises(ProviderError) as info:
        run(client(server).count_tokens(model="gpt-test", prompt="x"))
    assert info.value.kind is ErrorKind.UNSUPPORTED
    assert server.requests == []


def embeddings_reply(data: list[dict[str, Any]], tokens: int = 12) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "object": "list",
            "data": data,
            "model": "embed-test-v1",
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        },
    )


def test_embed_request_order_and_cost() -> None:
    server = Server(
        embeddings_reply(
            [
                {"object": "embedding", "index": 1, "embedding": [0.0, 1.0]},
                {"object": "embedding", "index": 0, "embedding": [1, 0]},
            ]
        )
    )
    result = run(client(server).embed(model="embed-test", texts=["a", "b"]))
    assert str(server.requests[0].url) == "https://api.openai.com/v1/embeddings"
    assert server.body() == {"model": "embed-test", "input": ["a", "b"], "encoding_format": "float"}
    assert result.vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert result.model == "embed-test-v1"
    assert result.usage.input_tokens == 12
    assert result.usage.cost_usd == pytest.approx(12 * 0.02 / 1e6)


def test_embed_without_index_uses_position_and_empty_input_sends_nothing() -> None:
    server = Server(embeddings_reply([{"embedding": [1]}, {"embedding": [2]}]))
    c = client(server)
    assert run(c.embed(model="embed-test", texts=["a", "b"])).vectors == [[1.0], [2.0]]
    empty = run(c.embed(model="embed-test", texts=[]))
    assert empty.vectors == []
    assert len(server.requests) == 1


@pytest.mark.parametrize(
    "data",
    [
        [{"index": 0, "embedding": [1]}],
        [{"index": 0, "embedding": [1]}, {"index": 0, "embedding": [2]}],
        [{"index": 5, "embedding": [1]}, {"index": 0, "embedding": [2]}],
        [{"index": 0, "embedding": ["x"]}, {"index": 1, "embedding": [2]}],
        [{"index": 0}, {"index": 1, "embedding": [2]}],
    ],
)
def test_embed_malformed(data: list[dict[str, Any]]) -> None:
    with pytest.raises(ProviderError) as info:
        run(client(Server(embeddings_reply(data))).embed(model="embed-test", texts=["a", "b"]))
    assert info.value.kind is ErrorKind.MALFORMED_RESPONSE


def test_price() -> None:
    from evalkit.pricing import TokenCounts

    c = client(Server(chat()))
    assert c.price("gpt-test", TokenCounts(1_000_000, 0)) == pytest.approx(2.0)
    assert c.price("gpt-test", TokenCounts(0, 1_000_000), batch=True) == pytest.approx(4.0)


def test_the_base_reads_no_error_body() -> None:
    from evalkit.providers._http import HttpJsonProvider

    base = HttpJsonProvider(name="x", prices=PRICES, base_url="https://x.test", headers={})
    assert base.error_details(429, {"error": "x"}) == ("", ErrorKind.RATE_LIMITED, None)
    run(base.aclose())
