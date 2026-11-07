"""The Gemini provider over raw HTTP: request bodies, safety blocks, usage and errors."""

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from pydantic import SecretStr

from evalkit.errors import InvalidInputError
from evalkit.pricing import ModelPrice, PriceTable, TokenCounts
from evalkit.providers import ErrorKind, Provider, ProviderError
from evalkit.providers.gemini import GeminiClient

BASE = "https://generativelanguage.googleapis.com/v1beta"
PRICES = PriceTable(
    {
        "gemini-test": ModelPrice(input=1.25, output=10.0, cache_read=0.125),
        "gemini-embed-test": ModelPrice(input=0.15, output=0.0),
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


def client(server: Server, **kwargs: Any) -> GeminiClient:
    kwargs.setdefault("api_key", SecretStr("g-key"))
    return GeminiClient(PRICES, transport=httpx2.MockTransport(server.handle), **kwargs)


def generated(
    parts: list[dict[str, Any]] | None = None,
    *,
    finish: str = "STOP",
    usage: dict[str, Any] | None = None,
    ratings: list[dict[str, Any]] | None = None,
) -> httpx2.Response:
    candidate: dict[str, Any] = {
        "content": {"role": "model", "parts": parts if parts is not None else [{"text": "hi"}]},
        "finishReason": finish,
        "index": 0,
    }
    if ratings is not None:
        candidate["safetyRatings"] = ratings
    return httpx2.Response(
        200,
        json={
            "candidates": [candidate],
            "usageMetadata": usage
            or {
                "promptTokenCount": 1000,
                "cachedContentTokenCount": 300,
                "candidatesTokenCount": 150,
                "thoughtsTokenCount": 50,
                "totalTokenCount": 1200,
            },
            "modelVersion": "gemini-test-001",
        },
    )


def complete(c: GeminiClient, **kwargs: Any) -> Any:
    async def go() -> Any:
        try:
            return await c.complete(model="gemini-test", prompt="Say hi", **kwargs)
        finally:
            await c.aclose()

    return run(go())


def test_satisfies_the_provider_protocol() -> None:
    c = client(Server(generated()))
    assert isinstance(c, Provider)
    assert c.name == "gemini"


def test_exact_request_body_and_headers() -> None:
    server = Server(generated())
    complete(client(server), system="Be brief.", max_tokens=300)
    request = server.requests[0]
    assert str(request.url) == f"{BASE}/models/gemini-test:generateContent"
    assert request.headers["x-goog-api-key"] == "g-key"
    assert "authorization" not in request.headers
    assert server.body() == {
        "contents": [{"role": "user", "parts": [{"text": "Say hi"}]}],
        "generationConfig": {"maxOutputTokens": 300},
        "systemInstruction": {"parts": [{"text": "Be brief."}]},
    }


def test_schema_and_thinking_level() -> None:
    schema = {"type": "object", "properties": {"score": {"type": "integer"}}}
    server = Server(generated([{"text": '{"score": 4}'}]))
    completion = complete(
        client(server, api_key=None, thinking_level=True), output_schema=schema, effort="max"
    )
    assert "x-goog-api-key" not in server.requests[0].headers
    assert server.body() == {
        "contents": [{"role": "user", "parts": [{"text": "Say hi"}]}],
        "generationConfig": {
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json",
            "responseJsonSchema": schema,
            "thinkingConfig": {"thinkingLevel": "high"},
        },
    }
    assert completion.json() == {"score": 4}


@pytest.mark.parametrize(
    ("effort", "level"),
    [("low", "low"), ("medium", "medium"), ("high", "high"), ("xhigh", "high"), ("max", "high")],
)
def test_thinking_levels(effort: str, level: str) -> None:
    server = Server(generated())
    complete(client(server, thinking_level=True), effort=effort)
    assert server.body()["generationConfig"]["thinkingConfig"] == {"thinkingLevel": level}


def test_model_name_is_quoted_into_the_path() -> None:
    prices = PriceTable({"tuned/../x:y": ModelPrice(input=1, output=1)})
    server = Server(generated())
    c = GeminiClient(prices, transport=httpx2.MockTransport(server.handle))
    run(c.complete(model="tuned/../x:y", prompt="x"))
    assert server.requests[0].url.raw_path == b"/v1beta/models/tuned%2F..%2Fx%3Ay:generateContent"


def test_text_usage_and_cost() -> None:
    completion = complete(
        client(Server(generated([{"text": "thinking...", "thought": True}, {"text": "hi"}, {}])))
    )
    assert completion.text == "hi"
    assert completion.stop_reason == "STOP"
    assert completion.model == "gemini-test-001"
    usage = completion.usage
    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens) == (700, 200, 300)
    assert usage.cost_usd == pytest.approx((700 * 1.25 + 200 * 10.0 + 300 * 0.125) / 1e6)


def test_max_tokens_keeps_partial_text_and_missing_usage_is_zero() -> None:
    response = httpx2.Response(
        200,
        json={
            "candidates": [{"content": {"parts": [{"text": "par"}]}, "finishReason": "MAX_TOKENS"}]
        },
    )
    completion = complete(client(Server(response)))
    assert (completion.text, completion.stop_reason) == ("par", "MAX_TOKENS")
    assert completion.model == "gemini-test"
    assert completion.usage.cost_usd == 0


def test_candidate_without_content_is_an_empty_answer() -> None:
    response = httpx2.Response(200, json={"candidates": [{"finishReason": "OTHER"}]})
    completion = complete(client(Server(response)))
    assert (completion.text, completion.stop_reason) == ("", "OTHER")


def test_blocked_prompt_is_a_safety_block() -> None:
    response = httpx2.Response(
        200,
        json={
            "promptFeedback": {
                "blockReason": "SAFETY",
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "HIGH",
                        "blocked": True,
                    },
                    {"category": "HARM_CATEGORY_HARASSMENT", "probability": "LOW"},
                ],
            },
            "usageMetadata": {"promptTokenCount": 12},
        },
    )
    with pytest.raises(ProviderError) as info:
        complete(client(Server(response)))
    error = info.value
    assert error.kind is ErrorKind.SAFETY_BLOCK
    assert not error.retryable
    assert error.provider == "gemini"
    assert "blocked the prompt (SAFETY)" in error.message
    assert "HARM_CATEGORY_DANGEROUS_CONTENT" in error.message
    assert "HARASSMENT" not in error.message


@pytest.mark.parametrize(
    "finish", ["SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY"]
)
def test_filtered_answer_is_a_safety_block(finish: str) -> None:
    with pytest.raises(ProviderError) as info:
        complete(client(Server(generated([], finish=finish, ratings=[]))))
    assert info.value.kind is ErrorKind.SAFETY_BLOCK
    assert f"stopped the answer ({finish})" in info.value.message
    assert "blocked categories" not in info.value.message


def test_unknown_model_fails_before_any_request() -> None:
    server = Server(generated())
    c = client(server)
    with pytest.raises(InvalidInputError):
        run(c.complete(model="gemini-unknown", prompt="x"))
    with pytest.raises(InvalidInputError):
        run(c.count_tokens(model="gemini-unknown", prompt="x"))
    with pytest.raises(InvalidInputError):
        run(c.embed(model="gemini-unknown", texts=["x"]))
    assert server.requests == []


def rpc_error(status: int, rpc: str, message: str = "nope", details: Any = None) -> httpx2.Response:
    error: dict[str, Any] = {"code": status, "message": message, "status": rpc}
    if details is not None:
        error["details"] = details
    return httpx2.Response(status, json={"error": error})


RETRY_INFO = "type.googleapis.com/google.rpc.RetryInfo"


@pytest.mark.parametrize(
    ("reply", "kind", "retry_after"),
    [
        (
            rpc_error(
                429,
                "RESOURCE_EXHAUSTED",
                details=[
                    {"@type": "type.googleapis.com/google.rpc.QuotaFailure"},
                    "junk",
                    {"@type": RETRY_INFO, "retryDelay": "37s"},
                ],
            ),
            ErrorKind.RATE_LIMITED,
            37.0,
        ),
        (
            rpc_error(
                429, "RESOURCE_EXHAUSTED", details=[{"@type": RETRY_INFO, "retryDelay": "x"}]
            ),
            ErrorKind.RATE_LIMITED,
            None,
        ),
        (rpc_error(503, "UNAVAILABLE", details={"not": "a list"}), ErrorKind.OVERLOADED, None),
        (rpc_error(500, "INTERNAL"), ErrorKind.SERVER, None),
        (rpc_error(504, "DEADLINE_EXCEEDED"), ErrorKind.TIMEOUT, None),
        (rpc_error(400, "INVALID_ARGUMENT"), ErrorKind.INVALID_REQUEST, None),
        (rpc_error(400, "FAILED_PRECONDITION"), ErrorKind.INVALID_REQUEST, None),
        (rpc_error(403, "PERMISSION_DENIED"), ErrorKind.AUTH, None),
        (rpc_error(404, "NOT_FOUND"), ErrorKind.NOT_FOUND, None),
        # An unusual HTTP status falls back to the rpc status name.
        (rpc_error(499, "ABORTED"), ErrorKind.CONFLICT, None),
        (rpc_error(499, "CANCELLED"), ErrorKind.UNKNOWN, None),
        (httpx2.Response(502, text="<html>gateway</html>"), ErrorKind.SERVER, None),
        (httpx2.Response(503, json={"error": "down"}), ErrorKind.OVERLOADED, None),
        (httpx2.Response(500, json={"error": {"message": "m"}}), ErrorKind.SERVER, None),
    ],
)
def test_errors(reply: httpx2.Response, kind: ErrorKind, retry_after: float | None) -> None:
    with pytest.raises(ProviderError) as info:
        complete(client(Server(reply)))
    error = info.value
    assert error.kind is kind
    assert error.retryable is kind.retryable
    assert error.retry_after == retry_after
    assert error.status == reply.status_code


def test_error_message_and_header_retry_after() -> None:
    reply = rpc_error(429, "RESOURCE_EXHAUSTED", "quota")
    reply.headers["retry-after"] = "4"
    with pytest.raises(ProviderError) as info:
        complete(client(Server(reply)))
    assert info.value.message == "the Gemini API returned 429: quota (RESOURCE_EXHAUSTED)"
    assert info.value.retry_after == 4.0
    reply = httpx2.Response(400, json={"error": {"code": 400, "status": "INVALID_ARGUMENT"}})
    with pytest.raises(ProviderError, match=r"returned 400: INVALID_ARGUMENT$"):
        complete(client(Server(reply)))


@pytest.mark.parametrize(
    "reply",
    [
        httpx2.Response(200, text="{"),
        httpx2.Response(200, json={"candidates": []}),
        httpx2.Response(200, json={}),
        httpx2.Response(200, json={"candidates": [{"content": {"parts": ["x"]}}]}),
        httpx2.Response(
            200,
            json={
                "candidates": [{"content": {"parts": []}}],
                "usageMetadata": {"promptTokenCount": 1.5},
            },
        ),
    ],
)
def test_malformed(reply: httpx2.Response) -> None:
    with pytest.raises(ProviderError) as info:
        complete(client(Server(reply)))
    assert info.value.kind is ErrorKind.MALFORMED_RESPONSE


def test_transport_failure() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused")

    with pytest.raises(ProviderError) as info:
        complete(client(Server(handler)))
    assert info.value.kind is ErrorKind.CONNECTION
    assert info.value.retryable


def test_count_tokens_request() -> None:
    server = Server(httpx2.Response(200, json={"totalTokens": 42}))
    c = client(server)
    assert run(c.count_tokens(model="gemini-test", prompt="Say hi", system="Be brief.")) == 42
    assert str(server.requests[0].url) == f"{BASE}/models/gemini-test:countTokens"
    assert server.body() == {
        "generateContentRequest": {
            "model": "models/gemini-test",
            "contents": [{"role": "user", "parts": [{"text": "Say hi"}]}],
            "systemInstruction": {"parts": [{"text": "Be brief."}]},
        }
    }
    run(c.count_tokens(model="gemini-test", prompt="x"))
    assert "systemInstruction" not in server.body()["generateContentRequest"]


@pytest.mark.parametrize("body", [{}, {"totalTokens": "many"}, {"totalTokens": -1}])
def test_count_tokens_malformed(body: dict[str, Any]) -> None:
    with pytest.raises(ProviderError) as info:
        run(
            client(Server(httpx2.Response(200, json=body))).count_tokens(
                model="gemini-test", prompt="x"
            )
        )
    assert info.value.kind is ErrorKind.MALFORMED_RESPONSE


def test_embed_request_and_vectors() -> None:
    server = Server(
        httpx2.Response(200, json={"embeddings": [{"values": [0.5, 1]}, {"values": [2, 0]}]})
    )
    c = client(server)
    result = run(c.embed(model="gemini-embed-test", texts=["a", "b"]))
    assert str(server.requests[0].url) == f"{BASE}/models/gemini-embed-test:batchEmbedContents"
    assert server.body() == {
        "requests": [
            {"model": "models/gemini-embed-test", "content": {"parts": [{"text": "a"}]}},
            {"model": "models/gemini-embed-test", "content": {"parts": [{"text": "b"}]}},
        ]
    }
    assert result.vectors == [[0.5, 1.0], [2.0, 0.0]]
    assert result.model == "gemini-embed-test"
    assert run(c.embed(model="gemini-embed-test", texts=[])).vectors == []
    assert len(server.requests) == 1


@pytest.mark.parametrize(
    "body",
    [
        {"embeddings": [{"values": [1]}]},
        {"embeddings": "x"},
        {"embeddings": [{"values": [1]}, {}]},
        {"embeddings": [{"values": [1]}, {"values": ["a"]}]},
        {},
    ],
)
def test_embed_malformed(body: dict[str, Any]) -> None:
    server = Server(httpx2.Response(200, json=body))
    with pytest.raises(ProviderError) as info:
        run(client(server).embed(model="gemini-embed-test", texts=["a", "b"]))
    assert info.value.kind is ErrorKind.MALFORMED_RESPONSE


def test_price() -> None:
    c = client(Server(generated()))
    assert c.price("gemini-test", TokenCounts(0, 1_000_000)) == pytest.approx(10.0)
