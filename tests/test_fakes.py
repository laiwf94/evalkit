"""The fake OpenAI-compatible and Gemini servers: every body the real clients
send is accepted and is exactly the body expected, and malformed requests are
refused the way the real APIs refuse them."""

import asyncio
import json
from typing import Any

import httpx2
import pytest
from pydantic import SecretStr

from evalkit.pricing import ModelPrice, PriceTable
from evalkit.providers import ErrorKind, ProviderError
from evalkit.providers.gemini import GeminiClient
from evalkit.providers.openai_compat import OpenAICompatClient
from tests.fake_gemini import FakeGemini, check_generate_body, count_text
from tests.fake_openai import BodyError, FakeOpenAI, check_chat_body

PRICES = PriceTable(
    {
        "gpt-test": ModelPrice(input=2.0, output=8.0),
        "embed-test": ModelPrice(input=0.1, output=0.0),
        "gemini-test": ModelPrice(input=1.0, output=4.0),
    }
)
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"score": {"type": "integer"}},
    "required": ["score"],
    "additionalProperties": False,
}
LOOSE_SCHEMA: dict[str, Any] = {"type": "object", "properties": {"x": {"type": "string"}}}


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def openai_client(fake: FakeOpenAI, **kwargs: Any) -> OpenAICompatClient:
    return OpenAICompatClient(
        PRICES,
        SecretStr("sk-test"),
        base_url="http://llm.test/v1",
        transport=fake.transport(),
        **kwargs,
    )


def gemini_client(fake: FakeGemini, **kwargs: Any) -> GeminiClient:
    return GeminiClient(
        PRICES,
        SecretStr("g-key"),
        base_url="http://gemini.test/v1beta",
        transport=fake.transport(),
        **kwargs,
    )


async def complete(client: Any, **kwargs: Any) -> Any:
    try:
        return await client.complete(**kwargs)
    finally:
        await client.aclose()


# OpenAI-compatible


@pytest.mark.parametrize(
    ("options", "client_options", "expected"),
    [
        (
            {},
            {},
            {
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_completion_tokens": 4096,
            },
        ),
        (
            {"system": "Be brief.", "max_tokens": 50, "effort": "high", "output_schema": SCHEMA},
            {"reasoning_effort": True},
            {
                "model": "gpt-test",
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Say hi"},
                ],
                "max_completion_tokens": 50,
                "reasoning_effort": "high",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "output", "schema": SCHEMA, "strict": True},
                },
            },
        ),
        (
            {"output_schema": LOOSE_SCHEMA, "max_tokens": 7, "effort": "max"},
            {"output_limit_field": "max_tokens", "reasoning_effort": True},
            {
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 7,
                "reasoning_effort": "high",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "output", "schema": LOOSE_SCHEMA, "strict": False},
                },
            },
        ),
    ],
    ids=["plain", "system-schema-effort", "older-server-loose-schema"],
)
def test_openai_client_sends_exactly_the_expected_body(
    options: dict[str, Any], client_options: dict[str, Any], expected: dict[str, Any]
) -> None:
    fake = FakeOpenAI(
        "http://llm.test/v1",
        output_limit_field=client_options.get("output_limit_field", "max_completion_tokens"),
        answer=lambda body: "hi",
    )
    done = run(
        complete(
            openai_client(fake, **client_options), model="gpt-test", prompt="Say hi", **options
        )
    )
    assert done.text == "hi"
    assert done.usage.cost_usd == pytest.approx((100 * 2.0 + 10 * 8.0) / 1e6)
    assert fake.rejected == []
    assert fake.bodies == [expected]
    (request,) = fake.requests
    assert request.url.path == "/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-test"


def test_openai_embeddings_body() -> None:
    fake = FakeOpenAI("http://llm.test/v1")
    client = openai_client(fake)

    async def go() -> Any:
        try:
            return await client.embed(model="embed-test", texts=["a b", "c"])
        finally:
            await client.aclose()

    found = run(go())
    assert found.vectors == [[3.0, 0.0], [1.0, 1.0]]
    assert fake.bodies == [
        {"model": "embed-test", "input": ["a b", "c"], "encoding_format": "float"}
    ]
    assert fake.rejected == []


USER = [{"role": "user", "content": "x"}]


def chat(**fields: Any) -> dict[str, Any]:
    """A valid chat body with ``fields`` changed (None removes a field)."""
    body = {"model": "m", "messages": USER, "max_completion_tokens": 1, **fields}
    return {k: v for k, v in body.items() if v is not None}


def schema(**fields: Any) -> dict[str, Any]:
    return chat(response_format={"type": "json_schema", "json_schema": fields})


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (chat(temperature=0), "Unrecognized request argument supplied: temperature"),
        (chat(model=None), "provide a model"),
        (chat(max_completion_tokens=None), "expected max_completion_tokens only, got []"),
        (chat(max_tokens=1), "expected max_completion_tokens only"),
        (chat(max_completion_tokens=0), "positive integer"),
        (chat(max_completion_tokens=True), "positive integer"),
        (chat(messages=[]), "non-empty array"),
        (chat(messages=[{"role": "user"}]), "exactly a role and a content"),
        (chat(messages=[{"role": "tool", "content": "x"}]), "role 'tool' is not valid"),
        (chat(messages=[{"role": "user", "content": 1}]), "must be a string"),
        (chat(messages=[*USER, {"role": "system", "content": "s"}]), "may only come first"),
        (chat(messages=[{"role": "system", "content": "s"}]), "last message must be from the user"),
        (chat(reasoning_effort="max"), "unsupported reasoning_effort"),
        (chat(response_format={"type": "json_object"}), "must have a type and a json_schema"),
        (chat(response_format={"type": "text", "json_schema": {}}), "response_format.type 'text'"),
        (schema(schema={}), "needs a name and a schema"),
        (schema(name="a b", schema={}), "name must match"),
        (schema(name="a", schema=[]), "schema must be an object"),
        (schema(name="a", schema={}, strict="yes"), "strict must be a boolean"),
    ],
)
def test_openai_fake_refuses_malformed_bodies(body: dict[str, Any], message: str) -> None:
    with pytest.raises(BodyError) as raised:
        check_chat_body(body, "max_completion_tokens")
    assert message in str(raised.value)


def test_openai_fake_answers_bad_requests_like_the_api() -> None:
    fake = FakeOpenAI("http://llm.test/v1", models={"gpt-test"})
    transport = fake.transport()

    def send(
        method: str, path: str, *, headers: dict[str, str] | None = None, content: bytes = b"{}"
    ) -> httpx2.Response:
        base = {"authorization": "Bearer sk-test", "content-type": "application/json"}
        request = httpx2.Request(
            method,
            f"http://llm.test/v1{path}",
            headers={**base, **(headers or {})},
            content=content,
        )
        return transport.handle_request(request)

    body = json.dumps(
        {
            "model": "gpt-x",
            "messages": [{"role": "user", "content": "x"}],
            "max_completion_tokens": 1,
        }
    ).encode()
    unknown_model = send("POST", "/chat/completions", content=body)
    assert unknown_model.status_code == 400
    assert unknown_model.json()["error"]["message"] == "The model `gpt-x` does not exist"
    assert (
        send("POST", "/chat/completions", headers={"authorization": "Bearer nope"}).status_code
        == 401
    )
    assert send("GET", "/chat/completions").status_code == 400
    assert (
        send("POST", "/chat/completions", headers={"content-type": "text/plain"}).status_code == 400
    )
    assert send("POST", "/chat/completions", content=b"not json").status_code == 400
    assert send("POST", "/chat/completions", content=b"[]").status_code == 400
    assert send("POST", "/completions").status_code == 404
    assert (
        send("POST", "/embeddings", content=b'{"model": "gpt-test", "input": []}').status_code
        == 400
    )
    assert (
        send(
            "POST",
            "/embeddings",
            content=b'{"model": "gpt-test", "input": ["a"], "encoding_format": "base64"}',
        ).status_code
        == 400
    )
    assert (
        send(
            "POST", "/embeddings", content=b'{"model": "gpt-test", "input": ["a"], "dimensions": 8}'
        ).status_code
        == 400
    )
    assert send("POST", "/embeddings", content=b'{"input": ["a"]}').status_code == 400
    other_host = transport.handle_request(
        httpx2.Request("POST", "http://elsewhere.test/v1/chat/completions")
    )
    assert other_host.status_code == 400
    assert len(fake.rejected) == 11
    keyless = FakeOpenAI("http://llm.test/v1", api_key=None).transport()
    keyed = httpx2.Request(
        "POST", "http://llm.test/v1/chat/completions", headers={"authorization": "Bearer x"}
    )
    assert keyless.handle_request(keyed).status_code == 400


def test_an_openai_rejection_reaches_the_client_as_an_invalid_request() -> None:
    fake = FakeOpenAI("http://llm.test/v1", output_limit_field="max_tokens")
    with pytest.raises(ProviderError) as raised:
        run(complete(openai_client(fake), model="gpt-test", prompt="x"))
    assert raised.value.kind is ErrorKind.INVALID_REQUEST
    assert fake.rejected == ["expected max_tokens only, got ['max_completion_tokens']"]


def test_a_wrong_openai_key_is_an_auth_error() -> None:
    fake = FakeOpenAI("http://llm.test/v1", api_key="sk-other")
    with pytest.raises(ProviderError) as raised:
        run(complete(openai_client(fake), model="gpt-test", prompt="x"))
    assert raised.value.kind is ErrorKind.AUTH


# Gemini


@pytest.mark.parametrize(
    ("options", "client_options", "expected"),
    [
        (
            {},
            {},
            {
                "contents": [{"role": "user", "parts": [{"text": "Say hi"}]}],
                "generationConfig": {"maxOutputTokens": 4096},
            },
        ),
        (
            {"system": "Be brief.", "max_tokens": 50, "effort": "xhigh", "output_schema": SCHEMA},
            {"thinking_level": True},
            {
                "contents": [{"role": "user", "parts": [{"text": "Say hi"}]}],
                "generationConfig": {
                    "maxOutputTokens": 50,
                    "responseMimeType": "application/json",
                    "responseJsonSchema": SCHEMA,
                    "thinkingConfig": {"thinkingLevel": "high"},
                },
                "systemInstruction": {"parts": [{"text": "Be brief."}]},
            },
        ),
    ],
    ids=["plain", "system-schema-thinking"],
)
def test_gemini_client_sends_exactly_the_expected_body(
    options: dict[str, Any], client_options: dict[str, Any], expected: dict[str, Any]
) -> None:
    fake = FakeGemini("http://gemini.test/v1beta", answer=lambda model, body: "hi")
    done = run(
        complete(
            gemini_client(fake, **client_options), model="gemini-test", prompt="Say hi", **options
        )
    )
    assert done.text == "hi"
    assert done.usage.cost_usd == pytest.approx((100 * 1.0 + 10 * 4.0) / 1e6)
    assert fake.rejected == []
    assert fake.bodies == [expected]
    (request,) = fake.requests
    assert request.url.path == "/v1beta/models/gemini-test:generateContent"
    assert request.headers["x-goog-api-key"] == "g-key"


def test_gemini_count_and_embed_bodies() -> None:
    fake = FakeGemini("http://gemini.test/v1beta")
    client = gemini_client(fake)

    async def go() -> Any:
        try:
            counted = await client.count_tokens(
                model="gemini-test", prompt="Say hi", system="Be brief."
            )
            vectors = await client.embed(model="gemini-test", texts=["ab", "c"])
            return counted, vectors
        finally:
            await client.aclose()

    counted, vectors = run(go())
    assert counted == count_text("Be brief.\nSay hi")
    assert vectors.vectors == [[2.0, 1.0], [1.0, 1.0]]
    assert fake.bodies == [
        {
            "generateContentRequest": {
                "model": "models/gemini-test",
                "contents": [{"role": "user", "parts": [{"text": "Say hi"}]}],
                "systemInstruction": {"parts": [{"text": "Be brief."}]},
            }
        },
        {
            "requests": [
                {"model": "models/gemini-test", "content": {"parts": [{"text": "ab"}]}},
                {"model": "models/gemini-test", "content": {"parts": [{"text": "c"}]}},
            ]
        },
    ]
    assert [r.url.path for r in fake.requests] == [
        "/v1beta/models/gemini-test:countTokens",
        "/v1beta/models/gemini-test:batchEmbedContents",
    ]
    assert fake.rejected == []


TEXT = {"role": "user", "parts": [{"text": "x"}]}


def config(**fields: Any) -> dict[str, Any]:
    return {"contents": [TEXT], "generationConfig": fields}


JSON = "application/json"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"contents": [TEXT], "safetySettings": []}, 'Unknown name "safetySettings"'),
        ({}, "contents is not specified"),
        ({"contents": [{**TEXT, "x": 1}]}, "exactly a role and parts"),
        ({"contents": [{**TEXT, "role": "system"}]}, "role 'system'"),
        ({"contents": [{**TEXT, "role": "model"}]}, "last content must be from the user"),
        ({"contents": [{"role": "user", "parts": []}]}, "non-empty array"),
        ({"contents": [{"role": "user", "parts": [{"inlineData": {}}]}]}, "must be text parts"),
        (
            {"contents": [TEXT], "systemInstruction": TEXT},
            "systemInstruction must have exactly parts",
        ),
        ({"contents": [TEXT], "generationConfig": []}, "generationConfig must be an object"),
        (config(temperature=0), 'Unknown name "temperature"'),
        (config(maxOutputTokens=0), "positive integer"),
        (config(responseJsonSchema={}), "needs responseMimeType application/json"),
        (config(responseMimeType=JSON, responseJsonSchema=[]), "must be an object"),
        (config(responseMimeType=JSON), "without a schema"),
        (config(thinkingConfig={"thinkingBudget": 1}), "exactly a thinkingLevel"),
        (config(thinkingConfig={"thinkingLevel": "max"}), "thinkingLevel 'max'"),
    ],
)
def test_gemini_fake_refuses_malformed_bodies(body: dict[str, Any], message: str) -> None:
    from tests.fake_gemini import BodyError as GeminiBodyError

    with pytest.raises(GeminiBodyError) as raised:
        check_generate_body(body)
    assert message in str(raised.value)
    check_generate_body(
        {"contents": [TEXT], "generationConfig": {"responseMimeType": "text/plain"}}
    )


def test_gemini_fake_answers_bad_requests_like_the_api() -> None:
    fake = FakeGemini("http://gemini.test/v1beta", models={"gemini-test"})
    transport = fake.transport()

    def send(
        path: str, body: Any = None, *, method: str = "POST", headers: dict[str, str] | None = None
    ) -> httpx2.Response:
        base = {"x-goog-api-key": "g-key", "content-type": "application/json"}
        content = (
            b"{}"
            if body is None
            else (body if isinstance(body, bytes) else json.dumps(body).encode())
        )
        request = httpx2.Request(
            method,
            f"http://gemini.test/v1beta{path}",
            headers={**base, **(headers or {})},
            content=content,
        )
        return transport.handle_request(request)

    good = {"contents": [TEXT]}
    assert send("/models/gemini-test:generateContent", good).status_code == 200
    missing = send("/models/gemini-other:generateContent", good)
    assert missing.status_code == 404
    assert missing.json()["error"]["status"] == "NOT_FOUND"
    assert (
        send(
            "/models/gemini-test:generateContent", good, headers={"x-goog-api-key": "bad"}
        ).status_code
        == 403
    )
    assert send("/models/gemini-test:generateContent", good, method="PUT").status_code == 400
    assert (
        send(
            "/models/gemini-test:generateContent", good, headers={"content-type": "text/plain"}
        ).status_code
        == 400
    )
    assert send("/models/gemini-test:generateContent", b"nope").status_code == 400
    assert send("/models/gemini-test:generateContent", b"[]").status_code == 400
    assert send("/models/gemini-test", good).status_code == 404
    assert send("/models/gemini-test:streamGenerateContent", good).status_code == 404
    assert send("/models/a%2Fb:generateContent", good).status_code == 400  # a quoted slash
    count = send(
        "/models/gemini-test:countTokens",
        {"generateContentRequest": {"model": "models/gemini-other", **good}},
    )
    assert count.status_code == 400
    assert send("/models/gemini-test:countTokens", {"contents": [TEXT]}).status_code == 400
    assert (
        send("/models/gemini-test:countTokens", {"generateContentRequest": []}).status_code == 400
    )
    embed = {"requests": [{"model": "models/gemini-test", "content": {"parts": [{"text": "a"}]}}]}
    assert send("/models/gemini-test:batchEmbedContents", embed).status_code == 200
    assert send("/models/gemini-test:batchEmbedContents", {"requests": []}).status_code == 400
    assert (
        send("/models/gemini-test:batchEmbedContents", {"requests": [], "x": 1}).status_code == 400
    )
    assert (
        send(
            "/models/gemini-test:batchEmbedContents",
            {"requests": [{"model": "models/gemini-test"}]},
        ).status_code
        == 400
    )
    assert (
        send(
            "/models/gemini-test:batchEmbedContents",
            {"requests": [{"model": "models/x", "content": {"parts": [{"text": "a"}]}}]},
        ).status_code
        == 400
    )
    assert (
        send(
            "/models/gemini-test:batchEmbedContents",
            {
                "requests": [
                    {
                        "model": "models/gemini-test",
                        "content": {"role": "user", "parts": [{"text": "a"}]},
                    }
                ]
            },
        ).status_code
        == 400
    )
    other_host = transport.handle_request(
        httpx2.Request("POST", "http://elsewhere.test/v1beta/models/m:generateContent")
    )
    assert other_host.status_code == 400
    assert len(fake.rejected) == 15


def test_a_wrong_gemini_key_is_an_auth_error() -> None:
    def answer(model: str, body: dict[str, Any]) -> str:
        raise AssertionError("a bad request must not be answered")

    fake = FakeGemini("http://gemini.test/v1beta", api_key="other", answer=answer)
    with pytest.raises(ProviderError) as raised:
        run(complete(gemini_client(fake), model="gemini-test", prompt="x"))
    assert raised.value.kind is ErrorKind.AUTH
    assert fake.rejected == ["bad key"]
