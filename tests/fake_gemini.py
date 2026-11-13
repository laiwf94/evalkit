"""A fake Gemini API behind httpx2.MockTransport.

Like ``tests.fake_openai``, it checks every request before answering it: the
method, the ``models/<model>:<method>`` path, the ``x-goog-api-key`` header, and
the exact shape of the JSON body for ``generateContent``, ``countTokens`` and
``batchEmbedContents`` (no unknown fields, user turns of text parts, a JSON
schema only with the JSON MIME type, a known thinking level). A request that
fails a check gets a 400 in the API's ``google.rpc.Status`` format and is
recorded in ``rejected``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any
from urllib.parse import unquote

import httpx2

Answer = Callable[[str, dict[str, Any]], str | httpx2.Response]

REQUEST_FIELDS = frozenset({"contents", "generationConfig", "systemInstruction", "tools"})
CONFIG_FIELDS = frozenset(
    {"maxOutputTokens", "responseMimeType", "responseJsonSchema", "thinkingConfig"}
)
THINKING_LEVELS = frozenset({"low", "medium", "high"})


class BodyError(Exception):
    pass


def generated(
    text: str = "ok",
    *,
    model: str = "gemini-test",
    finish: str = "STOP",
    prompt_tokens: int = 100,
    output_tokens: int = 10,
    cached_tokens: int = 0,
    thought_tokens: int = 0,
) -> httpx2.Response:
    usage = {
        "promptTokenCount": prompt_tokens,
        "candidatesTokenCount": output_tokens,
        "totalTokenCount": prompt_tokens + output_tokens + thought_tokens,
    }
    if cached_tokens:
        usage["cachedContentTokenCount"] = cached_tokens
    if thought_tokens:
        usage["thoughtsTokenCount"] = thought_tokens
    return httpx2.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": text}]},
                    "finishReason": finish,
                    "index": 0,
                }
            ],
            "usageMetadata": usage,
            "modelVersion": model,
        },
    )


def function_calls(
    calls: list[tuple[str, dict[str, Any]]],
    *,
    model: str = "gemini-test",
    signature: str | None = "sig-1",
    ids: bool = False,
) -> httpx2.Response:
    """An answer calling functions: (name, args) each, the first with a thought signature."""
    parts: list[dict[str, Any]] = []
    for n, (name, args) in enumerate(calls):
        call: dict[str, Any] = {"name": name, "args": args}
        if ids:
            call["id"] = f"g{n}"
        part: dict[str, Any] = {"functionCall": call}
        if n == 0 and signature is not None:
            part["thoughtSignature"] = signature
        parts.append(part)
    return httpx2.Response(
        200,
        json={
            "candidates": [
                {"content": {"role": "model", "parts": parts}, "finishReason": "STOP", "index": 0}
            ],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 10},
            "modelVersion": model,
        },
    )


def error_response(status: int, message: str, rpc_status: str) -> httpx2.Response:
    return httpx2.Response(
        status, json={"error": {"code": status, "message": message, "status": rpc_status}}
    )


def text_of(body: dict[str, Any]) -> str:
    """Every text part of a request, system instruction first."""
    parts = [*body.get("systemInstruction", {}).get("parts", [])]
    for content in body.get("contents", []):
        parts.extend(content["parts"])
    return "\n".join(p["text"] for p in parts)


def count_text(text: str) -> int:
    """The fake's token count: one per four bytes, rounded up."""
    return math.ceil(len(text.encode()) / 4)


class FakeGemini:
    """One Gemini API at ``base_url``. ``answer`` turns a validated
    ``generateContent`` body for a model into the answer text (or a response);
    ``models``, when given, are the only models it serves."""

    def __init__(
        self,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        *,
        api_key: str | None = "g-key",
        answer: Answer | None = None,
        models: set[str] | None = None,
        count: Callable[[dict[str, Any]], int] | None = None,
    ) -> None:
        url = httpx2.URL(base_url)
        self.host = url.host
        self.base_path = url.path.rstrip("/")
        self.api_key = api_key
        self.answer: Answer = answer or (lambda model, body: "ok")
        self.models = models
        # Tokens for a ``generateContentRequest``; by default from its text's length.
        self.count = count or (lambda request: count_text(text_of(request)))
        self.requests: list[httpx2.Request] = []
        self.rejected: list[str] = []

    @property
    def bodies(self) -> list[Any]:
        return [json.loads(r.content) for r in self.requests]

    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self.handle)

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        try:
            return self._serve(request)
        except BodyError as exc:
            self.rejected.append(str(exc))
            return error_response(400, str(exc), "INVALID_ARGUMENT")

    def _serve(self, request: httpx2.Request) -> httpx2.Response:
        if request.url.host != self.host:
            raise BodyError(f"request for host {request.url.host}, served {self.host}")
        if self.api_key is not None and request.headers.get("x-goog-api-key") != self.api_key:
            self.rejected.append("bad key")
            return error_response(403, "API key not valid", "PERMISSION_DENIED")
        if request.method != "POST":
            raise BodyError(f"{request.method} is not allowed")
        if request.headers.get("content-type") != "application/json":
            raise BodyError(f"content type {request.headers.get('content-type')!r}")
        path = request.url.raw_path.decode().removeprefix(self.base_path)
        if not path.startswith("/models/") or ":" not in path:
            return error_response(404, f"no such path {path}", "NOT_FOUND")
        model, method = path.removeprefix("/models/").rsplit(":", 1)
        model = unquote(model)
        if "/" in model:
            raise BodyError(f"model name {model!r} has a slash")
        if self.models is not None and model not in self.models:
            return error_response(404, f"models/{model} is not found", "NOT_FOUND")
        try:
            body = json.loads(request.content)
        except ValueError:
            raise BodyError("the body is not JSON") from None
        if not isinstance(body, dict):
            raise BodyError("the body is not a JSON object")
        if method == "generateContent":
            check_generate_body(body)
            reply = self.answer(model, body)
            return generated(reply, model=model) if isinstance(reply, str) else reply
        if method == "countTokens":
            check_count_body(body, model)
            return httpx2.Response(
                200, json={"totalTokens": self.count(body["generateContentRequest"])}
            )
        if method == "batchEmbedContents":
            texts = check_embed_body(body, model)
            return httpx2.Response(
                200, json={"embeddings": [{"values": [float(len(t)), 1.0]} for t in texts]}
            )
        return error_response(404, f"unknown method {method}", "NOT_FOUND")


def check_generate_body(body: dict[str, Any], *, allow_model: str | None = None) -> None:
    """Raise ``BodyError`` unless ``body`` is a ``generateContent`` request evalkit sends."""
    allowed = REQUEST_FIELDS | ({"model"} if allow_model else set())
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise BodyError(f'Invalid JSON payload received. Unknown name "{unknown[0]}"')
    if allow_model and body.get("model") != f"models/{allow_model}":
        raise BodyError(f"model {body.get('model')!r} does not match models/{allow_model}")
    contents = body.get("contents")
    if not isinstance(contents, list) or not contents:
        raise BodyError("contents is not specified")
    for i, content in enumerate(contents):
        if not isinstance(content, dict) or set(content) != {"role", "parts"}:
            raise BodyError(f"contents[{i}] must have exactly a role and parts")
        if content["role"] not in ("user", "model"):
            raise BodyError(f"contents[{i}].role {content['role']!r}")
        _check_parts(content["parts"], f"contents[{i}]", content["role"])
    if contents[-1]["role"] != "user":
        raise BodyError("the last content must be from the user")
    if "systemInstruction" in body:
        system = body["systemInstruction"]
        if not isinstance(system, dict) or set(system) != {"parts"}:
            raise BodyError("systemInstruction must have exactly parts")
        _check_parts(system["parts"], "systemInstruction")
    if "generationConfig" in body:
        _check_config(body["generationConfig"])
    if "tools" in body:
        _check_tools(body["tools"])


def _check_tools(tools: Any) -> None:
    if not isinstance(tools, list) or len(tools) != 1 or set(tools[0]) != {"functionDeclarations"}:
        raise BodyError("tools must be one functionDeclarations entry")
    for declaration in tools[0]["functionDeclarations"]:
        keys = set(declaration)
        if (
            not {"name", "parametersJsonSchema"}
            <= keys
            <= {
                "name",
                "parametersJsonSchema",
                "description",
            }
        ):
            raise BodyError("a function declaration needs a name and parametersJsonSchema")


def _check_parts(parts: Any, where: str, role: str = "user") -> None:
    if not isinstance(parts, list) or not parts:
        raise BodyError(f"{where}.parts must be a non-empty array")
    for part in parts:
        if not isinstance(part, dict):
            raise BodyError(f"{where}.parts must be objects")
        if "functionCall" in part and role == "model":
            if not set(part) <= {"functionCall", "thoughtSignature"}:
                raise BodyError(f"{where}: unknown fields next to functionCall")
            call = part["functionCall"]
            if not {"name", "args"} <= set(call) <= {"id", "name", "args"}:
                raise BodyError(f"{where}: a functionCall needs a name and args")
            if not isinstance(call["args"], dict):
                raise BodyError(f"{where}: functionCall.args must be an object")
            continue
        if "functionResponse" in part and role == "user":
            response = part["functionResponse"]
            if set(part) != {"functionResponse"} or not {"name", "response"} <= set(response) <= {
                "id",
                "name",
                "response",
            }:
                raise BodyError(f"{where}: a functionResponse needs a name and a response")
            if not isinstance(response["response"], dict):
                raise BodyError(f"{where}: functionResponse.response must be an object")
            continue
        if set(part) != {"text"} or not isinstance(part["text"], str):
            raise BodyError(f"{where}.parts must be text parts")


def _check_config(config: Any) -> None:
    if not isinstance(config, dict):
        raise BodyError("generationConfig must be an object")
    unknown = sorted(set(config) - CONFIG_FIELDS)
    if unknown:
        raise BodyError(f'Unknown name "{unknown[0]}" at generation_config')
    limit = config.get("maxOutputTokens")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise BodyError("maxOutputTokens must be a positive integer")
    if "responseJsonSchema" in config:
        if config.get("responseMimeType") != "application/json":
            raise BodyError("responseJsonSchema needs responseMimeType application/json")
        if not isinstance(config["responseJsonSchema"], dict):
            raise BodyError("responseJsonSchema must be an object")
    elif "responseMimeType" in config and config["responseMimeType"] != "text/plain":
        raise BodyError(f"responseMimeType {config['responseMimeType']!r} without a schema")
    if "thinkingConfig" in config:
        thinking = config["thinkingConfig"]
        if not isinstance(thinking, dict) or set(thinking) != {"thinkingLevel"}:
            raise BodyError("thinkingConfig must have exactly a thinkingLevel")
        if thinking["thinkingLevel"] not in THINKING_LEVELS:
            raise BodyError(f"thinkingLevel {thinking['thinkingLevel']!r}")


def check_count_body(body: dict[str, Any], model: str) -> None:
    if set(body) != {"generateContentRequest"}:
        raise BodyError("countTokens takes exactly a generateContentRequest")
    request = body["generateContentRequest"]
    if not isinstance(request, dict):
        raise BodyError("generateContentRequest must be an object")
    check_generate_body(request, allow_model=model)


def check_embed_body(body: dict[str, Any], model: str) -> list[str]:
    if set(body) != {"requests"}:
        raise BodyError("batchEmbedContents takes exactly requests")
    requests = body["requests"]
    if not isinstance(requests, list) or not requests:
        raise BodyError("requests must be a non-empty array")
    texts = []
    for i, item in enumerate(requests):
        if not isinstance(item, dict) or set(item) != {"model", "content"}:
            raise BodyError(f"requests[{i}] must have exactly a model and a content")
        if item["model"] != f"models/{model}":
            raise BodyError(f"requests[{i}].model {item['model']!r} is not models/{model}")
        content = item["content"]
        if not isinstance(content, dict) or set(content) != {"parts"}:
            raise BodyError(f"requests[{i}].content must have exactly parts")
        _check_parts(content["parts"], f"requests[{i}].content")
        texts.append("".join(p["text"] for p in content["parts"]))
    return texts
