"""A fake OpenAI-compatible server behind httpx2.MockTransport.

It checks every request the way a strict server would before answering it:
the method, the path, the bearer key, the content type, and the exact shape of
the JSON body (no unknown fields, one output limit field, messages in order, a
well-formed ``response_format``). A request that fails a check gets a 400 in
the API's own error format and is recorded in ``rejected``, so a test that
expects an answer fails loudly on any drift in what the client sends. Tests
compare ``bodies`` with the exact body they expect.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import httpx2

Answer = Callable[[dict[str, Any]], str | httpx2.Response]

CHAT_FIELDS = frozenset(
    {"model", "messages", "max_completion_tokens", "max_tokens", "reasoning_effort",
     "response_format", "tools"}
)  # fmt: skip
EMBEDDING_FIELDS = frozenset({"model", "input", "encoding_format"})
ROLES = frozenset({"system", "user", "assistant", "tool"})
EFFORTS = frozenset({"minimal", "low", "medium", "high"})
SCHEMA_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class BodyError(Exception):
    def __init__(self, message: str, param: str | None = None) -> None:
        super().__init__(message)
        self.param = param


def chat_response(
    text: str = "ok",
    *,
    model: str = "gpt-test",
    finish: str = "stop",
    prompt_tokens: int = 100,
    completion_tokens: int = 10,
    cached_tokens: int = 0,
) -> httpx2.Response:
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if cached_tokens:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return httpx2.Response(
        200,
        json={
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish,
                }
            ],
            "usage": usage,
        },
    )


def error_response(
    status: int, message: str, *, kind: str = "invalid_request_error", code: str | None = None
) -> httpx2.Response:
    return httpx2.Response(
        status,
        json={"error": {"message": message, "type": kind, "param": None, "code": code}},
    )


class FakeOpenAI:
    """One OpenAI-compatible server at ``base_url``.

    ``answer`` turns a validated chat body into the answer text (or a whole
    response); ``models``, when given, are the only models it serves.
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        *,
        api_key: str | None = "sk-test",
        answer: Answer | None = None,
        models: set[str] | None = None,
        output_limit_field: str = "max_completion_tokens",
    ) -> None:
        url = httpx2.URL(base_url)
        self.host = url.host
        self.base_path = url.path.rstrip("/")
        self.api_key = api_key
        self.answer: Answer = answer or (lambda body: "ok")
        self.models = models
        self.output_limit_field = output_limit_field
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
            return error_response(400, str(exc))

    def _serve(self, request: httpx2.Request) -> httpx2.Response:
        if request.url.host != self.host:
            raise BodyError(f"request for host {request.url.host}, served {self.host}")
        if self.api_key is not None:
            if request.headers.get("authorization") != f"Bearer {self.api_key}":
                self.rejected.append("bad key")
                return error_response(401, "Incorrect API key provided", code="invalid_api_key")
        elif "authorization" in request.headers:
            raise BodyError("this server takes no key")
        if request.method != "POST":
            raise BodyError(f"{request.method} is not allowed")
        if request.headers.get("content-type") != "application/json":
            raise BodyError(f"content type {request.headers.get('content-type')!r}")
        try:
            body = json.loads(request.content)
        except ValueError:
            raise BodyError("the body is not JSON") from None
        if not isinstance(body, dict):
            raise BodyError("the body is not a JSON object")
        path = request.url.path.removeprefix(self.base_path)
        if path == "/chat/completions":
            check_chat_body(body, self.output_limit_field)
            self._check_model(body["model"])
            reply = self.answer(body)
            return chat_response(reply, model=body["model"]) if isinstance(reply, str) else reply
        if path == "/embeddings":
            check_embedding_body(body)
            self._check_model(body["model"])
            return embedding_response(body["input"], model=body["model"])
        return error_response(404, f"Invalid URL (POST {request.url.path})")

    def _check_model(self, model: str) -> None:
        if self.models is not None and model not in self.models:
            raise BodyError(f"The model `{model}` does not exist", "model")


def check_chat_body(body: dict[str, Any], output_limit_field: str) -> None:
    """Raise ``BodyError`` unless ``body`` is a chat completions request evalkit may send."""
    unknown = sorted(set(body) - CHAT_FIELDS)
    if unknown:
        raise BodyError(f"Unrecognized request argument supplied: {', '.join(unknown)}")
    if not isinstance(body.get("model"), str) or not body["model"]:
        raise BodyError("you must provide a model parameter", "model")
    limits = [f for f in ("max_completion_tokens", "max_tokens") if f in body]
    if limits != [output_limit_field]:
        raise BodyError(f"expected {output_limit_field} only, got {limits}", output_limit_field)
    limit = body[output_limit_field]
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise BodyError(f"{output_limit_field} must be a positive integer", output_limit_field)
    _check_messages(body.get("messages"))
    if "tools" in body:
        _check_tools(body["tools"])
    if "reasoning_effort" in body and body["reasoning_effort"] not in EFFORTS:
        raise BodyError(f"unsupported reasoning_effort {body['reasoning_effort']!r}")
    if "response_format" in body:
        _check_response_format(body["response_format"])


def _check_messages(messages: Any) -> None:
    if not isinstance(messages, list) or not messages:
        raise BodyError("messages must be a non-empty array", "messages")
    pending: list[str] = []  # tool call ids the last assistant message is owed
    for i, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in ROLES:
            raise BodyError(f"messages[{i}].role is not valid", "messages")
        role = message["role"]
        if role == "tool":
            if set(message) != {"role", "tool_call_id", "content"}:
                raise BodyError(f"messages[{i}] must have a tool_call_id and content", "messages")
            if not pending or message["tool_call_id"] != pending.pop(0):
                raise BodyError(f"messages[{i}] answers no outstanding tool call", "messages")
            if not isinstance(message["content"], str):
                raise BodyError(f"messages[{i}].content must be a string", "messages")
            continue
        if pending:
            raise BodyError(f"tool calls {pending} were not answered", "messages")
        if role == "assistant" and "tool_calls" in message:
            if set(message) != {"role", "content", "tool_calls"}:
                raise BodyError(f"messages[{i}] has unknown fields", "messages")
            pending = _check_tool_calls(message["tool_calls"], i)
            if message["content"] is not None and not isinstance(message["content"], str):
                raise BodyError(f"messages[{i}].content must be a string or null", "messages")
            continue
        if set(message) != {"role", "content"}:
            raise BodyError(f"messages[{i}] must have exactly a role and a content", "messages")
        if not isinstance(message["content"], str):
            raise BodyError(f"messages[{i}].content must be a string", "messages")
        if role == "system" and i != 0:
            raise BodyError("a system message may only come first", "messages")
    if pending:
        raise BodyError(f"tool calls {pending} were not answered", "messages")
    if messages[-1]["role"] not in ("user", "tool"):
        raise BodyError("the last message must be from the user or a tool", "messages")


def _check_tool_calls(calls: Any, i: int) -> list[str]:
    if not isinstance(calls, list) or not calls:
        raise BodyError(f"messages[{i}].tool_calls must be a non-empty array", "messages")
    ids = []
    for call in calls:
        if not isinstance(call, dict) or set(call) != {"id", "type", "function"}:
            raise BodyError(f"messages[{i}].tool_calls entries need id, type, function")
        function = call["function"]
        if call["type"] != "function" or not isinstance(function, dict):
            raise BodyError(f"messages[{i}].tool_calls must be functions")
        if set(function) != {"name", "arguments"} or not isinstance(function["arguments"], str):
            raise BodyError(f"messages[{i}].tool_calls arguments must be a JSON string")
        ids.append(call["id"])
    return ids


def _check_tools(tools: Any) -> None:
    if not isinstance(tools, list) or not tools:
        raise BodyError("tools must be a non-empty array", "tools")
    for tool in tools:
        if not isinstance(tool, dict) or set(tool) != {"type", "function"}:
            raise BodyError("each tool needs exactly a type and a function", "tools")
        function = tool["function"]
        if tool["type"] != "function" or not isinstance(function, dict):
            raise BodyError("only function tools are supported", "tools")
        if not {"name", "parameters"} <= set(function) <= {"name", "parameters", "description"}:
            raise BodyError("a function needs a name and parameters", "tools")
        if not SCHEMA_NAME.match(function["name"]) or not isinstance(function["parameters"], dict):
            raise BodyError("function names match ^[A-Za-z0-9_-]{1,64}$", "tools")


def tool_call_response(
    calls: list[tuple[str, str, str]], *, model: str = "gpt-test", text: str | None = None
) -> httpx2.Response:
    """An answer asking for tools: (id, name, arguments as JSON text) each."""
    return httpx2.Response(
        200,
        json={
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text,
                        "tool_calls": [
                            {"id": i, "type": "function", "function": {"name": n, "arguments": a}}
                            for i, n, a in calls
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        },
    )


def _check_response_format(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"type", "json_schema"}:
        raise BodyError("response_format must have a type and a json_schema", "response_format")
    if value["type"] != "json_schema":
        raise BodyError(f"response_format.type {value['type']!r}", "response_format")
    spec = value["json_schema"]
    if not isinstance(spec, dict) or not {"name", "schema"} <= set(spec) <= {
        "name",
        "schema",
        "strict",
        "description",
    }:
        raise BodyError("json_schema needs a name and a schema", "response_format")
    if not isinstance(spec["name"], str) or not SCHEMA_NAME.match(spec["name"]):
        raise BodyError("json_schema.name must match ^[A-Za-z0-9_-]{1,64}$", "response_format")
    if not isinstance(spec["schema"], dict):
        raise BodyError("json_schema.schema must be an object", "response_format")
    if "strict" in spec and not isinstance(spec["strict"], bool):
        raise BodyError("json_schema.strict must be a boolean", "response_format")


def check_embedding_body(body: dict[str, Any]) -> None:
    unknown = sorted(set(body) - EMBEDDING_FIELDS)
    if unknown:
        raise BodyError(f"Unrecognized request argument supplied: {', '.join(unknown)}")
    if not isinstance(body.get("model"), str) or not body["model"]:
        raise BodyError("you must provide a model parameter", "model")
    texts = body.get("input")
    if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
        raise BodyError("input must be a non-empty array of strings", "input")
    if body.get("encoding_format", "float") != "float":
        raise BodyError("encoding_format must be float", "encoding_format")


def embedding_response(texts: list[str], *, model: str) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "object": "list",
            "model": model,
            "data": [
                {"object": "embedding", "index": i, "embedding": [float(len(t)), float(i)]}
                for i, t in enumerate(texts)
            ],
            "usage": {"prompt_tokens": sum(len(t.split()) for t in texts)},
        },
    )
