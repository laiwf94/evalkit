"""A fake Claude API behind httpx2.MockTransport, so tests exercise the real SDK."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import anthropic
import httpx2
from pydantic import SecretStr

from evalkit.llm import ClaudeClient
from evalkit.pricing import PriceTable

Reply = httpx2.Response | Callable[[dict[str, Any]], httpx2.Response]


def message(
    text: str = "ok",
    *,
    model: str = "claude-opus-5-5",
    stop_reason: str = "end_turn",
    usage: dict[str, Any] | None = None,
    extra_blocks: list[dict[str, Any]] | None = None,
    stop_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [*(extra_blocks or []), {"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage
        or {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }
    if stop_details is not None:
        body["stop_details"] = stop_details
    return body


def json_reply(
    body: dict[str, Any], status: int = 200, headers: dict[str, str] | None = None
) -> httpx2.Response:
    return httpx2.Response(status, json=body, headers=headers)


def error_reply(
    status: int, kind: str = "api_error", headers: dict[str, str] | None = None
) -> httpx2.Response:
    return httpx2.Response(
        status,
        json={"type": "error", "error": {"type": kind, "message": f"{kind} happened"}},
        headers=headers,
    )


def sse_reply(body: dict[str, Any]) -> httpx2.Response:
    start = {
        **body,
        "content": [],
        "stop_reason": None,
        "usage": {**body["usage"], "output_tokens": 0},
    }
    events: list[tuple[str, dict[str, Any]]] = [
        ("message_start", {"type": "message_start", "message": start})
    ]
    for i, block in enumerate(body["content"]):
        events.append(
            (
                "content_block_start",
                {"type": "content_block_start", "index": i, "content_block": {**block, "text": ""}},
            )
        )
        events.append(
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": i,
                    "delta": {"type": "text_delta", "text": block["text"]},
                },
            )
        )
        events.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
    events.append(
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": body["stop_reason"], "stop_sequence": None},
                "usage": body["usage"],
            },
        )
    )
    events.append(("message_stop", {"type": "message_stop"}))
    payload = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)
    return httpx2.Response(
        200, content=payload.encode(), headers={"content-type": "text/event-stream"}
    )


class FakeAnthropic:
    """Answers each request with the next scripted reply, recording what was sent."""

    def __init__(self, *replies: Reply) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.paths: list[str] = []
        self.headers: list[httpx2.Headers] = []
        self.default: Reply | None = None

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content) if request.content else {}
        self.requests.append(body)
        self.paths.append(request.url.path)
        self.headers.append(request.headers)
        reply = self.replies.pop(0) if self.replies else self.default
        if reply is None:
            raise AssertionError(f"unexpected request to {request.url.path}")
        return reply(body) if callable(reply) else reply

    def client(self, prices: PriceTable | None = None) -> ClaudeClient:
        http = anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(self.handler))
        return ClaudeClient(
            prices or PriceTable(),
            SecretStr("sk-ant-test-0000000000"),
            http_client=http,
            base_url="http://fake-anthropic",
        )
