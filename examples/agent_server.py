"""A reference agent for the evalkit session protocol (docs/agent-protocol.md).

It books restaurant tables with two tools, ``find_slots`` and ``book``, and
answers from the transcript alone, so it keeps no session state and the same
request always gets the same answer. It only needs the standard library:

    python examples/agent_server.py --port 8765
    evalkit agent conformance booker --sandbox examples/agents/booking.toml

with ``booker`` in EVALKIT_HTTP_TARGETS_FILE:

    [endpoints.booker]
    url = "http://127.0.0.1:8765/v1/step"
    allow_private = true

What it does in each turn of the user's:

* no date (YYYY-MM-DD) said yet: ask for one;
* look up the free slots on the date (``--skip-lookup`` leaves this out, to
  show a gate catching an agent that books blind);
* none free: say so; otherwise book the time the user asked for when it is
  free, else the first free slot, and confirm;
* a tool that fails is tried once more, then the agent apologises.

Protocol behaviour worth copying:

* a request in another protocol version, or one that is not a protocol
  request, gets 400;
* answers are kept by ``request_id`` (bounded, least recently used out), so a
  retried request gets the stored answer without doing the work twice;
* with one step left in the budget it answers in text instead of calling a tool.

``GET /healthz`` answers ``ok``; ``GET /stats`` counts answered and replayed requests.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PROTOCOL = "evalkit-agent/1"
MAX_BODY = 1024 * 1024
MAX_STORED = 10_000
DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
TIME = re.compile(r"\b([01]\d|2[0-3]):([0-5]\d)\b")


class BadRequestError(Exception):
    """The request is not one this agent can answer; the client gets 400."""


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _reply(text: str, calls: list[dict[str, Any]], messages: list[Any], **extra: Any) -> Any:
    seen = json.dumps(messages, sort_keys=True)
    out = json.dumps(calls, sort_keys=True) + text
    return {
        "protocol": PROTOCOL,
        "message": {"content": text, "tool_calls": calls},
        "usage": {"input_tokens": _tokens(seen), "output_tokens": _tokens(out), "cost_usd": 0.0},
        **extra,
    }


def _check(body: Any) -> tuple[list[dict[str, Any]], int, int]:
    """The transcript, the step and the steps left; BadRequestError unless ``body``
    is a protocol request this agent speaks."""
    if not isinstance(body, dict):
        raise BadRequestError("the body must be a JSON object")
    if body.get("protocol") != PROTOCOL:
        raise BadRequestError(f"this agent speaks {PROTOCOL}, not {body.get('protocol')!r}")
    messages, step, budget = body.get("messages"), body.get("step"), body.get("budget")
    if not isinstance(messages, list) or not messages:
        raise BadRequestError("messages must be a non-empty list")
    if not all(
        isinstance(m, dict) and m.get("role") in ("user", "assistant", "tool") for m in messages
    ):
        raise BadRequestError("every message needs a role: user, assistant or tool")
    if messages[0]["role"] != "user" or messages[-1]["role"] == "assistant":
        raise BadRequestError("messages must start with the user and not end with the assistant")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise BadRequestError("step must be a non-negative integer")
    if not isinstance(body.get("tools"), list):
        raise BadRequestError("tools must be a list")
    left = budget.get("steps_remaining") if isinstance(budget, dict) else None
    steps_left = left if isinstance(left, int) and not isinstance(left, bool) else 1_000_000
    return messages, step, steps_left


def respond(body: Any, *, skip_lookup: bool = False) -> Any:
    """The agent's answer to one request, from the request alone."""
    messages, step, steps_left = _check(body)
    offered = {t.get("name") for t in body["tools"] if isinstance(t, dict)}
    users = [str(m.get("content", "")) for m in messages if m["role"] == "user"]
    last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
    # This turn's tool results, oldest first: everything after the user's last message.
    results = [
        r
        for m in messages[last_user + 1 :]
        if m["role"] == "tool"
        for r in m.get("results", [])
        if isinstance(r, dict)
    ]
    date = next((d.group(1) for u in reversed(users) if (d := DATE.search(u))), None)
    if date is None:
        return _reply(
            "Which date would you like a table on? Please give it as YYYY-MM-DD.", [], messages
        )
    wanted = next(
        (f"{t.group(1)}:{t.group(2)}" for u in reversed(users) if (t := TIME.search(u))), None
    )

    def call(name: str, arguments: dict[str, Any]) -> Any:
        if name not in offered:
            return _reply(f"I cannot book tables here: there is no {name} tool.", [], messages)
        if steps_left <= 1:
            return _reply(
                "I ran out of steps before I could finish the booking.",
                [],
                messages,
                stop_reason="out_of_steps",
            )
        return _reply(
            "", [{"id": f"call-{step}-{name}", "name": name, "arguments": arguments}], messages
        )

    def failures(name: str) -> int:
        return sum(1 for r in results if r.get("name") == name and "error" in r)

    def answer(name: str) -> Any:
        return next(
            (r["result"] for r in reversed(results) if r.get("name") == name and "result" in r),
            None,
        )

    slots: Any = None
    if not skip_lookup:
        slots = answer("find_slots")
        if slots is None:
            if failures("find_slots") >= 2:
                return _reply(
                    "Sorry, I cannot check the free tables right now.",
                    [],
                    messages,
                    stop_reason="tool_failed",
                )
            return call("find_slots", {"date": date})
        slots = [s for s in slots if isinstance(s, str)] if isinstance(slots, list) else []
        if not slots:
            return _reply(f"There are no free tables on {date}.", [], messages)
    booked = answer("book")
    if booked is None:
        if failures("book") >= 2:
            return _reply(
                "Sorry, the booking did not go through. Please try again later.",
                [],
                messages,
                stop_reason="tool_failed",
            )
        slot = wanted if wanted is not None and (slots is None or wanted in slots) else None
        slot = slot or (slots[0] if slots else "19:00")
        return call("book", {"date": date, "slot": slot})
    slot = next(
        (
            c["arguments"].get("slot")
            for m in reversed(messages)
            if m["role"] == "assistant"
            for c in m.get("tool_calls", [])
            if c.get("name") == "book"
        ),
        None,
    )
    confirmation = booked.get("confirmation") if isinstance(booked, dict) else None
    return _reply(
        f"Booked a table on {date} at {slot}, confirmation {confirmation}.",
        [],
        messages,
        stop_reason="booked",
    )


class Agent:
    """The answers given so far, by request id, and counts for ``/stats``."""

    def __init__(self, *, skip_lookup: bool = False, delay_ms: float = 0.0) -> None:
        self.skip_lookup = skip_lookup
        self.delay_ms = delay_ms
        self.stored: OrderedDict[str, bytes] = OrderedDict()
        self.lock = threading.Lock()
        self.answered = 0
        self.replayed = 0

    def handle(self, raw: bytes, key: str | None) -> tuple[int, bytes]:
        try:
            body = json.loads(raw)
        except (ValueError, RecursionError):
            return HTTPStatus.BAD_REQUEST, _error("the body is not JSON")
        rid = body.get("request_id") if isinstance(body, dict) else None
        rid = rid if isinstance(rid, str) and rid else key
        if rid is not None:
            with self.lock:
                stored = self.stored.get(rid)
                if stored is not None:
                    self.stored.move_to_end(rid)
                    self.replayed += 1
                    return HTTPStatus.OK, stored
        try:
            answer = json.dumps(respond(body, skip_lookup=self.skip_lookup)).encode()
        except BadRequestError as exc:
            return HTTPStatus.BAD_REQUEST, _error(str(exc))
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000)  # stands in for model calls
        with self.lock:
            self.answered += 1
            if rid is not None:
                self.stored[rid] = answer
                while len(self.stored) > MAX_STORED:
                    self.stored.popitem(last=False)
        return HTTPStatus.OK, answer


def _error(message: str) -> bytes:
    return json.dumps({"error": message}).encode()


def make_server(host: str, port: int, agent: Agent) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True  # one small answer per request: send it now

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._send(HTTPStatus.OK, b"ok", "text/plain")
            elif self.path == "/stats":
                stats = {"answered": agent.answered, "replayed": agent.replayed}
                self._send(HTTPStatus.OK, json.dumps(stats).encode())
            else:
                self._send(HTTPStatus.NOT_FOUND, _error("not found"))

        def do_POST(self) -> None:
            length = self.headers.get("Content-Length", "")
            if not length.isdigit():
                self._refuse(HTTPStatus.LENGTH_REQUIRED, _error("send a Content-Length"))
                return
            if int(length) > MAX_BODY:
                self._refuse(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, _error("the body is too large"))
                return
            raw = self.rfile.read(int(length))
            status, payload = agent.handle(raw, self.headers.get("Idempotency-Key"))
            self._send(status, payload)

        def _refuse(self, status: int, payload: bytes) -> None:
            # The body is not read, so where the next request would start is
            # unknown: close the connection rather than parse the body as one.
            self.close_connection = True
            self._send(status, payload, close=True)

        def _send(
            self, status: int, payload: bytes, kind: str = "application/json", close: bool = False
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(payload)))
            if close:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            pass  # quiet: a run sends many requests

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="0 picks a free port")
    parser.add_argument("--port-file", help="write the port here once listening")
    parser.add_argument("--skip-lookup", action="store_true", help="book without find_slots")
    parser.add_argument("--delay-ms", type=float, default=0.0, help="wait this long per answer")
    args = parser.parse_args(argv)
    server = make_server(
        args.host, args.port, Agent(skip_lookup=args.skip_lookup, delay_ms=args.delay_ms)
    )
    port = server.server_address[1]
    if args.port_file:
        tmp = Path(args.port_file + ".tmp")
        tmp.write_text(str(port))
        tmp.replace(args.port_file)
    print(f"agent listening on http://{args.host}:{port}/v1/step", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
