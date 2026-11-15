"""``evalkit agent conformance``: does an agent speak the session protocol?

Before a run spends money on an agent behind an endpoint, this drives it
through short sessions and probes the parts of ``docs/agent-protocol.md`` that
a run relies on. Each check passes, warns (the agent works with evalkit but
does something the protocol advises against) or fails (a run would fail or
mislead):

``protocol_version``
    A request in an unknown protocol version is refused with 400. Any other
    4xx warns; accepting it warns; a 5xx or an unreachable agent fails.
``schemas``
    One session per task, answered by the sandbox: every reply is a valid
    protocol answer, every tool call names an offered tool with arguments that
    match its ``input_schema``, and each session ends within ``max_steps``.
``idempotent_retries``
    Every step of the first session is sent again with the same body and
    ``request_id``, and its first step twice at once, as a retry after a
    timeout would. Each must be answered; an answer that differs warns.
``ordering``
    The agent keys state by session and step, not by arrival: the first
    session's steps are sent last first under a new session, as a resumed run
    may, and every task runs at once under new sessions, interleaved. Each
    must be answered; answers that differ from the sequential ones warn.
``tool_errors``
    A session in which every tool call fails still gets valid replies and ends
    within ``max_steps``.
``timeouts``
    No request ran out of the endpoint's ``timeout_seconds``; the slowest
    taking more than 80% of it warns.

The checker does not retry: a 429 or 5xx is reported where it happened.
Requests go through the same SSRF-safe client as a run (``HttpTarget.post``).
"""

from __future__ import annotations

import asyncio
import secrets
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx2
from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match

from evalkit.datasets import canonical_json, sha256_hex
from evalkit.providers.tools import AgentMessage, ToolDefinition, ToolOutcome
from evalkit.targets.base import TargetError
from evalkit.targets.http import HttpEndpoint, HttpTarget, Resolver, system_resolver
from evalkit.targets.http_agent import (
    END_TURN,
    MAX_STEPS,
    PROTOCOL,
    AgentReply,
    parse_reply,
    request_id,
    wire_messages,
    wire_tools,
)
from evalkit.targets.specs import HttpTargetSpec
from evalkit.tools.sandbox import Fixtures, SandboxSpec, ToolSandbox

Status = Literal["pass", "warn", "fail", "skip"]
UNKNOWN_PROTOCOL = "evalkit-agent/999"
TOOL_FAILURE = "conformance probe: this tool call failed"
SLOW_FRACTION = 0.8
MAX_TASKS = 20
CHECKS = (
    "protocol_version",
    "schemas",
    "idempotent_retries",
    "ordering",
    "tool_errors",
    "timeouts",
)


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str


@dataclass(frozen=True)
class ConformanceReport:
    endpoint: str
    protocol: str
    checks: tuple[Check, ...]
    requests: int
    latency_ms_p50: float
    latency_ms_max: float
    timeout_seconds: float

    def passed(self, *, strict: bool = False) -> bool:
        """No check failed or was skipped (and, when ``strict``, none warned)."""
        bad = {"fail", "skip", "warn"} if strict else {"fail", "skip"}
        return not any(c.status in bad for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "protocol": self.protocol,
            "passed": self.passed(),
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail} for c in self.checks
            ],
            "requests": self.requests,
            "latency_ms": {"p50": self.latency_ms_p50, "max": self.latency_ms_max},
            "timeout_seconds": self.timeout_seconds,
        }


class ProtocolViolationError(Exception):
    """The agent broke the protocol; the message says how."""


@dataclass(frozen=True)
class Exchange:
    """One request of a session and the agent's reply to it."""

    body: dict[str, Any]
    reply: AgentReply

    def answer(self) -> str:
        """The reply as compared between runs: what it says and asks, not its usage."""
        return _answer(self.reply)


@dataclass
class Transcript:
    session_id: str
    exchanges: list[Exchange] = field(default_factory=list)
    end: str | None = None
    problems: list[str] = field(default_factory=list)

    def answers(self) -> list[str]:
        return [e.answer() for e in self.exchanges]


def _answer(reply: AgentReply) -> str:
    return canonical_json(
        {
            "text": reply.text,
            "tool_calls": [u.to_json() for u in reply.tool_uses],
            "stop_reason": reply.stop_reason,
        }
    )


class AgentConformance:
    """The checks above against one endpoint, with ``tasks`` as the users'
    messages and ``sandbox`` answering the agent's tool calls."""

    def __init__(
        self,
        name: str,
        endpoint: HttpEndpoint,
        sandbox: SandboxSpec,
        tasks: Sequence[str],
        *,
        fixtures: Fixtures | None = None,
        max_steps: int = 20,
        resolver: Resolver = system_resolver,
        transport: httpx2.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.perf_counter,
        nonce: str | None = None,
    ) -> None:
        if not tasks:
            raise ValueError("conformance needs at least one task")
        if len(tasks) > MAX_TASKS:
            raise ValueError(f"conformance takes at most {MAX_TASKS} tasks")
        if not 1 <= max_steps <= 500:
            raise ValueError("max_steps must be from 1 to 500")
        self.name = name
        self.endpoint = endpoint
        self.sandbox = sandbox
        self.fixtures = fixtures
        self.tasks = list(tasks)
        self.max_steps = max_steps
        self.http = HttpTarget(
            HttpTargetSpec(type="http", endpoint=name),
            endpoint,
            resolver=resolver,
            transport=transport,
        )
        self.tools = wire_tools([ToolDefinition.of(d) for d in sandbox.definitions()])
        self._inputs = {t.name: Draft202012Validator(t.input_schema) for t in sandbox.tools}
        self._clock = clock
        # Sessions are new to the agent on every invocation, so the ordering probes
        # never meet answers the agent kept from an earlier one.
        self._nonce = nonce or secrets.token_hex(8)
        self.latencies: list[float] = []
        self.timeouts: list[str] = []

    def session_id(self, label: str) -> str:
        return "s-" + sha256_hex(canonical_json(["conformance", self._nonce, label]))[:32]

    async def run(self) -> ConformanceReport:
        try:
            checks: dict[str, Check] = {}
            checks["protocol_version"] = await self._check_version()
            sequential: list[Transcript] = []
            checks["schemas"] = await self._check_schemas(sequential)
            if checks["schemas"].status == "fail" and not any(t.exchanges for t in sequential):
                reason = "the sessions of the schemas check got no valid reply"
                for name in ("idempotent_retries", "ordering", "tool_errors"):
                    checks[name] = Check(name, "skip", reason)
            else:
                checks["idempotent_retries"] = await self._check_retries(sequential[0])
                checks["ordering"] = await self._check_ordering(sequential)
                checks["tool_errors"] = await self._check_tool_errors()
            checks["timeouts"] = self._check_timeouts()
        finally:
            await self.http.aclose()
        lat = self.latencies
        return ConformanceReport(
            endpoint=self.name,
            protocol=PROTOCOL,
            checks=tuple(checks[n] for n in CHECKS),
            requests=len(lat),
            latency_ms_p50=round(statistics.median(lat), 3) if lat else 0.0,
            latency_ms_max=round(max(lat), 3) if lat else 0.0,
            timeout_seconds=self.endpoint.timeout_seconds,
        )

    # -- requests

    def body(self, session_id: str, messages: Sequence[AgentMessage], step: int) -> dict[str, Any]:
        """The request a run would send at ``step`` of a session, with its id."""
        body: dict[str, Any] = {
            "protocol": PROTOCOL,
            "session_id": session_id,
            "step": step,
            "messages": wire_messages(messages),
            "tools": self.tools,
            "budget": {"steps_remaining": self.max_steps - step, "tokens_remaining": None},
        }
        body["request_id"] = request_id(body)
        return body

    async def send(self, body: dict[str, Any]) -> Any:
        """POST one request with the protocol's headers; the parsed answer."""
        headers = {
            "Idempotency-Key": str(body.get("request_id", "")),
            "X-Evalkit-Protocol": str(body.get("protocol", "")),
            "X-Evalkit-Session": str(body.get("session_id", "")),
        }
        start = self._clock()
        try:
            return await self.http.post(body, headers)
        except TargetError as exc:
            if exc.timed_out:
                self.timeouts.append(f"step {body.get('step')} of {body.get('session_id')}")
            raise
        finally:
            self.latencies.append(max(0.0, (self._clock() - start) * 1000))

    async def ask(self, body: dict[str, Any]) -> AgentReply:
        """The agent's reply to ``body``: ProtocolViolationError when it is not a valid answer,
        or names a tool or arguments the request did not offer."""
        try:
            document = await self.send(body)
            reply = parse_reply(document, self.name)
        except TargetError as exc:
            raise ProtocolViolationError(f"step {body['step']}: {exc.message}") from None
        for use in reply.tool_uses:
            validator = self._inputs.get(use.name)
            if validator is None:
                raise ProtocolViolationError(
                    f"step {body['step']}: called {use.name!r}, which was not offered"
                )
            if not isinstance(use.arguments, dict):
                raise ProtocolViolationError(
                    f"step {body['step']}: arguments for {use.name} are not a JSON object"
                )
            error = best_match(validator.iter_errors(use.arguments))
            if error is not None:
                where = "/".join(str(p) for p in error.absolute_path) or "arguments"
                raise ProtocolViolationError(
                    f"step {body['step']}: arguments for {use.name} break its input_schema "
                    f"at {where}: {error.message[:200]}"
                )
        return reply

    async def session(self, task: str, session_id: str, *, fail_tools: bool = False) -> Transcript:
        """One task, step by step, as a run drives it. Tool calls are answered by
        a fresh sandbox, or all fail when ``fail_tools``. A ProtocolViolationError stops the
        session and is kept in ``problems``."""
        sandbox = ToolSandbox(self.sandbox, self.fixtures, session=session_id)
        messages = [AgentMessage("user", task)]
        out = Transcript(session_id)
        for step in range(self.max_steps):
            body = self.body(session_id, messages, step)
            try:
                reply = await self.ask(body)
            except ProtocolViolationError as exc:
                out.problems.append(str(exc))
                return out
            out.exchanges.append(Exchange(body, reply))
            messages.append(AgentMessage("assistant", reply.text, reply.tool_uses))
            if not reply.tool_uses:
                out.end = reply.stop_reason or END_TURN
                return out
            outcomes = []
            for use in reply.tool_uses:
                if fail_tools:
                    outcomes.append(ToolOutcome(use.id, use.name, TOOL_FAILURE, True))
                    continue
                call = await sandbox.call(use.name, use.arguments)
                content = call.error if call.is_error else call.result
                outcomes.append(ToolOutcome(use.id, use.name, content, call.is_error))
            messages.append(AgentMessage("tool", outcomes=tuple(outcomes)))
        out.end = MAX_STEPS
        return out

    def reissued(self, body: dict[str, Any], session_id: str) -> dict[str, Any]:
        """``body`` as the same step of another session, with its own request id."""
        moved = {k: v for k, v in body.items() if k != "request_id"}
        moved["session_id"] = session_id
        moved["request_id"] = request_id(moved)
        return moved

    # -- checks

    async def _check_version(self) -> Check:
        name = "protocol_version"
        body = self.body(self.session_id("version"), [AgentMessage("user", self.tasks[0])], 0)
        body.pop("request_id")
        body["protocol"] = UNKNOWN_PROTOCOL
        body["request_id"] = request_id(body)
        try:
            await self.send(body)
        except TargetError as exc:
            if exc.status == 400:
                return Check(name, "pass", f"answered 400 to {UNKNOWN_PROTOCOL}")
            if exc.status is not None and 400 < exc.status < 500:
                return Check(
                    name, "warn", f"answered {exc.status} to {UNKNOWN_PROTOCOL}; it should be 400"
                )
            return Check(name, "fail", f"a request in {UNKNOWN_PROTOCOL}: {exc.message}")
        return Check(
            name,
            "warn",
            f"accepted a request in {UNKNOWN_PROTOCOL}; an agent should answer 400 "
            "to a version it does not know",
        )

    async def _check_schemas(self, out: list[Transcript]) -> Check:
        name = "schemas"
        for i, task in enumerate(self.tasks):
            out.append(await self.session(task, self.session_id(f"task-{i}")))
        problems = [f"task {i + 1}: {p}" for i, t in enumerate(out) for p in t.problems]
        unfinished = [i + 1 for i, t in enumerate(out) if t.end == MAX_STEPS]
        if problems:
            more = f" (and {len(problems) - 1} more)" if len(problems) > 1 else ""
            return Check(name, "fail", problems[0] + more)
        if unfinished:
            listed = ", ".join(str(i) for i in unfinished)
            return Check(
                name, "fail", f"task {listed} did not finish within {self.max_steps} replies"
            )
        replies = sum(len(t.exchanges) for t in out)
        calls = sum(len(e.reply.tool_uses) for t in out for e in t.exchanges)
        return Check(
            name,
            "pass",
            f"{len(out)} sessions, {replies} replies and {calls} tool calls, all valid",
        )

    async def _check_retries(self, first: Transcript) -> Check:
        name = "idempotent_retries"
        if not first.exchanges:
            return Check(name, "skip", "the first session got no valid reply")
        differs: list[int] = []
        try:
            for e in first.exchanges:
                again = await self.ask(e.body)
                if _answer(again) != e.answer():
                    differs.append(e.body["step"])
            head = first.exchanges[0]
            pair = await asyncio.gather(self.ask(head.body), self.ask(head.body))
            if any(_answer(r) != head.answer() for r in pair) and 0 not in differs:
                differs.insert(0, 0)
        except ProtocolViolationError as exc:
            return Check(name, "fail", f"a repeated request_id was not answered: {exc}")
        sent = len(first.exchanges) + 2
        if differs:
            steps = ", ".join(str(s) for s in differs)
            return Check(
                name,
                "warn",
                f"a repeated request_id got a different answer at step {steps}; "
                "fine for an agent without side effects, otherwise return the stored answer",
            )
        return Check(name, "pass", f"{sent} repeated requests, each answered as before")

    async def _check_ordering(self, sequential: list[Transcript]) -> Check:
        name = "ordering"
        first = sequential[0]
        notes: list[str] = []
        # A resumed session: the agent sees later steps before (or without) earlier ones.
        resumed = self.session_id("resumed")
        try:
            for e in reversed(first.exchanges):
                reply = await self.ask(self.reissued(e.body, resumed))
                if _answer(reply) != e.answer():
                    notes.append(f"step {e.body['step']} of a resumed session")
        except ProtocolViolationError as exc:
            return Check(name, "fail", f"a session resumed at a later step: {exc}")
        # Interleaved sessions: every task at once (the first twice when it is alone).
        tasks = self.tasks if len(self.tasks) > 1 else self.tasks * 2
        runs = await asyncio.gather(
            *(self.session(t, self.session_id(f"concurrent-{i}")) for i, t in enumerate(tasks))
        )
        for i, run in enumerate(runs):
            if run.problems:
                return Check(name, "fail", f"concurrent session {i + 1}: {run.problems[0]}")
            expected = sequential[i % len(sequential)]
            if run.answers() != expected.answers():
                notes.append(f"concurrent session {i + 1}")
        steps = len(first.exchanges)
        if notes:
            more = f" (and {len(notes) - 1} more)" if len(notes) > 1 else ""
            return Check(
                name,
                "warn",
                f"{notes[0]}{more} answered differently from the sequential run; an agent "
                "that keeps state must key it by session_id and step",
            )
        return Check(
            name,
            "pass",
            f"a session resumed backwards over {steps} steps and {len(runs)} interleaved "
            "sessions answered as the sequential ones",
        )

    async def _check_tool_errors(self) -> Check:
        name = "tool_errors"
        run = await self.session(self.tasks[0], self.session_id("tool-errors"), fail_tools=True)
        if run.problems:
            return Check(name, "fail", f"with every tool failing: {run.problems[0]}")
        if run.end == MAX_STEPS:
            return Check(
                name,
                "fail",
                f"with every tool failing, the session did not finish within "
                f"{self.max_steps} replies",
            )
        calls = sum(len(e.reply.tool_uses) for e in run.exchanges)
        return Check(
            name,
            "pass",
            f"with every tool failing, finished after {len(run.exchanges)} replies "
            f"and {calls} failed calls",
        )

    def _check_timeouts(self) -> Check:
        name = "timeouts"
        limit = self.endpoint.timeout_seconds
        if self.timeouts:
            more = f" (and {len(self.timeouts) - 1} more)" if len(self.timeouts) > 1 else ""
            return Check(name, "fail", f"{self.timeouts[0]}{more} ran out of {limit:g} s")
        slowest = max(self.latencies, default=0.0)
        if slowest > SLOW_FRACTION * limit * 1000:
            return Check(
                name,
                "warn",
                f"the slowest reply took {slowest:.0f} ms, close to the {limit:g} s timeout",
            )
        return Check(
            name, "pass", f"the slowest reply took {slowest:.0f} ms of the {limit:g} s timeout"
        )
