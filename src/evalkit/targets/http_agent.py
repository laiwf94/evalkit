"""An agent behind an HTTP endpoint, driven over the session protocol.

``docs/agent-protocol.md`` is the protocol; this module is evalkit's side of
it. evalkit holds the session: each request is a POST with the whole
transcript so far, the tools on offer and what is left of the session's
budgets, and the agent answers with its next message. When that message asks
for tools, the run's sandbox (``evalkit.tools.sandbox``) answers each call, in
order, and the next request carries the answers. When it asks for none, the
agent's turn is over: a task session ends there, and a conversation session
hands the dialogue to the simulated user (``evalkit.simuser``), who replies or
ends it.

Requests go through ``HttpTarget.post``: the endpoint is named in operator
config, its addresses are checked and pinned (no private addresses unless the
endpoint allows them, no DNS rebinding, no redirects, a capped response).

Every request has a ``request_id`` derived from its content, also sent as the
``Idempotency-Key`` header, so a retry after a timeout or a crash sends the
same id for the same step and an agent can answer it without doing the work
twice. Each answer is cached by a hash of the request the moment it arrives,
so a retried case or a resumed run asks only for the steps it is missing.
Reply text is redacted before it is cached and before the session goes on;
tool arguments are not, since the sandbox matches on them.

The budgets are per session: ``max_steps`` agent replies and ``max_tokens``
input plus output tokens as the agent reports them. A replayed reply counts
the tokens it was answered with, so a resumed session spends its budget
exactly as an uninterrupted one would.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any

import httpx2
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from evalkit.datasets import (
    Case,
    Conversation,
    Message,
    canonical_json,
    conversation_of,
    sha256_hex,
)
from evalkit.providers.tools import (
    AgentMessage,
    ToolDefinition,
    ToolOutcome,
    ToolUse,
    check_transcript,
)
from evalkit.ratelimit import RateLimiter
from evalkit.redaction import Redactor
from evalkit.simuser import Simulator
from evalkit.targets.base import Generation, Step, Target, TargetError, Turn, Usage
from evalkit.targets.conversation import (
    MAX_TURNS,
    NO_REPLY,
    TurnStore,
    call_limited,
    simulated_turn,
    simulator_hash,
)
from evalkit.targets.http import HttpEndpoint, HttpTarget, Resolver, system_resolver
from evalkit.targets.specs import HttpAgentTargetSpec, HttpTargetSpec
from evalkit.targets.template import render
from evalkit.tools.sandbox import ToolSandbox

PROTOCOL = "evalkit-agent/1"
# How a session can end, besides the agent's own stop reason and the
# simulated user's verdicts.
MAX_STEPS = "max_steps"
TOKEN_BUDGET = "token_budget"  # noqa: S105 - a stop reason, not a secret
END_TURN = "end_turn"  # the agent finished its turn and named no stop reason
# Limits on one reply, beyond the endpoint's response size cap.
MAX_TOOL_CALLS = 64
MAX_NAME = 128

_Name = Annotated[str, Field(min_length=1, max_length=MAX_NAME)]


class _Wire(BaseModel):
    # Strict types, but unknown fields are ignored: a later protocol version may add some.
    model_config = ConfigDict(strict=True, extra="ignore", frozen=True)


class _WireCall(_Wire):
    id: _Name
    name: _Name
    arguments: Any = Field(default_factory=dict)


class _WireMessage(_Wire):
    content: str = ""
    tool_calls: list[_WireCall] = Field(default_factory=list, max_length=MAX_TOOL_CALLS)

    @model_validator(mode="after")
    def _unique_ids(self) -> _WireMessage:
        ids = [c.id for c in self.tool_calls]
        if len(set(ids)) != len(ids):
            raise ValueError("tool call ids must be unique within a message")
        return self


class _WireUsage(_Wire):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0, allow_inf_nan=False)


class _WireReply(_Wire):
    protocol: str | None = None
    message: _WireMessage
    usage: _WireUsage = Field(default_factory=_WireUsage)
    stop_reason: str | None = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def _same_protocol(self) -> _WireReply:
        if self.protocol is not None and self.protocol != PROTOCOL:
            raise ValueError(f"answered in protocol {self.protocol!r}, not {PROTOCOL!r}")
        return self


@dataclass(frozen=True)
class AgentReply:
    """One reply of the agent: its text, the tools it asks for, and what it says
    it spent (``usage``; nothing was paid by this run when ``cached``)."""

    text: str
    tool_uses: tuple[ToolUse, ...]
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    usage: Usage
    cached: bool = False

    @property
    def tokens(self) -> int:
        """What the reply spends from the session's token budget, cached or not."""
        return self.input_tokens + self.output_tokens

    def to_cache(self) -> str:
        return canonical_json(
            {
                "text": self.text,
                "tool_uses": [u.to_json() for u in self.tool_uses],
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            }
        )

    @classmethod
    def from_cache(cls, text: str, stop_reason: str | None) -> AgentReply:
        """A cached reply; ValueError or TypeError when it is not one this version wrote."""
        raw = json.loads(text)
        if not isinstance(raw, dict) or not isinstance(raw.get("text"), str):
            raise TypeError("not a cached agent reply")
        uses, tokens_in, tokens_out = (
            raw.get("tool_uses"),
            raw.get("input_tokens"),
            raw.get("output_tokens"),
        )
        if not isinstance(uses, list) or not isinstance(tokens_in, int):
            raise TypeError("not a cached agent reply")
        if not isinstance(tokens_out, int):
            raise TypeError("not a cached agent reply")
        return cls(
            text=raw["text"],
            tool_uses=tuple(ToolUse.from_json(u) for u in uses),
            stop_reason=stop_reason,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            usage=Usage(),
            cached=True,
        )


def wire_messages(messages: Sequence[AgentMessage]) -> list[dict[str, Any]]:
    """The transcript as the protocol sends it."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.text})
        elif m.role == "assistant":
            calls = [{"id": u.id, "name": u.name, "arguments": u.arguments} for u in m.tool_uses]
            out.append({"role": "assistant", "content": m.text, "tool_calls": calls})
        else:
            results = [{"id": o.id, "name": o.name, **o.payload()} for o in m.outcomes]
            out.append({"role": "tool", "results": results})
    return out


def wire_tools(tools: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]


def request_id(body: dict[str, Any]) -> str:
    """The idempotency key of one step: the same request always has the same id."""
    return sha256_hex(canonical_json(["evalkit-agent-request", body]))[:32]


def parse_reply(document: Any, endpoint: str) -> AgentReply:
    """An agent's answer, or a non-retryable TargetError saying how it breaks the
    protocol: an agent that answers malformed once usually does again."""
    try:
        reply = _WireReply.model_validate(document)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "answer"
        raise TargetError(
            f"agent {endpoint!r} broke the session protocol: {where}: {first['msg']}",
            retryable=False,
        ) from None
    usage = reply.usage
    return AgentReply(
        text=reply.message.content,
        tool_uses=tuple(ToolUse(c.id, c.name, c.arguments) for c in reply.message.tool_calls),
        stop_reason=reply.stop_reason,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        usage=Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=usage.cost_usd,
        ),
    )


@dataclass
class _Session:
    """One case's session: the transcript the agent sees, what it has spent, and
    the rows it produced."""

    id: str
    messages: list[AgentMessage]
    sandbox: ToolSandbox
    steps: list[Step] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    replies: int = 0
    tokens: int = 0
    end: str | None = None

    def dialogue(self) -> list[Message]:
        """What the simulated user has seen: its own messages and the agent's
        finished replies, without tool calls in between."""
        return [
            Message(role=m.role, content=m.text)
            for m in self.messages
            if m.role == "user" or (m.role == "assistant" and not m.tool_uses)
        ]

    def last_reply(self) -> str:
        return next((m.text for m in reversed(self.messages) if m.role == "assistant"), "")


class HttpAgentTarget(Target):
    caches_cases = False

    def __init__(
        self,
        spec: HttpAgentTargetSpec,
        endpoint: HttpEndpoint,
        simulator: Simulator | None = None,
        *,
        simulator_endpoint: str | None = None,
        resolver: Resolver = system_resolver,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        if (spec.simulator is None) != (simulator is None):
            raise ValueError("give a simulator exactly when the spec names one")
        self.spec = spec
        self.http = HttpTarget(
            HttpTargetSpec(type="http", endpoint=spec.endpoint),
            endpoint,
            resolver=resolver,
            transport=transport,
        )
        # Requests share rate limits and cool-downs with HTTP targets on the endpoint.
        self.provider = self.http.provider
        self.simulator = simulator
        self.fixtures = spec.recorded()
        self.tools = [ToolDefinition.of(d) for d in spec.sandbox.definitions()]
        self.wire_tools = wire_tools(self.tools)
        # Whose answers a step cache entry holds: the agent at this endpoint, offered
        # these tools under these budgets. Sessions are named after it and the case.
        self.agent_hash = sha256_hex(
            canonical_json(
                {
                    "http_agent": spec.endpoint,
                    "endpoint": endpoint.fingerprint(),
                    "tools": self.wire_tools,
                    "max_steps": spec.max_steps,
                    "max_tokens": spec.max_tokens,
                }
            )
        )
        self.simulator_hash = (
            simulator_hash(spec.simulator, simulator_endpoint) if spec.simulator is not None else ""
        )
        self.store: TurnStore | None = None
        self.redactor = Redactor()
        self.fingerprint = self.redactor.config.fingerprint()
        self.limiter: RateLimiter | None = None
        self.simulator_limiter: RateLimiter | None = None
        self.max_retry_after = 120.0
        # Requests answered by the agent for this target, for tests and logs.
        self.fresh_steps = 0

    def attach(
        self,
        *,
        store: TurnStore | None,
        redactor: Redactor,
        limiter: RateLimiter | None = None,
        simulator_limiter: RateLimiter | None = None,
        max_retry_after: float = 120.0,
    ) -> None:
        """What the run provides: the step cache (None when the run does not use
        the cache), its redaction, the endpoint's limiter and the simulator's."""
        self.store = store
        self.redactor = redactor
        self.fingerprint = redactor.config.fingerprint()
        self.limiter = limiter
        self.simulator_limiter = simulator_limiter
        self.max_retry_after = max_retry_after

    def session_id(self, case: Case) -> str:
        """Stable for a case and an agent: a retry or a resume continues the same session."""
        return "s-" + sha256_hex(canonical_json(["session", self.agent_hash, case.id]))[:32]

    async def generate(self, case: Case) -> Generation:
        # A fresh sandbox per attempt: a retried case sees the same answers and
        # faults, so its requests, and with them the step cache, are the same.
        sandbox = ToolSandbox(self.spec.sandbox, self.fixtures, session=case.id)
        if self.simulator is None:
            task = render(self.spec.template, case.input)
            session = _Session(self.session_id(case), [AgentMessage("user", task)], sandbox)
            await self._agent_turn(session)
        else:
            conversation = conversation_of(case)
            opening = [AgentMessage(m.role, m.content) for m in conversation.messages]
            session = _Session(self.session_id(case), opening, sandbox)
            await self._dialogue(session, conversation, self.simulator)
        usage = sum((s.usage for s in session.steps), Usage()) + sum(
            (t.usage for t in session.turns if t.role == "user"), Usage()
        )
        return Generation(
            text=session.last_reply(),
            stop_reason=session.end,
            usage=usage,
            turns=tuple(session.turns),
            steps=tuple(session.steps),
        )

    async def _dialogue(
        self, session: _Session, conversation: Conversation, simulator: Simulator
    ) -> None:
        finished = 0
        while True:
            if not session.messages or session.messages[-1].role == "assistant":
                user, turn = await simulated_turn(
                    simulator,
                    self.simulator_hash,
                    conversation,
                    session.dialogue(),
                    store=self.store,
                    redactor=self.redactor,
                    limiter=self.simulator_limiter,
                    max_retry_after=self.max_retry_after,
                )
                session.turns.append(turn)
                if user.status != "continue":
                    session.end = user.status
                    return
                if finished >= conversation.max_turns:
                    session.end = MAX_TURNS
                    return
                session.messages.append(AgentMessage("user", turn.content))
            index = len(session.dialogue())
            first = len(session.steps)
            done = await self._agent_turn(session)
            steps = session.steps[first:]
            models = [s for s in steps if s.kind == "model"]
            if not models:
                return  # a budget ran out before the agent's turn began
            text = models[-1].text
            session.turns.append(
                Turn(
                    index,
                    "assistant",
                    text,
                    session.end,
                    cached=bool(models) and all(s.cached for s in models),
                    latency_ms=sum(s.latency_ms for s in steps),
                    usage=sum((s.usage for s in steps), Usage()),
                )
            )
            if not done:
                return  # a budget ran out mid-turn
            if not text:
                session.end = NO_REPLY
                return
            finished += 1
            session.end = None

    async def _agent_turn(self, session: _Session) -> bool:
        """Ask the agent for replies, answering its tool calls, until it replies
        without one (True) or a budget runs out (False, ``session.end`` says which)."""
        spec = self.spec
        while True:
            if session.replies >= spec.max_steps:
                session.end = MAX_STEPS
                return False
            if spec.max_tokens is not None and session.tokens >= spec.max_tokens:
                session.end = TOKEN_BUDGET
                return False
            reply = await self._step(session)
            session.replies += 1
            session.tokens += reply.tokens
            session.messages.append(AgentMessage("assistant", reply.text, reply.tool_uses))
            if not reply.tool_uses:
                session.end = reply.stop_reason or END_TURN
                return True
            outcomes = []
            for use in reply.tool_uses:
                call = await session.sandbox.call(use.name, use.arguments)
                session.steps.append(
                    Step(
                        index=len(session.steps),
                        kind="tool",
                        tool=use.name,
                        call_id=use.id,
                        arguments=use.arguments,
                        status=call.status,
                        result=call.result,
                        error=call.error,
                        source=call.source,
                        latency_ms=call.latency_ms,
                    )
                )
                content = call.error if call.is_error else call.result
                outcomes.append(ToolOutcome(use.id, use.name, content, call.is_error))
            session.messages.append(AgentMessage("tool", outcomes=tuple(outcomes)))

    def request(self, session: _Session) -> dict[str, Any]:
        """The next request of ``session``, without its ``request_id``."""
        spec = self.spec
        return {
            "protocol": PROTOCOL,
            "session_id": session.id,
            "step": session.replies,
            "messages": wire_messages(session.messages),
            "tools": self.wire_tools,
            "budget": {
                "steps_remaining": spec.max_steps - session.replies,
                "tokens_remaining": (
                    None if spec.max_tokens is None else spec.max_tokens - session.tokens
                ),
            },
        }

    async def _step(self, session: _Session) -> AgentReply:
        check_transcript(session.messages)
        body = self.request(session)
        key = sha256_hex(
            canonical_json(["http-agent-step", self.agent_hash, body, self.fingerprint])
        )
        hit = await self.store.get(key) if self.store is not None else None
        if hit is not None:
            try:
                cached = AgentReply.from_cache(hit.output, hit.stop_reason)
            except (ValueError, TypeError):
                cached = None  # not a reply this version wrote; ask again
            if cached is not None:
                session.steps.append(
                    Step(len(session.steps), "model", cached.text, cached.stop_reason, cached=True)
                )
                return cached
        body["request_id"] = rid = request_id(body)
        headers = {
            "Idempotency-Key": rid,
            "X-Evalkit-Protocol": PROTOCOL,
            "X-Evalkit-Session": session.id,
        }
        start = time.perf_counter()
        document = await call_limited(
            self.limiter, self.max_retry_after, lambda: self.http.post(body, headers)
        )
        latency = (time.perf_counter() - start) * 1000
        reply = parse_reply(document, self.spec.endpoint)
        self.fresh_steps += 1
        redacted = self.redactor.redact(reply.text)
        reply = AgentReply(
            text=redacted.text,
            tool_uses=reply.tool_uses,
            stop_reason=reply.stop_reason,
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            usage=reply.usage,
        )
        # A redaction that timed out left a placeholder: keep it out of the cache.
        if self.store is not None and not redacted.failed:
            await self.store.put(
                key, self.agent_hash, reply.to_cache(), reply.stop_reason, reply.usage
            )
        session.steps.append(
            Step(
                len(session.steps),
                "model",
                reply.text,
                reply.stop_reason,
                latency_ms=latency,
                usage=reply.usage,
            )
        )
        return reply

    async def aclose(self) -> None:
        await self.http.aclose()
