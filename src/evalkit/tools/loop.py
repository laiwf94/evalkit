"""One tool loop for every provider that can call tools.

``ToolLoop.run`` gives the model a task and the sandbox's tools, then repeats:
ask the model for its next reply; if it asks for tools, answer each call from
the sandbox, in order, and send the answers back. It ends when the model
replies without asking for a tool, when a reply is cut off at the output limit
(a tool call in it may be incomplete, so none is run), or after ``max_steps``
replies (stop reason ``max_steps``). Every reply and every call becomes a
``Step``; the case's output is the model's last reply.

The loop speaks only the provider-neutral types of ``evalkit.providers.tools``;
each provider maps them onto its own API, so Claude, OpenAI-compatible servers
and Gemini run exactly the same loop.

Each model reply is cached by a hash of the transcript so far, the moment it is
paid for, as conversation turns are. The sandbox answers the same calls the
same way on every attempt (its faults are seeded by the case), so a case
retried after a transient failure, or a run resumed after a crash, replays the
replies it has for nothing and pays only for the rest. Reply text is redacted
before it is cached and before the loop goes on; tool arguments are not, since
the sandbox matches on them.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from evalkit.datasets import canonical_json, sha256_hex
from evalkit.providers.tools import (
    AgentMessage,
    ToolCompletion,
    ToolDefinition,
    ToolOutcome,
    ToolProvider,
    ToolUse,
)
from evalkit.ratelimit import RateLimiter
from evalkit.redaction import Redactor
from evalkit.targets.base import Step, TargetError, Usage
from evalkit.tools.sandbox import ToolSandbox

MAX_STEPS = "max_steps"


class StepHit(Protocol):
    @property
    def output(self) -> str: ...

    @property
    def stop_reason(self) -> str | None: ...


class StepStore(Protocol):
    """Where model replies are cached (``evalkit.cache.TurnStore``)."""

    async def get(self, key: str) -> StepHit | None: ...

    async def put(
        self, key: str, owner_hash: str, text: str, stop_reason: str | None, usage: Usage
    ) -> None: ...


@dataclass(frozen=True)
class LoopResult:
    text: str
    stop_reason: str | None
    usage: Usage
    steps: tuple[Step, ...] = field(default=())


def step_key(owner_hash: str, messages: Sequence[AgentMessage], fingerprint: str) -> str:
    """The cache key of the model's reply to ``messages``."""
    payload = [m.to_json() for m in messages]
    return sha256_hex(canonical_json(["agent-step", owner_hash, payload, fingerprint]))


def _reply_json(reply: ToolCompletion) -> str:
    return canonical_json(
        {
            "text": reply.text,
            "tool_uses": [u.to_json() for u in reply.tool_uses],
            "reasoning": list(reply.reasoning),
            "model": reply.model,
        }
    )


def _reply_from_json(text: str, stop_reason: str | None) -> ToolCompletion:
    """A cached reply; ValueError or TypeError when it is not one this version wrote."""
    raw = json.loads(text)
    if not isinstance(raw, dict) or not isinstance(raw.get("text"), str):
        raise TypeError("not a cached agent step")
    uses = raw.get("tool_uses")
    reasoning = raw.get("reasoning")
    if not isinstance(uses, list) or not isinstance(reasoning, list):
        raise TypeError("not a cached agent step")
    if not all(isinstance(r, dict) for r in reasoning):
        raise ValueError("not a cached agent step")
    return ToolCompletion(
        text=raw["text"],
        tool_uses=tuple(ToolUse.from_json(u) for u in uses),
        stop_reason=stop_reason,
        usage=Usage(),
        model=str(raw.get("model", "")),
        reasoning=tuple(reasoning),
    )


class ToolLoop:
    """A model working on a task with tools. One per target; ``run`` once per case."""

    def __init__(
        self,
        provider: ToolProvider,
        *,
        model: str,
        tools: Sequence[ToolDefinition],
        owner_hash: str,
        system: str = "",
        max_tokens: int = 4096,
        effort: str = "medium",
        fallbacks: bool = True,
        max_steps: int = 20,
    ) -> None:
        self.provider = provider
        self.model = model
        self.tools = tuple(tools)
        self.owner_hash = owner_hash
        self.system = system
        self.max_tokens = max_tokens
        self.effort = effort
        self.fallbacks = fallbacks
        self.max_steps = max_steps
        self.store: StepStore | None = None
        self.redactor = Redactor()
        self.fingerprint = self.redactor.config.fingerprint()
        self.limiter: RateLimiter | None = None
        self.max_retry_after = 120.0
        # Model replies paid for by this loop, for tests and logs.
        self.fresh_steps = 0

    def attach(
        self,
        *,
        store: StepStore | None,
        redactor: Redactor,
        limiter: RateLimiter | None = None,
        max_retry_after: float = 120.0,
    ) -> None:
        """What the run provides: the step cache (None when the run does not use
        the cache), its redaction and the provider's limiter."""
        self.store = store
        self.redactor = redactor
        self.fingerprint = redactor.config.fingerprint()
        self.limiter = limiter
        self.max_retry_after = max_retry_after

    async def run(self, task: str, sandbox: ToolSandbox) -> LoopResult:
        messages = [AgentMessage("user", task)]
        steps: list[Step] = []
        reply: ToolCompletion | None = None
        stop: str | None = MAX_STEPS
        for _ in range(self.max_steps):
            reply, step = await self._model_step(messages, len(steps))
            steps.append(step)
            messages.append(
                AgentMessage("assistant", reply.text, reply.tool_uses, reasoning=reply.reasoning)
            )
            if not reply.tool_uses or reply.truncated:
                stop = reply.stop_reason
                break
            outcomes = []
            for use in reply.tool_uses:
                call = await sandbox.call(use.name, use.arguments)
                steps.append(
                    Step(
                        index=len(steps),
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
            messages.append(AgentMessage("tool", outcomes=tuple(outcomes)))
        usage = sum((s.usage for s in steps), Usage())
        text = reply.text if reply is not None else ""
        return LoopResult(text=text, stop_reason=stop, usage=usage, steps=tuple(steps))

    async def _model_step(
        self, messages: list[AgentMessage], index: int
    ) -> tuple[ToolCompletion, Step]:
        key = step_key(self.owner_hash, messages, self.fingerprint)
        hit = await self.store.get(key) if self.store is not None else None
        if hit is not None:
            try:
                cached = _reply_from_json(hit.output, hit.stop_reason)
            except (ValueError, TypeError):
                cached = None  # not a reply this version wrote; ask again
            if cached is not None:
                step = Step(index, "model", cached.text, cached.stop_reason, cached=True)
                return cached, step
        start = time.perf_counter()
        reply = await self._call(
            lambda: self.provider.complete_tools(
                model=self.model,
                messages=messages,
                tools=self.tools,
                system=self.system,
                max_tokens=self.max_tokens,
                effort=self.effort,
                fallbacks=self.fallbacks,
            )
        )
        latency = (time.perf_counter() - start) * 1000
        self.fresh_steps += 1
        redacted = self.redactor.redact(reply.text)
        reply = ToolCompletion(
            text=redacted.text,
            tool_uses=reply.tool_uses,
            stop_reason=reply.stop_reason,
            usage=reply.usage,
            model=reply.model,
            reasoning=reply.reasoning,
            refusal_category=reply.refusal_category,
        )
        # A redaction that timed out left a placeholder: keep it out of the cache.
        if self.store is not None and not redacted.failed:
            await self.store.put(
                key, self.owner_hash, _reply_json(reply), reply.stop_reason, reply.usage
            )
        step = Step(
            index,
            "model",
            reply.text,
            reply.stop_reason,
            latency_ms=latency,
            usage=reply.usage,
        )
        return reply, step

    async def _call(self, call: Callable[[], Awaitable[ToolCompletion]]) -> ToolCompletion:
        """One paid call under the provider's limiter. A Retry-After holds back
        every caller of that provider; retrying is the run's job, and a retried
        case replays its cached replies for nothing."""
        limiter = self.limiter
        if limiter is not None:
            await limiter.acquire()
        try:
            return await call()
        except TargetError as exc:
            if limiter is not None and exc.retry_after is not None:
                await limiter.cool_down(min(exc.retry_after, self.max_retry_after))
            raise


def owner_hash(identity: Any, tools: Sequence[ToolDefinition]) -> str:
    """Whose replies a step cache entry holds: the model's settings and endpoint,
    and the tools it was offered."""
    offered = [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]
    return sha256_hex(canonical_json({"agent": identity, "tools": offered}))
