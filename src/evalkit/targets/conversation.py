"""A model holding a conversation with a simulated user.

Each case is one dialogue (its input is a ``Conversation``). The target loops:
the simulated user speaks (or ends the dialogue: goal reached or gave up), the
assistant replies, until the user ends it or the assistant has replied
``max_turns`` times. The case's output is the assistant's last reply and its
stop reason says how the dialogue ended; every generated turn comes back in
``Generation.turns`` for the ``result_turns`` table.

Every turn is cached by a hash of the conversation so far, the moment it is
paid for, in the response cache table. A run killed mid-dialogue, or a case
retried after a transient failure, replays the turns it already has for
nothing and pays only for the rest; a rerun of an unchanged target pays for
nothing at all. The assistant's turns are keyed by the assistant's identity
alone, so changing who plays the user keeps the replies it already gave to the
same dialogue. Turns are redacted before they are cached and before the
dialogue continues, so a resumed conversation goes on from exactly the
transcript it would have had.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol

from evalkit.datasets import (
    Case,
    Conversation,
    Message,
    canonical_json,
    conversation_of,
    sha256_hex,
)
from evalkit.providers.base import Provider
from evalkit.ratelimit import RateLimiter
from evalkit.redaction import Redactor
from evalkit.simuser import Simulator, UserTurn
from evalkit.targets.base import Generation, Target, TargetError, Turn, Usage
from evalkit.targets.specs import ConversationTargetSpec, SimulatorSpec

# How a dialogue ended, besides the simulated user's own verdicts.
MAX_TURNS = "max_turns"
NO_REPLY = "no_reply"  # the assistant answered with no text (a refusal, a filter)


class TurnHit(Protocol):
    @property
    def output(self) -> str: ...

    @property
    def stop_reason(self) -> str | None: ...


class TurnStore(Protocol):
    """Where turns are cached (``evalkit.cache.TurnStore``)."""

    async def get(self, key: str) -> TurnHit | None: ...

    async def put(
        self, key: str, owner_hash: str, text: str, stop_reason: str | None, usage: Usage
    ) -> None: ...


def turn_key(kind: str, owner_hash: str, payload: Any, fingerprint: str) -> str:
    """The cache key of one turn: who speaks, who they are, what they have seen."""
    return sha256_hex(canonical_json(["turn", kind, owner_hash, payload, fingerprint]))


def _dump(messages: Sequence[Message]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


class ConversationTarget(Target):
    # The run's outer limiter only paces dialogues; each turn takes its own
    # provider's limiter (``attach``), as a prompt target's call would.
    provider = "conversation"
    caches_cases = False

    def __init__(
        self,
        spec: ConversationTargetSpec,
        assistant: Provider,
        simulator: Simulator,
        *,
        assistant_endpoint: str | None = None,
        simulator_endpoint: str | None = None,
    ) -> None:
        # An unknown model would run with no cost accounting; refuse it up front.
        assistant.prices.require(spec.model)
        self.spec = spec
        self.assistant = assistant
        self.simulator = simulator
        self.assistant_hash = sha256_hex(
            canonical_json({"assistant": spec.assistant_part(), "extra": assistant_endpoint})
        )
        self.simulator_hash = simulator_hash(spec.simulator, simulator_endpoint)
        self.store: TurnStore | None = None
        self.redactor = Redactor()
        self.fingerprint = self.redactor.config.fingerprint()
        self.assistant_limiter: RateLimiter | None = None
        self.simulator_limiter: RateLimiter | None = None
        self.max_retry_after = 120.0
        # Turns paid for by this target, for tests and logs.
        self.fresh_turns = 0

    def attach(
        self,
        *,
        store: TurnStore | None,
        redactor: Redactor,
        assistant_limiter: RateLimiter | None = None,
        simulator_limiter: RateLimiter | None = None,
        max_retry_after: float = 120.0,
    ) -> None:
        """Give the target what the run provides: the turn cache (None when the
        run does not use the cache), the run's redaction and per-turn limiters."""
        self.store = store
        self.redactor = redactor
        self.fingerprint = redactor.config.fingerprint()
        self.assistant_limiter = assistant_limiter
        self.simulator_limiter = simulator_limiter
        self.max_retry_after = max_retry_after

    async def generate(self, case: Case) -> Generation:
        conversation = conversation_of(case)
        transcript = list(conversation.messages)
        turns: list[Turn] = []
        replies = 0
        end: str
        while True:
            if not transcript or transcript[-1].role == "assistant":
                user, turn = await self._user_turn(conversation, transcript)
                turns.append(turn)
                if user.status != "continue":
                    end = user.status
                    break
                if replies >= conversation.max_turns:
                    end = MAX_TURNS
                    break
                transcript.append(Message(role="user", content=turn.content))
            turn = await self._assistant_turn(transcript)
            turns.append(turn)
            if not turn.content:
                end = NO_REPLY
                break
            transcript.append(Message(role="assistant", content=turn.content))
            replies += 1
        usage = sum((t.usage for t in turns), Usage())
        last = next((m.content for m in reversed(transcript) if m.role == "assistant"), "")
        return Generation(text=last, stop_reason=end, usage=usage, turns=tuple(turns))

    async def _assistant_turn(self, transcript: list[Message]) -> Turn:
        index = len(transcript)
        key = turn_key("assistant", self.assistant_hash, _dump(transcript), self.fingerprint)
        hit = await self.store.get(key) if self.store is not None else None
        if hit is not None:
            return Turn(index, "assistant", hit.output, hit.stop_reason, cached=True)
        spec = self.spec
        start = time.perf_counter()
        completion = await self._call(
            self.assistant_limiter,
            lambda: self.assistant.complete(
                model=spec.model,
                prompt=transcript,
                system=spec.system,
                max_tokens=spec.max_tokens,
                effort=spec.effort,
                fallbacks=spec.fallbacks,
            ),
        )
        latency = (time.perf_counter() - start) * 1000
        text = await self._keep(key, completion.text, completion.stop_reason, completion.usage)
        return Turn(
            index,
            "assistant",
            text,
            completion.stop_reason,
            latency_ms=latency,
            usage=completion.usage,
        )

    async def _user_turn(
        self, conversation: Conversation, transcript: list[Message]
    ) -> tuple[UserTurn, Turn]:
        user, turn = await simulated_turn(
            self.simulator,
            self.simulator_hash,
            conversation,
            transcript,
            store=self.store,
            redactor=self.redactor,
            limiter=self.simulator_limiter,
            max_retry_after=self.max_retry_after,
        )
        if not turn.cached and self.simulator.provider is not None:
            self.fresh_turns += 1
        return user, turn

    async def _keep(self, key: str, text: str, stop_reason: str | None, usage: Usage) -> str:
        """Redact a fresh reply and cache it at once, so it is never paid for twice."""
        self.fresh_turns += 1
        redacted = self.redactor.redact(text)
        # A redaction that timed out left a placeholder: keep it out of the cache.
        if self.store is not None and not redacted.failed:
            await self.store.put(key, self.assistant_hash, redacted.text, stop_reason, usage)
        return redacted.text

    async def _call[T](self, limiter: RateLimiter | None, call: Callable[[], Awaitable[T]]) -> T:
        return await call_limited(limiter, self.max_retry_after, call)


def simulator_hash(spec: SimulatorSpec, endpoint: str | None) -> str:
    """Whose turns a user turn cache entry holds: the simulator and where it calls."""
    return sha256_hex(
        canonical_json({"simulator": spec.model_dump(mode="json"), "extra": endpoint})
    )


async def simulated_turn(
    simulator: Simulator,
    owner_hash: str,
    conversation: Conversation,
    transcript: Sequence[Message],
    *,
    store: TurnStore | None,
    redactor: Redactor,
    limiter: RateLimiter | None,
    max_retry_after: float,
) -> tuple[UserTurn, Turn]:
    """The simulated user's next turn of ``conversation`` after ``transcript``:
    from the turn cache when it has one, else asked for, redacted and cached at
    once. Any target that talks to a simulated user uses this, so a user turn is
    never paid for twice."""
    index = len(transcript)
    fingerprint = redactor.config.fingerprint()
    seen = {
        "case": conversation.model_dump(mode="json", exclude={"messages"}),
        "transcript": _dump(transcript),
    }
    key = turn_key("user", owner_hash, seen, fingerprint)
    hit = await store.get(key) if store is not None else None
    if hit is not None:
        try:
            cached = UserTurn.from_json(hit.output)
        except ValueError:
            cached = None  # not a turn this version wrote; ask again
        if cached is not None:
            return cached, Turn(index, "user", cached.message, cached.status, cached=True)
    start = time.perf_counter()
    snapshot = list(transcript)
    user = await call_limited(
        limiter, max_retry_after, lambda: simulator.next_turn(conversation, snapshot)
    )
    latency = (time.perf_counter() - start) * 1000
    redacted = redactor.redact(user.message)
    user = UserTurn(user.status, redacted.text, user.usage)
    if store is not None and not redacted.failed:
        await store.put(key, owner_hash, user.to_json(), user.status, user.usage)
    turn = Turn(index, "user", user.message, user.status, latency_ms=latency, usage=user.usage)
    return user, turn


async def call_limited[T](
    limiter: RateLimiter | None, max_retry_after: float, call: Callable[[], Awaitable[T]]
) -> T:
    """One paid call under its provider's limiter. A Retry-After holds back every
    caller of that provider; retrying is the run's job, and a retried dialogue
    replays its cached turns for nothing."""
    if limiter is not None:
        await limiter.acquire()
    try:
        return await call()
    except TargetError as exc:
        if limiter is not None and exc.retry_after is not None:
            await limiter.cool_down(min(exc.retry_after, max_retry_after))
        raise
