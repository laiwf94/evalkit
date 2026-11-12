"""What a run calls: anything that turns a case into text."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from evalkit.datasets import Case


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
            self.cost_usd + other.cost_usd,
        )


@dataclass(frozen=True)
class Turn:
    """One generated turn of a conversation, stored as a ``result_turns`` row.

    ``index`` is the turn's position in the whole dialogue, counting the
    messages the case opened with. ``usage`` is what this run paid for it:
    nothing when it came from the turn cache. A user turn's ``stop_reason`` is
    the simulated user's verdict (``continue``, ``goal_reached`` or ``gave_up``)."""

    index: int
    role: str
    content: str
    stop_reason: str | None = None
    cached: bool = False
    latency_ms: float = 0.0
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True)
class Generation:
    text: str
    stop_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    # The generated turns, for a conversation target; empty for a single answer.
    turns: tuple[Turn, ...] = ()


class TargetError(Exception):
    """A failed call. ``retryable`` failures go back through the retry policy."""

    def __init__(self, message: str, *, retryable: bool, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.retry_after = retry_after


class Target(ABC):
    #: Rate limits and cool-downs are shared by every target with the same provider key.
    provider: str
    #: Whether a case's whole output goes in the response cache. A target that
    #: caches its own steps (a conversation caches each turn) sets this False.
    caches_cases: bool = True

    @abstractmethod
    async def generate(self, case: Case) -> Generation:
        """Produce the output for one case, or raise TargetError."""

    async def aclose(self) -> None:
        """Release connections. Targets without resources need not override this."""
        return
