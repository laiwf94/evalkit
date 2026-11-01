"""Per-provider rate limiting and the retry policy.

Every target call goes through ``call_with_retries``. Two mechanisms work
together:

* a GCRA limiter per provider (the generic cell rate algorithm: one number, the
  theoretical arrival time, per provider) caps the steady request rate and
  allows a burst, and
* a provider-wide cool-down: when one worker gets a Retry-After, every worker
  calling that provider waits it out. Without this, 32 workers would each burn
  an attempt on the same 429 before backing off.

Limiters implement the ``RateLimiter`` protocol. ``ProviderLimiter`` keeps its
state in memory and serves one process; ``ratelimit_pg.PgRateLimiter`` keeps the
same state in a database row, so every process on one database shares the rate
and the cool-downs. Both run the arithmetic in ``Gcra``.

An admission reserves a slot and then sleeps until it. A slot that is not used,
because the caller was cancelled or a cool-down started while it slept, is
handed back if no later slot has been reserved since; otherwise it is simply
lost, which errs on the provider's side.

Failures without Retry-After back off locally with exponential delay and full
jitter, which spreads retries out instead of synchronising them.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from evalkit import metrics
from evalkit.targets.base import TargetError

log = logging.getLogger(__name__)


class Clock(Protocol):
    def monotonic(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...


class RealClock:
    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class RateLimit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    requests_per_second: float | None = Field(default=None, gt=0, le=10_000)
    burst: int = Field(default=1, ge=1, le=10_000)


class RetryPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_attempts: int = Field(default=5, ge=1, le=20)
    base_delay: float = Field(default=0.5, ge=0, le=60)
    max_delay: float = Field(default=30.0, ge=0, le=600)
    # A misbehaving server must not be able to park the run for an hour.
    max_retry_after: float = Field(default=120.0, ge=0, le=3600)


class RateLimiter(Protocol):
    """What ``call_with_retries`` needs from a limiter."""

    clock: Clock

    async def acquire(self) -> None:
        """Return when one call may be made: after any cool-down, at the next free slot."""

    async def cool_down(self, seconds: float) -> None:
        """Hold every caller of this provider back for ``seconds`` (a Retry-After)."""


@dataclass(frozen=True)
class Gcra:
    """The generic cell rate algorithm, as pure arithmetic on two numbers.

    ``tat`` is the theoretical arrival time: when the bucket would next be empty
    if calls kept coming at exactly the configured rate. ``cooldown_until`` is the
    end of the latest Retry-After. A call arriving at ``now`` may start at
    ``max(tat - tolerance, now, cooldown_until)``. Any sequence of start times this
    produces has at most ``burst + w * rate`` starts in any window of ``w`` seconds.
    """

    interval: float  # seconds per request; 0 means unlimited
    tolerance: float  # how far ahead of the rate a burst may run: (burst - 1) * interval

    @classmethod
    def of(cls, limit: RateLimit) -> Gcra:
        if limit.requests_per_second is None:
            return cls(0.0, 0.0)
        interval = 1.0 / limit.requests_per_second
        return cls(interval, (limit.burst - 1) * interval)

    def admit(self, tat: float, cooldown_until: float, now: float) -> tuple[float, float]:
        """The start of the reserved slot and the new ``tat``."""
        start = max(tat - self.tolerance, now, cooldown_until)
        return start, max(tat, start) + self.interval

    def cool_down(self, tat: float, cooldown_until: float, until: float) -> tuple[float, float]:
        """The new ``tat`` and ``cooldown_until``. Pushing ``tat`` past the end of the
        cool-down stops the whole burst from going out the moment it ends."""
        return max(tat, until + self.tolerance), max(cooldown_until, until)

    def refund(self, tat: float, cooldown_until: float) -> float:
        """``tat`` with the latest reservation handed back."""
        return max(tat - self.interval, cooldown_until + self.tolerance)


@dataclass(frozen=True)
class Reservation:
    """A reserved slot, in the limiter's own time base."""

    start: float
    now: float
    tat: float  # the tat this reservation left behind; a refund only applies while it is current
    cooldown_until: float

    @property
    def wait(self) -> float:
        return max(0.0, self.start - self.now)


class GcraLimiter(ABC):
    """Admission logic shared by the in-memory and database limiters.

    Backends keep ``tat`` and ``cooldown_until`` in their own time base (monotonic
    seconds in memory, the database server's clock for the shared limiter) and
    report that base's ``now`` with every answer, so callers only ever sleep for
    differences and clocks on different machines never need to agree.
    """

    #: How old this process's knowledge of the cool-down may get before a call
    #: re-reads it. Infinite for a limiter that sees every cool-down itself.
    peek_seconds: float = math.inf

    def __init__(self, limit: RateLimit, clock: Clock) -> None:
        self.limit = limit
        self.clock = clock
        self.gcra = Gcra.of(limit)
        self.refunds = 0
        self._cooldown_deadline = -math.inf  # on ``clock``
        self._known_at = -math.inf  # on ``clock``
        self._peek_lock = asyncio.Lock()
        self._background: asyncio.Task[None] | None = None

    @abstractmethod
    async def _reserve(self) -> Reservation: ...

    @abstractmethod
    async def _refund(self, reservation: Reservation) -> bool:
        """Hand ``reservation`` back if it is still the latest; report whether it was."""

    @abstractmethod
    async def _announce(self, seconds: float) -> tuple[float, float]:
        """Record a cool-down; return the backend's (cooldown_until, now)."""

    @abstractmethod
    async def _peek(self) -> tuple[float, float]:
        """Read the backend's (cooldown_until, now) without reserving anything."""

    def _learn(self, cooldown_until: float, now: float) -> None:
        self._known_at = self.clock.monotonic()
        self._cooldown_deadline = max(
            self._cooldown_deadline, self._known_at + cooldown_until - now
        )

    def _cooling(self) -> float:
        return self._cooldown_deadline - self.clock.monotonic()

    def _stale(self) -> bool:
        return self.clock.monotonic() - self._known_at >= self.peek_seconds

    async def _refresh(self) -> None:
        """Re-read the cool-down if this process's knowledge of it is stale."""
        if math.isinf(self.peek_seconds) or not self._stale():
            return
        async with self._peek_lock:
            # Concurrent callers share one read.
            if self._stale():
                self._learn(*await self._peek())

    def _refresh_soon(self) -> None:
        """Like ``_refresh``, without holding the caller up: an unlimited provider's
        callers must not queue behind a database read every ``peek_seconds``."""
        if math.isinf(self.peek_seconds) or not self._stale():
            return
        if self._background is None or self._background.done():
            self._background = asyncio.create_task(self._refresh_logged())

    async def _refresh_logged(self) -> None:
        try:
            await self._refresh()
        except Exception:
            log.warning("could not read the shared rate-limit state", exc_info=True)

    async def _wait_out_cool_down(self) -> None:
        while True:
            if self.gcra.interval == 0:
                self._refresh_soon()
            else:
                await self._refresh()
            wait = self._cooling()
            if wait <= 0:
                return
            await self.clock.sleep(wait)

    async def _give_back(self, reservation: Reservation) -> None:
        if await self._refund(reservation):
            self.refunds += 1
            metrics.RATE_LIMIT_REFUNDS.inc()

    async def cool_down(self, seconds: float) -> None:
        self._learn(*await self._announce(seconds))

    async def acquire(self) -> None:
        while True:
            await self._wait_out_cool_down()
            if self.gcra.interval == 0:
                return
            reservation = await self._reserve()
            self._learn(reservation.cooldown_until, reservation.now)
            try:
                if reservation.wait > 0:
                    await self.clock.sleep(reservation.wait)
                    await self._refresh()
            except BaseException:
                # Cancelled while waiting for the slot: nobody will use it.
                await asyncio.shield(self._give_back(reservation))
                raise
            if self._cooling() <= 0:
                return
            # A Retry-After arrived while this caller slept; its slot falls inside
            # the cool-down, so hand it back and queue again behind the cool-down.
            await self._give_back(reservation)


class ProviderLimiter(GcraLimiter):
    """A GCRA limiter whose state lives in this process."""

    def __init__(self, limit: RateLimit, clock: Clock) -> None:
        super().__init__(limit, clock)
        self._tat = -math.inf
        self._cooldown_until = -math.inf

    async def _reserve(self) -> Reservation:
        now = self.clock.monotonic()
        start, self._tat = self.gcra.admit(self._tat, self._cooldown_until, now)
        return Reservation(start, now, self._tat, self._cooldown_until)

    async def _refund(self, reservation: Reservation) -> bool:
        if self._tat != reservation.tat:
            return False
        self._tat = self.gcra.refund(self._tat, self._cooldown_until)
        return True

    async def _announce(self, seconds: float) -> tuple[float, float]:
        now = self.clock.monotonic()
        self._tat, self._cooldown_until = self.gcra.cool_down(
            self._tat, self._cooldown_until, now + seconds
        )
        return self._cooldown_until, now

    async def _peek(self) -> tuple[float, float]:  # pragma: no cover - peek_seconds is inf
        return self._cooldown_until, self.clock.monotonic()


class Limiters:
    """One in-memory limiter per provider key, created on first use."""

    def __init__(
        self, limits: dict[str, RateLimit] | None = None, clock: Clock | None = None
    ) -> None:
        self._limits = limits or {}
        self._clock = clock or RealClock()
        self._limiters: dict[str, ProviderLimiter] = {}

    def get(self, provider: str) -> ProviderLimiter:
        limiter = self._limiters.get(provider)
        if limiter is None:
            limit = self._limits.get(provider) or RateLimit()
            limiter = self._limiters[provider] = ProviderLimiter(limit, self._clock)
        return limiter


@dataclass
class RetryStats:
    attempts: int = 0
    waited_seconds: float = 0.0


async def call_with_retries[T](
    fn: Callable[[], Awaitable[T]],
    *,
    limiter: RateLimiter,
    policy: RetryPolicy,
    stats: RetryStats,
    rng: random.Random | None = None,
) -> T:
    """Run ``fn`` under the limiter, retrying retryable TargetErrors per ``policy``.

    ``stats.attempts`` is filled in on success and on failure.
    """
    rng = rng or random.Random()  # noqa: S311 - jitter, not security
    clock = limiter.clock
    for attempt in range(1, policy.max_attempts + 1):
        stats.attempts = attempt
        start = clock.monotonic()
        await limiter.acquire()
        stats.waited_seconds += clock.monotonic() - start
        try:
            return await fn()
        except TargetError as exc:
            if not exc.retryable or attempt == policy.max_attempts:
                raise
            if exc.retry_after is not None:
                await limiter.cool_down(min(exc.retry_after, policy.max_retry_after))
            else:
                ceiling = min(policy.max_delay, policy.base_delay * 2 ** (attempt - 1))
                delay = rng.uniform(0, ceiling)
                stats.waited_seconds += delay
                await clock.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def parse_retry_after(value: str | None, now: datetime | None = None) -> float | None:
    """Seconds to wait from a Retry-After header (delta-seconds or HTTP-date)."""
    if value is None or not value.strip():
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        seconds = (when - (now or datetime.now(UTC))).total_seconds()
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)
