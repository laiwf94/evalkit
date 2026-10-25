"""Per-provider rate limiting and the retry policy.

Every target call goes through ``call_with_retries``. Two mechanisms work
together:

* a token bucket per provider caps the steady request rate, and
* a provider-wide cool-down: when one worker gets a Retry-After, every worker
  calling that provider waits it out. Without this, 32 workers would each burn
  an attempt on the same 429 before backing off.

Failures without Retry-After back off locally with exponential delay and full
jitter, which spreads retries out instead of synchronising them.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from evalkit.targets.base import TargetError


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


class ProviderLimiter:
    def __init__(self, limit: RateLimit, clock: Clock) -> None:
        self.limit = limit
        self.clock = clock
        self._lock = asyncio.Lock()
        self._tokens = float(limit.burst)
        self._updated = clock.monotonic()
        self._cooldown_until = 0.0

    def cool_down(self, seconds: float) -> None:
        self._cooldown_until = max(self._cooldown_until, self.clock.monotonic() + seconds)

    async def acquire(self) -> None:
        while (wait := self._cooldown_until - self.clock.monotonic()) > 0:
            await self.clock.sleep(wait)
        rate = self.limit.requests_per_second
        if rate is None:
            return
        async with self._lock:
            now = self.clock.monotonic()
            self._tokens = min(self.limit.burst, self._tokens + (now - self._updated) * rate)
            self._updated = now
            if self._tokens < 1:
                await self.clock.sleep((1 - self._tokens) / rate)
                self._tokens = 1.0
                self._updated = self.clock.monotonic()
            self._tokens -= 1


class Limiters:
    """One limiter per provider key, created on first use."""

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
    limiter: ProviderLimiter,
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
                limiter.cool_down(min(exc.retry_after, policy.max_retry_after))
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
