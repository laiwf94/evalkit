import asyncio
import itertools
import random
from datetime import UTC, datetime

import pytest

from evalkit.ratelimit import (
    Limiters,
    RateLimit,
    RealClock,
    RetryPolicy,
    RetryStats,
    call_with_retries,
    parse_retry_after,
)
from evalkit.targets import TargetError
from tests.fakeclock import FakeClock


def test_token_bucket_paces_requests() -> None:
    clock = FakeClock()
    limiter = Limiters({"p": RateLimit(requests_per_second=10, burst=2)}, clock).get("p")

    async def main() -> list[float]:
        times = []
        for _ in range(12):
            await limiter.acquire()
            times.append(clock.now)
        return times

    times = asyncio.run(main())
    assert times[:2] == [0.0, 0.0]  # the burst
    assert times[-1] == pytest.approx(1.0)  # 10 more at 10/s
    gaps = [b - a for a, b in itertools.pairwise(times[2:])]
    assert all(g == pytest.approx(0.1) for g in gaps)


def test_bucket_refills_while_idle() -> None:
    clock = FakeClock()
    limiter = Limiters({"p": RateLimit(requests_per_second=1, burst=3)}, clock).get("p")

    async def main() -> None:
        for _ in range(3):
            await limiter.acquire()
        clock.now += 10
        for _ in range(3):
            await limiter.acquire()

    asyncio.run(main())
    assert clock.sleeps == []


def test_unlimited_provider_and_shared_instances() -> None:
    limiters = Limiters(clock=FakeClock())
    assert limiters.get("x") is limiters.get("x")
    assert limiters.get("x") is not limiters.get("y")
    asyncio.run(limiters.get("x").acquire())


def test_cool_down_delays_every_worker() -> None:
    clock = FakeClock()
    limiter = Limiters(clock=clock).get("anthropic")
    limiter.cool_down(5.0)
    starts: list[float] = []

    async def worker() -> None:
        await limiter.acquire()
        starts.append(clock.now)

    async def main() -> None:
        await asyncio.gather(*(worker() for _ in range(8)))

    asyncio.run(main())
    assert len(starts) == 8
    assert min(starts) >= 5.0


def test_retry_after_is_honoured_and_attempts_counted() -> None:
    clock = FakeClock()
    limiter = Limiters(clock=clock).get("p")
    calls = []

    async def flaky() -> str:
        calls.append(clock.now)
        if len(calls) < 3:
            raise TargetError("429", retryable=True, retry_after=2.0)
        return "ok"

    stats = RetryStats()
    result = asyncio.run(
        call_with_retries(flaky, limiter=limiter, policy=RetryPolicy(), stats=stats)
    )
    assert result == "ok"
    assert stats.attempts == 3
    assert calls == [0.0, 2.0, 4.0]
    assert stats.waited_seconds == pytest.approx(4.0)


def test_retry_after_is_capped() -> None:
    clock = FakeClock()
    limiter = Limiters(clock=clock).get("p")
    calls = []

    async def slow_server() -> str:
        calls.append(clock.now)
        if len(calls) == 1:
            raise TargetError("503", retryable=True, retry_after=86_400)
        return "ok"

    policy = RetryPolicy(max_retry_after=10)
    asyncio.run(call_with_retries(slow_server, limiter=limiter, policy=policy, stats=RetryStats()))
    assert calls == [0.0, 10.0]


def test_exponential_backoff_with_full_jitter() -> None:
    clock = FakeClock()
    limiter = Limiters(clock=clock).get("p")

    async def always_fails() -> str:
        raise TargetError("500", retryable=True)

    stats = RetryStats()
    policy = RetryPolicy(max_attempts=5, base_delay=1.0, max_delay=4.0)
    with pytest.raises(TargetError):
        asyncio.run(
            call_with_retries(
                always_fails, limiter=limiter, policy=policy, stats=stats, rng=random.Random(1)
            )
        )
    assert stats.attempts == 5
    ceilings = [1.0, 2.0, 4.0, 4.0]
    assert len(clock.sleeps) == 4
    assert all(0 <= s <= c for s, c in zip(clock.sleeps, ceilings, strict=True))


def test_non_retryable_errors_fail_immediately() -> None:
    limiter = Limiters(clock=FakeClock()).get("p")

    async def bad_request() -> str:
        raise TargetError("400", retryable=False)

    stats = RetryStats()
    with pytest.raises(TargetError, match="400"):
        asyncio.run(
            call_with_retries(bad_request, limiter=limiter, policy=RetryPolicy(), stats=stats)
        )
    assert stats.attempts == 1


def test_other_exceptions_propagate() -> None:
    limiter = Limiters(clock=FakeClock()).get("p")

    async def bug() -> str:
        raise KeyError("x")

    with pytest.raises(KeyError):
        asyncio.run(
            call_with_retries(bug, limiter=limiter, policy=RetryPolicy(), stats=RetryStats())
        )


def test_real_clock() -> None:
    clock = RealClock()
    before = clock.monotonic()
    asyncio.run(clock.sleep(0.01))
    assert clock.monotonic() - before >= 0.009


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("3", 3.0),
        ("1.5", 1.5),
        ("-4", 0.0),
        ("nan", None),
        ("inf", None),
        ("soon", None),
        ("Wed, 21 Oct 2015 07:28:30 GMT", 30.0),
        ("Wed, 21 Oct 2015 07:27:00 GMT", 0.0),
        ("Wed, 21 Oct 2015 07:28:30", 30.0),
    ],
)
def test_parse_retry_after(value: str | None, expected: float | None) -> None:
    now = datetime(2015, 10, 21, 7, 28, 0, tzinfo=UTC)
    assert parse_retry_after(value, now=now) == expected


def test_parse_retry_after_uses_current_time() -> None:
    assert parse_retry_after("Thu, 01 Jan 2071 00:00:00 GMT") is not None
