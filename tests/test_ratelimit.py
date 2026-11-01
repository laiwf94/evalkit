import asyncio
import itertools
import math
import random
from datetime import UTC, datetime
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evalkit.ratelimit import (
    Gcra,
    GcraLimiter,
    Limiters,
    ProviderLimiter,
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
    starts: list[float] = []

    async def worker() -> None:
        await limiter.acquire()
        starts.append(clock.now)

    async def main() -> None:
        await limiter.cool_down(5.0)
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


def test_gcra_arithmetic() -> None:
    assert Gcra.of(RateLimit()) == Gcra(0.0, 0.0)
    g = Gcra.of(RateLimit(requests_per_second=4, burst=3))
    assert g == Gcra(0.25, 0.5)
    # A fresh limiter admits at once; a full burst pushes the next start out.
    assert g.admit(tat=0.0, cooldown_until=0.0, now=10.0) == (10.0, 10.25)
    assert g.admit(tat=10.75, cooldown_until=0.0, now=10.0) == (10.25, 11.0)
    # A cool-down holds the start back and leaves no burst behind it.
    tat, until = g.cool_down(tat=10.25, cooldown_until=0.0, until=20.0)
    assert (tat, until) == (20.5, 20.0)
    assert g.admit(tat, until, now=10.0) == (20.0, 20.75)
    assert g.admit(20.75, until, now=10.0) == (20.25, 21.0)
    assert g.refund(21.0, until) == 20.75
    assert g.refund(20.6, until) == 20.5


class ManualClock:
    """Sleepers stay asleep until the test moves time past their deadline."""

    def __init__(self) -> None:
        self.now = 0.0
        self._sleepers: list[tuple[float, asyncio.Future[None]]] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        future = asyncio.get_running_loop().create_future()
        self._sleepers.append((self.now + seconds, future))
        await future

    async def advance_to(self, when: float) -> None:
        """Move time forward, waking sleepers in deadline order."""
        while True:
            await asyncio.sleep(0)
            due = [s for s in self._sleepers if s[0] <= when and not s[1].done()]
            if not due:
                break
            deadline, future = min(due, key=lambda s: s[0])
            self.now = max(self.now, deadline)
            future.set_result(None)
        self.now = max(self.now, when)
        await asyncio.sleep(0)

    async def run_until_done(self, *tasks: asyncio.Task[Any]) -> None:
        while not all(t.done() for t in tasks):
            # A task waiting on a database thread is not asleep yet; let it get
            # there before moving time, or it would wake late.
            for _ in range(5000):
                pending = [s[0] for s in self._sleepers if not s[1].done()]
                if len(pending) >= sum(not t.done() for t in tasks):
                    break
                await asyncio.sleep(0.001)
            await self.advance_to(min(pending) if pending else self.now)


def test_a_cancelled_caller_hands_its_slot_back() -> None:
    clock = ManualClock()
    limiter = Limiters({"p": RateLimit(requests_per_second=1)}, clock).get("p")
    starts: list[float] = []

    async def call() -> None:
        await limiter.acquire()
        starts.append(clock.now)

    async def main() -> None:
        await call()  # slot 0
        waiting = asyncio.create_task(call())  # slot 1
        await asyncio.sleep(0)
        last = asyncio.create_task(call())  # slot 2, the latest
        await asyncio.sleep(0)
        last.cancel()
        with pytest.raises(asyncio.CancelledError):
            await last
        assert limiter.refunds == 1
        again = asyncio.create_task(call())  # takes slot 2 again, not slot 3
        await clock.run_until_done(waiting, again)

    asyncio.run(main())
    assert sorted(starts) == [0.0, 1.0, 2.0]


def test_only_the_latest_slot_can_be_handed_back() -> None:
    clock = ManualClock()
    limiter = Limiters({"p": RateLimit(requests_per_second=1)}, clock).get("p")
    starts: list[float] = []

    async def call() -> None:
        await limiter.acquire()
        starts.append(clock.now)

    async def main() -> None:
        await call()
        middle = asyncio.create_task(call())  # slot 1
        await asyncio.sleep(0)
        last = asyncio.create_task(call())  # slot 2
        await asyncio.sleep(0)
        middle.cancel()
        with pytest.raises(asyncio.CancelledError):
            await middle
        # Handing slot 1 back now would let the next caller share slot 2.
        assert limiter.refunds == 0
        after = asyncio.create_task(call())
        await clock.run_until_done(last, after)

    asyncio.run(main())
    assert starts == [0.0, 2.0, 3.0]


def test_a_cool_down_reaches_callers_already_waiting_for_a_slot() -> None:
    clock = ManualClock()
    limiter = Limiters({"p": RateLimit(requests_per_second=2, burst=2)}, clock).get("p")
    starts: list[float] = []

    async def call() -> None:
        await limiter.acquire()
        starts.append(clock.now)

    async def main() -> None:
        await call()
        await call()  # the burst is spent
        waiting = [asyncio.create_task(call()) for _ in range(2)]  # slots at 0.5 and 1.0
        await clock.advance_to(0.25)
        await limiter.cool_down(10.0)  # a 429 arrives while they sleep
        await clock.run_until_done(*waiting)

    asyncio.run(main())
    # Nothing starts inside the cool-down, and no burst follows it.
    assert starts[:2] == [0.0, 0.0]
    assert starts[2:] == [pytest.approx(10.25), pytest.approx(10.75)]


def test_a_refund_after_a_cool_down_keeps_the_burst_spent() -> None:
    clock = ManualClock()
    limiter = Limiters({"p": RateLimit(requests_per_second=1, burst=3)}, clock).get("p")

    async def main() -> None:
        await limiter.cool_down(5.0)
        first = asyncio.create_task(limiter.acquire())
        await clock.run_until_done(first)
        assert clock.now == 5.0
        late = asyncio.create_task(limiter.acquire())  # slot 6
        await asyncio.sleep(0)
        late.cancel()
        with pytest.raises(asyncio.CancelledError):
            await late
        again = asyncio.create_task(limiter.acquire())
        await clock.run_until_done(again)
        assert clock.now == 6.0

    asyncio.run(main())
    assert limiter.refunds == 1


def conforms(starts: list[float], gcra: Gcra, burst: int) -> bool:
    """No window of w seconds holds more than burst + w / interval starts."""
    s = sorted(starts)
    return all(
        s[j] - s[i] >= (j - i - burst + 1) * gcra.interval - 1e-6
        for i in range(len(s))
        for j in range(i + 1, len(s))
    )


Op = tuple[str, float]
OPS = st.lists(
    st.tuples(
        st.sampled_from(["reserve", "reserve", "reserve", "refund", "cool_down"]),
        st.floats(min_value=0, max_value=2, allow_nan=False),
    ),
    max_size=60,
)
LIMITS = st.builds(
    RateLimit,
    requests_per_second=st.floats(min_value=0.5, max_value=50),
    burst=st.integers(min_value=1, max_value=8),
)


async def drive(limiter: GcraLimiter, clock: FakeClock, ops: list[Op], rng: random.Random) -> None:
    """Apply random admissions, refunds and cool-downs at random times, and check
    that the starts never exceed the rate and never fall inside a cool-down."""
    held: list[Any] = []
    cooling_until = -math.inf
    for op, value in ops:
        clock.now += value
        if op == "reserve":
            r = await limiter._reserve()
            assert r.start >= r.now - 1e-9
            assert r.start >= cooling_until - 1e-9
            held.append(r)
        elif op == "refund" and held:
            r = held[rng.randrange(len(held))]
            if await limiter._refund(r):
                held.remove(r)
        elif op == "cool_down":
            until, now = await limiter._announce(value * 3)
            cooling_until = max(cooling_until, clock.now + value * 3)
            assert until - now == pytest.approx(cooling_until - clock.now)
    assert conforms([r.start for r in held], limiter.gcra, limiter.limit.burst)


@settings(max_examples=150, deadline=None)
@given(limit=LIMITS, ops=OPS, seed=st.integers(0, 2**16))
def test_in_memory_limiter_never_exceeds_its_rate(
    limit: RateLimit, ops: list[Op], seed: int
) -> None:
    clock = FakeClock()
    limiter = ProviderLimiter(limit, clock)
    asyncio.run(drive(limiter, clock, ops, random.Random(seed)))


@settings(max_examples=60, deadline=None)
@given(limit=LIMITS, ops=OPS, seed=st.integers(0, 2**16))
def test_acquire_never_exceeds_the_rate(limit: RateLimit, ops: list[Op], seed: int) -> None:
    """End to end through acquire(): concurrent callers, cancellations and cool-downs."""
    clock = ManualClock()
    limiter = ProviderLimiter(limit, clock)
    rng = random.Random(seed)
    starts: list[float] = []
    blocked: list[tuple[float, float]] = []

    async def call() -> None:
        await limiter.acquire()
        starts.append(clock.now)

    async def main() -> None:
        tasks: list[asyncio.Task[None]] = []
        for op, value in ops:
            await clock.advance_to(clock.now + value / 4)
            if op == "reserve":
                tasks.append(asyncio.create_task(call()))
            elif op == "refund" and tasks:
                tasks[rng.randrange(len(tasks))].cancel()
            elif op == "cool_down":
                blocked.append((clock.now, clock.now + value))
                await limiter.cool_down(value)
            await asyncio.sleep(0)
        await clock.run_until_done(*tasks)

    asyncio.run(main())
    assert conforms(starts, limiter.gcra, limit.burst)
    for begin, end in blocked:
        assert not any(begin + 1e-9 < t < end - 1e-9 for t in starts)
