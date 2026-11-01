"""The database limiter, run on SQLite with a fake clock standing in for the server's.

The same SQL runs on PostgreSQL in tests/e2e/test_shared_limiter.py.
"""

import asyncio
import itertools
import random
import time
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker

from evalkit.config import Settings
from evalkit.db import make_session_factory
from evalkit.db.models import RateLimitState
from evalkit.ratelimit import (
    RateLimit,
    Reservation,
    RetryPolicy,
    RetryStats,
    call_with_retries,
)
from evalkit.ratelimit_pg import PgRateLimiter, limiter_key
from evalkit.runner import Runner
from evalkit.targets import TargetError
from tests.fakeclock import FakeClock
from tests.test_ratelimit import LIMITS, OPS, ManualClock, Op, conforms, drive

keys = itertools.count()


def shared(
    sessions: sessionmaker[Session], clock: FakeClock | ManualClock, limit: RateLimit, **kw: float
) -> PgRateLimiter:
    return PgRateLimiter(sessions, "p", limit, clock, now=clock.monotonic, **kw)


async def settle(limiter: PgRateLimiter) -> None:
    while limiter._background is not None and not limiter._background.done():
        await asyncio.sleep(0.001)


def test_database_limiter_never_exceeds_its_rate(engine: Engine) -> None:
    sessions = make_session_factory(engine)

    @settings(max_examples=60, deadline=None)
    @given(limit=LIMITS, ops=OPS, seed=st.integers(0, 2**16))
    def check(limit: RateLimit, ops: list[Op], seed: int) -> None:
        clock = FakeClock()
        clock.now = 1000.0
        limiter = PgRateLimiter(
            sessions, f"k{next(keys)}", limit, clock, now=clock.monotonic, peek_seconds=0
        )
        asyncio.run(drive(limiter, clock, ops, random.Random(seed)))

    check()


def test_processes_share_one_rate(sessions: sessionmaker[Session]) -> None:
    clock = ManualClock()
    limit = RateLimit(requests_per_second=10)
    a, b = shared(sessions, clock, limit), shared(sessions, clock, limit)
    starts: list[float] = []

    async def call(limiter: PgRateLimiter) -> None:
        await limiter.acquire()
        starts.append(clock.now)

    async def main() -> None:
        tasks = [asyncio.create_task(call(x)) for x in (a, b) * 5]
        await clock.run_until_done(*tasks)

    asyncio.run(main())
    assert starts == pytest.approx([i / 10 for i in range(10)])


def test_a_cool_down_in_one_process_pauses_the_others(sessions: sessionmaker[Session]) -> None:
    clock = ManualClock()
    limit = RateLimit(requests_per_second=2)
    a = shared(sessions, clock, limit)
    b = shared(sessions, clock, limit, peek_seconds=0.1)
    starts: list[float] = []

    async def call() -> None:
        await b.acquire()
        starts.append(clock.now)

    async def main() -> None:
        await b.acquire()  # b now knows the state as of t=0
        waiting = asyncio.create_task(call())  # b's slot at 0.5
        await clock.advance_to(0.2)
        await a.cool_down(5.0)  # until 5.2, which b has not seen
        await clock.run_until_done(waiting)
        # b re-reads the state before using a slot it waited 0.5 s for.
        later = asyncio.create_task(call())
        await clock.run_until_done(later)

    asyncio.run(main())
    assert starts == [pytest.approx(5.2), pytest.approx(5.7)]


def test_unlimited_providers_share_cool_downs_without_taking_slots(
    sessions: sessionmaker[Session],
) -> None:
    clock = ManualClock()
    a = shared(sessions, clock, RateLimit())
    b = shared(sessions, clock, RateLimit(), peek_seconds=0)

    async def main() -> float:
        await b.acquire()
        await a.cool_down(3.0)
        # b does not wait for its read of the shared state: this call goes at once
        # and the read finishes behind it.
        await b.acquire()
        assert clock.now == 0.0
        await settle(b)
        task = asyncio.create_task(b.acquire())
        await clock.run_until_done(task)
        return clock.now

    assert asyncio.run(main()) == pytest.approx(3.0)
    with sessions() as s:
        row = s.scalars(select(RateLimitState)).one()
    assert row.tat == pytest.approx(3.0)
    assert row.cooldown_until == pytest.approx(3.0)


def test_a_refund_only_applies_to_the_latest_slot_across_processes(
    sessions: sessionmaker[Session],
) -> None:
    clock = FakeClock()
    limit = RateLimit(requests_per_second=1)
    a, b = shared(sessions, clock, limit), shared(sessions, clock, limit)

    async def main() -> None:
        first = await a._reserve()
        second = await b._reserve()
        assert (first.start, second.start) == (0.0, 1.0)
        assert not await a._refund(first)  # b reserved after it
        assert await b._refund(second)
        third = await a._reserve()
        assert third.start == 1.0

    asyncio.run(main())


def test_retry_after_on_one_worker_holds_back_another(sessions: sessionmaker[Session]) -> None:
    clock = FakeClock()
    limit = RateLimit(requests_per_second=100)
    a = shared(sessions, clock, limit, peek_seconds=0)
    b = shared(sessions, clock, limit, peek_seconds=0)
    calls: list[tuple[str, float]] = []

    async def limited() -> str:
        calls.append(("a", clock.now))
        if len(calls) == 1:
            raise TargetError("429", retryable=True, retry_after=4.0)
        return "ok"

    async def other() -> str:
        calls.append(("b", clock.now))
        return "ok"

    async def main() -> None:
        await call_with_retries(limited, limiter=a, policy=RetryPolicy(), stats=RetryStats())
        await call_with_retries(other, limiter=b, policy=RetryPolicy(), stats=RetryStats())

    asyncio.run(main())
    assert calls == [("a", 0.0), ("a", 4.0), ("b", pytest.approx(4.01))]


def test_concurrent_callers_share_one_peek(sessions: sessionmaker[Session]) -> None:
    clock = FakeClock()
    limiter = shared(sessions, clock, RateLimit())
    peeks = 0
    original = limiter._peek

    async def counting() -> tuple[float, float]:
        nonlocal peeks
        peeks += 1
        return await original()

    limiter._peek = counting  # type: ignore[method-assign]

    async def main() -> None:
        await asyncio.gather(*(limiter.acquire() for _ in range(20)))
        await settle(limiter)
        await asyncio.gather(*(limiter.acquire() for _ in range(20)))  # still fresh

    asyncio.run(main())
    assert peeks == 1


def test_a_failed_background_read_is_logged_and_retried(
    sessions: sessionmaker[Session], caplog: pytest.LogCaptureFixture
) -> None:
    clock = FakeClock()
    limiter = shared(sessions, clock, RateLimit(), peek_seconds=0)

    async def broken() -> tuple[float, float]:
        raise OSError("database went away")

    limiter._peek = broken  # type: ignore[method-assign]

    async def main() -> None:
        await limiter.acquire()
        await settle(limiter)
        await limiter.acquire()
        await settle(limiter)

    asyncio.run(main())
    failures = [r for r in caplog.records if "shared rate-limit state" in r.getMessage()]
    assert len(failures) == 2


def test_server_time_by_default(sessions: sessionmaker[Session]) -> None:
    limiter = PgRateLimiter(sessions, "p", RateLimit(requests_per_second=1000, burst=5))
    before = time.time()

    async def main() -> None:
        for _ in range(5):
            await limiter.acquire()
        await limiter.cool_down(0.0)

    asyncio.run(main())
    with sessions() as s:
        row = s.scalars(select(RateLimitState)).one()
    assert before < row.tat < time.time() + 1
    assert before <= row.cooldown_until <= time.time()


def test_limiter_keys() -> None:
    assert limiter_key("anthropic") == "anthropic"
    long = "http:" + "x" * 400
    key = limiter_key(long)
    assert len(key) == 255
    assert key != limiter_key(long + "y")
    with pytest.raises(TypeError, match="bound"):
        PgRateLimiter(sessionmaker(), "p", RateLimit())


def test_runner_uses_the_shared_limiter_when_asked(sessions: sessionmaker[Session]) -> None:
    from evalkit.datasets import import_dataset
    from tests.test_runner import config, make_cases

    with sessions() as s:
        import_dataset(s, "qa", make_cases(20))
    runner = Runner(sessions, Settings.from_env({}), shared_limits=True)
    assert not Runner(sessions, Settings.from_env({})).shared_limits  # SQLite: off by default
    run = runner.create_run(
        config(options={"rate_limit": {"requests_per_second": 1000, "burst": 4}})
    )
    report = asyncio.run(runner.execute(run.id))
    assert report.executed == 20
    with sessions() as s:
        row = s.get(RateLimitState, "fake")
    assert row is not None
    assert row.tat > 0


def test_callers_waiting_together_are_admitted_by_one_statement(
    sessions: sessionmaker[Session],
) -> None:
    clock = FakeClock()
    limit = RateLimit(requests_per_second=10, burst=3)
    limiter = shared(sessions, clock, limit)
    statements = 0
    original = limiter._run

    async def counting(statement: Any, **extra: float) -> Any:
        nonlocal statements
        statements += statement is limiter._admit
        return await original(statement, **extra)

    limiter._run = counting  # type: ignore[method-assign]

    async def main() -> list[Reservation]:
        reservations = await asyncio.gather(*(limiter._reserve() for _ in range(40)))
        # Only the last slot handed out can be handed back.
        assert not await limiter._refund(reservations[-2])
        assert await limiter._refund(reservations[-1])
        assert await limiter._refund(reservations[-2])
        again = await limiter._reserve()
        assert again.start == pytest.approx(reservations[-2].start)
        return reservations

    reservations = asyncio.run(main())
    assert statements == 2  # the 40 callers together, then the one after
    starts = [r.start for r in reservations]
    assert starts[:3] == [0.0, 0.0, 0.0]  # the burst
    assert starts == sorted(starts)
    assert conforms(starts, limiter.gcra, limit.burst)
    assert starts[-1] == pytest.approx(3.7)


def test_a_failed_admission_fails_every_caller_in_it(sessions: sessionmaker[Session]) -> None:
    clock = FakeClock()
    limiter = shared(sessions, clock, RateLimit(requests_per_second=10))

    original = limiter._run

    async def broken(statement: Any, **extra: float) -> Any:
        if statement is limiter._admit:
            raise OSError("database went away")
        return await original(statement, **extra)

    limiter._run = broken  # type: ignore[method-assign]

    async def main() -> list[Any]:
        return await asyncio.gather(*(limiter.acquire() for _ in range(5)), return_exceptions=True)

    results = asyncio.run(main())
    assert all(isinstance(r, OSError) for r in results)


def test_an_admitter_stopped_mid_statement_cancels_its_callers(
    sessions: sessionmaker[Session],
) -> None:
    clock = FakeClock()
    limiter = shared(sessions, clock, RateLimit(requests_per_second=10))
    started = asyncio.Event()

    async def hangs(statement: Any, **extra: float) -> Any:
        started.set()
        await asyncio.sleep(3600)

    limiter._run = hangs  # type: ignore[method-assign]

    async def main() -> list[Any]:
        callers = [asyncio.create_task(limiter._reserve()) for _ in range(3)]
        await started.wait()
        assert limiter._admitter is not None
        limiter._admitter.cancel()
        return await asyncio.gather(*callers, return_exceptions=True)

    results = asyncio.run(main())
    assert all(isinstance(r, asyncio.CancelledError) for r in results)
