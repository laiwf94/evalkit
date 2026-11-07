"""Eight workers on PostgreSQL sharing one provider's rate limit and its cool-downs.

Each worker is a separate process with its own engine, event loop and limiter;
they meet only in the ``rate_limits`` row.
"""

import asyncio
import bisect
import itertools
import multiprocessing
import random
import time
from typing import Any, NamedTuple

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evalkit.db import make_engine, make_session_factory
from evalkit.ratelimit import Gcra, RateLimit, Reservation
from evalkit.ratelimit_pg import PgRateLimiter
from tests.fakeclock import FakeClock
from tests.test_ratelimit import LIMITS, OPS, Op, drive

pytestmark = pytest.mark.e2e
WORKERS = 8
CALLERS = 4  # concurrent callers in each worker, like a worker pool
keys = itertools.count()


class Job(NamedTuple):
    postgres_url: str
    key: str
    limit: RateLimit
    per_worker: int
    cool_down_after: int | None


def _worker(job: Job, n: int, gate: Any, results: Any) -> None:
    """One worker process: its own interpreter, engine, event loop and limiter.

    Records each call as (when it started on this machine's clock, the slot the
    limiter reserved for it on the database server's clock). Both clocks are
    the same kernel's wall clock here, so the two compare directly.
    """
    calls: list[tuple[float, float]] = []
    cooled: list[float] = []
    engine = make_engine(job.postgres_url)
    limiter = PgRateLimiter(make_session_factory(engine), job.key, job.limit)
    slots: dict[asyncio.Task[Any] | None, float] = {}
    reserve = limiter._reserve

    async def recording_reserve() -> Reservation:
        # Runs in the caller's task; a slot handed back for a cool-down is
        # replaced by the next reservation, so the last one is the one used.
        reservation = await reserve()
        slots[asyncio.current_task()] = reservation.start
        return reservation

    limiter._reserve = recording_reserve  # type: ignore[method-assign]

    async def caller(c: int) -> None:
        for i in range(job.per_worker // CALLERS):
            await limiter.acquire()
            calls.append((time.time(), slots[asyncio.current_task()]))
            if n == 0 and c == 0 and i == job.cool_down_after:
                # As if this call had come back 429 with Retry-After: 0.5.
                cooled.append(time.time())
                await limiter.cool_down(0.5)

    async def main() -> None:
        await limiter.acquire()  # creates the row and warms the connection
        gate.wait()
        await asyncio.gather(*(caller(c) for c in range(CALLERS)))

    error = None
    try:
        asyncio.run(main())
    except BaseException as exc:  # pragma: no cover - reported by the parent
        error = repr(exc)
    finally:
        engine.dispose()
    results.put((calls, cooled, error))


def run_workers(
    postgres_url: str, key: str, limit: RateLimit, per_worker: int, cool_down_after: int | None
) -> tuple[list[tuple[float, float]], list[float]]:
    """Run WORKERS separate processes against one limiter row, as a fleet of
    worker containers would. Returns every call as (start, reserved slot),
    sorted by start, and the times the cool-down was announced."""
    ctx = multiprocessing.get_context("spawn")
    gate = ctx.Barrier(WORKERS)
    results = ctx.Queue()
    job = Job(postgres_url, key, limit, per_worker, cool_down_after)
    procs = [ctx.Process(target=_worker, args=(job, n, gate, results)) for n in range(WORKERS)]
    for proc in procs:
        proc.start()
    try:
        # Read before joining: a child does not exit until its queued result is taken.
        outcomes = [results.get(timeout=120) for _ in procs]
    finally:
        for proc in procs:
            proc.join(timeout=30)
            if proc.is_alive():  # pragma: no cover - only after a failure above
                proc.kill()
    errors = [error for _, _, error in outcomes if error is not None]
    assert not errors, errors
    calls = sorted(call for worker_calls, _, _ in outcomes for call in worker_calls)
    cooled = [t for _, worker_cooled, _ in outcomes for t in worker_cooled]
    return calls, cooled


def most_in_any_window(times: list[float], window: float) -> int:
    ordered = sorted(times)
    return max(bisect.bisect_left(ordered, t + window) - i for i, t in enumerate(ordered))


def test_eight_workers_stay_within_five_percent_of_the_rate(postgres_url: str) -> None:
    rate = 200.0
    limit = RateLimit(requests_per_second=rate)
    calls, _ = run_workers(postgres_url, "rate-test", limit, per_worker=60, cool_down_after=None)
    # The warm-up admissions took the first slots; measure the 480 that follow.
    assert len(calls) == WORKERS * 60
    starts = [start for start, _ in calls]
    achieved = (len(starts) - 1) / (starts[-1] - starts[0])
    gcra = Gcra.of(limit)
    window = 0.1
    # The schedule the eight workers were given together: exactly the GCRA bound,
    # with no allowance for jitter.
    slots = [slot for _, slot in calls]
    assert most_in_any_window(slots, window) <= limit.burst + window / gcra.interval
    # No call starts before its slot (beyond clock-read resolution), and none
    # starts long after it.
    lateness = [start - slot for start, slot in calls]
    early, late = -min(lateness), max(lateness)
    print(
        f"\nachieved {achieved:.1f} requests/s against a limit of {rate:.0f}; "
        f"calls start up to {late * 1000:.1f} ms after their slots"
    )
    assert early < 0.002
    assert late < 0.25
    assert abs(achieved - rate) / rate < 0.05
    # What the provider sees: calls that start inside one window had their slots
    # inside that window widened by the lateness, so the bound follows from the
    # two above. A late caller (its worker's one connection busy with a statement
    # that waited for the row lock) shifts its call; it never adds one.
    widened = window + late + 0.002
    assert most_in_any_window(starts, window) <= limit.burst + widened / gcra.interval


def test_a_cool_down_on_one_worker_pauses_all_of_them(postgres_url: str) -> None:
    limit = RateLimit(requests_per_second=100)
    calls, cooled = run_workers(postgres_url, "cool-test", limit, per_worker=20, cool_down_after=2)
    (at,) = cooled
    starts = [start for start, _ in calls]
    # Slots reserved before the 429 may still go for up to peek_seconds (0.1 s);
    # after that nothing starts until the cool-down ends.
    quiet = [t for t in starts if at + PgRateLimiter.peek_seconds + 0.02 < t < at + 0.5]
    assert quiet == []
    assert any(t >= at + 0.5 for t in starts)


def test_the_postgresql_statements_never_exceed_the_rate(postgres_url: str) -> None:
    """The property tests/test_ratelimit_pg.py checks on SQLite, run through
    PostgreSQL's own statements (``greatest``, double precision arithmetic): over
    random admissions, refunds and cool-downs, no window of w seconds sees more
    than burst + w * rate starts and no start falls inside a cool-down. A fake
    clock stands in for the server's, so the schedule is exact."""
    engine = make_engine(postgres_url)
    sessions = make_session_factory(engine)

    @settings(max_examples=60, deadline=None)
    @given(limit=LIMITS, ops=OPS, seed=st.integers(0, 2**16))
    def check(limit: RateLimit, ops: list[Op], seed: int) -> None:
        clock = FakeClock()
        clock.now = 1000.0
        limiter = PgRateLimiter(
            sessions, f"property{next(keys)}", limit, clock, now=clock.monotonic, peek_seconds=0
        )
        asyncio.run(drive(limiter, clock, ops, random.Random(seed)))

    try:
        check()
    finally:
        engine.dispose()
