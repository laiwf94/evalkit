"""Eight workers on PostgreSQL sharing one provider's rate limit and its cool-downs.

Each worker has its own engine, event loop and limiter, as a separate process
would; they meet only in the ``rate_limits`` row.
"""

import asyncio
import threading
import time

import pytest

from evalkit.db import make_engine, make_session_factory
from evalkit.ratelimit import Gcra, RateLimit
from evalkit.ratelimit_pg import PgRateLimiter

pytestmark = pytest.mark.e2e
WORKERS = 8
CALLERS = 4  # concurrent callers in each worker, like a worker pool


def run_workers(
    postgres_url: str, key: str, limit: RateLimit, per_worker: int, cool_down_after: int | None
) -> tuple[list[float], list[float]]:
    starts: list[float] = []
    cooled: list[float] = []
    lock = threading.Lock()
    gate = threading.Barrier(WORKERS)
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        engine = make_engine(postgres_url)
        limiter = PgRateLimiter(make_session_factory(engine), key, limit)

        async def caller(c: int) -> None:
            for i in range(per_worker // CALLERS):
                await limiter.acquire()
                now = time.monotonic()
                with lock:
                    starts.append(now)
                if n == 0 and c == 0 and i == cool_down_after:
                    # As if this call had come back 429 with Retry-After: 0.5.
                    with lock:
                        cooled.append(time.monotonic())
                    await limiter.cool_down(0.5)

        async def main() -> None:
            await limiter.acquire()  # creates the row and warms the connection
            gate.wait()
            await asyncio.gather(*(caller(c) for c in range(CALLERS)))

        try:
            asyncio.run(main())
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)
        finally:
            engine.dispose()

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not errors, errors
    return sorted(starts), cooled


def test_eight_workers_stay_within_five_percent_of_the_rate(postgres_url: str) -> None:
    rate = 200.0
    limit = RateLimit(requests_per_second=rate)
    starts, _ = run_workers(postgres_url, "rate-test", limit, per_worker=60, cool_down_after=None)
    # The warm-up admissions took the first slots; measure the 480 that follow.
    assert len(starts) == WORKERS * 60
    achieved = (len(starts) - 1) / (starts[-1] - starts[0])
    print(f"\nachieved {achieved:.1f} requests/s against a limit of {rate:.0f}")
    assert abs(achieved - rate) / rate < 0.05
    # No 100 ms window holds more than the limit allows, give or take scheduling jitter.
    gcra = Gcra.of(limit)
    window = 0.1
    for i, t in enumerate(starts):
        inside = sum(1 for u in starts[i:] if u < t + window)
        assert inside <= limit.burst + window / gcra.interval + 2


def test_a_cool_down_on_one_worker_pauses_all_of_them(postgres_url: str) -> None:
    limit = RateLimit(requests_per_second=100)
    starts, cooled = run_workers(postgres_url, "cool-test", limit, per_worker=20, cool_down_after=2)
    (at,) = cooled
    # Slots reserved before the 429 may still go for up to peek_seconds (0.1 s);
    # after that nothing starts until the cool-down ends.
    quiet = [t for t in starts if at + PgRateLimiter.peek_seconds + 0.02 < t < at + 0.5]
    assert quiet == []
    assert any(t >= at + 0.5 for t in starts)
