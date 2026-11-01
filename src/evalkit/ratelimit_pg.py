"""A rate limiter shared by every process on one database.

Eight workers calling one provider with one API key must stay under that key's
rate together, and a 429 seen by any of them must hold all of them back. The
in-memory ``ProviderLimiter`` cannot do either across processes, so this limiter
keeps the same GCRA state (``ratelimit.Gcra``) in one row of ``rate_limits`` per
provider key:

* An admission is one ``UPDATE ... RETURNING`` in autocommit (one round trip):
  it advances the row's theoretical arrival time by one interval and returns the
  slot's start and the server's clock. Callers in one process that arrive while
  an admission is in flight are admitted together by the next statement, which
  advances the row by one interval per caller, so a process is not limited to
  one admission per round trip. The row lock serialises admissions from every
  process, no transaction outlives the statement, and callers sleep for the
  difference between two readings of the server's clock, so workers whose own
  clocks disagree still agree on the schedule.
* A cool-down (Retry-After) moves ``cooldown_until`` forward. Every admission
  returns it, and a caller whose knowledge of it is older than ``peek_seconds``
  when its slot arrives reads it again, so a 429 on one worker pauses the others
  within about that long, even those already asleep on a reserved slot.
* An unused slot is handed back with a compare-and-set on ``tat``: it only
  applies while no later slot has been reserved, which keeps the guarantee that
  no window ever sees more than ``burst + window * rate`` starts.

A provider without a configured rate takes no slots; it only shares cool-downs.
That costs one primary-key read per ``peek_seconds`` per process, made in the
background so no call waits for it.

The SQL also runs on SQLite (``max`` for ``greatest``, the client's wall clock
for the server's), which the unit tests use. The runner only uses this limiter
on PostgreSQL; with SQLite one process normally runs a run.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from sqlalchemy import Engine, text
from sqlalchemy.orm import Session, sessionmaker

from evalkit.ratelimit import Clock, GcraLimiter, RateLimit, RealClock, Reservation

MAX_KEY = 255
# The most callers one admission statement serves.
MAX_BATCH = 256
PG_NOW = "CAST(EXTRACT(EPOCH FROM clock_timestamp()) AS DOUBLE PRECISION)"


def limiter_key(provider: str) -> str:
    if len(provider) <= MAX_KEY:
        return provider
    return provider[:190] + ":" + hashlib.sha256(provider.encode()).hexdigest()


class PgRateLimiter(GcraLimiter):
    """A GCRA limiter whose state is a row in ``rate_limits``.

    ``now`` replaces the database server's clock with a client-side one; tests
    pass a fake clock's ``monotonic`` here. On SQLite it defaults to ``time.time``.
    """

    peek_seconds = 0.1

    def __init__(
        self,
        sessions: sessionmaker[Session],
        provider: str,
        limit: RateLimit,
        clock: Clock | None = None,
        *,
        now: Callable[[], float] | None = None,
        peek_seconds: float | None = None,
    ) -> None:
        super().__init__(limit, clock or RealClock())
        engine = sessions.kw.get("bind")
        if not isinstance(engine, Engine):
            raise TypeError("the session factory must be bound to an engine")
        self.engine = engine
        self.key = limiter_key(provider)
        if peek_seconds is not None:
            self.peek_seconds = peek_seconds
        postgres = engine.dialect.name == "postgresql"
        if now is None and not postgres:
            now = time.time
        self._now = now
        now_sql = ":now" if now is not None else PG_NOW
        names = {"GREATEST": "greatest" if postgres else "max", "NOW": now_sql}

        def sql(template: str) -> Any:
            # Only the two constants above are spliced in; values are bound parameters.
            for name, value in names.items():
                template = template.replace(name, value)
            return text(template)

        self._create = sql(
            "INSERT INTO rate_limits (key, tat, cooldown_until) VALUES (:key, 0, 0) "
            "ON CONFLICT (key) DO NOTHING"
        )
        # Admitting n callers at once is n admissions at the same instant (Gcra.admit
        # applied n times): the first starts at max(tat - tolerance, now,
        # cooldown_until), and tat ends at max(tat, first start) + n * interval. The
        # RETURNING clause sees the new row, from which the first start is
        # max(new_tat - n * interval - tolerance, now, cooldown_until).
        self._admit = sql(
            "UPDATE rate_limits "
            "SET tat = GREATEST(tat, GREATEST(tat - :tolerance, NOW, cooldown_until)) "
            "+ :count * :interval "
            "WHERE key = :key "
            "RETURNING GREATEST(tat - :count * :interval - :tolerance, NOW, cooldown_until), "
            "NOW, tat, cooldown_until"
        )
        self._refund_sql = sql(
            "UPDATE rate_limits SET tat = GREATEST(tat - :interval, cooldown_until + :tolerance) "
            "WHERE key = :key AND tat = :tat"
        )
        self._announce_sql = sql(
            "UPDATE rate_limits SET cooldown_until = GREATEST(cooldown_until, NOW + :seconds), "
            "tat = GREATEST(tat, NOW + :seconds + :tolerance) "
            "WHERE key = :key RETURNING cooldown_until, NOW"
        )
        self._peek_sql = sql("SELECT cooldown_until, NOW FROM rate_limits WHERE key = :key")
        self._ready = False
        # One statement at a time per process: the row lock would serialise them
        # anyway, and this keeps a large worker pool from holding every pooled
        # connection while it waits for that lock.
        self._lock = asyncio.Lock()
        self._waiting: list[asyncio.Future[Reservation]] = []
        self._admitter: asyncio.Task[None] | None = None

    def _params(self, extra: dict[str, float]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "key": self.key,
            "interval": self.gcra.interval,
            "tolerance": self.gcra.tolerance,
            **extra,
        }
        if self._now is not None:
            # Read the clock here, after waiting for the lock, not before.
            params["now"] = self._now()
        return params

    def _execute(self, statement: Any, extra: dict[str, float]) -> Any:
        # Autocommit: each statement is its own transaction, one round trip.
        with self.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            if not self._ready:
                conn.execute(self._create, {"key": self.key})
                self._ready = True
            result = conn.execute(statement, self._params(extra))
            return result.first() if result.returns_rows else result.rowcount

    async def _run(self, statement: Any, **extra: float) -> Any:
        async with self._lock:
            return await asyncio.to_thread(self._execute, statement, extra)

    async def _reserve(self) -> Reservation:
        # A caller cancelled while queued here may still be given a slot, which
        # then goes unused: lost capacity, never excess.
        future: asyncio.Future[Reservation] = asyncio.get_running_loop().create_future()
        self._waiting.append(future)
        if self._admitter is None or self._admitter.done():
            self._admitter = asyncio.create_task(self._admit_waiting())
        return await future

    async def _admit_waiting(self) -> None:
        while self._waiting:
            batch, self._waiting = self._waiting[:MAX_BATCH], self._waiting[MAX_BATCH:]
            try:
                row = await self._run(self._admit, count=len(batch))
            except Exception as exc:
                for future in batch:
                    if not future.done():
                        future.set_exception(exc)
                continue
            except BaseException:
                for future in batch + self._waiting:
                    future.cancel()
                raise
            for future, reservation in zip(batch, self._split(row, len(batch)), strict=True):
                with suppress(asyncio.InvalidStateError):  # its caller was cancelled
                    future.set_result(reservation)

    def _split(self, row: Any, count: int) -> list[Reservation]:
        """One reservation per caller admitted by one statement, in slot order."""
        first, now, tat, cooldown_until = (float(v) for v in row)
        interval, tolerance = self.gcra.interval, self.gcra.tolerance
        base = tat - count * interval  # max(old tat, first start)
        reservations = []
        for i in range(count):
            start = first if i == 0 else max(base + i * interval - tolerance, now, cooldown_until)
            # The tat this slot left behind; the last one's equals the row's exactly,
            # so the last slot of a batch can be handed back.
            after = tat - (count - 1 - i) * interval
            reservations.append(Reservation(start, now, after, cooldown_until))
        return reservations

    async def _refund(self, reservation: Reservation) -> bool:
        return bool(await self._run(self._refund_sql, tat=reservation.tat) == 1)

    async def _announce(self, seconds: float) -> tuple[float, float]:
        cooldown_until, now = await self._run(self._announce_sql, seconds=seconds)
        return float(cooldown_until), float(now)

    async def _peek(self) -> tuple[float, float]:
        cooldown_until, now = await self._run(self._peek_sql)
        return float(cooldown_until), float(now)
