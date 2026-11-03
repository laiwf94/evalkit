"""The worker daemon: ``evalkit worker`` executes runs from the queue.

A worker holds up to ``slots`` runs at once. It looks for work when it starts,
when a notification arrives, when one of its runs ends and every
``poll_seconds`` otherwise. Taking a slot (``runqueue.take_slot``) respects each
run's priority and worker cap; executing it is ``Runner.execute``, so several
workers share one run through case leases exactly as several CLI processes do.

On PostgreSQL a dedicated connection LISTENs on ``evalkit_runs``. ``queued`` and
``slot`` notifications wake the worker; ``cancel`` stops the named run at once
if the worker is executing it. If that connection drops, the worker logs it,
keeps polling and reconnects with backoff. Without notifications (SQLite, or
while disconnected) a cancellation is noticed at the runner's next heartbeat,
within a third of a lease length.

Stopping (SIGTERM or SIGINT; the API's shutdown for the embedded worker) drains:
the worker takes no new slots, each of its runs stops claiming cases and gives
back the ones it has not started, and the cases in flight get ``drain_seconds``
to finish and be saved. Whatever is still running then is cancelled, saving the
results it has. Each run is left ``queued`` for another worker (unless other
processes are still executing it), and the worker's slots are given back. A
second signal skips the wait.

When ``EVALKIT_CACHE_MAX_AGE`` or ``EVALKIT_CACHE_MAX_BYTES`` is set, or a blob
store is configured, the worker also sweeps the response cache and collects
unreferenced blobs every ``EVALKIT_CACHE_SWEEP_SECONDS`` (``cache.maintain``).
On PostgreSQL an advisory lock lets one worker at a time do it.

A worker started inside the API process (``EVALKIT_MAX_ACTIVE_RUNS`` slots) is
the same class, so a single ``evalkit serve`` still executes what it accepts.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

import psycopg
from psycopg import sql
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from evalkit import cache, metrics, runqueue
from evalkit.db.models import utcnow
from evalkit.logs import log_context
from evalkit.runner import Runner, wait_for_any

log = logging.getLogger(__name__)

LISTEN_BACKOFF_SECONDS = (0.5, 30.0)


def listen_url(database_url: str) -> str | None:
    """The libpq URL for a LISTEN connection, or None when the database has no
    LISTEN/NOTIFY (SQLite)."""
    url = make_url(database_url)
    if url.get_backend_name() != "postgresql":
        return None
    return url.set(drivername="postgresql").render_as_string(hide_password=False)


def _seconds(value: float | None) -> timedelta | None:
    return None if value is None else timedelta(seconds=value)


class Worker:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        runner: Runner,
        *,
        slots: int,
        poll_seconds: float,
        drain_seconds: float,
        listen: str | None = None,
        heartbeat_seconds: float | None = None,
    ) -> None:
        if slots < 1:
            raise ValueError("a worker needs at least one slot")
        self.sessions = sessions
        self.runner = runner
        self.id = runner.owner
        self.slots = slots
        self.poll_seconds = poll_seconds
        self.drain_seconds = drain_seconds
        self.listen = listen
        self.ttl = runner.lease_ttl
        self.heartbeat_seconds = heartbeat_seconds or self.ttl.total_seconds() / 3
        self.listening = asyncio.Event()
        self._wake = asyncio.Event()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._stopping = False
        settings = runner.settings
        # Sweep the response cache and collect blobs when there is anything to bound.
        self.maintain_cache = settings.cache_limited or runner.blobs is not None

    @property
    def runs(self) -> set[str]:
        """Runs this worker holds a slot on."""
        return set(self._tasks)

    def wake(self) -> None:
        """Look for work now (a run was queued in this process)."""
        self._wake.set()

    def cancel(self, run_id: str) -> bool:
        """Stop ``run_id`` now if this worker executes it (it was cancelled)."""
        return self.runner.request_cancel(run_id)

    def abort(self) -> None:
        """Stop waiting for in-flight cases while draining (a second signal)."""
        for task in self._tasks.values():
            task.cancel()

    async def serve(self, stop: asyncio.Event) -> None:
        """Execute queued runs until ``stop`` is set, then drain."""
        background = [asyncio.create_task(self._heartbeat(), name="worker-heartbeat")]
        if self.maintain_cache:
            background.append(asyncio.create_task(self._maintain_cache(), name="worker-cache"))
        if self.listen is not None:
            background.append(asyncio.create_task(self._listen(self.listen), name="worker-listen"))
        log.info("worker started", extra={"worker": self.id, "slots": self.slots})
        try:
            while not stop.is_set():
                self._wake.clear()
                try:
                    await asyncio.to_thread(self._sweep)
                    await self._fill()
                except Exception:
                    log.warning("cannot read the run queue; trying again", exc_info=True)
                await wait_for_any(self._wake, stop, self.poll_seconds)
        finally:
            await self._drain()
            for task in background:
                task.cancel()
            await asyncio.gather(*background, return_exceptions=True)
            log.info("worker stopped", extra={"worker": self.id})

    async def _fill(self) -> None:
        while len(self._tasks) < self.slots and not self._stopping:
            slot = await asyncio.to_thread(self._take_slot, set(self._tasks))
            if slot is None:
                return
            self._start(slot.run_id)

    def _start(self, run_id: str) -> None:
        task = asyncio.create_task(self._execute(run_id), name=f"run-{run_id}")
        self._tasks[run_id] = task
        metrics.WORKER_RUNS.set(len(self._tasks))

        def finished(_: asyncio.Task[None]) -> None:
            self._tasks.pop(run_id, None)
            metrics.WORKER_RUNS.set(len(self._tasks))
            self._wake.set()

        task.add_done_callback(finished)

    async def _execute(self, run_id: str) -> None:
        with log_context(run_id=run_id):
            try:
                report = await self.runner.execute(run_id)
                log.info("left the run", extra={"status": report.status})
            except asyncio.CancelledError:
                log.info("left the run: stopped while draining")
                raise
            except Exception:  # the runner has recorded the failure on the run
                log.warning("run stopped with an error", exc_info=True)
            finally:
                await asyncio.shield(self._leave(run_id))

    async def _leave(self, run_id: str) -> None:
        try:
            await asyncio.to_thread(self._leave_now, run_id)
        except Exception:  # the slot expires on its own after one lease length
            log.warning("could not give the slot back", extra={"run_id": run_id}, exc_info=True)

    async def _drain(self) -> None:
        self._stopping = True
        tasks = dict(self._tasks)
        if not tasks:
            return
        log.info("draining", extra={"runs": len(tasks), "seconds": self.drain_seconds})
        for run_id, task in tasks.items():
            if not self.runner.request_drain(run_id):
                task.cancel()  # not started yet: nothing in flight to wait for
        _, late = await asyncio.wait(tasks.values(), timeout=self.drain_seconds)
        for task in late:
            task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        # A task cancelled before it started never ran its own clean-up.
        for run_id in tasks:
            await self._leave(run_id)

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_seconds)
            try:
                await asyncio.to_thread(self._renew, list(self._tasks))
            except Exception:
                log.warning("could not renew worker slots", exc_info=True)

    async def _listen(self, url: str) -> None:
        delay, longest = LISTEN_BACKOFF_SECONDS
        while True:
            try:
                async with await psycopg.AsyncConnection.connect(url, autocommit=True) as conn:
                    await conn.execute(
                        sql.SQL("LISTEN {}").format(sql.Identifier(runqueue.CHANNEL))
                    )
                    self.listening.set()
                    # Anything queued while we were not listening is found now.
                    self._wake.set()
                    delay = LISTEN_BACKOFF_SECONDS[0]
                    async for note in conn.notifies():
                        self.on_notification(note.payload)
            except Exception:
                log.warning("lost the notification connection; polling meanwhile", exc_info=True)
            finally:
                self.listening.clear()
            await asyncio.sleep(delay)
            delay = min(delay * 2, longest)

    def on_notification(self, payload: str) -> None:
        event, _, run_id = payload.partition(":")
        if event == "cancel":
            self.cancel(run_id)
        self._wake.set()

    def _take_slot(self, busy: set[str]) -> runqueue.Slot | None:
        with self.sessions() as session:
            return runqueue.take_slot(session, self.id, busy=busy, ttl=self.ttl, now=utcnow())

    def _leave_now(self, run_id: str) -> None:
        with self.sessions() as session:
            runqueue.leave(session, run_id, self.id, utcnow())
            session.commit()

    def _renew(self, run_ids: list[str]) -> None:
        with self.sessions() as session:
            runqueue.renew_slots(session, self.id, run_ids, ttl=self.ttl, now=utcnow())
            session.commit()

    async def _maintain_cache(self) -> None:
        settings = self.runner.settings
        while True:
            try:
                done = await asyncio.to_thread(
                    cache.maintain,
                    self.sessions,
                    self.runner.blobs,
                    max_age=_seconds(settings.cache_max_age_seconds),
                    max_bytes=settings.cache_max_bytes,
                    grace=timedelta(seconds=settings.blob_grace_seconds),
                )
                if not done.skipped and (
                    done.sweep.removed or (done.blobs is not None and done.blobs.removed)
                ):
                    log.info(
                        "swept the response cache",
                        extra={
                            "entries": done.sweep.removed,
                            "bytes": done.sweep.removed_bytes,
                            "blobs": done.blobs.removed if done.blobs else 0,
                        },
                    )
            except Exception:
                log.warning("cannot sweep the response cache; trying later", exc_info=True)
            await asyncio.sleep(settings.cache_sweep_seconds)

    def _sweep(self) -> None:
        with self.sessions() as session:
            if runqueue.sweep(session, utcnow()):
                log.info("finished cancellations whose processes had stopped")
