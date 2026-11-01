"""The run queue: runs submitted for worker daemons, their slots and cancellation.

A run submitted through the API (or ``evalkit run --queue``) is ``queued``. Any
worker daemon may take a slot on it, up to the run's ``max_workers``, highest
priority first and oldest first within a priority. A worker joins the run
through the case leases in ``leases.py``, so several workers share a run
without executing a case twice, and a worker can also join a ``running`` run
that still has cases nobody holds (the cases of a worker that died, once their
leases expire). A slot is a row in ``run_workers`` that its worker renews with a
heartbeat, so the slot of a dead worker frees itself after one lease length.

Taking a slot first locks the run row with an UPDATE, then counts the run's live
slots and adds one only if the cap allows. The lock makes the count and the
insert one step: on PostgreSQL a second worker waits for the row and, under
READ COMMITTED, counts again with the first worker's slot in view; on SQLite the
UPDATE takes the database write lock.

States::

    pending     created by the CLI, executed by the process that created it
    queued      waiting for a worker
    running     at least one process has joined it
    cancelling  cancel was requested while processes were executing it
    cancelled   final; every process has stopped
    interrupted every process stopped (Ctrl-C, shutdown) with cases left
    failed      a process failed; finished cases are kept
    completed   final; every case has a result

On PostgreSQL every change a worker should react to sends a NOTIFY on
``evalkit_runs`` in the same transaction (delivered on commit), with the payload
``queued:<run id>``, ``cancel:<run id>`` or ``slot:<run id>``. Workers also poll,
so a lost notification only delays work by one poll interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import (
    ColumnElement,
    and_,
    case,
    delete,
    exists,
    false,
    func,
    insert,
    or_,
    select,
    update,
)
from sqlalchemy.orm import Session

from evalkit.db.models import CaseLease, Run, RunWorker
from evalkit.errors import ConflictError, NotFoundError

CHANNEL = "evalkit_runs"
JOINABLE = ("queued", "running")
FINAL = ("completed", "cancelled")
# Statuses a resume puts back in the queue.
REQUEUEABLE = ("pending", "interrupted", "failed")
CANDIDATES = 8


@dataclass(frozen=True)
class Slot:
    """A slot a worker took on a run."""

    run_id: str
    priority: int


def notify(session: Session, event: str, run_id: str) -> None:
    """Tell listening workers about ``event`` when the transaction commits.
    Only PostgreSQL has LISTEN/NOTIFY; elsewhere workers find out by polling."""
    if session.get_bind().dialect.name == "postgresql":
        session.execute(select(func.pg_notify(CHANNEL, f"{event}:{run_id}")))


def enqueue(session: Session, run_id: str, now: datetime) -> str:
    """Put a run (back) in the queue and return its status. A run that is queued or
    running stays as it is but becomes visible to workers; a completed run is left
    alone; a cancelled one cannot come back. The caller commits."""
    run = session.get(Run, run_id)
    if run is None:
        raise NotFoundError(f"run {run_id} not found")
    if run.status in ("cancelling", "cancelled"):
        raise ConflictError(f"run {run_id} is {run.status}")
    if run.status == "completed":
        return run.status
    session.execute(
        update(Run)
        .where(Run.id == run_id, Run.status.in_((*REQUEUEABLE, *JOINABLE)))
        .values(
            status=case((Run.status.in_(REQUEUEABLE), "queued"), else_=Run.status),
            error=case((Run.status.in_(REQUEUEABLE), ""), else_=Run.error),
            queued_at=func.coalesce(Run.queued_at, now),
        )
        .execution_options(synchronize_session=False)
    )
    notify(session, "queued", run_id)
    return _status(session, run_id)


def cancel(session: Session, run_id: str, now: datetime) -> str:
    """Request cancellation and return the new status: ``cancelled`` at once when no
    process is executing the run, else ``cancelling`` until the last one stops.
    Cancelling a cancelled run is a no-op; a completed run cannot be cancelled.
    The caller commits."""
    run = session.get(Run, run_id)
    if run is None:
        raise NotFoundError(f"run {run_id} not found")
    if run.status == "completed":
        raise ConflictError(f"run {run_id} is already completed")
    session.execute(
        update(Run)
        .where(Run.id == run_id, Run.status.not_in((*FINAL, "cancelling")))
        .values(status="cancelling")
        .execution_options(synchronize_session=False)
    )
    finalize_cancelled(session, now, run_id)
    status = _status(session, run_id)
    if status == "cancelling":
        notify(session, "cancel", run_id)
    return status


def nobody_working(now: datetime) -> ColumnElement[bool]:
    """No live worker slot, case lease or (batch mode) run lease on the run."""
    live_slot = exists().where(RunWorker.run_id == Run.id, RunWorker.expires_at >= now)
    live_lease = exists().where(
        CaseLease.run_id == Run.id,
        CaseLease.done == false(),
        CaseLease.owner.is_not(None),
        CaseLease.expires_at >= now,
    )
    live_run_lease = and_(Run.lease_owner.is_not(None), Run.lease_expires_at >= now)
    return and_(~live_slot, ~live_lease, ~live_run_lease)


def finalize_cancelled(session: Session, now: datetime, run_id: str | None = None) -> int:
    """Mark ``cancelling`` runs that nobody is executing any more ``cancelled``: the
    last process to stop normally does this itself, but it may have died, or two
    may have stopped at the same moment and each seen the other still working."""
    query = update(Run).where(Run.status == "cancelling", nobody_working(now))
    if run_id is not None:
        query = query.where(Run.id == run_id)
    result = session.execute(
        query.values(
            status="cancelled", finished_at=now, lease_owner=None, lease_expires_at=None
        ).execution_options(synchronize_session=False)
    )
    return int(result.rowcount)  # type: ignore[attr-defined]


def take_slot(
    session: Session, worker_id: str, *, busy: set[str], ttl: timedelta, now: datetime
) -> Slot | None:
    """Take a slot on the best run this worker can join, or return None. ``busy``
    holds the runs this worker already executes. Commits."""
    live_slots = (
        select(func.count())
        .select_from(RunWorker)
        .where(RunWorker.run_id == Run.id, RunWorker.expires_at >= now)
        .scalar_subquery()
    )
    # A running run is worth joining only while it has cases nobody holds:
    # otherwise its workers already have everything and this one would wait idle.
    claimable = exists().where(
        CaseLease.run_id == Run.id,
        CaseLease.done == false(),
        or_(CaseLease.owner.is_(None), CaseLease.expires_at < now),
    )
    query = (
        select(Run.id, Run.priority, Run.max_workers)
        .where(
            Run.status.in_(JOINABLE),
            Run.queued_at.is_not(None),
            or_(Run.max_workers.is_(None), live_slots < Run.max_workers),
            or_(Run.status == "queued", Run.leases_seeded.is_(False), claimable),
        )
        .order_by(Run.priority.desc(), Run.queued_at, Run.id)
        .limit(CANDIDATES + len(busy))
    )
    candidates = [row for row in session.execute(query) if row.id not in busy]
    session.rollback()
    for row in candidates[:CANDIDATES]:
        if _try_slot(session, row.id, row.max_workers, worker_id, ttl=ttl, now=now):
            return Slot(row.id, row.priority)
    return None


def _try_slot(
    session: Session,
    run_id: str,
    max_workers: int | None,
    worker_id: str,
    *,
    ttl: timedelta,
    now: datetime,
) -> bool:
    locked = session.execute(
        update(Run)
        .where(Run.id == run_id, Run.status.in_(JOINABLE))
        .values(priority=Run.priority)
        .execution_options(synchronize_session=False)
    )
    if not locked.rowcount:  # type: ignore[attr-defined]
        session.rollback()
        return False
    session.execute(
        delete(RunWorker).where(
            RunWorker.run_id == run_id,
            or_(RunWorker.expires_at < now, RunWorker.worker_id == worker_id),
        )
    )
    if max_workers is not None:
        taken = session.scalar(
            select(func.count()).select_from(RunWorker).where(RunWorker.run_id == run_id)
        )
        if (taken or 0) >= max_workers:
            session.rollback()
            return False
    session.execute(
        insert(RunWorker).values(
            run_id=run_id, worker_id=worker_id, joined_at=now, expires_at=now + ttl
        )
    )
    session.commit()
    return True


def renew_slots(
    session: Session, worker_id: str, run_ids: list[str], *, ttl: timedelta, now: datetime
) -> int:
    """Extend this worker's slots on ``run_ids``. Returns how many it still had.
    The caller commits."""
    if not run_ids:
        return 0
    result = session.execute(
        update(RunWorker)
        .where(RunWorker.worker_id == worker_id, RunWorker.run_id.in_(run_ids))
        .values(expires_at=now + ttl)
        .execution_options(synchronize_session=False)
    )
    return int(result.rowcount)  # type: ignore[attr-defined]


def leave(session: Session, run_id: str, worker_id: str, now: datetime) -> str:
    """Give the slot back. A run this worker left ``interrupted`` (it drained on
    shutdown) goes back to the queue for another worker. Returns the run's status.
    The caller commits."""
    session.execute(
        delete(RunWorker).where(RunWorker.run_id == run_id, RunWorker.worker_id == worker_id)
    )
    session.execute(
        update(Run)
        .where(Run.id == run_id, Run.status == "interrupted", Run.queued_at.is_not(None))
        .values(status="queued")
        .execution_options(synchronize_session=False)
    )
    finalize_cancelled(session, now, run_id)
    notify(session, "slot", run_id)
    return _status(session, run_id)


def sweep(session: Session, now: datetime) -> int:
    """Housekeeping any worker may do: drop expired slots and finish cancellations
    whose processes are gone. Returns how many runs became cancelled. Commits."""
    session.execute(delete(RunWorker).where(RunWorker.expires_at < now))
    cancelled = finalize_cancelled(session, now)
    session.commit()
    return cancelled


def live_workers(session: Session, run_id: str, now: datetime) -> int:
    return int(
        session.scalar(
            select(func.count())
            .select_from(RunWorker)
            .where(RunWorker.run_id == run_id, RunWorker.expires_at >= now)
        )
        or 0
    )


def _status(session: Session, run_id: str) -> str:
    return session.scalars(select(Run.status).where(Run.id == run_id)).one()
