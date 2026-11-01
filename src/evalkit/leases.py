"""Case leases: several workers execute one run without running a case twice.

When a run is first joined, every case that still needs executing gets a row in
``case_leases``. A worker claims a batch of rows with ``SELECT ... FOR UPDATE
SKIP LOCKED`` inside the claiming UPDATE, so two workers claiming at the same
moment take disjoint batches instead of waiting on each other. A claim sets the
owner, an expiry and a new fence token (the previous token plus one).

The fence is what makes a late result harmless. A worker that stalls past its
expiry can lose a case to another worker, which claims it with a higher token.
Results are written in the same transaction that marks their leases done, and
only for leases whose owner and token still match the claim, so the stalled
worker's result is refused and the case is recorded once, by the new owner.
The unique (run, case) constraint on results stays as a second line of defence.

Times come from the caller (the worker's clock, in UTC), so worker clocks must
agree to well within one lease length.

On SQLite there is no row locking: the claiming UPDATE takes the database write
lock before it chooses its rows, which serialises claims.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import ColumnElement, and_, false, func, insert, or_, select, tuple_, update
from sqlalchemy.orm import Session

from evalkit.db.models import CaseLease, Run

SEED_CHUNK = 5000
WRITE_CHUNK = 500


@dataclass(frozen=True, order=True)
class Claim:
    """A case this worker may execute, and the token its result must carry."""

    position: int
    case_id: str
    fence: int


@dataclass(frozen=True)
class Outstanding:
    """Unfinished cases of a run, from one worker's point of view."""

    unowned: int  # about to be claimed, or skipped because a claim had them locked
    held: int  # held by other workers
    mine: int  # held by this worker
    next_expiry: datetime | None  # the earliest expiry among other workers' leases

    @property
    def total(self) -> int:
        """Unfinished cases this worker does not hold."""
        return self.unowned + self.held


def seed(session: Session, run_id: str, case_ids: Sequence[str]) -> bool:
    """Create the run's leases, once. Returns whether this call created them.

    The flag on the run is flipped by a conditional UPDATE in the same
    transaction as the inserts, so concurrent first joiners cannot both seed:
    the second waits on the run row and then finds the flag set. The caller
    commits.
    """
    flipped = session.execute(
        update(Run)
        .where(Run.id == run_id, Run.leases_seeded.is_(False))
        .values(leases_seeded=True)
        .execution_options(synchronize_session=False)
    )
    if not flipped.rowcount:  # type: ignore[attr-defined]
        return False
    for start in range(0, len(case_ids), SEED_CHUNK):
        rows = [
            {
                "run_id": run_id,
                "case_id": case_id,
                "position": start + offset,
                "fence": 0,
                "done": False,
            }
            for offset, case_id in enumerate(case_ids[start : start + SEED_CHUNK])
        ]
        session.execute(insert(CaseLease), rows)
    return True


def _unfinished() -> ColumnElement[bool]:
    # "done = false" rather than "done IS false", so PostgreSQL can use the index.
    return CaseLease.done == false()


def _claimable(owner: str, now: datetime) -> ColumnElement[bool]:
    # A worker never re-claims its own expired lease: that case is still in its queue.
    return and_(
        _unfinished(),
        or_(
            CaseLease.owner.is_(None),
            and_(CaseLease.expires_at < now, CaseLease.owner != owner),
        ),
    )


def claim(
    session: Session, run_id: str, owner: str, limit: int, *, ttl: timedelta, now: datetime
) -> list[Claim]:
    """Claim up to ``limit`` claimable cases in dataset order. The caller commits."""
    if limit < 1:
        return []
    candidates = (
        select(CaseLease.case_id)
        .where(CaseLease.run_id == run_id, _claimable(owner, now))
        .order_by(CaseLease.position)
        .limit(limit)
        .with_for_update(skip_locked=True)
        .cte("candidates")
        # Locked rows must be chosen exactly once; a materialised CTE guarantees
        # the planner cannot re-run the locking subquery.
        .prefix_with("MATERIALIZED", dialect="postgresql")
    )
    rows = session.execute(
        update(CaseLease)
        # No second claimability check here: the candidates are chosen inside this
        # statement, under SQLite's write lock or PostgreSQL's row locks (which
        # re-check the condition on the latest row version), and without one the
        # planner looks the rows up by primary key instead of scanning the run.
        .where(CaseLease.run_id == run_id, CaseLease.case_id.in_(select(candidates.c.case_id)))
        .values(owner=owner, expires_at=now + ttl, fence=CaseLease.fence + 1)
        .returning(CaseLease.position, CaseLease.case_id, CaseLease.fence)
        .execution_options(synchronize_session=False)
    ).all()
    return sorted(Claim(row.position, row.case_id, row.fence) for row in rows)


def renew(session: Session, run_id: str, owner: str, *, ttl: timedelta, now: datetime) -> int:
    """Extend every unfinished lease ``owner`` still holds. Returns how many."""
    result = session.execute(
        update(CaseLease)
        .where(
            CaseLease.run_id == run_id,
            CaseLease.owner == owner,
            _unfinished(),
        )
        .values(expires_at=now + ttl)
        .execution_options(synchronize_session=False)
    )
    return int(result.rowcount)  # type: ignore[attr-defined]


def release(
    session: Session, run_id: str, owner: str, case_ids: Sequence[str] | None = None
) -> int:
    """Give back unfinished leases (all of them, or those of ``case_ids``) so other
    workers can claim them at once. Returns how many."""
    if case_ids is not None and not case_ids:
        return 0
    where = [CaseLease.run_id == run_id, CaseLease.owner == owner, _unfinished()]
    released = 0
    chunks = (
        [None]
        if case_ids is None
        else [case_ids[i : i + WRITE_CHUNK] for i in range(0, len(case_ids), WRITE_CHUNK)]
    )
    for chunk in chunks:
        condition = where if chunk is None else [*where, CaseLease.case_id.in_(chunk)]
        result = session.execute(
            update(CaseLease)
            .where(*condition)
            .values(owner=None, expires_at=None)
            .execution_options(synchronize_session=False)
        )
        released += int(result.rowcount)  # type: ignore[attr-defined]
    return released


def finish(
    session: Session, run_id: str, owner: str, claims: Iterable[tuple[str, int]]
) -> set[str]:
    """Mark leases done for (case id, fence) pairs that ``owner`` still holds.

    Returns the case ids that were accepted. A pair whose lease has since been
    claimed by someone else (a higher fence) or finished is refused. Run this in
    the transaction that inserts the results, so a refused case gets no row.
    """
    pairs = list(claims)
    accepted: set[str] = set()
    for start in range(0, len(pairs), WRITE_CHUNK):
        chunk = pairs[start : start + WRITE_CHUNK]
        accepted.update(
            session.scalars(
                update(CaseLease)
                .where(
                    CaseLease.run_id == run_id,
                    CaseLease.owner == owner,
                    _unfinished(),
                    tuple_(CaseLease.case_id, CaseLease.fence).in_(chunk),
                )
                .values(done=True, owner=None, expires_at=None)
                .returning(CaseLease.case_id)
                .execution_options(synchronize_session=False)
            )
        )
    return accepted


def outstanding(session: Session, run_id: str, owner: str) -> Outstanding:
    others = and_(CaseLease.owner.is_not(None), CaseLease.owner != owner)
    unowned, held, mine, next_expiry = session.execute(
        select(
            func.count().filter(CaseLease.owner.is_(None)),
            func.count().filter(others),
            func.count().filter(CaseLease.owner == owner),
            func.min(CaseLease.expires_at).filter(others),
        ).where(CaseLease.run_id == run_id, _unfinished())
    ).one()
    return Outstanding(int(unowned), int(held), int(mine), next_expiry)


def remaining(session: Session, run_id: str) -> int:
    """Cases of the run that have no result yet."""
    count = session.scalar(
        select(func.count()).select_from(CaseLease).where(CaseLease.run_id == run_id, _unfinished())
    )
    return int(count or 0)
