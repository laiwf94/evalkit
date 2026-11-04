"""Claim races on PostgreSQL, forced into a chosen order.

Each test opens transactions on separate connections and holds one open while
another statement runs, so the race happens in exactly the order under test
rather than whenever the scheduler allows. Where the second statement has to
wait for a row lock, the test checks that it is really waiting (in
pg_stat_activity) before letting the first transaction commit. Where it must
not wait, ``lock_timeout`` turns any wait into an error.

The random-order counterpart on SQLite is tests/test_lease_interleavings.py.
"""

import itertools
import threading
import time
from collections.abc import Callable, Iterator
from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session, sessionmaker

from evalkit import aggregates, leases
from evalkit.config import Settings
from evalkit.datasets import Case, import_dataset
from evalkit.db import make_engine, make_session_factory
from evalkit.db.models import CaseLease, Result, Run, utcnow
from evalkit.runner import CaseResult, Runner

pytestmark = pytest.mark.e2e
TTL = timedelta(seconds=60)
runs = itertools.count()


@pytest.fixture
def sessions(postgres_url: str) -> Iterator[sessionmaker[Session]]:
    engine = make_engine(postgres_url)
    yield make_session_factory(engine)
    engine.dispose()


def make_run(sessions: sessionmaker[Session], n: int, *, seed: bool = True) -> str:
    run_id = f"race{next(runs)}"
    with sessions() as s:
        dataset, _ = import_dataset(s, run_id, [Case(id=f"k{i}", input=i) for i in range(n)])
        s.add(
            Run(
                id=run_id,
                dataset_id=dataset.id,
                target_spec={},
                target_hash="t" * 64,
                scorers=[{"type": "exact"}],
                options={},
                seed=1,
                status="running",
                cases_total=n,
            )
        )
        s.commit()
        if seed:
            leases.seed(s, run_id, [f"k{i}" for i in range(n)])
            s.commit()
    return run_id


def runner(sessions: sessionmaker[Session], owner: str) -> Runner:
    return Runner(sessions, Settings.from_env({}), owner=owner, lease_ttl=TTL)


def no_waiting(s: Session) -> None:
    """Make this transaction fail instead of waiting for any row lock."""
    s.execute(text("SET LOCAL lock_timeout = '200ms'"))


def claim_expired(
    sessions: sessionmaker[Session], run_id: str, owner: str, n: int
) -> list[leases.Claim]:
    """Claim ``n`` cases with leases that have already run out (a stalled owner)."""
    with sessions() as s:
        claims = leases.claim(
            s, run_id, owner, n, ttl=timedelta(seconds=1), now=utcnow() - timedelta(seconds=30)
        )
        s.commit()
    return claims


def lease_rows(sessions: sessionmaker[Session], run_id: str) -> list[tuple[str, str | None, int]]:
    with sessions() as s:
        rows = s.execute(
            select(CaseLease.case_id, CaseLease.owner, CaseLease.fence)
            .where(CaseLease.run_id == run_id)
            .order_by(CaseLease.position)
        ).all()
    return [(r.case_id, r.owner, r.fence) for r in rows]


def saved(sessions: sessionmaker[Session], run_id: str) -> int:
    with sessions() as s:
        return int(
            s.scalar(select(func.count()).select_from(Result).where(Result.run_id == run_id)) or 0
        )


class Background:
    """Run ``fn`` on another thread and collect what it returns or raises."""

    def __init__(self, fn: Callable[[], Any]) -> None:
        self._value: Any = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, args=(fn,), daemon=True)
        self._thread.start()

    def _run(self, fn: Callable[[], Any]) -> None:
        try:
            self._value = fn()
        except BaseException as e:
            self._error = e

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def result(self, timeout: float = 30) -> Any:
        self._thread.join(timeout)
        assert not self._thread.is_alive(), "the background statement never finished"
        if self._error is not None:
            raise self._error
        return self._value


def wait_for_lock_waiters(sessions: sessionmaker[Session], count: int, timeout: float = 10) -> None:
    """Block until ``count`` other backends are waiting for a lock."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with sessions() as s:
            waiting = s.scalar(
                text(
                    "SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock' "
                    "AND datname = current_database() AND pid <> pg_backend_pid()"
                )
            )
        if waiting is not None and waiting >= count:
            return
        time.sleep(0.01)
    raise AssertionError(f"expected {count} statement(s) waiting for a lock")


def results_for(claims: list[leases.Claim]) -> list[CaseResult]:
    return [
        CaseResult(c.case_id, "ok", scores={"exact": {"value": 1.0}}, fence=c.fence) for c in claims
    ]


def save(s: Session, run_id: str, owner: str, results: list[CaseResult]) -> None:
    """What a result write does after its leases are finished, short of committing."""
    s.add_all(Result(**r.row(run_id)) for r in results)
    aggregates.add(s, run_id, owner, aggregates.fold(results, ["exact"]))
    s.flush()


def test_a_claim_skips_rows_locked_by_a_claim_in_flight(sessions: sessionmaker[Session]) -> None:
    run_id = make_run(sessions, 10)
    with sessions() as first, sessions() as second:
        held = leases.claim(first, run_id, "a", 4, ttl=TTL, now=utcnow())
        assert [c.case_id for c in held] == ["k0", "k1", "k2", "k3"]
        # The first claim has not committed. The second neither waits for it nor
        # takes its rows.
        no_waiting(second)
        other = leases.claim(second, run_id, "b", 4, ttl=TTL, now=utcnow())
        assert [(c.case_id, c.fence) for c in other] == [(f"k{i}", 1) for i in range(4, 8)]
        second.commit()
        # A claim that rolls back leaves nothing behind: the cases are claimable
        # again under the fence they had.
        first.rollback()
    again = runner(sessions, "c")._claim_cases(run_id, 10)
    assert [(c.case_id, c.fence) for c in again] == [
        ("k0", 1),
        ("k1", 1),
        ("k2", 1),
        ("k3", 1),
        ("k8", 1),
        ("k9", 1),
    ]


def test_a_late_result_waits_for_a_takeover_in_flight_and_is_refused(
    sessions: sessionmaker[Session],
) -> None:
    run_id = make_run(sessions, 4)
    stale = claim_expired(sessions, run_id, "a", 4)
    with sessions() as taking_over:
        taken = leases.claim(taking_over, run_id, "b", 4, ttl=TTL, now=utcnow())
        assert {c.fence for c in taken} == {2}
        # The stalled owner wakes up and saves its results while the takeover is
        # still uncommitted: its write has to wait for the takeover's row locks.
        late = Background(lambda: runner(sessions, "a")._write(run_id, results_for(stale)))
        wait_for_lock_waiters(sessions, 1)
        assert late.running
        taking_over.commit()
    # PostgreSQL re-checks the waiting UPDATE against the committed rows, which
    # now belong to "b" under fence 2: every result is refused, none is saved.
    assert late.result() == 4
    assert saved(sessions, run_id) == 0
    assert runner(sessions, "b")._write(run_id, results_for(taken)) == 0
    assert saved(sessions, run_id) == 4
    with sessions() as s:
        assert aggregates.check(s, s.get_one(Run, run_id))[2] == []


def test_a_takeover_skips_cases_whose_results_are_being_saved(
    sessions: sessionmaker[Session],
) -> None:
    run_id = make_run(sessions, 6)
    stale = claim_expired(sessions, run_id, "a", 4)
    with sessions() as saving, sessions() as taking_over:
        accepted = leases.finish(saving, run_id, "a", [(c.case_id, c.fence) for c in stale])
        assert accepted == {"k0", "k1", "k2", "k3"}
        save(saving, run_id, "a", results_for(stale))
        # The leases have run out, but their results are being saved: a takeover
        # passes over them without waiting and takes only the unowned cases.
        no_waiting(taking_over)
        taken = leases.claim(taking_over, run_id, "b", 10, ttl=TTL, now=utcnow())
        assert [c.case_id for c in taken] == ["k4", "k5"]
        taking_over.commit()
        saving.commit()
    # Once the save commits, the cases are done and nobody can claim them.
    assert runner(sessions, "c")._claim_cases(run_id, 10) == []
    assert saved(sessions, run_id) == 4
    assert lease_rows(sessions, run_id)[:4] == [(f"k{i}", None, 1) for i in range(4)]
    with sessions() as s:
        assert aggregates.check(s, s.get_one(Run, run_id))[2] == []


def test_a_renewal_after_a_takeover_does_not_win_the_cases_back(
    sessions: sessionmaker[Session],
) -> None:
    run_id = make_run(sessions, 4)
    claim_expired(sessions, run_id, "a", 4)
    with sessions() as taking_over:
        leases.claim(taking_over, run_id, "b", 4, ttl=TTL, now=utcnow())

        def renew() -> int:
            with sessions() as s:
                n = leases.renew(s, run_id, "a", ttl=TTL, now=utcnow())
                s.commit()
                return n

        renewal = Background(renew)
        wait_for_lock_waiters(sessions, 1)
        taking_over.commit()
    assert renewal.result() == 0
    assert lease_rows(sessions, run_id) == [(f"k{i}", "b", 2) for i in range(4)]


def test_a_claim_skips_cases_being_released_and_gets_them_after(
    sessions: sessionmaker[Session],
) -> None:
    run_id = make_run(sessions, 4)
    runner(sessions, "a")._claim_cases(run_id, 4)
    with sessions() as releasing, sessions() as claiming:
        assert leases.release(releasing, run_id, "a") == 4
        no_waiting(claiming)
        assert leases.claim(claiming, run_id, "b", 4, ttl=TTL, now=utcnow()) == []
        claiming.commit()
        releasing.commit()
    again = runner(sessions, "b")._claim_cases(run_id, 4)
    assert [(c.case_id, c.fence) for c in again] == [(f"k{i}", 2) for i in range(4)]


def test_concurrent_first_joiners_seed_the_leases_once(sessions: sessionmaker[Session]) -> None:
    run_id = make_run(sessions, 50, seed=False)
    ids = [f"k{i}" for i in range(50)]
    with sessions() as first:
        assert leases.seed(first, run_id, ids)

        def second() -> bool:
            with sessions() as s:
                seeded = leases.seed(s, run_id, ids)
                s.commit()
                return seeded

        joiner = Background(second)
        # The second joiner waits on the run row the first one flipped...
        wait_for_lock_waiters(sessions, 1)
        first.commit()
    # ...then finds the flag set and creates nothing.
    assert joiner.result() is False
    assert [r[0] for r in lease_rows(sessions, run_id)] == ids


def test_processes_finishing_together_complete_the_run_once(
    sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = make_run(sessions, 8)
    a, b, c = (runner(sessions, owner) for owner in "abc")
    assert a._write(run_id, results_for(a._claim_cases(run_id, 8))) == 0
    compactions: list[str] = []
    real_compact = aggregates.compact

    def counting(session: Session, run: Run) -> None:
        compactions.append(run.id)
        real_compact(session, run)

    monkeypatch.setattr(aggregates, "compact", counting)
    with sessions() as holding:
        # Hold the run row so all three completions queue up behind it and then
        # run their compare-and-set one after another on the same row.
        holding.execute(select(Run.id).where(Run.id == run_id).with_for_update())
        finishing = [Background(lambda r=r: r._complete(run_id)) for r in (a, b, c)]  # type: ignore[misc]
        wait_for_lock_waiters(sessions, 3)
        holding.commit()
    assert [f.result() for f in finishing] == ["completed"] * 3
    assert compactions == [run_id]
    with sessions() as s:
        assert aggregates.verify(s, run_id).ok


def test_a_completion_check_during_the_last_save_waits_for_the_next_one(
    sessions: sessionmaker[Session],
) -> None:
    run_id = make_run(sessions, 3)
    a, b = runner(sessions, "a"), runner(sessions, "b")
    first, last = a._claim_cases(run_id, 2), b._claim_cases(run_id, 1)
    assert a._write(run_id, results_for(first)) == 0
    with sessions() as saving:
        assert leases.finish(saving, run_id, "b", [(last[0].case_id, last[0].fence)])
        save(saving, run_id, "b", results_for(last))
        # The last result is not committed: "a" sees an unfinished case, and its
        # check neither waits nor completes the run.
        assert a._complete(run_id) == "running"
        saving.commit()
    # The process that saved the last result checks after its own commit.
    assert b._complete(run_id) == "completed"
    with sessions() as s:
        assert aggregates.verify(s, run_id).ok
