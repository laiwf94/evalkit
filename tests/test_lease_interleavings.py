"""Random interleavings of several processes on one run, checked against a model.

Three runners share one run on SQLite. Hypothesis drives them through every
operation the lease protocol has (claim, fenced result write, renew, release,
a lease running out because its holder stalled, and the completion
compare-and-set) in random orders, and after every step checks the database
against a small model of what the protocol promises:

* a claim takes exactly the claimable cases, in dataset order, each with the
  next fence token;
* a result is saved if and only if its writer still holds the case under the
  fence it was claimed with, so each case is recorded at most once, by the
  holder of its latest lease;
* the run's aggregates always equal a recomputation from its results;
* the run completes exactly when no case is left, and once it does a
  verification of its aggregates and score arrays finds nothing to repair.

The PostgreSQL counterpart, which forces the same races inside overlapping
transactions, is tests/e2e/test_claim_races.py.
"""

import itertools
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    rule,
    run_state_machine_as_test,
)
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from evalkit import aggregates, leases
from evalkit.config import Settings
from evalkit.datasets import Case, import_dataset
from evalkit.db.models import CaseLease, Result, Run, utcnow
from evalkit.runner import CaseResult, Runner
from evalkit.runspec import RunConfig

OWNERS = ("w0", "w1", "w2")
names = itertools.count()


@dataclass
class Lease:
    """The model of one case's lease."""

    fence: int = 0
    owner: str | None = None
    expired: bool = False
    done: bool = False


def outcome(draw: Any) -> dict[str, Any]:
    status = draw(st.sampled_from(["ok", "ok", "ok", "error"]))
    value = draw(st.one_of(st.none(), st.sampled_from([0.0, 0.25, 0.5, 1.0])))
    return {
        "status": status,
        "scores": {} if value is None else {"exact": {"value": value}},
        "cached": draw(st.booleans()),
        "input_tokens": draw(st.integers(0, 50)),
        "cost_usd": draw(st.sampled_from([0.0, 0.001, 0.25])),
        "latency_ms": draw(st.sampled_from([1.0, 5.0, 40.0])),
    }


def test_random_interleavings_keep_the_lease_protocol(sessions: sessionmaker[Session]) -> None:
    class Interleavings(RuleBasedStateMachine):
        def __init__(self) -> None:
            super().__init__()
            self.runners = {
                o: Runner(sessions, Settings.from_env({}), owner=o, lease_ttl=timedelta(minutes=10))
                for o in OWNERS
            }
            self.model: dict[str, Lease] = {}
            self.order: list[str] = []
            self.held: dict[str, list[leases.Claim]] = {o: [] for o in OWNERS}
            self.results: dict[str, dict[str, Any]] = {}
            self.completions = 0
            self.run_id = ""

        @initialize(n=st.integers(1, 12))
        def create_run(self, n: int) -> None:
            name = f"il{next(names)}"
            with sessions() as s:
                import_dataset(s, name, [Case(id=f"c{i:02d}", input=i) for i in range(n)])
            run = self.runners["w0"].create_run(
                RunConfig.model_validate(
                    {
                        "dataset": {"name": name},
                        "target": {"type": "fake"},
                        "scorers": [{"type": "exact"}],
                    }
                )
            )
            self.run_id = run.id
            self.order = [f"c{i:02d}" for i in range(n)]
            with sessions() as s:
                assert leases.seed(s, run.id, self.order)
                s.commit()
            self.model = {case_id: Lease() for case_id in self.order}

        def claimable(self, owner: str) -> list[str]:
            return [
                c
                for c in self.order
                if not self.model[c].done
                and (
                    self.model[c].owner is None
                    or (self.model[c].expired and self.model[c].owner != owner)
                )
            ]

        @rule(owner=st.sampled_from(OWNERS), limit=st.integers(1, 5))
        def claim(self, owner: str, limit: int) -> None:
            want = self.claimable(owner)[:limit]
            got = self.runners[owner]._claim_cases(self.run_id, limit)
            assert [c.case_id for c in got] == want
            for c in got:
                lease = self.model[c.case_id]
                assert c.fence == lease.fence + 1
                lease.fence, lease.owner, lease.expired = c.fence, owner, False
            self.held[owner] += got

        @rule(owner=st.sampled_from(OWNERS))
        def stall(self, owner: str) -> None:
            """The owner stops renewing and its leases run out."""
            with sessions() as s:
                s.execute(
                    update(CaseLease)
                    .where(
                        CaseLease.run_id == self.run_id,
                        CaseLease.owner == owner,
                        CaseLease.done.is_(False),
                    )
                    .values(expires_at=utcnow() - timedelta(seconds=1))
                )
                s.commit()
            for lease in self.model.values():
                if lease.owner == owner and not lease.done:
                    lease.expired = True

        @rule(owner=st.sampled_from(OWNERS))
        def renew(self, owner: str) -> None:
            mine = [x for x in self.model.values() if x.owner == owner and not x.done]
            with sessions() as s:
                renewed = leases.renew(
                    s, self.run_id, owner, ttl=timedelta(minutes=10), now=utcnow()
                )
                s.commit()
            assert renewed == len(mine)
            for lease in mine:
                lease.expired = False

        @rule(owner=st.sampled_from(OWNERS), data=st.data())
        def release(self, owner: str, data: st.DataObject) -> None:
            everything = data.draw(st.booleans())
            ids = (
                None
                if everything
                else data.draw(st.lists(st.sampled_from(self.order), unique=True, max_size=4))
            )
            chosen = [
                c
                for c in self.order
                if self.model[c].owner == owner
                and not self.model[c].done
                and (ids is None or c in ids)
            ]
            with sessions() as s:
                assert leases.release(s, self.run_id, owner, ids) == len(chosen)
                s.commit()
            for c in chosen:
                self.model[c].owner, self.model[c].expired = None, False

        @rule(owner=st.sampled_from(OWNERS), data=st.data())
        def write(self, owner: str, data: st.DataObject) -> None:
            """Save a batch of this owner's claims, some of which may be stale."""
            held = self.held[owner]
            if not held:
                return
            picked = data.draw(
                st.lists(st.integers(0, len(held) - 1), unique=True, min_size=1, max_size=6)
            )
            claims = [held[i] for i in picked]
            batch = [
                CaseResult(case_id=c.case_id, fence=c.fence, **outcome(data.draw)) for c in claims
            ]
            accepted: dict[str, CaseResult] = {}
            for r in batch:
                lease = self.model[r.case_id]
                if not lease.done and lease.owner == owner and lease.fence == r.fence:
                    accepted[r.case_id] = r
            refused = self.runners[owner]._write(self.run_id, batch)
            assert refused == len(batch) - len(accepted)
            for case_id, r in accepted.items():
                lease = self.model[case_id]
                lease.done, lease.owner, lease.expired = True, None, False
                self.results[case_id] = {"status": r.status, "scores": r.scores, "fence": r.fence}
            self.held[owner] = [c for i, c in enumerate(held) if i not in picked]

        @rule(owner=st.sampled_from(OWNERS))
        def complete(self, owner: str) -> None:
            with sessions() as s:
                before = s.scalars(select(Run.status).where(Run.id == self.run_id)).one()
            status = self.runners[owner]._complete(self.run_id)
            everything_done = all(x.done for x in self.model.values())
            assert status == ("completed" if everything_done else before)
            if before != "completed" and status == "completed":
                self.completions += 1
            assert self.completions <= 1

        @invariant()
        def database_matches_the_model(self) -> None:
            if not self.run_id:
                return
            with sessions() as s:
                rows = s.execute(
                    select(CaseLease.case_id, CaseLease.owner, CaseLease.fence, CaseLease.done)
                    .where(CaseLease.run_id == self.run_id)
                    .order_by(CaseLease.position)
                ).all()
                assert [(r.case_id, r.owner, r.fence, r.done) for r in rows] == [
                    (c, self.model[c].owner, self.model[c].fence, self.model[c].done)
                    for c in self.order
                ]
                saved = {
                    r.case_id: {"status": r.status, "scores": r.scores}
                    for r in s.scalars(select(Result).where(Result.run_id == self.run_id))
                }
                assert saved == {
                    c: {"status": v["status"], "scores": v["scores"]}
                    for c, v in self.results.items()
                }
                _, _, problems = aggregates.check(s, s.get_one(Run, self.run_id))
                assert problems == []

        def teardown(self) -> None:
            if not self.run_id:
                return
            # The run is finished whatever state it was left in: every lease runs
            # out, the runners take over what is left and save it, and it completes.
            for owner in OWNERS:
                self.stall(owner)
            progress = True
            while progress:
                progress = False
                for runner in self.runners.values():
                    claims = runner._claim_cases(self.run_id, 100)
                    batch = [CaseResult(c.case_id, "ok", fence=c.fence) for c in claims]
                    assert runner._write(self.run_id, batch) == 0
                    progress = progress or bool(claims)
            last = self.runners["w0"]
            assert last._complete(self.run_id) == "completed"
            with sessions() as s:
                assert leases.remaining(s, self.run_id) == 0
                ids = list(s.scalars(select(Result.case_id).where(Result.run_id == self.run_id)))
                assert sorted(ids) == self.order
                check = aggregates.verify(s, self.run_id)
                assert check.ok, check.problems
                assert check.compacted

    run_state_machine_as_test(  # type: ignore[no-untyped-call]
        Interleavings,
        settings=settings(
            max_examples=75,
            stateful_step_count=30,
            deadline=None,
            suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
        ),
    )
