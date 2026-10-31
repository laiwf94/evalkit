from datetime import timedelta

import pytest
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from evalkit import leases
from evalkit.datasets import Case, import_dataset
from evalkit.db.models import CaseLease, Run, utcnow

TTL = timedelta(seconds=30)


@pytest.fixture
def run_id(sessions: sessionmaker[Session]) -> str:
    with sessions() as s:
        dataset, _ = import_dataset(s, "d", [Case(id=f"k{i}", input=i) for i in range(10)])
        s.add(
            Run(
                id="r1",
                dataset_id=dataset.id,
                target_spec={},
                target_hash="t" * 64,
                scorers=[],
                options={},
                seed=1,
            )
        )
        s.commit()
        assert leases.seed(s, "r1", [f"k{i}" for i in range(10)])
        s.commit()
    return "r1"


def claim(s: Session, owner: str, limit: int, now_offset: float = 0) -> list[leases.Claim]:
    now = utcnow() + timedelta(seconds=now_offset)
    claims = leases.claim(s, "r1", owner, limit, ttl=TTL, now=now)
    s.commit()
    return claims


def test_seed_happens_once(sessions: sessionmaker[Session], run_id: str) -> None:
    with sessions() as s:
        assert not leases.seed(s, run_id, ["other"])
        s.commit()
        assert leases.remaining(s, run_id) == 10
        positions = list(s.scalars(select(CaseLease.position).order_by(CaseLease.position)))
    assert positions == list(range(10))


def test_claims_are_disjoint_and_in_dataset_order(
    sessions: sessionmaker[Session], run_id: str
) -> None:
    with sessions() as s:
        a = claim(s, "a", 4)
        b = claim(s, "b", 4)
        c = claim(s, "c", 4)
        assert claim(s, "d", 4) == []
    assert [x.case_id for x in a] == ["k0", "k1", "k2", "k3"]
    assert [x.case_id for x in b] == ["k4", "k5", "k6", "k7"]
    assert [x.case_id for x in c] == ["k8", "k9"]
    assert {x.fence for x in a + b + c} == {1}


def test_zero_limit_claims_nothing(sessions: sessionmaker[Session], run_id: str) -> None:
    with sessions() as s:
        assert claim(s, "a", 0) == []


def test_expired_leases_move_to_another_owner_with_a_higher_fence(
    sessions: sessionmaker[Session], run_id: str
) -> None:
    with sessions() as s:
        first = claim(s, "a", 10)
        # Not expired yet: nobody can take them.
        assert claim(s, "b", 10, now_offset=29) == []
        # Expired: the owner itself does not re-claim (the cases are still in its
        # queue), anyone else does, with a new token.
        assert claim(s, "a", 10, now_offset=31) == []
        second = claim(s, "b", 3, now_offset=31)
    assert [x.case_id for x in second] == ["k0", "k1", "k2"]
    assert {x.fence for x in second} == {2}
    with sessions() as s:
        # a's results for the taken cases are refused; the rest are accepted.
        accepted = leases.finish(s, run_id, "a", [(x.case_id, x.fence) for x in first])
        s.commit()
        assert accepted == {f"k{i}" for i in range(3, 10)}
        assert leases.finish(s, run_id, "b", [(x.case_id, x.fence) for x in second]) == {
            "k0",
            "k1",
            "k2",
        }
        s.commit()
        assert leases.remaining(s, run_id) == 0


def test_finish_refuses_a_stale_fence_even_for_the_same_owner(
    sessions: sessionmaker[Session], run_id: str
) -> None:
    with sessions() as s:
        (lease,) = claim(s, "a", 1)
        assert leases.finish(s, run_id, "a", [(lease.case_id, lease.fence - 1)]) == set()
        assert leases.finish(s, run_id, "b", [(lease.case_id, lease.fence)]) == set()
        assert leases.finish(s, run_id, "a", [(lease.case_id, lease.fence)]) == {"k0"}
        # Finished once; a second finish (a duplicate write) is refused.
        assert leases.finish(s, run_id, "a", [(lease.case_id, lease.fence)]) == set()
        s.commit()


def test_finish_handles_more_pairs_than_one_statement(
    sessions: sessionmaker[Session], run_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(leases, "WRITE_CHUNK", 3)
    with sessions() as s:
        claims = claim(s, "a", 10)
        accepted = leases.finish(s, run_id, "a", [(c.case_id, c.fence) for c in claims])
    assert len(accepted) == 10


def test_renew_keeps_leases_from_expiring(sessions: sessionmaker[Session], run_id: str) -> None:
    with sessions() as s:
        claim(s, "a", 5)
        later = utcnow() + timedelta(seconds=25)
        assert leases.renew(s, run_id, "a", ttl=TTL, now=later) == 5
        assert leases.renew(s, run_id, "b", ttl=TTL, now=later) == 0
        s.commit()
        # Past the original expiry, but within the renewed one.
        assert claim(s, "b", 10, now_offset=40) == [
            leases.Claim(i, f"k{i}", 1) for i in range(5, 10)
        ]


def test_release_makes_cases_claimable_at_once(
    sessions: sessionmaker[Session], run_id: str
) -> None:
    with sessions() as s:
        claim(s, "a", 4)
        assert leases.release(s, run_id, "a") == 4
        s.commit()
        again = claim(s, "b", 2)
    assert [(x.case_id, x.fence) for x in again] == [("k0", 2), ("k1", 2)]


def test_outstanding_counts_what_others_hold(sessions: sessionmaker[Session], run_id: str) -> None:
    with sessions() as s:
        assert leases.outstanding(s, run_id, "a") == leases.Outstanding(10, 0, 0, None)
        claim(s, "a", 4)
        claim(s, "b", 4)
        left = leases.outstanding(s, run_id, "a")
        assert (left.unowned, left.held, left.mine, left.total) == (2, 4, 4, 6)
        assert left.next_expiry is not None
        s.execute(update(CaseLease).values(done=True))
        s.commit()
        assert leases.outstanding(s, run_id, "a").total == 0
