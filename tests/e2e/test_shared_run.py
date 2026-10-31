"""Several processes on one run against PostgreSQL: disjoint claims, one killed."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select

from evalkit import leases
from evalkit.datasets import Case, import_dataset
from evalkit.db import make_engine, make_session_factory
from evalkit.db.models import CaseLease, Result, Run, utcnow
from tests.e2e.conftest import evalkit_env

pytestmark = pytest.mark.e2e
CASES = 800


def test_concurrent_claims_are_disjoint(postgres_url: str) -> None:
    engine = make_engine(postgres_url)
    sessions = make_session_factory(engine)
    with sessions() as s:
        dataset, _ = import_dataset(s, "claims", [Case(id=f"x{i}", input=i) for i in range(2000)])
        s.add(
            Run(
                id="claims-race",
                dataset_id=dataset.id,
                target_spec={},
                target_hash="t" * 64,
                scorers=[],
                options={},
                seed=1,
            )
        )
        s.commit()
        leases.seed(s, "claims-race", [f"x{i}" for i in range(2000)])
        s.commit()
    start = threading.Barrier(8)
    claimed: dict[str, list[leases.Claim]] = {}

    def worker(owner: str) -> None:
        mine: list[leases.Claim] = []
        start.wait()
        while True:
            with sessions() as s:
                batch = leases.claim(
                    s, "claims-race", owner, 7, ttl=timedelta(seconds=60), now=utcnow()
                )
                s.commit()
            if not batch:
                break
            mine += batch
        claimed[owner] = mine

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    ids = [c.case_id for batch in claimed.values() for c in batch]
    assert len(ids) == len(set(ids)) == 2000
    assert {c.fence for batch in claimed.values() for c in batch} == {1}
    # SKIP LOCKED means nobody waited for anybody: every worker got a share.
    assert all(claimed[f"w{i}"] for i in range(8))
    engine.dispose()


def test_four_processes_one_killed(postgres_url: str, tmp_path: Path) -> None:
    data = tmp_path / "shared.jsonl"
    data.write_text(
        "\n".join(
            json.dumps({"id": f"s{i:03d}", "input": f"q{i}", "expected": f"a{i}"})
            for i in range(CASES)
        )
    )
    config = tmp_path / "shared.toml"
    config.write_text(
        'label = "shared"\n[dataset]\nname = "shared"\npath = "shared.jsonl"\n'
        '[target]\ntype = "fake"\naccuracy = 0.8\nlatency_ms = 100\n'
        '[[scorers]]\ntype = "exact"\n[options]\nconcurrency = 2\n'
    )
    env = evalkit_env(postgres_url, EVALKIT_LEASE_SECONDS="3")
    first = subprocess.Popen(
        [sys.executable, "-m", "evalkit", "run", str(config), "--json"],
        env=env,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert first.stderr is not None
    run_id = first.stderr.readline().split()[2]
    others = [
        subprocess.Popen(
            [sys.executable, "-m", "evalkit", "run", "--resume", run_id, "--json"],
            env=env,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        for _ in range(3)
    ]
    engine = make_engine(postgres_url)
    sessions = make_session_factory(engine)

    def saved() -> int:
        with sessions() as s:
            query = select(func.count()).select_from(Result).where(Result.run_id == run_id)
            return s.scalar(query) or 0

    def holders() -> set[str]:
        with sessions() as s:
            query = select(CaseLease.owner).where(
                CaseLease.run_id == run_id, CaseLease.owner.is_not(None)
            )
            return set(s.scalars(query))

    deadline = time.monotonic() + 60
    while (len(holders()) < 4 or saved() < 20) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert len(holders()) == 4, "all four processes should be working on the run"
    victim = others[0]
    os.kill(victim.pid, signal.SIGKILL)
    victim.wait(timeout=10)
    assert saved() < CASES

    survivors = [first, *others[1:]]
    for proc in survivors:
        out, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, err
        summary = json.loads(out)
        assert summary["status"] == "completed"
        assert summary["completed"] == CASES
    for proc in others:
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()

    with sessions() as s:
        ids = list(s.scalars(select(Result.case_id).where(Result.run_id == run_id)))
        tokens = s.scalar(select(func.sum(Result.input_tokens)).where(Result.run_id == run_id))
        fences = list(s.scalars(select(CaseLease.fence).where(CaseLease.run_id == run_id)))
        run = s.get_one(Run, run_id)
    assert run.status == "completed"
    assert len(ids) == len(set(ids)) == CASES
    # Each case is recorded once and its usage counted once.
    assert tokens == sum(max(1, len(f"q{i}") // 4) for i in range(CASES))
    # The killed process held cases; they were claimed again after its leases expired.
    assert max(fences) == 2
    engine.dispose()
