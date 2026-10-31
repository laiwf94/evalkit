"""kill -9 a CLI run against PostgreSQL, then resume it from another process."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from sqlalchemy import func, select

from evalkit.db import make_engine, make_session_factory
from evalkit.db.models import CaseLease, Result, Run
from tests.e2e.conftest import evalkit_env

pytestmark = pytest.mark.e2e
CASES = 200


def test_kill_minus_nine_then_resume(postgres_url: str, tmp_path: Path) -> None:
    data = tmp_path / "crash.jsonl"
    data.write_text(
        "\n".join(
            json.dumps({"id": f"k{i:03d}", "input": f"q{i}", "expected": f"a{i}"})
            for i in range(CASES)
        )
    )
    config = tmp_path / "crash.toml"
    config.write_text(
        'label = "crash"\n[dataset]\nname = "crash"\npath = "crash.jsonl"\n'
        '[target]\ntype = "fake"\naccuracy = 0.8\nlatency_ms = 40\n'
        '[[scorers]]\ntype = "exact"\n[options]\nconcurrency = 4\n'
    )
    env = evalkit_env(postgres_url, EVALKIT_LEASE_SECONDS="3")
    proc = subprocess.Popen(
        [sys.executable, "-m", "evalkit", "run", str(config)],
        env=env,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        text=True,
    )
    assert proc.stderr is not None
    first_line = proc.stderr.readline()
    run_id = first_line.split()[2]
    engine = make_engine(postgres_url)
    sessions = make_session_factory(engine)

    def saved() -> int:
        with sessions() as s:
            return (
                s.scalar(select(func.count()).select_from(Result).where(Result.run_id == run_id))
                or 0
            )

    while saved() < 40:
        time.sleep(0.05)
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait(timeout=10)
    proc.stderr.close()
    before = saved()
    with sessions() as s:
        run = s.get_one(Run, run_id)
        assert run.status == "running"  # nobody got to mark it: the process died
        held = s.scalar(
            select(func.count())
            .select_from(CaseLease)
            .where(CaseLease.run_id == run_id, CaseLease.owner.is_not(None))
        )
        assert held  # and its claimed cases are still leased to it
    assert 40 <= before < CASES

    # A resume joins at once, runs what nobody holds, and takes the dead
    # process's cases over when their leases expire.
    started = time.monotonic()
    resumed = subprocess.run(
        [sys.executable, "-m", "evalkit", "run", "--resume", run_id, "--json"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert time.monotonic() - started < 60
    assert resumed.returncode == 0, resumed.stderr
    summary = json.loads(resumed.stdout)
    assert summary["status"] == "completed"
    assert summary["completed"] == CASES
    with sessions() as s:
        ids = list(s.scalars(select(Result.case_id).where(Result.run_id == run_id)))
        tokens = s.scalar(select(func.sum(Result.input_tokens)).where(Result.run_id == run_id))
    assert len(ids) == len(set(ids)) == CASES
    # Usage is counted once per case: a case recorded before the crash was not paid for again.
    assert tokens == sum(max(1, len(f"q{i}") // 4) for i in range(CASES))
    engine.dispose()
