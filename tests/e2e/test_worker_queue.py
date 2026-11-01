"""The run queue on PostgreSQL: an API that only queues, worker daemons that
execute, LISTEN/NOTIFY wake-ups and cancellation, SIGKILL and SIGTERM."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx2
import pytest
from sqlalchemy import func, select, text

from evalkit import runqueue
from evalkit.config import Settings
from evalkit.datasets import Case, import_dataset
from evalkit.db import make_engine, make_session_factory
from evalkit.db.models import CaseLease, Result, Run, utcnow
from evalkit.runner import Runner
from evalkit.runspec import RunConfig
from tests.e2e.conftest import TOKEN, evalkit_env, free_port

pytestmark = pytest.mark.e2e
AUTH = {"Authorization": f"Bearer {TOKEN}"}
CASES = 800


def until(check: Callable[[], bool], seconds: float, what: str) -> float:
    """Wait for ``check``; returns how long it took."""
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        if check():
            return time.monotonic() - start
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


@contextmanager
def api(postgres_url: str, workdir: Path) -> Iterator[httpx2.Client]:
    port = free_port()
    env = evalkit_env(
        postgres_url,
        EVALKIT_API_TOKENS=TOKEN,
        EVALKIT_PORT=str(port),
        EVALKIT_MAX_ACTIVE_RUNS="0",
    )
    log = (workdir / "api.log").open("w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "evalkit", "serve"], env=env, stdout=log, stderr=log
    )
    client = httpx2.Client(base_url=f"http://127.0.0.1:{port}", headers=AUTH, timeout=30)

    def healthy() -> bool:
        try:
            return client.get("/healthz").status_code == 200
        except httpx2.TransportError:
            return False

    try:
        until(healthy, 30, "the API")
        yield client
    finally:
        client.close()
        proc.terminate()
        proc.wait(timeout=30)
        log.close()


STARTED: list[subprocess.Popen[str]] = []


@pytest.fixture(autouse=True)
def stop_workers() -> Iterator[None]:
    yield
    for proc in STARTED:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
    STARTED.clear()


def start_worker(postgres_url: str, workdir: Path, name: str, lease: str) -> subprocess.Popen[str]:
    env = evalkit_env(
        postgres_url,
        EVALKIT_LEASE_SECONDS=lease,
        # Polling is slow on purpose: work must arrive by notification.
        EVALKIT_WORKER_POLL_SECONDS="60",
        EVALKIT_DRAIN_SECONDS="20",
        EVALKIT_LOG_LEVEL="INFO",
    )
    with (workdir / f"{name}.log").open("w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "evalkit", "worker", "--runs", "1"],
            env=env,
            stdout=log,
            stderr=log,
            text=True,
        )
    STARTED.append(proc)
    return proc


def listeners(sessions: Any) -> int:
    with sessions() as s:
        query = text(
            "SELECT count(*) FROM pg_stat_activity WHERE query = 'LISTEN \"evalkit_runs\"'"
        )
        return int(s.scalar(query) or 0)


def body(label: str, latency_ms: int, concurrency: int, **queue: Any) -> dict[str, Any]:
    # A seed per label keeps the response cache from serving one run from another.
    return {
        "label": label,
        "dataset": {"name": "fleet"},
        "target": {"type": "fake", "accuracy": 0.8, "latency_ms": latency_ms, "seed": len(label)},
        "scorers": [{"type": "exact"}],
        "options": {"concurrency": concurrency},
        "queue": queue,
    }


def test_worker_slot_caps_hold_under_concurrency(postgres_url: str) -> None:
    engine = make_engine(postgres_url)
    sessions = make_session_factory(engine)
    with sessions() as s:
        import_dataset(s, "caps", [Case(id=f"k{i}", input=i) for i in range(10)])

    config = RunConfig.model_validate(
        {
            "dataset": {"name": "caps"},
            "target": {"type": "fake"},
            "scorers": [{"type": "exact"}],
            "queue": {"max_workers": 3, "priority": 1000},
        }
    )
    run = Runner(sessions, Settings.from_env({})).create_run(config, queued=True)
    barrier = threading.Barrier(8)
    taken: list[str] = []

    def take(worker_id: str) -> None:
        barrier.wait()
        with sessions() as s:
            slot = runqueue.take_slot(
                s, worker_id, busy=set(), ttl=timedelta(seconds=30), now=utcnow()
            )
        if slot is not None and slot.run_id == run.id:
            taken.append(worker_id)

    threads = [threading.Thread(target=take, args=(f"cap{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert len(taken) == 3
    with sessions() as s:
        assert runqueue.live_workers(s, run.id, utcnow()) == 3
        runqueue.cancel(s, run.id, utcnow())
        s.commit()
    engine.dispose()


def test_api_and_four_workers(postgres_url: str, tmp_path: Path) -> None:
    engine = make_engine(postgres_url)
    sessions = make_session_factory(engine)
    data = "\n".join(
        json.dumps({"id": f"f{i:03d}", "input": f"q{i}", "expected": f"a{i}"}) for i in range(CASES)
    )
    workers: list[subprocess.Popen[str]] = []
    with api(postgres_url, tmp_path) as client:
        assert client.post("/v1/datasets/fleet", content=data).status_code == 201
        workers = [start_worker(postgres_url, tmp_path, f"w{i}", "3") for i in range(4)]
        until(lambda: listeners(sessions) >= 4, 30, "four listening workers")

        # Four workers share one run; one is killed with SIGKILL partway.
        run_id = client.post("/v1/runs", json=body("fleet", 100, 2)).json()["run_id"]
        until(lambda: client.get(f"/v1/runs/{run_id}").json()["workers"] == 4, 10, "4 slots")
        until(lambda: client.get(f"/v1/runs/{run_id}").json()["completed"] >= 80, 30, "progress")
        os.kill(workers[0].pid, signal.SIGKILL)
        workers[0].wait(timeout=10)
        until(
            lambda: client.get(f"/v1/runs/{run_id}").json()["status"] == "completed",
            120,
            "the shared run",
        )
        with sessions() as s:
            ids = list(s.scalars(select(Result.case_id).where(Result.run_id == run_id)))
            fences = list(s.scalars(select(CaseLease.fence).where(CaseLease.run_id == run_id)))
        assert len(ids) == len(set(ids)) == CASES
        assert max(fences) == 2, "the killed worker's cases were claimed again"

        # SIGTERM drains: the other three exit cleanly and give their slots back.
        for proc in workers[1:]:
            proc.send_signal(signal.SIGTERM)
        for proc in workers[1:]:
            assert proc.wait(timeout=60) == 0
        cancel_and_drain(client, sessions, postgres_url, tmp_path)
    for name in ("w1", "v0"):
        log = (tmp_path / f"{name}.log").read_text()
        assert "worker stopped" in log, log
    engine.dispose()


def cancel_and_drain(
    client: httpx2.Client, sessions: Any, postgres_url: str, tmp_path: Path
) -> None:
    def run(run_id: str) -> dict[str, Any]:
        summary: dict[str, Any] = client.get(f"/v1/runs/{run_id}").json()
        return summary

    # With a 60 s lease the heartbeat would take 20 s to see a cancellation, and
    # the poll is 60 s: work and cancellations must arrive by notification.
    workers = [start_worker(postgres_url, tmp_path, f"v{i}", "60") for i in range(2)]
    until(lambda: listeners(sessions) >= 2, 30, "two listening workers")
    slow = client.post("/v1/runs", json=body("slow", 200, 4)).json()["run_id"]
    assert until(lambda: run(slow)["workers"] == 2, 10, "slow run started") < 5
    until(lambda: run(slow)["completed"] >= 16, 30, "progress")
    assert client.post(f"/v1/runs/{slow}/cancel").status_code == 202
    assert until(lambda: run(slow)["status"] == "cancelled", 10, "the cancellation") < 5
    cancelled = run(slow)
    assert cancelled["completed"] < CASES
    assert cancelled["workers"] == 0

    # A worker drained mid-run hands its cases to the other, losing nothing.
    third = client.post("/v1/runs", json=body("drained", 100, 4, max_workers=2)).json()["run_id"]
    until(lambda: run(third)["workers"] == 2, 10, "two slots")
    until(lambda: run(third)["completed"] >= 40, 30, "progress")
    workers[0].send_signal(signal.SIGTERM)
    assert workers[0].wait(timeout=60) == 0
    until(lambda: run(third)["status"] == "completed", 120, "the drained run")
    with sessions() as s:
        ids = list(s.scalars(select(Result.case_id).where(Result.run_id == third)))
        usage = s.scalar(select(func.sum(Result.input_tokens)).where(Result.run_id == third))
        statuses = dict(s.execute(select(Run.id, Run.status).where(Run.label == "slow")).all())
    assert len(ids) == len(set(ids)) == CASES
    assert usage == sum(max(1, len(f"q{i}") // 4) for i in range(CASES))
    assert statuses == {slow: "cancelled"}
    workers[1].send_signal(signal.SIGTERM)
    assert workers[1].wait(timeout=60) == 0
