import asyncio
import os
import signal
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, sessionmaker

from evalkit import runqueue
from evalkit.config import Settings
from evalkit.datasets import import_dataset
from evalkit.db import make_engine, make_session_factory, upgrade
from evalkit.db.models import CaseLease, Result, Run, RunWorker, utcnow
from evalkit.errors import ConflictError, NotFoundError
from evalkit.runner import Runner
from evalkit.runspec import RunConfig
from evalkit.worker import Worker, listen_url
from tests.test_cli import run_cli, write_config
from tests.test_runner import CountingFactory, make_cases

pytestmark = pytest.mark.anyio

LEASE = timedelta(seconds=2)


@pytest.fixture
def dataset(sessions: sessionmaker[Session]) -> str:
    with sessions() as s:
        import_dataset(s, "qa", make_cases(40))
    return "qa"


def config(*, priority: int = 0, max_workers: int | None = None, **target: Any) -> RunConfig:
    options = {"concurrency": 4, **target.pop("options", {})}
    return RunConfig.model_validate(
        {
            "label": target.pop("label", "queued"),
            "dataset": {"name": "qa"},
            "target": {"type": "fake", **target},
            "scorers": [{"type": "exact"}],
            "options": options,
            "queue": {"priority": priority, "max_workers": max_workers},
        }
    )


def make_runner(
    sessions: sessionmaker[Session], factory: CountingFactory | None = None, **kw: Any
) -> Runner:
    return Runner(
        sessions,
        Settings.from_env({}),
        target_factory=factory or CountingFactory(),
        lease_ttl=kw.pop("lease_ttl", LEASE),
        **kw,
    )


def make_worker(sessions: sessionmaker[Session], runner: Runner | None = None, **kw: Any) -> Worker:
    return Worker(
        sessions,
        runner or make_runner(sessions),
        slots=kw.pop("slots", 1),
        poll_seconds=kw.pop("poll_seconds", 0.05),
        drain_seconds=kw.pop("drain_seconds", 10.0),
        **kw,
    )


@asynccontextmanager
async def running(worker: Worker) -> AsyncIterator[asyncio.Event]:
    stop = asyncio.Event()
    task = asyncio.create_task(worker.serve(stop))
    try:
        yield stop
    finally:
        stop.set()
        await task


def status(sessions: sessionmaker[Session], run_id: str) -> str:
    with sessions() as s:
        return s.get_one(Run, run_id).status


def results(sessions: sessionmaker[Session], run_id: str) -> list[str]:
    with sessions() as s:
        return list(s.scalars(select(Result.case_id).where(Result.run_id == run_id)))


async def until(check: Any, seconds: float = 10.0) -> None:
    for _ in range(int(seconds / 0.01)):
        if check():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached")


def queue(sessions: sessionmaker[Session], cfg: RunConfig) -> str:
    return make_runner(sessions).create_run(cfg, queued=True).id


async def test_a_worker_executes_queued_runs_but_not_cli_runs(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    cli_run = make_runner(sessions).create_run(config())
    queued = queue(sessions, config(label="q"))
    worker = make_worker(sessions)
    async with running(worker):
        await until(lambda: status(sessions, queued) == "completed")
        await until(lambda: not worker.runs)
    assert len(set(results(sessions, queued))) == 40
    assert status(sessions, cli_run.id) == "pending"
    with sessions() as s:
        assert s.scalar(select(func.count()).select_from(RunWorker)) == 0


async def test_higher_priority_runs_go_first(sessions: sessionmaker[Session], dataset: str) -> None:
    low = queue(sessions, config(priority=-1, label="low"))
    normal = queue(sessions, config(label="normal"))
    high = queue(sessions, config(priority=10, label="high"))
    order: list[str] = []
    runner = make_runner(sessions)
    execute = runner.execute

    async def recording(run_id: str) -> Any:
        order.append(run_id)
        return await execute(run_id)

    runner.execute = recording  # type: ignore[method-assign]
    async with running(make_worker(sessions, runner)):
        await until(lambda: status(sessions, low) == "completed")
    assert order == [high, normal, low]


def test_worker_caps_hold_under_concurrent_slot_taking(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    capped = queue(sessions, config(max_workers=3))
    barrier = threading.Barrier(8)
    taken: list[str] = []

    def take(worker_id: str) -> None:
        barrier.wait()
        with sessions() as s:
            slot = runqueue.take_slot(s, worker_id, busy=set(), ttl=LEASE, now=utcnow())
        if slot is not None:
            taken.append(worker_id)

    threads = [threading.Thread(target=take, args=(f"w{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(taken) == 3
    with sessions() as s:
        assert runqueue.live_workers(s, capped, utcnow()) == 3
        # A worker never takes a second slot on a run it already executes.
        busy = runqueue.take_slot(s, taken[0], busy={capped}, ttl=LEASE, now=utcnow())
        assert busy is None
        # An expired slot no longer counts, so a new worker may take its place.
        later = utcnow() + LEASE * 2
        assert runqueue.take_slot(s, "w-late", busy=set(), ttl=LEASE, now=later) is not None


def test_batch_runs_get_one_worker(sessions: sessionmaker[Session], dataset: str) -> None:
    cfg = RunConfig.model_validate(
        {
            "dataset": {"name": "qa"},
            "target": {"type": "prompt", "model": "claude-opus-5-5", "template": "{{q}}"},
            "scorers": [{"type": "exact"}],
            "options": {"mode": "batch"},
            "queue": {"max_workers": 8},
        }
    )
    run = make_runner(sessions).create_run(cfg, queued=True)
    assert run.max_workers == 1


def test_a_running_run_is_joined_only_while_it_has_unheld_cases(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    run_id = queue(sessions, config())
    now = utcnow()
    with sessions() as s:
        s.execute(update(Run).where(Run.id == run_id).values(status="running", leases_seeded=True))
        s.add_all(
            CaseLease(
                run_id=run_id,
                case_id=f"c{i:03d}",
                position=i,
                owner="other",
                expires_at=now + LEASE,
                fence=1,
                done=False,
            )
            for i in range(3)
        )
        s.commit()
        assert runqueue.take_slot(s, "w1", busy=set(), ttl=LEASE, now=now) is None
        # Once the holder's leases expire, its cases are there to take over.
        later = now + LEASE * 2
        assert runqueue.take_slot(s, "w1", busy=set(), ttl=LEASE, now=later) is not None


async def test_cancel_reaches_a_worker_through_a_notification(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    run_id = queue(sessions, config(latency_ms=100))
    # A long lease: the heartbeat would take minutes, so only the notification
    # can stop the run this quickly.
    runner = make_runner(sessions, lease_ttl=timedelta(seconds=600))
    worker = make_worker(sessions, runner)
    async with running(worker):
        await until(lambda: len(results(sessions, run_id)) >= 4)
        with sessions() as s:
            assert runqueue.cancel(s, run_id, utcnow()) == "cancelling"
            s.commit()
        worker.on_notification(f"cancel:{run_id}")
        await until(lambda: status(sessions, run_id) == "cancelled", seconds=2)
        await until(lambda: not worker.runs)
    saved = results(sessions, run_id)
    assert 4 <= len(saved) < 40
    assert len(saved) == len(set(saved))
    with sessions() as s:
        run = s.get_one(Run, run_id)
        assert run.finished_at is not None
        held = s.scalar(
            select(func.count()).select_from(CaseLease).where(CaseLease.owner.is_not(None))
        )
        assert held == 0


async def test_cancel_is_noticed_by_the_heartbeat_without_notifications(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    run_id = queue(sessions, config(latency_ms=50))
    runner = make_runner(sessions, lease_ttl=timedelta(seconds=2), heartbeat_seconds=0.1)
    worker = make_worker(sessions, runner)
    async with running(worker):
        await until(lambda: len(results(sessions, run_id)) >= 4)
        with sessions() as s:
            runqueue.cancel(s, run_id, utcnow())
            s.commit()
        await until(lambda: status(sessions, run_id) == "cancelled", seconds=3)
    assert len(results(sessions, run_id)) < 40


async def test_sigterm_drain_finishes_cases_in_flight_and_requeues(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    run_id = queue(sessions, config(latency_ms=300, options={"concurrency": 8}))
    factory = CountingFactory()
    first = make_worker(sessions, make_runner(sessions, factory))
    async with running(first):
        await until(lambda: factory.calls >= 8)
    # The eight cases in flight finished and were saved; nothing else ran.
    saved = results(sessions, run_id)
    assert len(saved) == factory.finished == 8
    assert status(sessions, run_id) == "queued"
    with sessions() as s:
        assert runqueue.live_workers(s, run_id, utcnow()) == 0
        held = s.scalar(
            select(func.count()).select_from(CaseLease).where(CaseLease.owner.is_not(None))
        )
        assert held == 0
    # Another worker picks the run up and executes only the rest.
    second_factory = CountingFactory()
    second = make_worker(sessions, make_runner(sessions, second_factory))
    async with running(second):
        await until(lambda: status(sessions, run_id) == "completed")
    assert second_factory.calls == 32
    final = results(sessions, run_id)
    assert len(final) == len(set(final)) == 40


async def test_drain_gives_up_after_its_time_limit(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    run_id = queue(sessions, config(latency_ms=5000))
    factory = CountingFactory()
    worker = make_worker(sessions, make_runner(sessions, factory), drain_seconds=0.1)
    async with running(worker):
        await until(lambda: factory.calls >= 4)
    # The calls in flight were abandoned, not saved, and the run went back.
    assert results(sessions, run_id) == []
    assert status(sessions, run_id) == "queued"


async def test_abort_skips_the_drain_wait(sessions: sessionmaker[Session], dataset: str) -> None:
    run_id = queue(sessions, config(latency_ms=5000))
    factory = CountingFactory()
    worker = make_worker(sessions, make_runner(sessions, factory), drain_seconds=60)
    stop = asyncio.Event()
    task = asyncio.create_task(worker.serve(stop))
    await until(lambda: factory.calls >= 4)
    stop.set()
    await asyncio.sleep(0.05)
    worker.abort()
    await asyncio.wait_for(task, 5)
    assert status(sessions, run_id) == "queued"


async def test_a_run_queued_while_a_worker_waits_starts_on_wake(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    worker = make_worker(sessions, poll_seconds=60)
    async with running(worker):
        await asyncio.sleep(0.05)
        run_id = queue(sessions, config())
        worker.on_notification(f"queued:{run_id}")
        await until(lambda: status(sessions, run_id) == "completed", seconds=5)


async def test_several_workers_share_a_run_and_respect_its_cap(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    run_id = queue(sessions, config(max_workers=2, latency_ms=20))
    workers = [make_worker(sessions) for _ in range(3)]
    seen: set[int] = set()
    async with running(workers[0]), running(workers[1]), running(workers[2]):

        def check() -> bool:
            with sessions() as s:
                seen.add(runqueue.live_workers(s, run_id, utcnow()))
            return status(sessions, run_id) == "completed"

        await until(check)
    assert max(seen) == 2
    final = results(sessions, run_id)
    assert len(final) == len(set(final)) == 40


async def test_queue_errors_do_not_stop_the_worker(
    sessions: sessionmaker[Session], dataset: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = make_worker(sessions)
    failures = iter([True])
    take = worker._take_slot

    def flaky(busy: set[str]) -> Any:
        if next(failures, False):
            raise RuntimeError("database went away")
        return take(busy)

    monkeypatch.setattr(worker, "_take_slot", flaky)
    run_id = queue(sessions, config())
    async with running(worker):
        await until(lambda: status(sessions, run_id) == "completed")


async def test_a_failing_run_frees_its_slot(sessions: sessionmaker[Session], dataset: str) -> None:
    def broken(spec: Any) -> Any:
        raise RuntimeError("target exploded")

    runner = Runner(sessions, Settings.from_env({}), target_factory=broken, lease_ttl=LEASE)
    run_id = queue(sessions, config())
    worker = make_worker(sessions, runner)
    async with running(worker):
        await until(lambda: status(sessions, run_id) == "failed")
        await until(lambda: not worker.runs)
    with sessions() as s:
        assert runqueue.live_workers(s, run_id, utcnow()) == 0


def test_cancel_and_enqueue_states(sessions: sessionmaker[Session], dataset: str) -> None:
    runner = make_runner(sessions)
    now = utcnow()
    with sessions() as s:
        with pytest.raises(NotFoundError):
            runqueue.cancel(s, "nope", now)
        with pytest.raises(NotFoundError):
            runqueue.enqueue(s, "nope", now)
    failed = runner.create_run(config())
    running_run = runner.create_run(config())
    with sessions() as s:
        s.execute(update(Run).where(Run.id == failed.id).values(status="failed", error="boom"))
        s.execute(update(Run).where(Run.id == running_run.id).values(status="running"))
        s.commit()
        assert runqueue.enqueue(s, failed.id, now) == "queued"
        assert runqueue.enqueue(s, running_run.id, now) == "running"
        s.commit()
        assert s.get_one(Run, failed.id).error == ""
        assert s.get_one(Run, running_run.id).queued_at is not None
        # A process holds a case: the run is cancelling until it stops.
        s.add(
            CaseLease(
                run_id=running_run.id,
                case_id="c000",
                position=0,
                owner="cli",
                expires_at=now + LEASE,
                fence=1,
                done=False,
            )
        )
        s.commit()
        assert runqueue.cancel(s, running_run.id, now) == "cancelling"
        s.commit()
        with pytest.raises(ConflictError):
            runqueue.enqueue(s, running_run.id, now)
        # It died: the sweep finishes the cancellation once its lease has expired.
        assert runqueue.sweep(s, now) == 0
        assert runqueue.sweep(s, now + LEASE * 2) == 1
        assert s.get_one(Run, running_run.id).status == "cancelled"
        s.execute(update(Run).where(Run.id == failed.id).values(status="completed"))
        s.commit()
        assert runqueue.enqueue(s, failed.id, now) == "completed"
        with pytest.raises(ConflictError):
            runqueue.cancel(s, failed.id, now)


async def test_runner_stops_at_once_on_a_cancelled_run(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    runner = make_runner(sessions)
    run_id = runner.create_run(config()).id
    with sessions() as s:
        runqueue.cancel(s, run_id, utcnow())
        s.commit()
    report = await runner.execute(run_id)
    assert report.status == "cancelled"
    assert results(sessions, run_id) == []
    assert not runner.request_cancel(run_id)
    assert not runner.request_drain(run_id)


def test_worker_needs_a_slot(sessions: sessionmaker[Session]) -> None:
    with pytest.raises(ValueError, match="at least one slot"):
        make_worker(sessions, slots=0)


def test_listen_url() -> None:
    assert listen_url("sqlite:///x.db") is None
    assert (
        listen_url("postgresql+psycopg://u:p%40ss@db:5433/evalkit?sslmode=require")
        == "postgresql://u:p%40ss@db:5433/evalkit?sslmode=require"
    )


async def test_listener_failures_fall_back_to_polling(
    sessions: sessionmaker[Session], dataset: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("evalkit.worker.LISTEN_BACKOFF_SECONDS", (0.01, 0.02))
    worker = make_worker(sessions, listen="postgresql://u@127.0.0.1:1/none")
    run_id = queue(sessions, config())
    async with running(worker):
        await until(lambda: status(sessions, run_id) == "completed")
        assert not worker.listening.is_set()


def test_cli_queue_cancel_and_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = f"sqlite:///{tmp_path / 'w.db'}"
    cfg = write_config(tmp_path, "queued", 0.9, "[queue]\npriority = 3\n")
    code, out = run_cli(db, "run", str(cfg), "--queue", "--json")
    assert code == 0
    assert '"status": "queued"' in out
    assert '"priority": 3' in out
    second = run_cli(db, "run", str(cfg), "--queue", "--json")[1]
    cancelled = second.split('"run_id": "')[1].split('"')[0]
    assert run_cli(db, "cancel", cancelled) == (0, f"run {cancelled} cancelled\n")
    assert run_cli(db, "worker", "--runs", "0")[0] == 2

    engine = make_engine(db)
    upgrade(engine)
    sessions = make_session_factory(engine)

    def stop_when_done() -> None:
        # The worker stops on SIGTERM once the queued run is completed.
        for _ in range(3000):
            with sessions() as s:
                if s.scalar(select(func.count()).select_from(Run).where(Run.status == "completed")):
                    break
            threading.Event().wait(0.01)
        os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setenv("EVALKIT_WORKER_POLL_SECONDS", "0.05")
    stopper = threading.Thread(target=stop_when_done)
    stopper.start()
    assert run_cli(db, "worker", "--runs", "2")[0] == 0
    stopper.join()
    with sessions() as s:
        statuses = sorted(s.scalars(select(Run.status)))
    assert statuses == ["cancelled", "completed"]
    engine.dispose()
