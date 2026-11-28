"""Annotation queues under concurrent annotators on PostgreSQL.

Many threads, each with its own connection, are released together by a barrier.
Claims never put more than ``overlap`` annotators on an item or one annotator on
it twice; the last labels of an item, submitted at the same moment, settle it
exactly once with the right status; and two writers creating one queue end with
one queue.
"""

import threading
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from evalkit.annotation import MAX_OVERLAP, QueueSpec, claim_next, create_queue, list_tasks, submit
from evalkit.datasets import Case, import_dataset
from evalkit.db import make_engine, make_session_factory
from evalkit.db.models import AnnotationAssignment, AnnotationQueue

pytestmark = pytest.mark.e2e
WORKERS = 12


@pytest.fixture
def sessions(postgres_url: str) -> Iterator[sessionmaker[Session]]:
    engine = make_engine(postgres_url)
    yield make_session_factory(engine)
    engine.dispose()


def together(fn: Callable[[int], Any], n: int = WORKERS) -> list[Any]:
    barrier = threading.Barrier(n)
    out: list[Any] = [None] * n

    def work(i: int) -> None:
        barrier.wait()
        try:
            out[i] = fn(i)
        except Exception as exc:
            out[i] = exc

    threads = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    return out


def queue(sessions: sessionmaker[Session], cases: int, overlap: int) -> str:
    name = f"d-{uuid.uuid4().hex[:8]}"
    with sessions() as s:
        import_dataset(s, name, [Case(id=f"c{i:02d}", input={"q": i}) for i in range(cases)])
        spec = QueueSpec(name=name, dataset=name, labels=("yes", "no"), overlap=overlap)
        create_queue(s, spec)
    return name


def test_claims_and_labels_from_many_annotators(sessions: sessionmaker[Session]) -> None:
    name = queue(sessions, 20, 3)

    def work(i: int) -> int:
        done = 0
        with sessions() as s:
            while (claim := claim_next(s, name, f"ann{i}")) is not None:
                submit(s, name, f"ann{i}", claim.case_id, "yes" if i % 5 else "no")
                done += 1
        return done

    out = together(work)
    assert all(isinstance(o, int) for o in out), out
    assert sum(out) == 60
    with sessions() as s:
        qid = s.scalar(select(AnnotationQueue.id).where(AnnotationQueue.name == name))
        rows = list(
            s.scalars(select(AnnotationAssignment).where(AnnotationAssignment.queue_id == qid))
        )
        tasks = list_tasks(s, name)
    by_task: dict[int, list[str]] = {}
    for r in rows:
        by_task.setdefault(r.task_id, []).append(r.annotator)
    assert len(by_task) == 20
    assert all(len(v) == len(set(v)) == 3 for v in by_task.values())
    assert {r.slot for r in rows} == {0, 1, 2}
    for t in tasks:
        labels = {a.label for a in t.assignments}
        assert t.status == ("agreed" if len(labels) == 1 else "disputed"), t


def test_the_last_labels_at_once_settle_the_item(sessions: sessionmaker[Session]) -> None:
    n = MAX_OVERLAP
    for agree in (True, False):
        name = queue(sessions, 1, n)
        with sessions() as s:
            for i in range(n):
                assert claim_next(s, name, f"ann{i}") is not None

        def label(i: int, name: str = name, agree: bool = agree) -> str:
            with sessions() as s:
                value = "yes" if agree or i else "no"
                return submit(s, name, f"ann{i}", "c00", value).status

        out = together(label, n)
        # Exactly one submit saw every label and settled the item.
        settled = "agreed" if agree else "disputed"
        assert sorted(out) == sorted(["open"] * (n - 1) + [settled]), out
        with sessions() as s:
            assert list_tasks(s, name)[0].status == settled


def test_one_queue_from_concurrent_creates(sessions: sessionmaker[Session]) -> None:
    name = f"d-{uuid.uuid4().hex[:8]}"
    with sessions() as s:
        import_dataset(s, name, [Case(id=f"c{i}", input={"q": i}) for i in range(50)])
    spec = QueueSpec(name=name, dataset=name, labels=("yes", "no"))

    def make(_: int) -> tuple[str, bool]:
        with sessions() as s:
            summary, created = create_queue(s, spec)
            return summary.id, created

    out = together(make, 6)
    assert all(isinstance(o, tuple) for o in out), out
    assert len({o[0] for o in out}) == 1
    assert sum(o[1] for o in out) == 1
