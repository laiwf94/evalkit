"""Case-level dataset edits under concurrent writers on PostgreSQL.

Many threads, each with its own connection, edit the same version at once,
released together by a barrier. Exactly one edit becomes the next version; every
other writer is told the dataset moved on, unless its edit is the one that won
(then it gets that version back). Writers that chain their edits, each retrying
on the newest version after a conflict, lose none of them.
"""

import threading
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from sqlalchemy.orm import Session, sessionmaker

from evalkit.dataset_edits import DatasetEdit, apply_edit, dataset_log, lineage
from evalkit.datasets import Case, get_dataset, import_dataset, load_cases
from evalkit.db import make_engine, make_session_factory
from evalkit.errors import ConflictError

pytestmark = pytest.mark.e2e
WRITERS = 12


@pytest.fixture
def sessions(postgres_url: str) -> Iterator[sessionmaker[Session]]:
    engine = make_engine(postgres_url)
    yield make_session_factory(engine)
    engine.dispose()


def together(fn: Callable[[int], Any]) -> list[Any]:
    barrier = threading.Barrier(WRITERS)
    out: list[Any] = [None] * WRITERS

    def work(i: int) -> None:
        barrier.wait()
        try:
            out[i] = fn(i)
        except Exception as exc:
            out[i] = exc

    threads = [threading.Thread(target=work, args=(i,)) for i in range(WRITERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    return out


def seed(sessions: sessionmaker[Session]) -> tuple[str, str]:
    name = f"d-{uuid.uuid4().hex[:8]}"
    cases = [Case(id=f"c{i:02d}", input={"q": i}, expected=f"a{i}") for i in range(WRITERS)]
    with sessions() as s:
        return name, import_dataset(s, name, cases)[0].content_hash


def test_one_of_many_edits_from_one_version_wins(sessions: sessionmaker[Session]) -> None:
    name, base = seed(sessions)

    def write(i: int) -> tuple[bool, bool, int]:
        body = {"base": base, "update": [{"id": "c00", "expected": f"writer {i}"}]}
        with sessions() as s:
            result = apply_edit(s, name, DatasetEdit.model_validate(body))
            return result.created, result.existing, result.dataset.version

    out = together(write)
    assert [o for o in out if isinstance(o, tuple)] == [(True, False, 2)], out
    assert sum(isinstance(o, ConflictError) for o in out) == WRITERS - 1
    with sessions() as s:
        assert [v.version for v in dataset_log(s, name)] == [2, 1]


def test_identical_edits_mint_one_version(sessions: sessionmaker[Session]) -> None:
    name, base = seed(sessions)

    def write(_: int) -> tuple[bool, bool, int]:
        with sessions() as s:
            body = DatasetEdit.model_validate({"base": base, "delete": ["c05"]})
            result = apply_edit(s, name, body)
            return result.created, result.existing, result.dataset.version

    out = together(write)
    assert sorted(out) == [(False, True, 2)] * (WRITERS - 1) + [(True, False, 2)]


def test_writers_that_retry_on_conflict_lose_no_edit(sessions: sessionmaker[Session]) -> None:
    name, _ = seed(sessions)

    def write(i: int) -> int:
        for _ in range(10 * WRITERS):
            with sessions() as s:
                newest = get_dataset(s, name).content_hash
                body = {"base": newest, "update": [{"id": f"c{i:02d}", "expected": f"done {i}"}]}
                try:
                    return apply_edit(s, name, DatasetEdit.model_validate(body)).dataset.version
                except ConflictError:
                    continue
        raise AssertionError(f"writer {i} gave up after {10 * WRITERS} attempts")

    out = together(write)
    assert sorted(out) == list(range(2, WRITERS + 2)), out
    with sessions() as s:
        final = load_cases(s, get_dataset(s, name))
        assert [c.expected for c in final] == [f"done {i}" for i in range(WRITERS)]
        chain = lineage(s, name)
        assert [v.version for v in chain] == list(range(WRITERS + 1, 0, -1))
        assert all(v.changed == 1 for v in chain[:-1])
