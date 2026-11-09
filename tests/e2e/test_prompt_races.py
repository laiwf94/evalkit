"""The prompt registry under concurrent writers on PostgreSQL.

Many threads, each with its own connection, push the same content or move the
same label from the same expected version at once, released together by a
barrier. Exactly one version is minted per distinct content, and exactly one
label move wins; every loser is told where the label is now.
"""

import threading
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from evalkit.db import make_engine, make_session_factory
from evalkit.db.models import PromptVersion
from evalkit.errors import ConflictError
from evalkit.prompts import get_prompt, labels_of, move_label, push_version

pytestmark = pytest.mark.e2e
WRITERS = 12


@pytest.fixture
def sessions(postgres_url: str) -> Iterator[sessionmaker[Session]]:
    engine = make_engine(postgres_url)
    yield make_session_factory(engine)
    engine.dispose()


def together(fn: Callable[[int], Any]) -> list[Any]:
    """Call ``fn(i)`` on WRITERS threads released at once; results or exceptions."""
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


def test_concurrent_first_pushes_of_the_same_content(sessions: sessionmaker[Session]) -> None:
    name = f"p-{uuid.uuid4().hex[:8]}"

    def push(_: int) -> tuple[int, bool]:
        with sessions() as s:
            version, created = push_version(s, name, template="Answer: {{q}}")
            return version.version, created

    outcomes = together(push)
    assert all(isinstance(o, tuple) for o in outcomes), outcomes
    assert {o[0] for o in outcomes} == {1}
    assert sum(o[1] for o in outcomes) == 1


def test_concurrent_pushes_of_different_content(sessions: sessionmaker[Session]) -> None:
    name = f"p-{uuid.uuid4().hex[:8]}"
    with sessions() as s:
        push_version(s, name, template="First: {{q}}")

    def push(i: int) -> int:
        with sessions() as s:
            # Two writers per content: each pair must share one version.
            return push_version(s, name, template=f"Variant {i % 6}: {{{{q}}}}")[0].version

    outcomes = together(push)
    errors = [o for o in outcomes if isinstance(o, Exception)]
    # A writer that loses five races in a row gives up with a conflict, never a duplicate.
    assert all(isinstance(e, ConflictError) for e in errors), errors
    with sessions() as s:
        prompt = get_prompt(s, name)
        rows = s.execute(
            select(PromptVersion.version, PromptVersion.template).where(
                PromptVersion.prompt_id == prompt.id
            )
        ).all()
        versions = sorted(v for v, _ in rows)
        templates = {t for _, t in rows}
        count = s.scalar(
            select(func.count())
            .select_from(PromptVersion)
            .where(PromptVersion.prompt_id == prompt.id)
        )
    assert versions == list(range(1, len(versions) + 1))
    assert count == len(templates)
    by_variant: dict[int, set[int]] = {}
    for i, o in enumerate(outcomes):
        if isinstance(o, int):
            by_variant.setdefault(i % 6, set()).add(o)
    assert all(len(v) == 1 for v in by_variant.values()), by_variant


def test_exactly_one_label_move_wins(sessions: sessionmaker[Session]) -> None:
    name = f"p-{uuid.uuid4().hex[:8]}"
    with sessions() as s:
        for i in range(WRITERS + 1):
            push_version(s, name, template=f"Version {i}: {{{{q}}}}")
        move_label(s, name, "production", 1, expected=None)

    def promote(i: int) -> int:
        with sessions() as s:
            # Every writer read the label at version 1 and promotes its own version.
            return move_label(s, name, "production", i + 2, expected=1).version

    outcomes = together(promote)
    winners = [o for o in outcomes if isinstance(o, int)]
    losers = [o for o in outcomes if isinstance(o, Exception)]
    assert len(winners) == 1, outcomes
    assert all(isinstance(e, ConflictError) for e in losers)
    assert all(f"is at version {winners[0]}" in str(e) for e in losers)
    with sessions() as s:
        assert labels_of(s, get_prompt(s, name)) == {"production": winners[0]}


def test_exactly_one_label_creation_wins(sessions: sessionmaker[Session]) -> None:
    name = f"p-{uuid.uuid4().hex[:8]}"
    with sessions() as s:
        for i in range(WRITERS):
            push_version(s, name, template=f"Version {i}: {{{{q}}}}")

    def create(i: int) -> int:
        with sessions() as s:
            return move_label(s, name, "canary", i + 1, expected=None).version

    outcomes = together(create)
    winners = [o for o in outcomes if isinstance(o, int)]
    assert len(winners) == 1, outcomes
    assert all(isinstance(o, ConflictError) for o in outcomes if not isinstance(o, int))
