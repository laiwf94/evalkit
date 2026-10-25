"""Response cache: the same target given the same input returns the stored output.

The key is SHA-256 over (target hash, canonical input, redaction fingerprint).
The target hash already covers model, prompt, parameters and endpoint, so a
cache hit is only possible when nothing that could change the output changed.
The redaction fingerprint is there because the cache stores redacted text.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session, sessionmaker

from evalkit.datasets import canonical_json, sha256_hex
from evalkit.db.models import CachedResponse

_LOOKUP_CHUNK = 500


def cache_key(target_hash: str, case_input: Any, redaction_fingerprint: str) -> str:
    return sha256_hex(canonical_json([target_hash, case_input, redaction_fingerprint]))


def lookup(session: Session, keys: Sequence[str]) -> dict[str, CachedResponse]:
    found: dict[str, CachedResponse] = {}
    for start in range(0, len(keys), _LOOKUP_CHUNK):
        chunk = keys[start : start + _LOOKUP_CHUNK]
        for row in session.scalars(select(CachedResponse).where(CachedResponse.key.in_(chunk))):
            found[row.key] = row
    return found


def store(session: Session, rows: Iterable[dict[str, Any]]) -> None:
    """Insert entries, keeping the existing one when two runs race on a key."""
    values = list(rows)
    if not values:
        return
    dialect = session.get_bind().dialect.name
    insert = pg_insert if dialect == "postgresql" else sqlite_insert
    session.execute(
        insert(CachedResponse).values(values).on_conflict_do_nothing(index_elements=["key"])
    )


class VerdictStore:
    """Judge answers in the response cache table, keyed by the judge's own hash."""

    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self.sessions = sessions

    async def get(self, key: str) -> str | None:
        def load() -> str | None:
            with self.sessions() as session:
                row = session.get(CachedResponse, key)
                return row.output if row is not None else None

        return await asyncio.to_thread(load)

    async def put(self, key: str, judge_hash: str, text: str, usage: dict[str, Any]) -> None:
        entry = {
            "key": key,
            "target_hash": judge_hash,
            "output": text,
            "stop_reason": "end_turn",
            "usage": usage,
        }

        def save() -> None:
            with self.sessions() as session:
                store(session, [entry])
                session.commit()

        await asyncio.to_thread(save)
