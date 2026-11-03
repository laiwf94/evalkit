"""Response cache: the same target given the same input returns the stored output.

The key is SHA-256 over (target hash, canonical input, redaction fingerprint).
The target hash already covers model, prompt, parameters and endpoint, so a
cache hit is only possible when nothing that could change the output changed.
The redaction fingerprint is there because the cache stores redacted text.

The cache is bounded. Every entry records when it was last used (a lookup
refreshes that at most once per ``TOUCH_INTERVAL``, so a hot entry costs one
write an hour, not one per run) and its output's size. ``sweep`` deletes
entries unused for longer than a maximum age, then the least recently used
ones until the total size is under a maximum. Outputs of ``blob_min_bytes`` or
more can live in a content-addressed blob store (``blobs.py``) with only the
digest in the row; ``collect_blobs`` then removes blobs no entry refers to.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import and_, delete, func, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session, sessionmaker

from evalkit import blobs, metrics
from evalkit.datasets import canonical_json, sha256_hex
from evalkit.db.models import CachedResponse, utcnow

log = logging.getLogger(__name__)

_LOOKUP_CHUNK = 500
SWEEP_CHUNK = 1000
TOUCH_INTERVAL = timedelta(hours=1)


@dataclass(frozen=True)
class Hit:
    output: str
    stop_reason: str | None


def cache_key(target_hash: str, case_input: Any, redaction_fingerprint: str) -> str:
    return sha256_hex(canonical_json([target_hash, case_input, redaction_fingerprint]))


def existing(session: Session, keys: Sequence[str]) -> set[str]:
    """Which keys have an entry, without reading outputs or marking them used."""
    found: set[str] = set()
    for start in range(0, len(keys), _LOOKUP_CHUNK):
        chunk = keys[start : start + _LOOKUP_CHUNK]
        found.update(
            session.scalars(select(CachedResponse.key).where(CachedResponse.key.in_(chunk)))
        )
    return found


def lookup(
    session: Session,
    keys: Sequence[str],
    store: blobs.BlobStore | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, Hit]:
    """The entries for ``keys``, with blob outputs read back, and marks them used.

    An entry whose blob is missing or corrupt is deleted and treated as a miss,
    so the next fresh output replaces it. One whose blob cannot be read right
    now (store unreachable, or no store configured) is a miss and kept. The
    caller commits."""
    now = now or utcnow()
    found: dict[str, Hit] = {}
    broken: list[str] = []
    for start in range(0, len(keys), _LOOKUP_CHUNK):
        chunk = keys[start : start + _LOOKUP_CHUNK]
        rows = session.execute(
            select(
                CachedResponse.key,
                CachedResponse.output,
                CachedResponse.stop_reason,
                CachedResponse.output_blob,
            ).where(CachedResponse.key.in_(chunk))
        )
        for key, output, stop_reason, blob in rows:
            if blob is None:
                found[key] = Hit(output, stop_reason)
                continue
            text = _read_blob(store, key, blob)
            if text is _BROKEN:
                broken.append(key)
            elif text is not None:
                found[key] = Hit(str(text), stop_reason)
    hit_keys = list(found)
    for start in range(0, len(hit_keys), _LOOKUP_CHUNK):
        session.execute(
            update(CachedResponse)
            .where(
                CachedResponse.key.in_(hit_keys[start : start + _LOOKUP_CHUNK]),
                CachedResponse.last_used_at < now - TOUCH_INTERVAL,
            )
            .values(last_used_at=now)
        )
    if broken:
        session.execute(delete(CachedResponse).where(CachedResponse.key.in_(broken)))
    return found


_BROKEN = object()


def _read_blob(store: blobs.BlobStore | None, key: str, digest: str) -> object:
    if store is None:
        log.warning("cache entry is in the blob store, but none is configured", extra={"key": key})
        return None
    try:
        return store.get(digest).decode("utf-8")
    except (blobs.BlobNotFoundError, blobs.BlobCorruptError, UnicodeDecodeError) as exc:
        log.warning("cache entry's blob is unusable; dropping it", extra={"key": key, "error": exc})
        metrics.BLOB_FAILURES.labels("unusable").inc()
        return _BROKEN
    except blobs.BlobError:
        log.warning("cannot read a cache entry's blob", extra={"key": key}, exc_info=True)
        metrics.BLOB_FAILURES.labels("unreachable").inc()
        return None


def offload(
    entries: Iterable[dict[str, Any]], store: blobs.BlobStore | None, min_bytes: int
) -> list[dict[str, Any]]:
    """Entries ready for ``store``: each gets its size, and outputs of
    ``min_bytes`` or more are written to the blob store first, leaving the
    digest in the entry. If the blob store fails, the output stays inline."""
    ready = []
    for entry in entries:
        data = entry["output"].encode("utf-8")
        row = {**entry, "size_bytes": len(data)}
        if store is not None and len(data) >= min_bytes and data:
            try:
                row["output_blob"] = store.put(data)
                row["output"] = ""
            except blobs.BlobError:
                log.warning("cannot write to the blob store; caching inline", exc_info=True)
                metrics.BLOB_FAILURES.labels("write").inc()
        ready.append(row)
    return ready


def store(session: Session, rows: Iterable[dict[str, Any]], *, now: datetime | None = None) -> None:
    """Insert entries, keeping the existing one when two runs race on a key."""
    now = now or utcnow()
    values = []
    for row in rows:
        value = {"created_at": now, "last_used_at": now, **row}
        value.setdefault("size_bytes", len(value["output"].encode("utf-8")))
        value.setdefault("output_blob", None)  # a multi-row insert needs every column
        values.append(value)
    if not values:
        return
    dialect = session.get_bind().dialect.name
    insert = pg_insert if dialect == "postgresql" else sqlite_insert
    session.execute(
        insert(CachedResponse).values(values).on_conflict_do_nothing(index_elements=["key"])
    )


@dataclass
class SweepReport:
    removed: int = 0
    removed_bytes: int = 0
    kept: int = 0
    kept_bytes: int = 0
    dry_run: bool = False


def sweep(
    session: Session,
    *,
    max_age: timedelta | None = None,
    max_bytes: int | None = None,
    now: datetime | None = None,
    dry_run: bool = False,
    chunk: int = SWEEP_CHUNK,
) -> SweepReport:
    """Delete entries unused for longer than ``max_age``, then the least
    recently used ones until the rest add up to at most ``max_bytes``.

    Deletes go in chunks of ``chunk`` keys, each in its own transaction, so a
    large sweep never holds many locks. A chunk only deletes entries whose
    last use is no newer than when the sweep saw them, so an entry that a run
    uses meanwhile survives."""
    now = now or utcnow()
    report = SweepReport(dry_run=dry_run)
    if max_age is not None:
        cutoff = now - max_age
        while True:
            rows = session.execute(
                select(CachedResponse.key, CachedResponse.size_bytes)
                .where(CachedResponse.last_used_at < cutoff)
                .order_by(CachedResponse.last_used_at, CachedResponse.key)
                .offset(report.removed if dry_run else 0)
                .limit(chunk)
            ).all()
            if not rows:
                break
            report.removed += len(rows)
            report.removed_bytes += sum(size for _, size in rows)
            if not dry_run:
                keys = [key for key, _ in rows]
                session.execute(
                    delete(CachedResponse).where(
                        CachedResponse.key.in_(keys), CachedResponse.last_used_at < cutoff
                    )
                )
                session.commit()
    if max_bytes is not None:
        _sweep_size(session, max_bytes, report, dry_run=dry_run, chunk=chunk)
    count, total = session.execute(
        select(func.count(), func.coalesce(func.sum(CachedResponse.size_bytes), 0))
    ).one()
    report.kept, report.kept_bytes = int(count), int(total)
    if dry_run:
        # Nothing was deleted: what would be kept is what is there minus what would go.
        report.kept -= report.removed
        report.kept_bytes -= report.removed_bytes
    else:
        metrics.CACHE_SWEPT.inc(report.removed)
    return report


def _sweep_size(
    session: Session, max_bytes: int, report: SweepReport, *, dry_run: bool, chunk: int
) -> None:
    total = int(session.scalar(select(func.coalesce(func.sum(CachedResponse.size_bytes), 0))) or 0)
    if dry_run:
        total -= report.removed_bytes
    excess = total - max_bytes
    # Keyset pagination in least-recently-used order; in a dry run the age
    # phase's entries come first in that order and are skipped by position.
    position: tuple[datetime, str] | None = None
    skip = report.removed if dry_run else 0
    while excess > 0:
        query = select(
            CachedResponse.key, CachedResponse.size_bytes, CachedResponse.last_used_at
        ).order_by(CachedResponse.last_used_at, CachedResponse.key)
        if position is not None:
            used, key = position
            query = query.where(
                or_(
                    CachedResponse.last_used_at > used,
                    and_(CachedResponse.last_used_at == used, CachedResponse.key > key),
                )
            )
        elif skip:
            query = query.offset(skip)
        rows = session.execute(query.limit(chunk)).all()
        if not rows:
            break
        victims = []
        for key, size, used in rows:
            if excess <= 0:
                break
            victims.append(key)
            excess -= size
            report.removed += 1
            report.removed_bytes += size
            position = (used, key)
        if not dry_run and position is not None:
            session.execute(
                delete(CachedResponse).where(
                    CachedResponse.key.in_(victims), CachedResponse.last_used_at <= position[0]
                )
            )
            session.commit()


def referenced_blobs(session: Session, digests: Sequence[str]) -> set[str]:
    if not digests:
        return set()
    return set(
        session.scalars(
            select(CachedResponse.output_blob).where(CachedResponse.output_blob.in_(digests))
        )
    )


def collect_blobs(
    session: Session,
    store: blobs.BlobStore,
    *,
    grace: timedelta,
    now: datetime | None = None,
    dry_run: bool = False,
) -> blobs.GarbageReport:
    """Remove blobs that no cache entry refers to and that are older than ``grace``."""
    now = now or utcnow()

    def used(digests: Sequence[str]) -> set[str]:
        found = referenced_blobs(session, digests)
        session.rollback()  # end the read transaction between pages
        return found

    report = blobs.collect_garbage(store, used, before=now - grace, dry_run=dry_run)
    if not dry_run:
        metrics.BLOBS_COLLECTED.inc(report.removed)
    return report


# Session-level advisory lock id on PostgreSQL: one sweeper at a time per database.
SWEEP_LOCK_ID = 0x65766B_6361  # "evk" "ca"


@dataclass
class Maintenance:
    sweep: SweepReport
    blobs: blobs.GarbageReport | None
    skipped: bool = False


def maintain(
    sessions: sessionmaker[Session],
    store: blobs.BlobStore | None,
    *,
    max_age: timedelta | None,
    max_bytes: int | None,
    grace: timedelta,
    dry_run: bool = False,
    now: datetime | None = None,
) -> Maintenance:
    """Sweep the cache, then collect unreferenced blobs. On PostgreSQL this
    takes an advisory lock first and does nothing (``skipped``) if another
    process is already maintaining the cache."""
    engine = sessions.kw["bind"]
    postgres = engine.dialect.name == "postgresql"
    with engine.connect() as lock:
        if postgres and not lock.scalar(
            text("SELECT pg_try_advisory_lock(:id)"), {"id": SWEEP_LOCK_ID}
        ):
            return Maintenance(SweepReport(dry_run=dry_run), None, skipped=True)
        try:
            with sessions() as session:
                report = sweep(
                    session, max_age=max_age, max_bytes=max_bytes, now=now, dry_run=dry_run
                )
                collected = None
                if store is not None:
                    collected = collect_blobs(session, store, grace=grace, now=now, dry_run=dry_run)
            return Maintenance(report, collected)
        finally:
            if postgres:
                lock.execute(text("SELECT pg_advisory_unlock(:id)"), {"id": SWEEP_LOCK_ID})
                lock.commit()


class VerdictStore:
    """Judge answers in the response cache table, keyed by the judge's own hash."""

    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self.sessions = sessions

    async def get(self, key: str) -> str | None:
        def load() -> str | None:
            with self.sessions() as session:
                hit = lookup(session, [key]).get(key)
                session.commit()
                return hit.output if hit is not None else None

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
