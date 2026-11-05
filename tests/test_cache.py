import asyncio
import io
import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import select, text, update
from sqlalchemy.orm import Session, sessionmaker

from evalkit import blobs, cache
from evalkit.blobs import BlobError, FilesystemBlobStore
from evalkit.cli import main
from evalkit.config import ConfigError, Settings, parse_duration, parse_size
from evalkit.datasets import import_dataset
from evalkit.db import make_engine, make_session_factory, upgrade
from evalkit.db.models import CachedResponse, as_utc
from evalkit.runner import Runner
from evalkit.runspec import RunConfig
from evalkit.worker import Worker
from tests.test_runner import make_cases

T0 = datetime(2026, 11, 1, tzinfo=UTC)


def entry(key: str, output: str = "out", **extra: Any) -> dict[str, Any]:
    return {
        "key": key,
        "target_hash": "t" * 64,
        "output": output,
        "stop_reason": "end_turn",
        "usage": {},
        **extra,
    }


def put(sessions: sessionmaker[Session], *rows: dict[str, Any], now: datetime = T0) -> None:
    with sessions() as s:
        cache.store(s, rows, now=now)
        s.commit()


def rows(sessions: sessionmaker[Session]) -> dict[str, CachedResponse]:
    with sessions() as s:
        return {r.key: r for r in s.scalars(select(CachedResponse))}


# --- store, lookup and last use ---------------------------------------------------


def test_store_records_size_and_times(sessions: sessionmaker[Session]) -> None:
    put(sessions, entry("a", "héllo"), entry("b", ""))
    stored = rows(sessions)
    assert stored["a"].size_bytes == len("héllo".encode())
    assert stored["b"].size_bytes == 0
    assert as_utc(stored["a"].last_used_at) == as_utc(stored["a"].created_at) == T0


def test_lookup_marks_entries_used_at_most_once_per_interval(
    sessions: sessionmaker[Session],
) -> None:
    put(sessions, entry("a"), entry("b"))
    soon = T0 + cache.TOUCH_INTERVAL / 2
    later = T0 + cache.TOUCH_INTERVAL * 2
    with sessions() as s:
        assert cache.lookup(s, ["a", "missing"], now=soon) == {"a": cache.Hit("out", "end_turn")}
        s.commit()
    assert as_utc(rows(sessions)["a"].last_used_at) == T0
    with sessions() as s:
        cache.lookup(s, ["a"], now=later)
        s.commit()
    stored = rows(sessions)
    assert as_utc(stored["a"].last_used_at) == later
    assert as_utc(stored["b"].last_used_at) == T0


def test_existing_does_not_mark_entries_used(sessions: sessionmaker[Session]) -> None:
    put(sessions, entry("a"))
    with sessions() as s:
        assert cache.existing(s, ["a", "b"]) == {"a"}
        s.commit()
    assert as_utc(rows(sessions)["a"].last_used_at) == T0


def test_lookup_reads_in_chunks(sessions: sessionmaker[Session]) -> None:
    keys = [f"k{i:04d}" for i in range(1200)]
    put(sessions, *(entry(k) for k in keys))
    with sessions() as s:
        assert len(cache.lookup(s, keys, now=T0 + timedelta(days=1))) == 1200
        s.commit()
    assert {as_utc(r.last_used_at) for r in rows(sessions).values()} == {T0 + timedelta(days=1)}


def test_verdicts_are_marked_used(sessions: sessionmaker[Session]) -> None:
    verdicts = cache.VerdictStore(sessions)
    asyncio.run(verdicts.put("v", "j" * 64, "PASS", {"input_tokens": 1}))
    with sessions() as s:
        s.execute(update(CachedResponse).values(last_used_at=T0))
        s.commit()
    assert asyncio.run(verdicts.get("v")) == "PASS"
    assert asyncio.run(verdicts.get("nope")) is None
    assert as_utc(rows(sessions)["v"].last_used_at) > T0


# --- blobs in the cache ---------------------------------------------------------------


def test_long_outputs_go_to_the_blob_store(sessions: sessionmaker[Session], tmp_path: Path) -> None:
    store = FilesystemBlobStore(tmp_path / "blobs")
    long, short = "x" * 100 + "é", "short"
    ready = cache.offload([entry("long", long), entry("short", short)], store, min_bytes=100)
    put(sessions, *ready)
    stored = rows(sessions)
    digest = blobs.digest_of(long.encode())
    assert (stored["long"].output, stored["long"].output_blob) == ("", digest)
    assert stored["long"].size_bytes == len(long.encode())
    assert (stored["short"].output, stored["short"].output_blob) == (short, None)
    with sessions() as s:
        hits = cache.lookup(s, ["long", "short"], store)
    assert hits["long"].output == long
    assert hits["short"].output == short


def test_offload_keeps_output_inline_when_the_store_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Broken(FilesystemBlobStore):
        def put(self, data: bytes) -> str:
            raise BlobError("disk full")

    ready = cache.offload([entry("a", "x" * 10)], Broken("/nonexistent"), min_bytes=1)
    assert ready[0]["output"] == "x" * 10
    assert "output_blob" not in ready[0]
    assert "caching inline" in caplog.text
    assert cache.offload([entry("b", "")], Broken("/x"), min_bytes=0)[0]["output"] == ""


def test_unusable_blob_drops_the_entry(sessions: sessionmaker[Session], tmp_path: Path) -> None:
    store = FilesystemBlobStore(tmp_path)
    put(sessions, *cache.offload([entry("gone", "a" * 50), entry("bad", "b" * 50)], store, 10))
    stored = rows(sessions)
    store.delete(stored["gone"].output_blob or "")
    (tmp_path / blobs.blob_key(stored["bad"].output_blob or "")).write_bytes(b"tampered")
    with sessions() as s:
        assert cache.lookup(s, ["gone", "bad"], store) == {}
        s.commit()
    assert rows(sessions) == {}


def test_unreadable_blob_is_a_miss_but_kept(
    sessions: sessionmaker[Session], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = FilesystemBlobStore(tmp_path)
    put(sessions, *cache.offload([entry("a", "a" * 50)], store, 10))

    class Down(FilesystemBlobStore):
        def get(self, digest: str) -> bytes:
            raise BlobError("unreachable")

    with sessions() as s:
        assert cache.lookup(s, ["a"], Down(tmp_path)) == {}
        assert cache.lookup(s, ["a"], None) == {}
        s.commit()
    assert "a" in rows(sessions)
    assert "none is configured" in caplog.text


def test_runner_round_trips_long_outputs_through_the_blob_store(
    sessions: sessionmaker[Session], tmp_path: Path
) -> None:
    with sessions() as s:
        import_dataset(s, "qa", make_cases(12))
    settings = Settings(blob_url=str(tmp_path / "blobs"), blob_min_bytes=0)
    runner = Runner(sessions, settings, allow_import_paths=True)
    assert isinstance(runner.blobs, FilesystemBlobStore)
    config = RunConfig.model_validate(
        {"dataset": {"name": "qa"}, "target": {"type": "fake"}, "scorers": [{"type": "exact"}]}
    )
    first = asyncio.run(runner.execute(runner.create_run(config).id))
    assert first.executed == 12
    stored = rows(sessions)
    assert all(r.output_blob and r.output == "" for r in stored.values())
    assert len(list(runner.blobs.list())) == 12
    second = asyncio.run(runner.execute(runner.create_run(config).id))
    assert (second.executed, second.cached) == (0, 12)


# --- sweeping ---------------------------------------------------------------------------


def sweep(sessions: sessionmaker[Session], **kw: Any) -> cache.SweepReport:
    with sessions() as session:
        return cache.sweep(session, **kw)


def aged(sessions: sessionmaker[Session], spec: list[tuple[int, int]]) -> list[str]:
    """Entries k000.. with (size, hours since last use)."""
    keys = []
    for i, (size, hours) in enumerate(spec):
        key = f"k{i:03d}"
        keys.append(key)
        put(sessions, entry(key, "x" * size), now=T0 - timedelta(hours=hours))
    return keys


def test_sweep_by_age(sessions: sessionmaker[Session]) -> None:
    keys = aged(sessions, [(10, 1), (20, 50), (30, 100), (40, 24)])
    report = sweep(sessions, max_age=timedelta(hours=24), now=T0)
    assert (report.removed, report.removed_bytes, report.kept, report.kept_bytes) == (2, 50, 2, 50)
    assert set(rows(sessions)) == {keys[0], keys[3]}


def test_sweep_by_size_removes_least_recently_used_first(sessions: sessionmaker[Session]) -> None:
    keys = aged(sessions, [(10, 1), (20, 50), (30, 100), (40, 24), (5, 2)])
    report = sweep(sessions, max_bytes=60, now=T0, chunk=2)
    # 105 bytes, 45 too many. Oldest first: k002 (30) and k001 (20) go.
    assert (report.removed, report.removed_bytes, report.kept_bytes) == (2, 50, 55)
    assert set(rows(sessions)) == {keys[0], keys[3], keys[4]}


def test_sweep_keeps_entries_used_since_it_looked(sessions: sessionmaker[Session]) -> None:
    keys = aged(sessions, [(10, 100), (10, 90)])
    session = sessions()
    real_execute = session.execute

    def touch_then_execute(statement: Any, *args: Any, **kw: Any) -> Any:
        if getattr(statement, "is_delete", False):
            with sessions() as other:  # a run uses k000 between the select and the delete
                cache.lookup(other, [keys[0]], now=T0)
                other.commit()
        return real_execute(statement, *args, **kw)

    session.execute = touch_then_execute  # type: ignore[method-assign]
    with session:
        cache.sweep(session, max_age=timedelta(hours=1), now=T0)
    assert set(rows(sessions)) == {keys[0]}


def test_sweep_with_no_limits_only_counts(sessions: sessionmaker[Session]) -> None:
    aged(sessions, [(10, 1), (20, 2)])
    report = sweep(sessions, now=T0)
    assert (report.removed, report.kept, report.kept_bytes) == (0, 2, 30)


@settings(
    max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    spec=st.lists(st.tuples(st.integers(0, 50), st.integers(0, 200)), max_size=25),
    max_age=st.one_of(st.none(), st.integers(1, 200)),
    max_bytes=st.one_of(st.none(), st.integers(0, 600)),
    chunk=st.integers(1, 7),
)
def test_sweep_properties(
    tmp_path: Path,
    spec: list[tuple[int, int]],
    max_age: int | None,
    max_bytes: int | None,
    chunk: int,
) -> None:
    engine = make_engine(f"sqlite:///{tmp_path / f'sweep-{time.monotonic_ns()}.db'}")
    try:
        upgrade(engine)
        check_sweep(make_session_factory(engine), spec, max_age, max_bytes, chunk)
    finally:
        engine.dispose()


def check_sweep(
    sessions: sessionmaker[Session],
    spec: list[tuple[int, int]],
    max_age: int | None,
    max_bytes: int | None,
    chunk: int,
) -> None:
    aged(sessions, spec)
    age = None if max_age is None else timedelta(hours=max_age)
    dry = sweep(sessions, max_age=age, max_bytes=max_bytes, now=T0, dry_run=True, chunk=chunk)
    assert len(rows(sessions)) == len(spec)  # a dry run deletes nothing
    real = sweep(sessions, max_age=age, max_bytes=max_bytes, now=T0, chunk=chunk)
    # A dry run reports exactly what the real sweep does.
    assert (dry.removed, dry.removed_bytes, dry.kept, dry.kept_bytes) == (
        real.removed,
        real.removed_bytes,
        real.kept,
        real.kept_bytes,
    )
    left = rows(sessions)
    assert real.kept == len(left)
    assert real.kept_bytes == sum(r.size_bytes for r in left.values())
    if max_bytes is not None:
        assert real.kept_bytes <= max_bytes
    if age is not None:
        assert all(as_utc(r.last_used_at) >= T0 - age for r in left.values())
    # What is left is always the most recently used entries.
    removed_newest = max(
        (T0 - timedelta(hours=h) for i, (_, h) in enumerate(spec) if f"k{i:03d}" not in left),
        default=None,
    )
    if removed_newest is not None:
        assert all(as_utc(r.last_used_at) >= removed_newest for r in left.values())


# --- blob garbage collection and maintenance ----------------------------------------------


def test_sweep_then_collect_removes_the_blobs_of_swept_entries(
    sessions: sessionmaker[Session], tmp_path: Path
) -> None:
    store = FilesystemBlobStore(tmp_path)
    put(sessions, *cache.offload([entry("old", "o" * 40)], store, 10), now=T0 - timedelta(days=60))
    put(sessions, *cache.offload([entry("new", "n" * 40)], store, 10), now=T0)
    old_blob = blobs.digest_of(b"o" * 40)
    stamp = (T0 - timedelta(days=60)).timestamp()
    os.utime(tmp_path / blobs.blob_key(old_blob), (stamp, stamp))
    done = cache.maintain(
        sessions,
        store,
        max_age=timedelta(days=30),
        max_bytes=None,
        grace=timedelta(hours=1),
        now=T0,
    )
    assert not done.skipped
    assert done.sweep.removed == 1
    assert done.blobs is not None
    assert (done.blobs.scanned, done.blobs.removed, done.blobs.referenced) == (2, 1, 1)
    assert [b.digest for b in store.list()] == [blobs.digest_of(b"n" * 40)]


def test_referenced_blobs(sessions: sessionmaker[Session]) -> None:
    put(sessions, entry("a", "", output_blob="1" * 64), entry("b"))
    with sessions() as s:
        assert cache.referenced_blobs(s, ["1" * 64, "2" * 64]) == {"1" * 64}
        assert cache.referenced_blobs(s, []) == set()


# --- migration ----------------------------------------------------------------------------


def test_migration_0006_fills_in_last_use_and_size(tmp_path: Path) -> None:
    engine = make_engine(f"sqlite:///{tmp_path / 'old.db'}")
    upgrade(engine, "0005")
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO response_cache"
                " (key, target_hash, output, stop_reason, usage, created_at)"
                " VALUES ('k', 't', 'héllo ✓', 'end_turn', '{}', '2026-01-01 00:00:00')"
            )
        )
    upgrade(engine)
    with make_session_factory(engine)() as s:
        row = s.get_one(CachedResponse, "k")
        assert row.size_bytes == len("héllo ✓".encode())
        assert row.last_used_at == row.created_at
        assert row.output_blob is None
    engine.dispose()


# --- configuration ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1024", 1024), ("500MB", 500_000_000), ("20GiB", 20 * 2**30), ("1.5 KiB", 1536), ("0", 0)],
)
def test_parse_size(raw: str, expected: int) -> None:
    assert parse_size(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("90", 90), ("45m", 2700), ("12h", 43200), ("30d", 2592000), ("2W", 1209600)],
)
def test_parse_duration(raw: str, expected: float) -> None:
    assert parse_duration(raw) == expected


@pytest.mark.parametrize("bad", ["", "-1", "ten", "5 parsecs", "1e9"])
def test_parse_size_and_duration_reject_garbage(bad: str) -> None:
    with pytest.raises(ValueError, match="not a"):
        parse_size(bad)
    with pytest.raises(ValueError, match="not a"):
        parse_duration(bad)
    with pytest.raises(ValueError, match="not a duration"):
        parse_duration("0h")


def test_cache_and_blob_settings_from_env(tmp_path: Path) -> None:
    secret = tmp_path / "secret"
    secret.write_text("blob-secret-value\n")
    s = Settings.from_env(
        {
            "EVALKIT_CACHE_MAX_AGE": "30d",
            "EVALKIT_CACHE_MAX_BYTES": "20GiB",
            "EVALKIT_CACHE_SWEEP_SECONDS": "600",
            "EVALKIT_BLOB_URL": "s3+https://store.test/bkt",
            "EVALKIT_BLOB_ACCESS_KEY": "blob-access-value",
            "EVALKIT_BLOB_SECRET_KEY_FILE": str(secret),
            "EVALKIT_BLOB_REGION": "eu-west-9",
            "EVALKIT_BLOB_MIN_BYTES": "1MiB",
            "EVALKIT_BLOB_GRACE": "2h",
        }
    )
    assert s.cache_max_age_seconds == 30 * 86400
    assert s.cache_max_bytes == 20 * 2**30
    assert s.cache_sweep_seconds == 600
    assert s.cache_limited
    assert (s.blob_min_bytes, s.blob_grace_seconds, s.blob_region) == (2**20, 7200, "eu-west-9")
    assert {"blob-access-value", "blob-secret-value"} <= set(s.secret_values())
    store = s.open_blob_store()
    assert isinstance(store, blobs.S3BlobStore)
    store.close()
    plain = Settings.from_env({})
    assert not plain.cache_limited
    assert plain.open_blob_store() is None


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"EVALKIT_CACHE_MAX_AGE": "soon"}, "EVALKIT_CACHE_MAX_AGE"),
        ({"EVALKIT_CACHE_MAX_BYTES": "lots"}, "EVALKIT_CACHE_MAX_BYTES"),
        ({"EVALKIT_BLOB_URL": "s3+https://store.test/bkt"}, "EVALKIT_BLOB_URL: an S3"),
        ({"EVALKIT_BLOB_URL": "ftp://x/y"}, "EVALKIT_BLOB_URL"),
        ({"EVALKIT_BLOB_REGION": "Not A Region"}, "blob_region"),
    ],
)
def test_bad_cache_and_blob_settings(env: dict[str, str], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        Settings.from_env(env)


# --- CLI ------------------------------------------------------------------------------------


@pytest.fixture
def cli_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    for var in list(os.environ):
        if var.startswith("EVALKIT_"):
            monkeypatch.delenv(var)
    url = f"sqlite:///{tmp_path / 'cli.db'}"
    engine = make_engine(url)
    upgrade(engine)
    sessions = make_session_factory(engine)
    now = datetime.now(UTC)
    for i in range(4):
        put(sessions, entry(f"k{i}", "x" * 100), now=now - timedelta(days=10 * i))
    engine.dispose()
    return url


def cli(db: str, *args: str) -> tuple[int, str]:
    out = io.StringIO()
    return main(["--database-url", db, "cache", *args], out=out), out.getvalue()


def test_cli_sweep(cli_db: str) -> None:
    code, out = cli(cli_db, "sweep", "--max-age", "15d", "--dry-run")
    assert code == 0
    assert out == "cache: would remove 2 entries (200 B), 2 left (200 B)\n"
    code, out = cli(cli_db, "sweep", "--max-age", "15d", "--max-size", "100", "--json")
    assert code == 0
    report = json.loads(out)
    assert report["sweep"]["removed"] == 3
    assert report["sweep"]["kept_bytes"] == 100
    assert report["blobs"] is None


def test_cli_sweep_uses_configured_limits_and_collects_blobs(
    cli_db: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FilesystemBlobStore(tmp_path / "blobs")
    orphan = store.put(b"nobody refers to me")
    stamp = (datetime.now(UTC) - timedelta(days=1)).timestamp()
    os.utime(store.root / blobs.blob_key(orphan), (stamp, stamp))
    monkeypatch.setenv("EVALKIT_CACHE_MAX_BYTES", "2KiB")
    monkeypatch.setenv("EVALKIT_BLOB_URL", f"file://{store.root}")
    code, out = cli(cli_db, "sweep")
    assert code == 0
    assert out.splitlines() == [
        "cache: removed 0 entries (0 B), 4 left (400 B)",
        "blobs: scanned 1, removed 1 (19 B), kept 0 referenced and 0 within the grace period",
    ]
    (store.root / "tmp" / "x.part").write_bytes(b"x")
    os.utime(store.root / "tmp" / "x.part", (stamp, stamp))
    code, out = cli(cli_db, "sweep", "--blob-grace", "1h")
    assert "removed 1 unfinished writes" in out


def test_cli_sweep_errors(cli_db: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli(cli_db, "sweep")[0] == 2
    assert "nothing to sweep" in capsys.readouterr().err
    assert cli(cli_db, "sweep", "--max-age", "a while")[0] == 2
    assert "--max-age" in capsys.readouterr().err
    assert cli(cli_db, "scrub")[0] == 2
    assert "no blob store" in capsys.readouterr().err


def test_cli_scrub(cli_db: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = FilesystemBlobStore(tmp_path / "blobs")
    store.put(b"fine")
    monkeypatch.setenv("EVALKIT_BLOB_URL", str(store.root))
    assert cli(cli_db, "scrub") == (0, "blobs: checked 1, corrupt 0\n")
    bad = store.put(b"will rot")
    (store.root / blobs.blob_key(bad)).write_bytes(b"rotten")
    code, out = cli(cli_db, "scrub", "--json")
    assert code == 3
    assert json.loads(out) == {"scanned": 2, "corrupt": [bad]}
    code, out = cli(cli_db, "scrub")
    assert out.splitlines()[1] == f"  corrupt {bad}"


def test_byte_formatting() -> None:
    from evalkit.cli import _bytes

    assert [_bytes(n) for n in (0, 1023, 1024, 5 * 2**20, 3 * 2**30, 2**50)] == [
        "0 B",
        "1023 B",
        "1.0 KiB",
        "5.0 MiB",
        "3.0 GiB",
        "1024.0 TiB",
    ]


# --- the worker sweeps periodically ---------------------------------------------------------


def test_worker_sweeps_the_cache(
    sessions: sessionmaker[Session], caplog: pytest.LogCaptureFixture
) -> None:
    now = datetime.now(UTC)
    put(sessions, entry("old"), now=now - timedelta(days=40))
    put(sessions, entry("new"), now=now)
    settings = Settings(cache_max_age_seconds=30 * 86400, cache_sweep_seconds=1)
    worker = Worker(
        sessions,
        Runner(sessions, settings),
        slots=1,
        poll_seconds=0.05,
        drain_seconds=1,
    )
    assert worker.maintain_cache

    async def go() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(worker.serve(stop))
        # Wait for the log line, not for the row: the row is gone as soon as the
        # sweep commits, and stopping the worker then can cancel the sweep's
        # thread await before it logs.
        for _ in range(200):
            if "swept the response cache" in caplog.text:
                break
            await asyncio.sleep(0.02)
        stop.set()
        await task

    with caplog.at_level(logging.INFO, logger="evalkit.worker"):
        asyncio.run(go())
    assert set(rows(sessions)) == {"new"}
    assert "swept the response cache" in caplog.text


def test_worker_survives_a_failing_sweep(
    sessions: sessionmaker[Session],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def broken(*_a: object, **_k: object) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("database gone")

    monkeypatch.setattr(cache, "maintain", broken)
    settings = Settings(cache_max_bytes=10, cache_sweep_seconds=1)
    worker = Worker(
        sessions, Runner(sessions, settings), slots=1, poll_seconds=0.05, drain_seconds=1
    )

    async def go() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(worker.serve(stop))
        for _ in range(100):
            if calls:
                break
            await asyncio.sleep(0.02)
        stop.set()
        await task

    asyncio.run(go())
    assert calls >= 1
    assert "cannot sweep the response cache" in caplog.text
    assert not Worker(
        sessions, Runner(sessions, Settings()), slots=1, poll_seconds=1, drain_seconds=1
    ).maintain_cache
