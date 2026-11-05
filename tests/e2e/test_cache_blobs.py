"""The bounded cache on PostgreSQL and the S3 blob backend on a real server
(versitygw: see ``conftest.py``)."""

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker

from evalkit import blobs, cache
from evalkit.blobs import BlobCorruptError, BlobError, S3BlobStore, open_store, scrub
from evalkit.db import make_engine, make_session_factory
from evalkit.db.models import CachedResponse, as_utc
from tests.e2e.conftest import ACCESS, SECRET

pytestmark = pytest.mark.e2e


def store_for(endpoint: str, bucket: str, **kw: str) -> S3BlobStore:
    options = {"access_key": ACCESS, "secret_key": SECRET, **kw}
    store = S3BlobStore(endpoint, bucket, **options)  # type: ignore[arg-type]
    store.create_bucket()
    return store


def test_s3_blobs_on_a_real_server(s3_endpoint: str) -> None:
    store = store_for(s3_endpoint, "evalkit-blobs", prefix="team a/x+y")
    store.create_bucket()  # already there: still fine
    big = os.urandom(3 * 1024 * 1024)
    digest = store.put(big)
    assert store.put(big) == digest
    assert store.exists(digest)
    assert store.get(digest) == big
    small = sorted(store.put(f"blob {i}".encode()) for i in range(5))
    listed = list(store.list())
    assert sorted(b.digest for b in listed) == sorted([digest, *small])
    assert {b.size for b in listed} == {len(big), len(b"blob 0")}
    report = blobs.collect_garbage(
        store, lambda page: {digest} & set(page), before=datetime.now(UTC) + timedelta(minutes=1)
    )
    assert (report.scanned, report.referenced, report.removed) == (6, 1, 5)
    assert [b.digest for b in store.list()] == [digest]
    assert scrub(store).corrupt == []
    store.delete(digest)
    store.delete(digest)
    assert not store.exists(digest)
    store.close()


def test_s3_wrong_secret_is_refused_without_leaking_it(s3_endpoint: str) -> None:
    good = store_for(s3_endpoint, "evalkit-auth")
    digest = good.put(b"data")
    bad = S3BlobStore(s3_endpoint, "evalkit-auth", access_key=ACCESS, secret_key="wrong-secret")
    with pytest.raises(BlobError, match="HTTP 403") as err:
        bad.get(digest)
    assert "wrong-secret" not in str(err.value)
    good.close()
    bad.close()


def test_s3_listing_pages_on_a_real_server(
    s3_endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(blobs, "GC_PAGE", 4)
    store = store_for(s3_endpoint, "evalkit-pages")
    digests = sorted(store.put(bytes([i])) for i in range(10))
    assert [b.digest for b in store.list()] == digests
    store.close()


def test_s3_corrupt_object_is_detected(s3_endpoint: str) -> None:
    store = store_for(s3_endpoint, "evalkit-corrupt")
    digest = store.put(b"original")
    # Overwrite the object behind the store's back with a correctly signed PUT.
    tamper = S3BlobStore(s3_endpoint, "evalkit-corrupt", access_key=ACCESS, secret_key=SECRET)
    response = tamper._request(
        "PUT",
        blobs.blob_key(digest),
        body=b"changed",
        payload_sha256=blobs.digest_of(b"changed"),
    )
    assert response.status_code == 200
    with pytest.raises(BlobCorruptError):
        store.get(digest)
    assert scrub(store).corrupt == [digest]
    store.close()
    tamper.close()


def test_s3_store_from_settings_url(s3_endpoint: str) -> None:
    url = s3_endpoint.replace("http://", "s3+http://") + "/evalkit-url/prefix"
    store = open_store(url, access_key=ACCESS, secret_key=SECRET)
    assert isinstance(store, S3BlobStore)
    store.create_bucket()
    assert store.get(store.put(b"via url")) == b"via url"
    store.close()


# --- PostgreSQL ----------------------------------------------------------------


@pytest.fixture
def pg_sessions(postgres_url: str) -> Iterator[sessionmaker[Session]]:
    engine = make_engine(postgres_url)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM response_cache"))
    yield make_session_factory(engine)
    engine.dispose()


def entry(key: str, output: str) -> dict[str, object]:
    return {"key": key, "target_hash": "t" * 64, "output": output, "usage": {}}


def test_sweep_and_blob_collection_on_postgres(
    pg_sessions: sessionmaker[Session], s3_endpoint: str
) -> None:
    store = store_for(s3_endpoint, "evalkit-pg")
    now = datetime.now(UTC)
    with pg_sessions() as s:
        for i in range(30):
            ready = cache.offload([entry(f"k{i:02d}", f"{i:02d}" * 50)], store, 50)
            cache.store(s, ready, now=now - timedelta(days=i))
        s.commit()
    with pg_sessions() as s:
        hits = cache.lookup(s, ["k00", "k29"], store, now=now)
        s.commit()
    assert hits["k29"].output == "29" * 50
    later = now + timedelta(minutes=1)
    done = cache.maintain(
        pg_sessions,
        store,
        max_age=timedelta(days=20),
        max_bytes=5 * 100,
        grace=timedelta(0),
        now=later,
    )
    assert not done.skipped
    with pg_sessions() as s:
        left = {r.key: r for r in s.scalars(select(CachedResponse))}
    # k29 was used just now; then the five most recently used fit in 500 bytes.
    assert set(left) == {"k29", "k00", "k01", "k02", "k03"}
    assert as_utc(left["k29"].last_used_at) == now
    assert done.sweep.kept_bytes == 500
    assert done.blobs is not None
    assert done.blobs.removed == 25
    assert sorted(b.digest for b in store.list()) == sorted(
        r.output_blob or "" for r in left.values()
    )
    store.close()


def test_one_maintainer_at_a_time_on_postgres(
    pg_sessions: sessionmaker[Session], postgres_url: str
) -> None:
    holder = create_engine(postgres_url)
    with holder.connect() as conn:
        assert conn.scalar(text("SELECT pg_try_advisory_lock(:id)"), {"id": cache.SWEEP_LOCK_ID})
        busy = cache.maintain(
            pg_sessions, None, max_age=timedelta(days=1), max_bytes=None, grace=timedelta(0)
        )
        assert busy.skipped
        conn.execute(text("SELECT pg_advisory_unlock(:id)"), {"id": cache.SWEEP_LOCK_ID})
        conn.commit()
    holder.dispose()
    free = cache.maintain(
        pg_sessions, None, max_age=timedelta(days=1), max_bytes=None, grace=timedelta(0)
    )
    assert not free.skipped
