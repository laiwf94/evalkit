import hashlib
import os
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx2
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from evalkit import blobs
from evalkit.blobs import (
    EMPTY_SHA256,
    BlobCorruptError,
    BlobError,
    BlobNotFoundError,
    FilesystemBlobStore,
    S3BlobStore,
    collect_garbage,
    open_store,
    scrub,
    sign_request,
)
from tests.s3fake import ACCESS, REGION, SECRET, FakeS3

T0 = datetime(2026, 11, 1, tzinfo=UTC)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def s3(fake: FakeS3, **kw: object) -> S3BlobStore:
    options: dict[str, object] = {
        "access_key": ACCESS,
        "secret_key": SECRET,
        "region": REGION,
        "transport": fake.transport(),
        "sleep": lambda _s: None,
        **kw,
    }
    return S3BlobStore("http://store.test:9000", "evalkit", **options)  # type: ignore[arg-type]


@pytest.fixture(params=["filesystem", "s3"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[blobs.BlobStore]:
    if request.param == "filesystem":
        yield FilesystemBlobStore(tmp_path / "blobs")
    else:
        backend = s3(FakeS3(), prefix="team/a")
        yield backend
        backend.close()


# --- both backends -----------------------------------------------------------


def test_put_get_round_trip_and_digest(store: blobs.BlobStore) -> None:
    data = "naïve output ✓".encode() * 100
    digest = store.put(data)
    assert digest == sha(data)
    assert store.get(digest) == data
    assert store.exists(digest)
    assert store.put(data) == digest  # repeating a put is harmless


def test_missing_blob(store: blobs.BlobStore) -> None:
    digest = sha(b"never stored")
    assert not store.exists(digest)
    with pytest.raises(BlobNotFoundError):
        store.get(digest)
    store.delete(digest)  # deleting what is not there is fine


def test_list_and_delete(store: blobs.BlobStore) -> None:
    digests = sorted(store.put(f"blob {i}".encode()) for i in range(7))
    listed = list(store.list())
    assert [b.digest for b in listed] == digests
    assert all(b.size == len(b"blob 0") for b in listed)
    assert all(b.modified.tzinfo is not None for b in listed)
    store.delete(digests[3])
    assert [b.digest for b in store.list()] == digests[:3] + digests[4:]


def test_empty_blob(store: blobs.BlobStore) -> None:
    digest = store.put(b"")
    assert digest == EMPTY_SHA256
    assert store.get(digest) == b""


@pytest.mark.parametrize("bad", ["../../etc/passwd", "ABC", "0" * 63, "g" * 64, "0" * 64 + "/x"])
def test_names_must_be_digests(store: blobs.BlobStore, bad: str) -> None:
    for call in (store.get, store.exists, store.delete):
        with pytest.raises(ValueError, match="SHA-256"):
            call(bad)


def test_scrub_reports_corrupt_blobs(tmp_path: Path) -> None:
    fs = FilesystemBlobStore(tmp_path)
    good, bad = fs.put(b"good"), fs.put(b"bad")
    (tmp_path / blobs.blob_key(bad)).write_bytes(b"tampered")
    report = scrub(fs)
    assert (report.scanned, report.corrupt) == (2, [bad])
    assert fs.get(good) == b"good"


@settings(max_examples=40, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.lists(st.binary(max_size=2048), max_size=12))
def test_any_bytes_round_trip_on_both_backends(tmp_path: Path, items: list[bytes]) -> None:
    fs = FilesystemBlobStore(tmp_path / f"fs-{time.monotonic_ns()}")
    remote = s3(FakeS3(page_size=2))
    for data in items:
        assert fs.put(data) == remote.put(data) == sha(data)
    for data in items:
        assert fs.get(sha(data)) == remote.get(sha(data)) == data
    expected = sorted({sha(d) for d in items})
    assert [b.digest for b in fs.list()] == [b.digest for b in remote.list()] == expected
    remote.close()


# --- filesystem ----------------------------------------------------------------


def test_filesystem_detects_tampering(tmp_path: Path) -> None:
    fs = FilesystemBlobStore(tmp_path)
    digest = fs.put(b"original")
    path = tmp_path / "sha256" / digest[:2] / digest[2:4] / digest
    assert path.read_bytes() == b"original"
    path.write_bytes(b"changed")
    with pytest.raises(BlobCorruptError, match=digest):
        fs.get(digest)


def test_filesystem_repeated_put_refreshes_the_time(tmp_path: Path) -> None:
    fs = FilesystemBlobStore(tmp_path)
    digest = fs.put(b"x")
    path = tmp_path / blobs.blob_key(digest)
    os.utime(path, (1_000_000, 1_000_000))
    assert fs.modified(digest) == datetime.fromtimestamp(1_000_000, tz=UTC)
    fs.put(b"x")
    modified = fs.modified(digest)
    assert modified is not None
    assert modified > datetime.now(UTC) - timedelta(minutes=1)
    assert fs.modified(sha(b"y")) is None


def test_filesystem_write_failure_leaves_nothing(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "sha256").write_text("a file where the blob directory should be")
    fs = FilesystemBlobStore(root)
    with pytest.raises(BlobError, match="cannot write blob"):
        fs.put(b"data")
    assert list((root / "tmp").iterdir()) == []


def test_filesystem_failure_after_temporary_file_is_cleaned_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fs = FilesystemBlobStore(tmp_path)

    def broken(*_a: object) -> None:
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(BlobError, match="Input/output error"):
        fs.put(b"data")
    assert list((tmp_path / "tmp").iterdir()) == []


def test_filesystem_read_and_delete_errors(tmp_path: Path) -> None:
    fs = FilesystemBlobStore(tmp_path)
    digest = fs.put(b"data")
    path = tmp_path / blobs.blob_key(digest)
    path.unlink()
    path.mkdir()  # a directory where the blob was: reading and unlinking fail
    with pytest.raises(BlobError, match="cannot read"):
        fs.get(digest)
    with pytest.raises(BlobError, match="cannot delete"):
        fs.delete(digest)


def test_filesystem_listing_ignores_strangers(tmp_path: Path) -> None:
    fs = FilesystemBlobStore(tmp_path)
    assert list(fs.list()) == []
    digest = fs.put(b"data")
    (tmp_path / "sha256" / "zz-not-a-shard").mkdir()
    (tmp_path / "sha256" / digest[:2] / digest[2:4] / "notes.txt").write_text("hi")
    assert [b.digest for b in fs.list()] == [digest]


def test_filesystem_discards_old_unfinished_writes(tmp_path: Path) -> None:
    fs = FilesystemBlobStore(tmp_path)
    assert fs.discard_incomplete(datetime.now(UTC)) == 0
    fs.put(b"creates tmp/")
    old, new = tmp_path / "tmp" / "old.part", tmp_path / "tmp" / "new.part"
    old.write_bytes(b"x")
    new.write_bytes(b"y")
    os.utime(old, (1_000_000, 1_000_000))
    assert fs.discard_incomplete(datetime.now(UTC) - timedelta(hours=1)) == 1
    assert not old.exists()
    assert new.exists()


# --- request signing ----------------------------------------------------------


def test_canonical_request_follows_the_specification() -> None:
    signed = sign_request(
        "GET",
        "store.test:9000",
        "/bkt/a b/c+d~é",
        query=[("prefix", "x/y z"), ("list-type", "2"), ("a", "")],
        headers={"Range": "  bytes=0-9   ", "X-Custom": "two   spaces"},
        payload_sha256=EMPTY_SHA256,
        access_key="key",
        secret_key="secret",
        region="here-1",
        now=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
    )
    assert signed.canonical_request == "\n".join(
        [
            "GET",
            "/bkt/a%20b/c%2Bd~%C3%A9",
            "a=&list-type=2&prefix=x%2Fy%20z",
            "host:store.test:9000",
            "range:bytes=0-9",
            f"x-amz-content-sha256:{EMPTY_SHA256}",
            "x-amz-date:20260102T030405Z",
            "x-custom:two spaces",
            "",
            "host;range;x-amz-content-sha256;x-amz-date;x-custom",
            EMPTY_SHA256,
        ]
    )
    assert signed.string_to_sign.split("\n")[:3] == [
        "AWS4-HMAC-SHA256",
        "20260102T030405Z",
        "20260102/here-1/s3/aws4_request",
    ]
    assert signed.string_to_sign.split("\n")[3] == sha(signed.canonical_request.encode())
    auth = signed.headers["authorization"]
    assert auth.startswith("AWS4-HMAC-SHA256 Credential=key/20260102/here-1/s3/aws4_request, ")
    assert auth.endswith(f"Signature={signed.signature}")
    assert "host" not in signed.headers
    assert signed.headers["x-amz-date"] == "20260102T030405Z"


def test_signature_depends_on_every_signed_part() -> None:
    base = {
        "query": [("a", "1")],
        "headers": {},
        "payload_sha256": EMPTY_SHA256,
        "access_key": "key",
        "secret_key": "secret",
        "region": "here-1",
        "now": T0,
    }
    ref = sign_request("GET", "h", "/b/k", **base).signature  # type: ignore[arg-type]
    variants: list[tuple[str, str, str, dict[str, Any]]] = [
        ("PUT", "h", "/b/k", {}),
        ("GET", "h2", "/b/k", {}),
        ("GET", "h", "/b/k2", {}),
        ("GET", "h", "/b/k", {"query": [("a", "2")]}),
        ("GET", "h", "/b/k", {"payload_sha256": sha(b"x")}),
        ("GET", "h", "/b/k", {"secret_key": "other"}),
        ("GET", "h", "/b/k", {"region": "there-2"}),
        ("GET", "h", "/b/k", {"now": T0 + timedelta(seconds=1)}),
    ]
    for method, host, path, change in variants:
        signature = sign_request(method, host, path, **{**base, **change}).signature
        assert signature != ref, change


# --- S3 backend -----------------------------------------------------------------


def test_s3_requests_are_path_style_and_signed() -> None:
    fake = FakeS3()
    store = s3(fake, prefix="/team/a/")
    digest = store.put(b"hello")
    assert fake.requests[-1] == (
        "PUT",
        f"/evalkit/team/a/sha256/{digest[:2]}/{digest[2:4]}/{digest}",
    )
    assert fake.objects[f"team/a/{blobs.blob_key(digest)}"].data == b"hello"
    store.close()


def test_s3_listing_pages_and_skips_foreign_keys() -> None:
    fake = FakeS3(page_size=2)
    store = s3(fake)
    digests = sorted(store.put(bytes([i])) for i in range(5))
    fake.objects["sha256/ab/cd/README"] = fake.objects[blobs.blob_key(digests[0])]
    fake.objects["other/thing"] = fake.objects[blobs.blob_key(digests[0])]
    assert [b.digest for b in store.list()] == digests
    lists = [r for r in fake.requests if r == ("GET", "/evalkit")]
    assert len(lists) == 3  # 6 keys under the prefix, 2 per page


def test_s3_detects_tampering_and_wrong_credentials() -> None:
    fake = FakeS3()
    store = s3(fake)
    digest = store.put(b"original")
    fake.objects[blobs.blob_key(digest)].data = b"changed"
    with pytest.raises(BlobCorruptError):
        store.get(digest)
    wrong = s3(fake, secret_key="not-the-secret")
    with pytest.raises(BlobError, match="HTTP 403 SignatureDoesNotMatch") as err:
        wrong.get(digest)
    assert "not-the-secret" not in str(err.value)
    with pytest.raises(BlobError, match=r"cannot write blob .* 403"):
        wrong.put(b"x")
    with pytest.raises(BlobError, match="cannot list blobs: HTTP 403"):
        list(wrong.list())
    with pytest.raises(BlobError, match=r"cannot check blob .* 403"):
        wrong.exists(digest)
    with pytest.raises(BlobError, match=r"cannot delete blob .* 403"):
        wrong.delete(digest)


def test_s3_retries_server_errors_and_transport_failures() -> None:
    fake = FakeS3()
    # One 503 and one refused connection, then the real server.
    failures: Iterator[httpx2.Response | Exception] = iter(
        [httpx2.Response(503), httpx2.ConnectError("refused")]
    )
    slept: list[float] = []

    def flaky(request: httpx2.Request) -> httpx2.Response | None:
        item = next(failures, None)
        if isinstance(item, Exception):
            raise item
        return item

    fake.intercept = flaky
    store = s3(fake, sleep=slept.append)
    assert store.put(b"data") == sha(b"data")
    assert slept == list(blobs.RETRY_DELAYS)

    fake.intercept = lambda _r: httpx2.Response(500)
    with pytest.raises(BlobError, match="HTTP 500"):
        store.get(sha(b"data"))

    def down(_r: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused")

    fake.intercept = down
    with pytest.raises(BlobError, match="unreachable: ConnectError"):
        store.exists(sha(b"data"))


def test_s3_rejects_bad_listings() -> None:
    fake = FakeS3()
    store = s3(fake)
    fake.intercept = lambda _r: httpx2.Response(200, content=b"<not xml")
    with pytest.raises(BlobError, match="malformed XML"):
        list(store.list())
    fake.intercept = lambda _r: httpx2.Response(
        200, content=b"<x>" + b" " * (blobs.MAX_LISTING_BYTES + 1) + b"</x>"
    )
    with pytest.raises(BlobError, match="too large"):
        list(store.list())
    bad_time = (
        "<ListBucketResult><IsTruncated>false</IsTruncated><Contents>"
        f"<Key>sha256/00/00/{'0' * 64}</Key><LastModified>yesterday</LastModified>"
        "<Size>1</Size></Contents></ListBucketResult>"
    )
    fake.intercept = lambda _r: httpx2.Response(200, content=bad_time.encode())
    with pytest.raises(BlobError, match="unreadable time"):
        list(store.list())
    naive = bad_time.replace("yesterday", "2026-11-01T10:00:00")
    fake.intercept = lambda _r: httpx2.Response(200, content=naive.encode())
    assert next(store.list()).modified == datetime(2026, 11, 1, 10, tzinfo=UTC)
    fake.intercept = lambda _r: httpx2.Response(418, content=b"<Error><Code>Teapot")
    with pytest.raises(BlobError, match=r"HTTP 418$"):
        list(store.list())


def test_s3_bucket_creation_is_idempotent() -> None:
    fake = FakeS3()
    store = S3BlobStore(
        "https://store.test",
        "fresh-bucket",
        access_key=ACCESS,
        secret_key=SECRET,
        region=REGION,
        transport=fake.transport(),
    )
    store.create_bucket()
    store.create_bucket()
    assert "fresh-bucket" in fake.buckets
    fake.intercept = lambda _r: httpx2.Response(
        403, content=b"<Error><Code>AccessDenied</Code></Error>"
    )
    with pytest.raises(BlobError, match="AccessDenied"):
        store.create_bucket()
    assert store.discard_incomplete(datetime.now(UTC)) == 0
    store.close()


def test_s3_configuration_is_checked() -> None:
    with pytest.raises(ValueError, match="http or https"):
        S3BlobStore("ftp://x", "bucket", access_key="a", secret_key="b")
    with pytest.raises(ValueError, match="valid bucket"):
        S3BlobStore("http://x", "Bad_Bucket", access_key="a", secret_key="b")


# --- configuration ----------------------------------------------------------------


def test_open_store_from_urls(tmp_path: Path) -> None:
    fs = open_store(f"file://{tmp_path}")
    assert isinstance(fs, FilesystemBlobStore)
    assert fs.root == tmp_path
    assert isinstance(open_store(str(tmp_path)), FilesystemBlobStore)
    remote = open_store(
        "s3+https://store.test:9000/bkt/some/prefix", access_key="a", secret_key="b"
    )
    assert isinstance(remote, S3BlobStore)
    assert (remote.base, remote.bucket, remote.prefix) == (
        "https://store.test:9000",
        "bkt",
        "some/prefix",
    )
    remote.close()
    plain = open_store("s3+http://store.test/bkt", access_key="a", secret_key="b", region="eu-1")
    assert isinstance(plain, S3BlobStore)
    assert (plain.base, plain.prefix, plain.region) == ("http://store.test", "", "eu-1")
    plain.close()


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("s3+https://store.test/", "no bucket"),
        ("s3+https://store.test/bkt?secret=1", "own variables"),
        ("s3+https://user:pw@store.test/bkt", "own variables"),
        ("file://host/path", "file:///absolute"),
        ("relative/path", "must be"),
        ("https://store.test/bkt", "must be"),
    ],
)
def test_open_store_rejects_bad_urls(url: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        open_store(url, access_key="a", secret_key="b")


def test_open_store_needs_credentials_for_s3() -> None:
    with pytest.raises(ValueError, match="EVALKIT_BLOB_ACCESS_KEY"):
        open_store("s3+https://store.test/bkt")


# --- garbage collection -------------------------------------------------------------


def test_garbage_collection_keeps_referenced_and_recent_blobs(store: blobs.BlobStore) -> None:
    kept = store.put(b"referenced")
    gone = store.put(b"unreferenced")
    now = datetime.now(UTC)
    calls: list[list[str]] = []

    def referenced(digests: object) -> set[str]:
        calls.append(list(digests))  # type: ignore[call-overload]
        return {kept}

    recent = collect_garbage(store, referenced, before=now - timedelta(hours=1))
    assert (recent.scanned, recent.referenced, recent.recent, recent.removed) == (2, 1, 1, 0)
    dry = collect_garbage(store, referenced, before=now + timedelta(seconds=5), dry_run=True)
    assert (dry.removed, dry.removed_bytes, dry.dry_run) == (1, len(b"unreferenced"), True)
    assert store.exists(gone)
    real = collect_garbage(store, referenced, before=now + timedelta(seconds=5))
    assert (real.removed, real.removed_bytes) == (1, len(b"unreferenced"))
    assert store.exists(kept)
    assert not store.exists(gone)
    assert all(sorted(c) == sorted({kept, gone}) for c in calls[:3])


def test_garbage_collection_pages_the_reference_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(blobs, "GC_PAGE", 3)
    fs = FilesystemBlobStore(tmp_path)
    digests = [fs.put(bytes([i])) for i in range(8)]
    pages: list[int] = []

    def referenced(page: object) -> set[str]:
        pages.append(len(page))  # type: ignore[arg-type]
        return set(digests[:4])

    report = collect_garbage(fs, referenced, before=datetime.now(UTC) + timedelta(seconds=5))
    assert pages == [3, 3, 2]
    assert report.removed == 4
    assert sorted(b.digest for b in fs.list()) == sorted(digests[:4])


def test_garbage_collection_spares_a_blob_written_again_during_the_scan(tmp_path: Path) -> None:
    fs = FilesystemBlobStore(tmp_path)
    digest = fs.put(b"about to be cached again")
    os.utime(tmp_path / blobs.blob_key(digest), (1_000_000, 1_000_000))

    def referenced(_page: object) -> set[str]:
        fs.put(b"about to be cached again")  # a run writes it before its row commits
        return set()

    report = collect_garbage(fs, referenced, before=datetime.now(UTC) - timedelta(minutes=5))
    assert (report.removed, report.recent) == (0, 1)
    assert fs.exists(digest)


def test_garbage_collection_removes_unfinished_writes(tmp_path: Path) -> None:
    fs = FilesystemBlobStore(tmp_path)
    (tmp_path / "tmp").mkdir()
    stale = tmp_path / "tmp" / "abc.part"
    stale.write_bytes(b"half")
    os.utime(stale, (1_000_000, 1_000_000))
    report = collect_garbage(fs, lambda _d: set(), before=datetime.now(UTC))
    assert report.incomplete_removed == 1
    assert (
        collect_garbage(
            fs, lambda _d: set(), before=datetime.now(UTC), dry_run=True
        ).incomplete_removed
        == 0
    )


@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    st.lists(
        st.tuples(st.binary(min_size=1, max_size=16), st.booleans(), st.booleans()),
        max_size=20,
    )
)
def test_garbage_collection_never_removes_a_referenced_or_recent_blob(
    tmp_path: Path, blobs_spec: list[tuple[bytes, bool, bool]]
) -> None:
    fs = FilesystemBlobStore(tmp_path / f"gc-{time.monotonic_ns()}")
    cutoff = datetime.now(UTC) - timedelta(hours=1)
    wanted: dict[str, tuple[bool, bool]] = {}
    for data, used, old in blobs_spec:
        digest = fs.put(data)
        prev_used, prev_old = wanted.get(digest, (False, True))
        wanted[digest] = (used or prev_used, old and prev_old)
        stamp = (cutoff - timedelta(hours=1)).timestamp() if wanted[digest][1] else time.time()
        os.utime(fs.root / blobs.blob_key(digest), (stamp, stamp))
    used_set = {d for d, (u, _) in wanted.items() if u}
    collect_garbage(fs, lambda page: used_set & set(page), before=cutoff)
    survivors = {b.digest for b in fs.list()}
    assert survivors == {d for d, (u, old) in wanted.items() if u or not old}
