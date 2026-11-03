"""A content-addressed blob store: bytes in, SHA-256 hex digest out.

A blob's name is the SHA-256 of its bytes, so a write can be repeated safely,
two writers of the same bytes agree on the name, and every read checks that
the bytes it got hash to the name it asked for (``BlobCorruptError`` if not).
Blobs live under ``sha256/<2 hex>/<2 hex>/<digest>`` in one of two backends:

- ``FilesystemBlobStore``: a directory. A write goes to a temporary file in
  ``tmp/``, is flushed to disk and renamed into place, so a reader never sees
  a partial blob and a crash leaves at most a temporary file, which garbage
  collection removes once it is older than the grace period.
- ``S3BlobStore``: a bucket on any S3-compatible object store (MinIO, Ceph,
  SeaweedFS, versitygw), addressed path-style. Requests are signed here with
  S3 signature version 4 (``sign_request``), so there is no vendor SDK. A
  single PUT is atomic on these servers, and the payload hash the signature
  covers is the blob's own name, so the server rejects a body that was
  changed in transit.

Garbage collection (``collect_garbage``) lists the store page by page, asks the
caller which of each page's digests are still referenced, and deletes the rest
that are older than a grace period. The grace period covers the window between
writing a blob and committing the row that refers to it: writers put the blob
first, and a repeated put refreshes the blob's modification time, so a blob
that is about to be referenced again is never old enough to collect.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import logging
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import quote, urlsplit

import httpx2

log = logging.getLogger(__name__)

DIGEST = re.compile(r"^[0-9a-f]{64}$")
NAMESPACE = "sha256"
GC_PAGE = 1000
# A listing page of 1,000 keys is about 300 KB of XML; anything far larger is
# not a listing.
MAX_LISTING_BYTES = 16 * 1024 * 1024
RETRY_STATUSES = frozenset({500, 502, 503, 504})
RETRY_DELAYS = (0.2, 1.0)


class BlobError(Exception):
    """The blob store could not be read or written."""


class BlobNotFoundError(BlobError):
    pass


class BlobCorruptError(BlobError):
    """The stored bytes do not hash to the blob's name."""


@dataclass(frozen=True)
class BlobInfo:
    digest: str
    size: int
    modified: datetime


class BlobStore(Protocol):
    def put(self, data: bytes) -> str:
        """Store ``data`` and return its digest. Repeating a put is harmless and
        refreshes the blob's modification time."""
        ...

    def get(self, digest: str) -> bytes:
        """The blob's bytes, checked against its digest."""
        ...

    def exists(self, digest: str) -> bool: ...

    def delete(self, digest: str) -> None:
        """Remove the blob; removing one that is not there is not an error."""
        ...

    def list(self) -> Iterator[BlobInfo]:
        """Every blob, in digest order."""
        ...

    def discard_incomplete(self, before: datetime) -> int:
        """Remove writes that never finished and started before ``before``."""
        ...

    def close(self) -> None: ...


def digest_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_digest(digest: str) -> str:
    if not DIGEST.fullmatch(digest):
        raise ValueError(f"not a SHA-256 hex digest: {digest[:80]!r}")
    return digest


def verified(digest: str, data: bytes) -> bytes:
    actual = digest_of(data)
    if actual != digest:
        raise BlobCorruptError(f"blob {digest} holds bytes that hash to {actual}")
    return data


def blob_key(digest: str) -> str:
    return f"{NAMESPACE}/{digest[:2]}/{digest[2:4]}/{digest}"


class FilesystemBlobStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._blobs = self.root / NAMESPACE
        self._tmp = self.root / "tmp"

    def _path(self, digest: str) -> Path:
        return self.root / blob_key(check_digest(digest))

    def put(self, data: bytes) -> str:
        digest = digest_of(data)
        path = self._path(digest)
        try:
            if path.exists():
                os.utime(path)
                return digest
            self._tmp.mkdir(parents=True, exist_ok=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(dir=self._tmp, prefix=digest[:16] + ".")
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_name, path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_name)
                raise
            _fsync_dir(path.parent)
        except OSError as exc:
            raise BlobError(f"cannot write blob {digest}: {exc.strerror}") from exc
        return digest

    def get(self, digest: str) -> bytes:
        path = self._path(digest)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise BlobNotFoundError(f"blob {digest} is not in the store") from None
        except OSError as exc:
            raise BlobError(f"cannot read blob {digest}: {exc.strerror}") from exc
        return verified(digest, data)

    def exists(self, digest: str) -> bool:
        return self._path(digest).is_file()

    def delete(self, digest: str) -> None:
        try:
            self._path(digest).unlink(missing_ok=True)
        except OSError as exc:
            raise BlobError(f"cannot delete blob {digest}: {exc.strerror}") from exc

    def list(self) -> Iterator[BlobInfo]:
        for first in _sorted_dirs(self._blobs):
            for second in _sorted_dirs(first):
                for entry in sorted(os.scandir(second), key=lambda e: e.name):
                    if not (DIGEST.fullmatch(entry.name) and entry.is_file()):
                        continue
                    try:
                        st = entry.stat()
                    except FileNotFoundError:  # deleted while listing
                        continue
                    yield BlobInfo(entry.name, st.st_size, _from_timestamp(st.st_mtime))

    def discard_incomplete(self, before: datetime) -> int:
        removed = 0
        if not self._tmp.is_dir():
            return 0
        for entry in os.scandir(self._tmp):
            with contextlib.suppress(FileNotFoundError):
                if entry.is_file() and _from_timestamp(entry.stat().st_mtime) < before:
                    os.unlink(entry.path)
                    removed += 1
        return removed

    def modified(self, digest: str) -> datetime | None:
        try:
            return _from_timestamp(self._path(digest).stat().st_mtime)
        except FileNotFoundError:
            return None

    def close(self) -> None:
        return


def _sorted_dirs(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_dir() and len(p.name) == 2)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _from_timestamp(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=UTC)


# --- S3 signature version 4 -------------------------------------------------

ALGORITHM = "AWS4-HMAC-SHA256"  # the protocol's fixed name for this scheme
EMPTY_SHA256 = digest_of(b"")
_UNRESERVED = "-_.~"


def uri_encode(value: str, *, keep_slash: bool = False) -> str:
    """RFC 3986 percent-encoding of everything but unreserved characters, as
    the signature's canonical form requires (spaces as %20, never +)."""
    return quote(value, safe=_UNRESERVED + ("/" if keep_slash else ""))


def signing_key(secret_key: str, date: str, region: str, service: str) -> bytes:
    key = hmac.new(f"AWS4{secret_key}".encode(), date.encode(), hashlib.sha256).digest()
    for part in (region, service, "aws4_request"):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    return key


@dataclass(frozen=True)
class Signed:
    headers: dict[str, str]
    canonical_request: str
    string_to_sign: str
    signature: str


def sign_request(
    method: str,
    host: str,
    path: str,
    *,
    query: Sequence[tuple[str, str]],
    headers: Mapping[str, str],
    payload_sha256: str,
    access_key: str,
    secret_key: str,
    region: str,
    now: datetime,
    service: str = "s3",
) -> Signed:
    """Sign one request. ``path`` is the decoded object path; ``headers`` are
    extra headers to sign besides host, date and payload hash. Returns the
    headers to send (including ``Authorization``)."""
    stamp = now.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    date = stamp[:8]
    send = {k.lower(): " ".join(v.split()) for k, v in headers.items()}
    send.update({"host": host, "x-amz-date": stamp, "x-amz-content-sha256": payload_sha256})
    names = sorted(send)
    canonical_query = "&".join(
        f"{k}={v}" for k, v in sorted((uri_encode(k), uri_encode(v)) for k, v in query)
    )
    canonical = "\n".join(
        [
            method,
            uri_encode(path, keep_slash=True) or "/",
            canonical_query,
            "".join(f"{n}:{send[n]}\n" for n in names),
            ";".join(names),
            payload_sha256,
        ]
    )
    scope = f"{date}/{region}/{service}/aws4_request"
    to_sign = "\n".join([ALGORITHM, stamp, scope, digest_of(canonical.encode())])
    signature = hmac.new(
        signing_key(secret_key, date, region, service), to_sign.encode(), hashlib.sha256
    ).hexdigest()
    send["authorization"] = (
        f"{ALGORITHM} Credential={access_key}/{scope}, "
        f"SignedHeaders={';'.join(names)}, Signature={signature}"
    )
    del send["host"]  # the client sets it from the URL
    return Signed(send, canonical, to_sign, signature)


class S3BlobStore:
    """Blobs in ``bucket`` under ``prefix`` on an S3-compatible endpoint."""

    def __init__(
        self,
        endpoint: str,
        bucket: str,
        *,
        access_key: str,
        secret_key: str,
        prefix: str = "",
        region: str = "us-east-1",
        timeout: float = 30.0,
        transport: httpx2.BaseTransport | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        parts = urlsplit(endpoint)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError("the blob endpoint must be an http or https URL")
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket):
            raise ValueError(f"{bucket!r} is not a valid bucket name")
        self.base = f"{parts.scheme}://{parts.netloc}"
        self.host = parts.netloc
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region = region
        self._access_key = access_key
        self._secret_key = secret_key
        self._clock = clock or (lambda: datetime.now(UTC))
        self._sleep = sleep
        self._client = httpx2.Client(
            timeout=httpx2.Timeout(timeout), transport=transport, follow_redirects=False
        )

    def _key(self, digest: str) -> str:
        key = blob_key(check_digest(digest))
        return f"{self.prefix}/{key}" if self.prefix else key

    def _request(
        self,
        method: str,
        key: str | None,
        *,
        query: Sequence[tuple[str, str]] = (),
        body: bytes = b"",
        payload_sha256: str = EMPTY_SHA256,
        headers: Mapping[str, str] | None = None,
    ) -> httpx2.Response:
        path = f"/{self.bucket}" + (f"/{key}" if key is not None else "")
        url = self.base + uri_encode(path, keep_slash=True)
        if query:
            url += "?" + "&".join(f"{uri_encode(k)}={uri_encode(v)}" for k, v in query)
        last: Exception | None = None
        for attempt in range(len(RETRY_DELAYS) + 1):
            if attempt:
                self._sleep(RETRY_DELAYS[attempt - 1])
            signed = sign_request(
                method,
                self.host,
                path,
                query=query,
                headers=headers or {},
                payload_sha256=payload_sha256,
                access_key=self._access_key,
                secret_key=self._secret_key,
                region=self.region,
                now=self._clock(),
            )
            try:
                response = self._client.request(method, url, content=body, headers=signed.headers)
            except httpx2.TransportError as exc:
                last = exc
                continue
            if response.status_code in RETRY_STATUSES:
                last = BlobError(_describe(response))
                continue
            return response
        if isinstance(last, httpx2.TransportError):
            raise BlobError(f"blob store unreachable: {type(last).__name__}") from last
        raise last or BlobError("blob store request failed")

    def put(self, data: bytes) -> str:
        digest = digest_of(data)
        response = self._request(
            "PUT",
            self._key(digest),
            body=data,
            payload_sha256=digest,
            headers={"content-type": "application/octet-stream"},
        )
        if response.status_code != 200:
            raise BlobError(f"cannot write blob {digest}: {_describe(response)}")
        return digest

    def get(self, digest: str) -> bytes:
        response = self._request("GET", self._key(digest))
        if response.status_code == 404:
            raise BlobNotFoundError(f"blob {digest} is not in the store")
        if response.status_code != 200:
            raise BlobError(f"cannot read blob {digest}: {_describe(response)}")
        return verified(digest, response.content)

    def exists(self, digest: str) -> bool:
        response = self._request("HEAD", self._key(digest))
        if response.status_code == 404:
            return False
        if response.status_code != 200:
            raise BlobError(f"cannot check blob {digest}: {_describe(response)}")
        return True

    def delete(self, digest: str) -> None:
        response = self._request("DELETE", self._key(digest))
        if response.status_code not in (200, 204, 404):
            raise BlobError(f"cannot delete blob {digest}: {_describe(response)}")

    def list(self) -> Iterator[BlobInfo]:
        prefix = (f"{self.prefix}/" if self.prefix else "") + f"{NAMESPACE}/"
        token: str | None = None
        while True:
            query = [("list-type", "2"), ("prefix", prefix), ("max-keys", str(GC_PAGE))]
            if token is not None:
                query.append(("continuation-token", token))
            response = self._request("GET", None, query=query)
            if response.status_code != 200:
                raise BlobError(f"cannot list blobs: {_describe(response)}")
            if len(response.content) > MAX_LISTING_BYTES:
                raise BlobError("the blob listing response is too large")
            root = _parse_xml(response.content)
            for item in _children(root, "Contents"):
                name = _text(item, "Key").rsplit("/", 1)[-1]
                if DIGEST.fullmatch(name):
                    yield BlobInfo(
                        name,
                        int(_text(item, "Size") or 0),
                        _parse_time(_text(item, "LastModified")),
                    )
            token = _text(root, "NextContinuationToken") or None
            if _text(root, "IsTruncated") != "true" or token is None:
                return

    def discard_incomplete(self, before: datetime) -> int:
        # Blobs are written with a single PUT, which is atomic: nothing to discard.
        return 0

    def create_bucket(self) -> None:
        """Create the bucket if it does not exist (for set-up and tests)."""
        response = self._request("PUT", None)
        if response.status_code == 200:
            return
        if response.status_code == 409 and _error_code(response) in (
            "BucketAlreadyOwnedByYou",
            "BucketAlreadyExists",
        ):
            return
        raise BlobError(f"cannot create bucket {self.bucket}: {_describe(response)}")

    def close(self) -> None:
        self._client.close()


def _parse_xml(data: bytes) -> ET.Element:
    try:
        # The body comes from the configured object store, is size-capped above,
        # and the bundled expat refuses entity-expansion attacks.
        return ET.fromstring(data)  # noqa: S314
    except ET.ParseError as exc:
        raise BlobError(f"the blob store sent malformed XML: {exc}") from None


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(node: ET.Element, name: str) -> Iterator[ET.Element]:
    return (child for child in node if _local(child.tag) == name)


def _text(node: ET.Element, name: str) -> str:
    for child in _children(node, name):
        return (child.text or "").strip()
    return ""


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise BlobError(f"the blob store sent an unreadable time {value[:40]!r}") from None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _error_code(response: httpx2.Response) -> str:
    if not response.content or len(response.content) > 64 * 1024:
        return ""
    try:
        return _text(_parse_xml(response.content), "Code")
    except BlobError:
        return ""


def _describe(response: httpx2.Response) -> str:
    code = _error_code(response)
    return f"HTTP {response.status_code}" + (f" {code}" if code else "")


# --- configuration and garbage collection -----------------------------------


def open_store(
    url: str,
    *,
    access_key: str | None = None,
    secret_key: str | None = None,
    region: str = "us-east-1",
) -> BlobStore:
    """A store from ``EVALKIT_BLOB_URL``: ``file:///path``, a plain absolute
    path, or ``s3+http(s)://host[:port]/bucket[/prefix]``."""
    parts = urlsplit(url)
    if parts.scheme in ("s3+http", "s3+https"):
        bucket, _, prefix = parts.path.lstrip("/").partition("/")
        if not bucket:
            raise ValueError("the blob URL names no bucket: s3+https://host/bucket[/prefix]")
        if not access_key or not secret_key:
            raise ValueError(
                "an S3 blob store needs EVALKIT_BLOB_ACCESS_KEY and EVALKIT_BLOB_SECRET_KEY"
            )
        if parts.query or parts.fragment or parts.username or parts.password:
            raise ValueError("put blob store credentials in their own variables, not the URL")
        return S3BlobStore(
            f"{parts.scheme.removeprefix('s3+')}://{parts.netloc}",
            bucket,
            prefix=prefix,
            access_key=access_key,
            secret_key=secret_key,
            region=region,
        )
    if parts.scheme == "file":
        if parts.netloc not in ("", "localhost"):
            raise ValueError("a file blob URL must be file:///absolute/path")
        return FilesystemBlobStore(parts.path)
    if parts.scheme == "" and Path(url).is_absolute():
        return FilesystemBlobStore(url)
    raise ValueError(
        "the blob URL must be file:///path, an absolute path or s3+https://host/bucket"
    )


@dataclass
class GarbageReport:
    scanned: int = 0
    referenced: int = 0
    recent: int = 0
    removed: int = 0
    removed_bytes: int = 0
    incomplete_removed: int = 0
    dry_run: bool = False


def collect_garbage(
    store: BlobStore,
    referenced: Callable[[Sequence[str]], set[str]],
    *,
    before: datetime,
    dry_run: bool = False,
) -> GarbageReport:
    """Delete blobs that ``referenced`` does not name and that were last written
    before ``before``. ``referenced`` is called once per page of up to 1,000
    digests and returns the ones still in use."""
    report = GarbageReport(dry_run=dry_run)
    page: list[BlobInfo] = []

    def flush() -> None:
        used = referenced([b.digest for b in page])
        for blob in page:
            if blob.digest in used:
                report.referenced += 1
            elif blob.modified >= before:
                report.recent += 1
            elif dry_run:
                report.removed += 1
                report.removed_bytes += blob.size
            elif _still_old(store, blob.digest, before):
                store.delete(blob.digest)
                report.removed += 1
                report.removed_bytes += blob.size
            else:
                report.recent += 1
        page.clear()

    for blob in store.list():
        report.scanned += 1
        page.append(blob)
        if len(page) >= GC_PAGE:
            flush()
    if page:
        flush()
    if not dry_run:
        report.incomplete_removed = store.discard_incomplete(before)
    return report


def _still_old(store: BlobStore, digest: str, before: datetime) -> bool:
    """Re-check a filesystem blob's time just before deleting it, so a put that
    refreshed it since the listing keeps it."""
    if isinstance(store, FilesystemBlobStore):
        modified = store.modified(digest)
        return modified is not None and modified < before
    return True


@dataclass
class ScrubReport:
    scanned: int = 0
    corrupt: list[str] = field(default_factory=list)


def scrub(store: BlobStore) -> ScrubReport:
    """Read every blob and check it against its digest."""
    report = ScrubReport()
    for blob in store.list():
        report.scanned += 1
        try:
            store.get(blob.digest)
        except BlobCorruptError:
            report.corrupt.append(blob.digest)
        except BlobNotFoundError:  # deleted since the listing
            report.scanned -= 1
    return report
