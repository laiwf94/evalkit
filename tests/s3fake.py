"""An in-memory S3-compatible server behind ``httpx2.MockTransport``.

It checks every request's signature by recomputing it from what arrived on
the wire (method, path, query, the signed headers and the declared payload
hash), checks the payload hash against the body, and pages listings in pages
of ``page_size`` so continuation tokens are exercised.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import parse_qsl, unquote, urlsplit
from xml.sax.saxutils import escape

import httpx2

from evalkit.blobs import sign_request

ACCESS = "test-access-key"
SECRET = "test-secret-key/with+symbols"
REGION = "test-region-1"
AUTH = re.compile(
    r"^AWS4-HMAC-SHA256 Credential=(?P<key>[^/]+)/(?P<date>\d{8})/(?P<region>[^/]+)/s3/"
    r"aws4_request, SignedHeaders=(?P<signed>[a-z0-9;-]+), Signature=(?P<sig>[0-9a-f]{64})$"
)


@dataclass
class Obj:
    data: bytes
    modified: datetime


@dataclass
class FakeS3:
    bucket: str = "evalkit"
    page_size: int = 3
    objects: dict[str, Obj] = field(default_factory=dict)
    buckets: set[str] = field(default_factory=lambda: {"evalkit"})
    requests: list[tuple[str, str]] = field(default_factory=list)
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    # Hooks: return a response to short-circuit a request.
    intercept: Callable[[httpx2.Request], httpx2.Response | None] | None = None

    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self.handle)

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append((request.method, request.url.path))
        if self.intercept is not None:
            hooked = self.intercept(request)
            if hooked is not None:
                return hooked
        problem = self._check_signature(request)
        if problem:
            return _error(403, "SignatureDoesNotMatch", problem)
        parts = [unquote(p) for p in request.url.path.lstrip("/").split("/", 1)]
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else None
        if key is None and request.method == "PUT":
            if bucket in self.buckets:
                return _error(409, "BucketAlreadyOwnedByYou", "yours already")
            self.buckets.add(bucket)
            return httpx2.Response(200)
        if bucket not in self.buckets:
            return _error(404, "NoSuchBucket", "no bucket")
        if key is None:
            return self._list(request)
        if request.method == "PUT":
            self.objects[key] = Obj(request.content, self.clock())
            return httpx2.Response(200, headers={"ETag": '"x"'})
        obj = self.objects.get(key)
        if request.method == "DELETE":
            self.objects.pop(key, None)
            return httpx2.Response(204)
        if obj is None:
            if request.method == "HEAD":
                return httpx2.Response(404)
            return _error(404, "NoSuchKey", "no such key")
        if request.method == "HEAD":
            return httpx2.Response(200, headers={"Content-Length": str(len(obj.data))})
        return httpx2.Response(200, content=obj.data)

    def _check_signature(self, request: httpx2.Request) -> str | None:
        match = AUTH.fullmatch(request.headers.get("authorization", ""))
        if match is None:
            return "no or malformed authorization"
        if match["key"] != ACCESS or match["region"] != REGION:
            return "wrong credential scope"
        declared = request.headers.get("x-amz-content-sha256", "")
        if declared != hashlib.sha256(request.content).hexdigest():
            return "payload hash does not match the body"
        names = match["signed"].split(";")
        if not {"host", "x-amz-date", "x-amz-content-sha256"} <= set(names):
            return "required headers are not signed"
        extra = {n: request.headers[n] for n in names if n not in ("host", "x-amz-date")}
        extra.pop("x-amz-content-sha256")
        now = datetime.strptime(request.headers["x-amz-date"], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        url = urlsplit(str(request.url))
        expected = sign_request(
            request.method,
            request.headers["host"],
            unquote(url.path),
            query=parse_qsl(url.query, keep_blank_values=True),
            headers=extra,
            payload_sha256=declared,
            access_key=ACCESS,
            secret_key=SECRET,
            region=REGION,
            now=now,
        )
        if expected.signature != match["sig"]:
            return "signature mismatch"
        return None

    def _list(self, request: httpx2.Request) -> httpx2.Response:
        query = dict(request.url.params)
        if query.get("list-type") != "2":
            return _error(400, "InvalidArgument", "list-type")
        prefix = query.get("prefix", "")
        keys = sorted(k for k in self.objects if k.startswith(prefix))
        start = int(query.get("continuation-token", "0"))
        size = min(self.page_size, int(query.get("max-keys", "1000")))
        page = keys[start : start + size]
        more = start + size < len(keys)
        items = "".join(
            f"<Contents><Key>{escape(k)}</Key>"
            f"<LastModified>{self.objects[k].modified.strftime('%Y-%m-%dT%H:%M:%S.000Z')}"
            f"</LastModified><Size>{len(self.objects[k].data)}</Size></Contents>"
            for k in page
        )
        token = f"<NextContinuationToken>{start + size}</NextContinuationToken>" if more else ""
        body = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<ListBucketResult xmlns="urn:example:s3">'
            f"<Name>{self.bucket}</Name><Prefix>{escape(prefix)}</Prefix>"
            f"<IsTruncated>{'true' if more else 'false'}</IsTruncated>{token}{items}"
            "</ListBucketResult>"
        )
        return httpx2.Response(200, content=body.encode())


def _error(status: int, code: str, message: str) -> httpx2.Response:
    body = f"<Error><Code>{code}</Code><Message>{escape(message)}</Message></Error>"
    return httpx2.Response(status, content=body.encode())
