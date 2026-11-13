"""The application under test, called over HTTP.

URLs come from operator configuration only. A run names an endpoint; it never
supplies a URL, so an API caller cannot point the evaluator at an internal
service or a cloud metadata address. On top of that, each call:

* resolves the host and refuses non-public addresses (loopback, private,
  link-local including 169.254.169.254, CGNAT, multicast, reserved, and the
  IPv4 forms hidden in IPv6) unless the endpoint sets ``allow_private``;
* connects to the address it just checked, sending the original Host header
  and TLS server name, so a DNS answer that changes between the check and the
  connect (DNS rebinding) cannot redirect the request;
* ignores proxy environment variables, follows no redirects and caps the
  response size before buffering.

Request body: ``{"input": <case input>}``. The answer is read from
``response_field`` (dotted path) of the JSON response.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import socket
from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from urllib.parse import urlsplit

import httpx2
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from evalkit.datasets import Case
from evalkit.ratelimit import parse_retry_after
from evalkit.targets.base import Generation, Target, TargetError
from evalkit.targets.specs import HttpTargetSpec

Resolver = Callable[[str, int], Awaitable[list[str]]]
IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

HEADER_NAME = r"^[A-Za-z0-9-]{1,64}$"


class HttpEndpoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str = Field(max_length=2048)
    timeout_seconds: float = Field(default=30.0, gt=0, le=600)
    max_response_bytes: int = Field(default=1024 * 1024, ge=1024, le=64 * 1024 * 1024)
    headers: dict[str, str] = Field(default_factory=dict)
    secret_headers: dict[str, SecretStr] = Field(default_factory=dict)
    allow_private: bool = False

    @field_validator("url")
    @classmethod
    def _valid_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https"):
            raise ValueError("url must be http or https")
        if not parts.hostname:
            raise ValueError("url must have a host")
        if parts.username or parts.password:
            raise ValueError("put credentials in secret_headers, not in the url")
        if parts.fragment:
            raise ValueError("url must not have a fragment")
        return value

    @field_validator("headers", "secret_headers")
    @classmethod
    def _valid_header_names(cls, value: dict[str, Any]) -> dict[str, Any]:
        for name in value:
            if not re.fullmatch(HEADER_NAME, name) or name.lower() in ("host", "content-length"):
                raise ValueError(f"header {name!r} is not allowed")
        return value

    def fingerprint(self) -> dict[str, Any]:
        """What identifies the endpoint for caching: never the secret values."""
        return {
            "url": self.url,
            "headers": self.headers,
            "secret_headers": sorted(self.secret_headers),
        }


def is_public(ip: IPAddress) -> bool:
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = ip.ipv4_mapped or ip.sixtofour or (ip.teredo[1] if ip.teredo else None)
        if embedded is not None:
            return is_public(embedded)
    return ip.is_global and not ip.is_multicast


async def system_resolver(host: str, port: int) -> list[str]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return [str(info[4][0]) for info in infos]


class HttpTarget(Target):
    def __init__(
        self,
        spec: HttpTargetSpec,
        endpoint: HttpEndpoint,
        *,
        resolver: Resolver = system_resolver,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self.spec = spec
        self.endpoint = endpoint
        self.provider = f"http:{spec.endpoint}"
        self._resolver = resolver
        parts = urlsplit(endpoint.url)
        self._scheme = parts.scheme
        self._host = parts.hostname or ""
        self._port = parts.port or (443 if parts.scheme == "https" else 80)
        self._path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        default_port = self._port == (443 if parts.scheme == "https" else 80)
        host_header = f"[{self._host}]" if ":" in self._host else self._host
        self._host_header = host_header if default_port else f"{host_header}:{self._port}"
        self._client = httpx2.AsyncClient(
            transport=transport,
            follow_redirects=False,
            trust_env=False,
            timeout=httpx2.Timeout(
                endpoint.timeout_seconds, connect=min(10.0, endpoint.timeout_seconds)
            ),
        )

    async def _pinned_address(self) -> str:
        try:
            literal = ipaddress.ip_address(self._host)
        except ValueError:
            literal = None
        if literal is not None:
            candidates = [str(literal)]
        else:
            try:
                candidates = await self._resolver(self._host, self._port)
            except OSError as exc:
                raise TargetError(f"cannot resolve {self._host}: {exc}", retryable=True) from None
        if not candidates:
            raise TargetError(f"{self._host} has no addresses", retryable=True)
        addresses = [ipaddress.ip_address(a.split("%")[0]) for a in candidates]
        if not self.endpoint.allow_private:
            blocked = [str(a) for a in addresses if not is_public(a)]
            if blocked:
                raise TargetError(
                    f"endpoint {self.spec.endpoint!r} resolves to a non-public address "
                    f"({', '.join(blocked)}); set allow_private for this endpoint to permit it",
                    retryable=False,
                )
        first = addresses[0]
        return f"[{first}]" if first.version == 6 else str(first)

    async def generate(self, case: Case) -> Generation:
        document = await self.post({"input": case.input})
        return Generation(text=self._extract(document), stop_reason=None)

    async def post(self, payload: Any) -> Any:
        """POST ``payload`` as JSON to the endpoint, under every check above, and
        return the parsed JSON answer. The tool recorder uses it as well."""
        address = await self._pinned_address()
        url = f"{self._scheme}://{address}:{self._port}{self._path}"
        headers = {**self.endpoint.headers, "Host": self._host_header}
        headers.update({k: v.get_secret_value() for k, v in self.endpoint.secret_headers.items()})
        extensions = {"sni_hostname": self._host} if self._scheme == "https" else {}
        request = self._client.build_request(
            "POST", url, json=payload, headers=headers, extensions=extensions
        )
        try:
            response = await self._client.send(request, stream=True)
        except httpx2.TimeoutException:
            raise TargetError(
                f"endpoint {self.spec.endpoint!r} timed out", retryable=True
            ) from None
        except httpx2.TransportError as exc:
            raise TargetError(
                f"cannot reach endpoint {self.spec.endpoint!r}: {type(exc).__name__}",
                retryable=True,
            ) from None
        except httpx2.HTTPError as exc:
            raise TargetError(
                f"request to endpoint {self.spec.endpoint!r} failed: {type(exc).__name__}",
                retryable=False,
            ) from None
        try:
            body = await self._read_capped(response)
        finally:
            await response.aclose()
        try:
            return json.loads(body)
        except (ValueError, RecursionError):
            raise TargetError(
                f"endpoint {self.spec.endpoint!r} did not return JSON "
                f"(content-type {response.headers.get('content-type', 'missing')})",
                retryable=False,
            ) from None

    async def _read_capped(self, response: httpx2.Response) -> bytes:
        status = response.status_code
        if 300 <= status < 400:
            raise TargetError(
                f"endpoint {self.spec.endpoint!r} answered with a redirect ({status}); "
                "redirects are not followed",
                retryable=False,
            )
        if status >= 400:
            retry_after = parse_retry_after(response.headers.get("retry-after"))
            raise TargetError(
                f"endpoint {self.spec.endpoint!r} returned {status}",
                retryable=status in (408, 429) or status >= 500,
                retry_after=retry_after,
            )
        limit = self.endpoint.max_response_bytes
        declared = response.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            raise TargetError(f"response is larger than {limit} bytes", retryable=False)
        chunks: list[bytes] = []
        size = 0
        try:
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > limit:
                    raise TargetError(f"response is larger than {limit} bytes", retryable=False)
                chunks.append(chunk)
        except httpx2.DecodingError:
            raise TargetError(
                f"response from {self.spec.endpoint!r} could not be decoded", retryable=False
            ) from None
        except httpx2.TransportError as exc:
            raise TargetError(
                f"response from {self.spec.endpoint!r} was cut off: {exc}", retryable=True
            ) from None
        return b"".join(chunks)

    def _extract(self, document: Any) -> str:
        value: Any = document
        for part in self.spec.response_field.split("."):
            if not isinstance(value, Mapping) or part not in value:
                raise TargetError(
                    f"response has no field {self.spec.response_field!r}", retryable=False
                )
            value = value[part]
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    async def aclose(self) -> None:
        await self._client.aclose()
