"""What the raw HTTP providers share: one JSON-over-HTTPS call and its failures.

The OpenAI-compatible and Gemini providers speak plain HTTP through httpx2
rather than vendor SDKs, which keeps the dependency list short and puts every
request byte under evalkit's control (and under test). Each call:

* sends one JSON body and follows no redirects, so a misconfigured base URL
  fails loudly instead of silently posting the prompt somewhere else;
* reads at most ``max_response_bytes`` of the answer before giving up, so a
  broken or hostile server cannot exhaust memory;
* turns transport failures, HTTP error statuses and unparseable bodies into a
  ``ProviderError`` whose kind decides whether the run retries.

Retries are not done here. They belong to the run's retry policy, which shares
cool-downs across workers when a provider asks to slow down.
"""

from __future__ import annotations

import json
import math
from typing import Any

import httpx2

from evalkit.pricing import PriceTable, TokenCounts
from evalkit.providers.base import ErrorKind, ProviderError, kind_for_status
from evalkit.ratelimit import parse_retry_after

DEFAULT_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
ERROR_MESSAGE_LIMIT = 500


class HttpJsonProvider:
    """Base for providers reached over a JSON HTTP API.

    Subclasses set ``label`` (the provider's human name, for messages) and
    override ``error_details`` to read their API's error body.
    """

    label = "provider"

    def __init__(
        self,
        *,
        name: str,
        prices: PriceTable,
        base_url: str,
        headers: dict[str, str],
        timeout: float = 600.0,
        transport: httpx2.AsyncBaseTransport | None = None,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        self.name = name
        self.prices = prices
        self.base_url = base_url.rstrip("/")
        self.max_response_bytes = max_response_bytes
        self._client = httpx2.AsyncClient(
            headers={"content-type": "application/json", **headers},
            timeout=httpx2.Timeout(timeout, connect=min(timeout, 30.0)),
            follow_redirects=False,
            transport=transport,
        )

    def price(self, model: str, tokens: TokenCounts, *, batch: bool = False) -> float:
        return self.prices.cost(model, tokens, batch=batch)

    async def post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST ``body`` to ``base_url + path`` and return the decoded JSON object."""
        try:
            async with self._client.stream(
                "POST", f"{self.base_url}{path}", content=json.dumps(body).encode()
            ) as response:
                raw = await self._read_capped(response)
        except httpx2.HTTPError as exc:
            raise self.classify_error(exc) from None
        if response.status_code >= 400:
            raise self.status_error(response.status_code, response.headers, raw)
        if 300 <= response.status_code < 400:
            raise ProviderError(
                f"{self.label} answered {response.status_code} (a redirect); "
                "check the provider's base URL",
                kind=ErrorKind.INVALID_REQUEST,
                provider=self.name,
                status=response.status_code,
            )
        payload = self._decode(raw)
        if not isinstance(payload, dict):
            raise self.malformed("the response body is not a JSON object")
        return payload

    async def _read_capped(self, response: httpx2.Response) -> bytes:
        chunks: list[bytes] = []
        size = 0
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > self.max_response_bytes:
                raise ProviderError(
                    f"{self.label} response exceeds {self.max_response_bytes} bytes",
                    kind=ErrorKind.MALFORMED_RESPONSE,
                    provider=self.name,
                    status=response.status_code,
                )
            chunks.append(chunk)
        return b"".join(chunks)

    def _decode(self, raw: bytes) -> Any:
        try:
            return json.loads(raw)
        except (ValueError, RecursionError):
            raise self.malformed("the response body is not JSON") from None

    def malformed(self, detail: str) -> ProviderError:
        return ProviderError(
            f"unexpected response from {self.label}: {detail}",
            kind=ErrorKind.MALFORMED_RESPONSE,
            provider=self.name,
        )

    def status_error(self, status: int, headers: httpx2.Headers, raw: bytes) -> ProviderError:
        try:
            payload = json.loads(raw)
        except (ValueError, RecursionError):
            payload = None
        message, kind, retry_after = self.error_details(status, payload)
        if retry_after is None:
            retry_after = retry_after_from_headers(headers)
        if not message:
            message = raw[:ERROR_MESSAGE_LIMIT].decode("utf-8", "replace").strip()
        message = message[:ERROR_MESSAGE_LIMIT]
        return ProviderError(
            f"{self.label} returned {status}" + (f": {message}" if message else ""),
            kind=kind,
            provider=self.name,
            status=status,
            retry_after=retry_after,
        )

    def error_details(self, status: int, payload: Any) -> tuple[str, ErrorKind, float | None]:
        """Message, kind and server-requested delay from an error body."""
        return "", kind_for_status(status), None

    def classify_error(self, exc: BaseException) -> ProviderError:
        """Map transport and parsing exceptions onto error kinds."""
        match exc:
            case ProviderError():
                return exc
            case httpx2.TimeoutException():
                return ProviderError(
                    f"{self.label} request timed out", kind=ErrorKind.TIMEOUT, provider=self.name
                )
            case httpx2.TransportError():
                return ProviderError(
                    f"cannot reach {self.label}: {type(exc).__name__}: {exc}",
                    kind=ErrorKind.CONNECTION,
                    provider=self.name,
                )
            case httpx2.HTTPError() | ValueError() | TypeError() | KeyError() | IndexError():
                return self.malformed(f"{type(exc).__name__}: {exc}")
            case _:
                return ProviderError(
                    f"{self.label} call failed: {type(exc).__name__}: {exc}",
                    kind=ErrorKind.UNKNOWN,
                    provider=self.name,
                )

    async def aclose(self) -> None:
        await self._client.aclose()


def retry_after_from_headers(headers: httpx2.Headers) -> float | None:
    """Seconds to wait: ``retry-after-ms`` when present, else ``Retry-After``."""
    millis = headers.get("retry-after-ms")
    if millis is not None:
        try:
            value = float(millis)
        except ValueError:
            value = -1.0
        if math.isfinite(value) and value >= 0:
            return value / 1000
    return parse_retry_after(headers.get("retry-after"))


def non_negative_int(value: Any) -> int:
    """A token count from a usage block; missing or null counts as zero."""
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"token count {value!r} is not a non-negative integer")
    return value
