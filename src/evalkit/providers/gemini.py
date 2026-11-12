"""The Gemini API provider, over raw HTTP (``generativelanguage.googleapis.com``).

* ``complete`` posts to ``models/{model}:generateContent``. The system prompt
  goes in ``systemInstruction`` and ``max_tokens`` in
  ``generationConfig.maxOutputTokens``.
* Structured output sets ``responseMimeType: application/json`` and passes the
  JSON schema as ``responseJsonSchema``.
* ``effort`` becomes ``thinkingConfig.thinkingLevel`` only when the provider is
  built with ``thinking_level=True`` (models without thinking levels reject
  it): ``low`` stays ``low``, ``medium`` stays ``medium``, and ``high``,
  ``xhigh`` and ``max`` become ``high``.
* Safety blocks are their own error kind, ``safety_block``, never an empty
  answer: a prompt blocked before generation (``promptFeedback.blockReason``)
  and an answer stopped by a filter (finish reasons ``SAFETY``,
  ``PROHIBITED_CONTENT``, ``BLOCKLIST``, ``SPII``, ``RECITATION`` and
  ``IMAGE_SAFETY``). The message names the reason and the categories rated as
  blocked, so a report can tell a filter from a model that refused.
* Thought parts (``"thought": true``) are not part of the answer.
* Usage: ``promptTokenCount`` includes ``cachedContentTokenCount``, which is
  priced as cache reads; ``thoughtsTokenCount`` is billed as output, so it is
  added to ``candidatesTokenCount``.
* ``count_tokens`` posts the same request to ``models/{model}:countTokens``.
* ``embed`` posts to ``models/{model}:batchEmbedContents``. Its response
  carries no token counts, so embedding usage is recorded as zero tokens.
* Errors carry a ``google.rpc`` status; a ``RetryInfo`` detail's
  ``retryDelay`` becomes the retry delay when no Retry-After header is sent.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any
from urllib.parse import quote

import httpx2
from pydantic import SecretStr

from evalkit.pricing import PriceTable, TokenCounts
from evalkit.providers._http import (
    DEFAULT_MAX_RESPONSE_BYTES,
    HttpJsonProvider,
    non_negative_int,
)
from evalkit.providers.base import (
    Completion,
    Embeddings,
    ErrorKind,
    Prompt,
    ProviderError,
    as_messages,
    kind_for_status,
)
from evalkit.targets.base import Usage

PROVIDER = "gemini"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
THINKING_LEVEL = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}
SAFETY_FINISH_REASONS = frozenset(
    {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY"}
)
# google.rpc status names, used when the HTTP status alone does not say enough.
RPC_STATUS_KINDS = {
    "RESOURCE_EXHAUSTED": ErrorKind.RATE_LIMITED,
    "UNAVAILABLE": ErrorKind.OVERLOADED,
    "DEADLINE_EXCEEDED": ErrorKind.TIMEOUT,
    "INTERNAL": ErrorKind.SERVER,
    "ABORTED": ErrorKind.CONFLICT,
    "INVALID_ARGUMENT": ErrorKind.INVALID_REQUEST,
    "FAILED_PRECONDITION": ErrorKind.INVALID_REQUEST,
    "OUT_OF_RANGE": ErrorKind.INVALID_REQUEST,
    "UNAUTHENTICATED": ErrorKind.AUTH,
    "PERMISSION_DENIED": ErrorKind.AUTH,
    "NOT_FOUND": ErrorKind.NOT_FOUND,
}
RETRY_DELAY = re.compile(r"^(\d+(?:\.\d+)?)s$")


class GeminiClient(HttpJsonProvider):
    """The ``Provider`` for the Gemini API."""

    label = "the Gemini API"

    def __init__(
        self,
        prices: PriceTable,
        api_key: SecretStr | None = None,
        *,
        name: str = PROVIDER,
        base_url: str = DEFAULT_BASE_URL,
        thinking_level: bool = False,
        timeout: float = 600.0,
        transport: httpx2.AsyncBaseTransport | None = None,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        headers = {}
        if api_key is not None:
            headers["x-goog-api-key"] = api_key.get_secret_value()
        super().__init__(
            name=name,
            prices=prices,
            base_url=base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
            max_response_bytes=max_response_bytes,
        )
        self.thinking_level = thinking_level

    def request_body(
        self,
        *,
        model: str,
        prompt: Prompt,
        system: str,
        max_tokens: int,
        effort: str,
        output_schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        config: dict[str, Any] = {"maxOutputTokens": max_tokens}
        if output_schema is not None:
            config["responseMimeType"] = "application/json"
            config["responseJsonSchema"] = output_schema
        if self.thinking_level:
            config["thinkingConfig"] = {"thinkingLevel": THINKING_LEVEL.get(effort, "high")}
        body: dict[str, Any] = {"contents": _contents(prompt), "generationConfig": config}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    async def complete(
        self,
        *,
        model: str,
        prompt: Prompt,
        system: str = "",
        max_tokens: int = 4096,
        effort: str = "medium",
        output_schema: dict[str, Any] | None = None,
        fallbacks: bool = True,
    ) -> Completion:
        # ``fallbacks`` is a Claude API feature; Gemini has none.
        self.prices.require(model)
        body = self.request_body(
            model=model,
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            effort=effort,
            output_schema=output_schema,
        )
        payload = await self.post_json(f"/{_model_path(model)}:generateContent", body)
        try:
            return self.completion_from_payload(payload, requested_model=model)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            raise self.malformed(f"{type(exc).__name__}: {exc}") from None

    def completion_from_payload(
        self, payload: dict[str, Any], *, requested_model: str
    ) -> Completion:
        feedback = payload.get("promptFeedback") or {}
        block = feedback.get("blockReason")
        if block:
            raise self._safety_block(
                f"blocked the prompt ({block})", feedback.get("safetyRatings"), requested_model
            )
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("no candidates in the response")
        candidate = candidates[0]
        finish = candidate.get("finishReason")
        if finish in SAFETY_FINISH_REASONS:
            raise self._safety_block(
                f"stopped the answer ({finish})", candidate.get("safetyRatings"), requested_model
            )
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(
            part["text"]
            for part in parts
            if isinstance(part.get("text"), str) and not part.get("thought")
        )
        return Completion(
            text=text,
            stop_reason=finish if isinstance(finish, str) else None,
            usage=self._usage(payload.get("usageMetadata") or {}, requested_model),
            model=str(payload.get("modelVersion") or requested_model),
        )

    def _safety_block(self, what: str, ratings: Any, model: str) -> ProviderError:
        blocked = [
            str(rating.get("category"))
            for rating in ratings or []
            if isinstance(rating, dict) and rating.get("blocked")
        ]
        categories = f"; blocked categories: {', '.join(blocked)}" if blocked else ""
        return ProviderError(
            f"{self.label} safety filter {what} on model {model}{categories}",
            kind=ErrorKind.SAFETY_BLOCK,
            provider=self.name,
        )

    def _usage(self, usage: dict[str, Any], model: str) -> Usage:
        prompt_tokens = non_negative_int(usage.get("promptTokenCount"))
        cached = min(non_negative_int(usage.get("cachedContentTokenCount")), prompt_tokens)
        counts = TokenCounts(
            input_tokens=prompt_tokens - cached,
            output_tokens=non_negative_int(usage.get("candidatesTokenCount"))
            + non_negative_int(usage.get("thoughtsTokenCount")),
            cache_read_tokens=cached,
        )
        return Usage(
            counts.input_tokens,
            counts.output_tokens,
            counts.cache_read_tokens,
            0,
            self.price(model, counts),
        )

    async def count_tokens(self, *, model: str, prompt: str, system: str = "") -> int:
        self.prices.require(model)
        request: dict[str, Any] = {"model": _model_path(model), "contents": _contents(prompt)}
        if system:
            request["systemInstruction"] = {"parts": [{"text": system}]}
        payload = await self.post_json(
            f"/{_model_path(model)}:countTokens", {"generateContentRequest": request}
        )
        try:
            return non_negative_int(payload["totalTokens"])
        except (ValueError, KeyError) as exc:
            raise self.malformed(f"{type(exc).__name__}: {exc}") from None

    async def embed(self, *, model: str, texts: Sequence[str]) -> Embeddings:
        self.prices.require(model)
        if not texts:
            return Embeddings(vectors=[], usage=Usage(), model=model)
        path = _model_path(model)
        body = {
            "requests": [{"model": path, "content": {"parts": [{"text": text}]}} for text in texts]
        }
        payload = await self.post_json(f"/{path}:batchEmbedContents", body)
        try:
            vectors = _vectors(payload, len(texts))
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise self.malformed(f"{type(exc).__name__}: {exc}") from None
        return Embeddings(vectors=vectors, usage=Usage(), model=model)

    def error_details(self, status: int, payload: Any) -> tuple[str, ErrorKind, float | None]:
        kind = kind_for_status(status)
        error = payload.get("error") if isinstance(payload, dict) else None
        if not isinstance(error, dict):
            return "", kind, None
        rpc_status = error.get("status")
        if kind is ErrorKind.UNKNOWN and rpc_status in RPC_STATUS_KINDS:
            kind = RPC_STATUS_KINDS[rpc_status]
        message = error.get("message")
        text = message if isinstance(message, str) else ""
        if isinstance(rpc_status, str) and rpc_status:
            text = f"{text} ({rpc_status})" if text else rpc_status
        return text, kind, _retry_delay(error.get("details"))


def _vectors(payload: dict[str, Any], count: int) -> list[list[float]]:
    embeddings = payload["embeddings"]
    if not isinstance(embeddings, list) or len(embeddings) != count:
        raise ValueError(f"expected {count} embeddings")
    return [[float(x) for x in item["values"]] for item in embeddings]


def _contents(prompt: Prompt) -> list[dict[str, Any]]:
    # Gemini calls the assistant's side of a conversation "model".
    return [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in as_messages(prompt)
    ]


def _model_path(model: str) -> str:
    # Model names come from the price table, but are quoted so a name with a
    # slash or a colon cannot reach another endpoint.
    return f"models/{quote(model, safe='')}"


def _retry_delay(details: Any) -> float | None:
    """Seconds from a ``google.rpc.RetryInfo`` detail ("37s", "0.5s")."""
    if not isinstance(details, list):
        return None
    for detail in details:
        if not isinstance(detail, dict) or not str(detail.get("@type", "")).endswith(
            "google.rpc.RetryInfo"
        ):
            continue
        match = RETRY_DELAY.match(str(detail.get("retryDelay", "")))
        if match:
            return float(match.group(1))
    return None
