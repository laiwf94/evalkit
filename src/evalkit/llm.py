"""The one place that talks to the Claude API.

Prompt targets, the judge and token counting all go through ``ClaudeClient``,
so request shape, refusal handling, usage and cost are decided once:

* SDK retries are off (``max_retries=0``). Retries belong to the run's retry
  policy, which shares a cool-down across workers when the API sends
  Retry-After; two retry layers would multiply attempts.
* The system prompt carries ``cache_control``. It is identical for every case
  of a run, so after the first request it is read from the prompt cache.
* Structured output uses ``output_config.format`` (forced tool choice is not
  accepted by current models).
* Server-side refusal fallback (``fallbacks: "default"``) is on by default for
  the models that support it; cost is then priced per iteration, by the model
  that actually ran it.
* Requests with a large ``max_tokens`` are streamed, because the SDK refuses
  non-streaming requests that may run past ten minutes.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import anthropic
import httpx2
from pydantic import SecretStr

from evalkit.pricing import PriceTable, TokenCounts
from evalkit.ratelimit import parse_retry_after
from evalkit.targets.base import TargetError, Usage

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models that accept fallbacks="default" on the Claude API.
FALLBACK_MODELS = frozenset(
    {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
)
STREAMING_THRESHOLD = 16_000
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


@dataclass(frozen=True)
class Completion:
    text: str
    stop_reason: str | None
    usage: Usage
    model: str
    refusal_category: str | None = None

    def json(self) -> Any:
        return json.loads(self.text)


class ClaudeClient:
    def __init__(
        self,
        prices: PriceTable,
        api_key: SecretStr | None = None,
        *,
        http_client: httpx2.AsyncClient | None = None,
        base_url: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        self.prices = prices
        kwargs: dict[str, Any] = {"max_retries": 0, "timeout": timeout}
        if api_key is not None:
            kwargs["api_key"] = api_key.get_secret_value()
        if http_client is not None:
            kwargs["http_client"] = http_client
        if base_url is not None:
            kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)

    @property
    def sdk(self) -> anthropic.AsyncAnthropic:
        return self._client

    def request_params(
        self,
        *,
        model: str,
        prompt: str,
        system: str,
        max_tokens: int,
        effort: str,
        output_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Messages API parameters shared by online and batch requests."""
        output_config: dict[str, Any] = {"effort": effort}
        if output_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": output_schema}
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": output_config,
        }
        if system:
            params["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        return params

    async def complete(
        self,
        *,
        model: str,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
        effort: str = "medium",
        output_schema: dict[str, Any] | None = None,
        fallbacks: bool = True,
    ) -> Completion:
        self.prices.require(model)
        params = self.request_params(
            model=model,
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            effort=effort,
            output_schema=output_schema,
        )
        if fallbacks and model in FALLBACK_MODELS:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        async with _mapped_errors():
            if max_tokens > STREAMING_THRESHOLD:
                async with self._client.beta.messages.stream(**params) as stream:
                    message = await stream.get_final_message()
            else:
                message = await self._client.beta.messages.create(**params)
        try:
            return self.completion_from_message(message, requested_model=model)
        except (TypeError, AttributeError) as exc:
            raise TargetError(
                f"unexpected response from the Claude API: {exc}", retryable=False
            ) from None

    def completion_from_message(
        self, message: Any, *, requested_model: str, batch: bool = False
    ) -> Completion:
        text = "".join(block.text for block in message.content if block.type == "text")
        category = None
        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None)
            # A refusal can arrive after partial output; the partial text is not an answer.
            text = ""
        return Completion(
            text=text,
            stop_reason=message.stop_reason,
            usage=self._usage(message.usage, requested_model, batch=batch),
            model=message.model,
            refusal_category=category,
        )

    def _usage(self, usage: Any, requested_model: str, *, batch: bool) -> Usage:
        counts = _counts(usage)
        iterations = getattr(usage, "iterations", None) or []
        priced = [it for it in iterations if it.type in ("message", "fallback_message")]
        if priced:
            cost = sum(
                self.prices.cost(
                    getattr(it, "model", None) or requested_model, _counts(it), batch=batch
                )
                for it in priced
            )
        else:
            cost = self.prices.cost(requested_model, counts, batch=batch)
        return Usage(
            counts.input_tokens,
            counts.output_tokens,
            counts.cache_read_tokens,
            counts.cache_write_tokens,
            cost,
        )

    async def create_batch(self, requests: list[dict[str, Any]]) -> Any:
        """Submit Messages requests as one Message Batch (half price, results within 24 h)."""
        async with _mapped_errors():
            return await self._client.messages.batches.create(requests=requests)  # type: ignore[arg-type]

    async def get_batch(self, batch_id: str) -> Any:
        async with _mapped_errors():
            return await self._client.messages.batches.retrieve(batch_id)

    async def batch_results(self, batch_id: str) -> list[Any]:
        """All result entries of an ended batch. Results arrive in any order; match on custom_id."""
        async with _mapped_errors():
            decoder = await self._client.messages.batches.results(batch_id)
            return [entry async for entry in decoder]

    async def count_tokens(self, *, model: str, prompt: str, system: str = "") -> int:
        self.prices.require(model)
        params: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            params["system"] = system
        async with _mapped_errors():
            result = await self._client.messages.count_tokens(**params)
        return int(result.input_tokens)

    async def aclose(self) -> None:
        await self._client.close()


@contextlib.asynccontextmanager
async def _mapped_errors() -> AsyncIterator[None]:
    """Turn SDK exceptions into TargetErrors that the retry policy understands."""
    try:
        yield
    except anthropic.APIStatusError as exc:
        raise _status_error(exc) from None
    except anthropic.APITimeoutError:
        raise TargetError("Claude API request timed out", retryable=True) from None
    except anthropic.APIConnectionError as exc:
        raise TargetError(f"cannot reach the Claude API: {exc}", retryable=True) from None
    except (anthropic.APIError, ValueError) as exc:
        # A response the SDK cannot parse: specific to this case, not worth retrying.
        raise TargetError(
            f"unexpected response from the Claude API: {exc}", retryable=False
        ) from None


def _counts(usage: Any) -> TokenCounts:
    return TokenCounts(
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", None) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", None) or 0,
    )


def _status_error(exc: anthropic.APIStatusError) -> TargetError:
    status = exc.status_code
    retry_after = parse_retry_after(exc.response.headers.get("retry-after"))
    request_id = exc.response.headers.get("request-id", "")
    suffix = f" (request {request_id})" if request_id else ""
    return TargetError(
        f"Claude API returned {status}: {exc.message}{suffix}",
        retryable=status in _RETRYABLE_STATUS,
        retry_after=retry_after,
    )
