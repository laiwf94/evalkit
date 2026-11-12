"""A provider for any server that speaks the OpenAI chat completions API.

This covers OpenAI itself and the self-hosted and hosted servers that copy its
wire format (vLLM, llama.cpp, Ollama, LiteLLM, TGI and others). It is raw HTTP
over httpx2, with no vendor SDK:

* ``complete`` posts to ``/chat/completions``. The system prompt becomes a
  ``system`` message. ``max_tokens`` is sent as ``max_completion_tokens``
  (current OpenAI models), or as ``max_tokens`` for older servers that only
  know that name.
* Structured output sends ``response_format`` with a JSON schema. The schema is
  marked ``strict`` only when strict mode can accept it (every object closed
  with ``additionalProperties: false`` and every property required); otherwise
  the server is asked for best-effort JSON and the caller validates it.
* ``effort`` becomes ``reasoning_effort`` only when the provider is built with
  ``reasoning_effort=True``, since servers and models without reasoning reject
  the field. evalkit's ``xhigh`` and ``max`` map to ``high``.
* A ``refusal`` in the message gives an empty answer with stop reason
  ``refusal``, like a refusal from Claude. A ``content_filter`` finish is the
  server's own filter, raised as a ``safety_block`` error.
* Usage reads ``prompt_tokens`` and ``completion_tokens``; cached prompt tokens
  (``prompt_tokens_details.cached_tokens``) are priced as cache reads.
  Reasoning tokens are already inside ``completion_tokens``.
* ``embed`` posts to ``/embeddings`` and returns vectors in input order.
* The chat completions API has no token counting endpoint, so
  ``count_tokens`` raises an ``unsupported`` error.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

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

DEFAULT_BASE_URL = "https://api.openai.com/v1"
REASONING_EFFORT = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}
# Error codes OpenAI-style servers send with a status that undersells them.
CODE_KINDS = {
    "context_length_exceeded": ErrorKind.INVALID_REQUEST,
    "insufficient_quota": ErrorKind.AUTH,
    "invalid_api_key": ErrorKind.AUTH,
    "model_not_found": ErrorKind.NOT_FOUND,
    "content_filter": ErrorKind.SAFETY_BLOCK,
    "content_policy_violation": ErrorKind.SAFETY_BLOCK,
}


class OpenAICompatClient(HttpJsonProvider):
    """The ``Provider`` for OpenAI-compatible chat completions servers."""

    label = "the OpenAI-compatible API"

    def __init__(
        self,
        prices: PriceTable,
        api_key: SecretStr | None = None,
        *,
        name: str = "openai",
        base_url: str = DEFAULT_BASE_URL,
        output_limit_field: Literal["max_completion_tokens", "max_tokens"] = (
            "max_completion_tokens"
        ),
        reasoning_effort: bool = False,
        timeout: float = 600.0,
        transport: httpx2.AsyncBaseTransport | None = None,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        headers = {}
        if api_key is not None:
            headers["authorization"] = f"Bearer {api_key.get_secret_value()}"
        super().__init__(
            name=name,
            prices=prices,
            base_url=base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
            max_response_bytes=max_response_bytes,
        )
        self.output_limit_field = output_limit_field
        self.reasoning_effort = reasoning_effort

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
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(as_messages(prompt))
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            self.output_limit_field: max_tokens,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = REASONING_EFFORT.get(effort, "medium")
        if output_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "output",
                    "schema": output_schema,
                    "strict": strict_compatible(output_schema),
                },
            }
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
        # ``fallbacks`` is a Claude API feature; chat completions servers have none.
        self.prices.require(model)
        body = self.request_body(
            model=model,
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            effort=effort,
            output_schema=output_schema,
        )
        payload = await self.post_json("/chat/completions", body)
        try:
            return self.completion_from_payload(payload, requested_model=model)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            raise self.malformed(f"{type(exc).__name__}: {exc}") from None

    def completion_from_payload(
        self, payload: dict[str, Any], *, requested_model: str
    ) -> Completion:
        choices = payload["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError("no choices in the response")
        choice = choices[0]
        message = choice["message"]
        finish = choice.get("finish_reason")
        usage = self._usage(payload.get("usage") or {}, requested_model)
        if finish == "content_filter":
            raise ProviderError(
                f"{self.label} blocked the answer with its content filter "
                f"(model {requested_model})",
                kind=ErrorKind.SAFETY_BLOCK,
                provider=self.name,
            )
        refusal = message.get("refusal")
        if refusal:
            return Completion(
                text="",
                stop_reason="refusal",
                usage=usage,
                model=str(payload.get("model") or requested_model),
            )
        content = message.get("content")
        if content is None:
            text = ""
        elif isinstance(content, str):
            text = content
        elif isinstance(content, list):
            # Some servers return content parts even for text-only answers.
            text = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        else:
            raise TypeError(f"message content is {type(content).__name__}")
        return Completion(
            text=text,
            stop_reason=finish if isinstance(finish, str) else None,
            usage=usage,
            model=str(payload.get("model") or requested_model),
        )

    def _usage(self, usage: dict[str, Any], model: str) -> Usage:
        prompt_tokens = non_negative_int(usage.get("prompt_tokens"))
        details = usage.get("prompt_tokens_details") or {}
        cached = min(non_negative_int(details.get("cached_tokens")), prompt_tokens)
        counts = TokenCounts(
            input_tokens=prompt_tokens - cached,
            output_tokens=non_negative_int(usage.get("completion_tokens")),
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
        raise ProviderError(
            "the chat completions API has no token counting endpoint",
            kind=ErrorKind.UNSUPPORTED,
            provider=self.name,
        )

    async def embed(self, *, model: str, texts: Sequence[str]) -> Embeddings:
        self.prices.require(model)
        if not texts:
            return Embeddings(vectors=[], usage=Usage(), model=model)
        payload = await self.post_json(
            "/embeddings", {"model": model, "input": list(texts), "encoding_format": "float"}
        )
        try:
            return self._embeddings(payload, model, len(texts))
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            raise self.malformed(f"{type(exc).__name__}: {exc}") from None

    def _embeddings(self, payload: dict[str, Any], model: str, count: int) -> Embeddings:
        vectors: list[list[float] | None] = [None] * count
        for position, item in enumerate(payload["data"]):
            index = item.get("index", position)
            if not isinstance(index, int) or not 0 <= index < count or vectors[index] is not None:
                raise ValueError(f"embedding index {index!r} is out of place")
            vectors[index] = [float(x) for x in item["embedding"]]
        if any(v is None for v in vectors):
            raise ValueError(f"{sum(v is None for v in vectors)} of {count} embeddings missing")
        tokens = non_negative_int((payload.get("usage") or {}).get("prompt_tokens"))
        counts = TokenCounts(input_tokens=tokens)
        return Embeddings(
            vectors=[v for v in vectors if v is not None],
            usage=Usage(tokens, 0, 0, 0, self.price(model, counts)),
            model=str(payload.get("model") or model),
        )

    def error_details(self, status: int, payload: Any) -> tuple[str, ErrorKind, float | None]:
        kind = kind_for_status(status)
        error = payload.get("error") if isinstance(payload, dict) else None
        if not isinstance(error, dict):
            # Some servers put the message at the top level, or send a bare string.
            if isinstance(error, str):
                return error, kind, None
            detail = payload.get("message") if isinstance(payload, dict) else None
            return (detail if isinstance(detail, str) else ""), kind, None
        message = error.get("message")
        code = error.get("code")
        if isinstance(code, str) and code in CODE_KINDS:
            kind = CODE_KINDS[code]
        text = message if isinstance(message, str) else ""
        if isinstance(code, str) and code:
            text = f"{text} ({code})" if text else code
        return text, kind, None


def strict_compatible(schema: Any) -> bool:
    """Whether OpenAI strict structured output accepts ``schema`` as it is.

    Strict mode requires every object to forbid additional properties and to
    list all of its properties as required.
    """
    if isinstance(schema, bool):
        return True
    if not isinstance(schema, dict):
        return False
    properties = schema.get("properties")
    if schema.get("type") == "object" or properties is not None:
        if schema.get("additionalProperties") is not False:
            return False
        properties = {} if properties is None else properties
        if not isinstance(properties, dict):
            return False
        if set(schema.get("required") or []) != set(properties):
            return False
    children: list[Any] = []
    for key in ("properties", "$defs", "definitions"):
        mapping = schema.get(key)
        if isinstance(mapping, dict):
            children.extend(mapping.values())
    for key in ("items", "not", "if", "then", "else", "additionalProperties"):
        if isinstance(schema.get(key), dict):
            children.append(schema[key])
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        if isinstance(schema.get(key), list):
            children.extend(schema[key])
    return all(strict_compatible(child) for child in children)
