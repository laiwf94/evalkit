"""What evalkit needs from a model provider.

Prompt targets, judges and cost estimates talk to a provider through the
``Provider`` protocol, so a new provider is one module that implements five
operations:

* ``complete``: one prompt, or a conversation of alternating user and
  assistant messages ending with the user, in; one ``Completion`` out, with
  usage and cost.
* ``count_tokens``: input tokens for a prompt, for ``evalkit estimate``.
* ``embed``: vectors for a list of texts, with usage and cost.
* ``price``: USD for a set of token counts on one of the provider's models.
* ``classify_error``: any exception raised while calling the provider, turned
  into a ``ProviderError`` with an ``ErrorKind``.

Every failure a provider raises is a ``ProviderError``. It is a ``TargetError``,
so the run's retry policy, shared cool-downs and per-case error records work
unchanged; the kind says what went wrong in terms that do not depend on the
provider's own status codes or error bodies, and decides whether a retry can
help.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from evalkit.datasets import Message, check_alternation
from evalkit.errors import InvalidInputError
from evalkit.pricing import PriceTable, TokenCounts
from evalkit.targets.base import TargetError, Usage

#: What ``complete`` answers: a single user prompt, or a whole conversation.
Prompt = str | Sequence[Message]


def as_messages(prompt: Prompt) -> list[dict[str, str]]:
    """``prompt`` as role and content pairs: a string is one user message, and a
    conversation must alternate from the user and end with the user. Raises
    InvalidInputError before anything is sent otherwise."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    messages = list(prompt)
    try:
        check_alternation(messages)
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from None
    if not messages or messages[-1].role != "user":
        raise InvalidInputError("a conversation sent to a model must end with a user message")
    return [{"role": m.role, "content": m.content} for m in messages]


class ErrorKind(StrEnum):
    RATE_LIMITED = "rate_limited"  # the provider asked us to slow down (429)
    OVERLOADED = "overloaded"  # the provider is out of capacity (529, 503)
    SERVER = "server"  # the provider failed (5xx)
    CONFLICT = "conflict"  # a transient conflict on the provider's side (409)
    TIMEOUT = "timeout"  # no answer in time (client timeout or 408)
    CONNECTION = "connection"  # the provider could not be reached
    AUTH = "auth"  # the key is missing, wrong or not allowed (401, 403)
    NOT_FOUND = "not_found"  # unknown model or endpoint (404)
    INVALID_REQUEST = "invalid_request"  # the provider rejected the request (400, 413, 422)
    SAFETY_BLOCK = "safety_block"  # the provider's own filter blocked the prompt or answer
    MALFORMED_RESPONSE = "malformed_response"  # an answer that cannot be parsed
    UNSUPPORTED = "unsupported"  # the provider does not offer the operation
    UNKNOWN = "unknown"  # anything else

    @property
    def retryable(self) -> bool:
        return self in RETRYABLE_KINDS


RETRYABLE_KINDS = frozenset(
    {
        ErrorKind.RATE_LIMITED,
        ErrorKind.OVERLOADED,
        ErrorKind.SERVER,
        ErrorKind.CONFLICT,
        ErrorKind.TIMEOUT,
        ErrorKind.CONNECTION,
    }
)


class ProviderError(TargetError):
    """A failed provider call. Retryable exactly when its kind is."""

    def __init__(
        self,
        message: str,
        *,
        kind: ErrorKind,
        provider: str,
        status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, retryable=kind.retryable, retry_after=retry_after)
        self.kind = kind
        self.provider = provider
        self.status = status


def kind_for_status(status: int) -> ErrorKind:
    """The error kind an HTTP status means, for providers that use plain HTTP semantics."""
    match status:
        case 429:
            return ErrorKind.RATE_LIMITED
        case 503 | 529:
            return ErrorKind.OVERLOADED
        case 409:
            return ErrorKind.CONFLICT
        case 408 | 504:
            return ErrorKind.TIMEOUT
        case 401 | 403:
            return ErrorKind.AUTH
        case 404:
            return ErrorKind.NOT_FOUND
        case 500 | 502:
            return ErrorKind.SERVER
        case 400 | 413 | 422:
            return ErrorKind.INVALID_REQUEST
        case _:
            # 501, 505 and other unusual statuses: retrying will not change the answer.
            return ErrorKind.UNKNOWN


# Each provider's stop reason for an answer that ended on its own, and for one cut
# off at the output limit: Claude, OpenAI-compatible servers, Gemini.
FINISHED_STOP_REASONS = frozenset({"end_turn", "stop", "STOP"})
TRUNCATED_STOP_REASONS = frozenset({"max_tokens", "length", "MAX_TOKENS"})


@dataclass(frozen=True)
class Completion:
    text: str
    stop_reason: str | None
    usage: Usage
    model: str
    refusal_category: str | None = None

    def json(self) -> Any:
        return json.loads(self.text)

    @property
    def finished(self) -> bool:
        """The model ended the answer itself (not cut off, refused or filtered)."""
        return self.stop_reason in FINISHED_STOP_REASONS

    @property
    def truncated(self) -> bool:
        """The answer was cut off at the output token limit."""
        return self.stop_reason in TRUNCATED_STOP_REASONS


@dataclass(frozen=True)
class Embeddings:
    """One vector per input text, in input order."""

    vectors: list[list[float]]
    usage: Usage
    model: str


@runtime_checkable
class Provider(Protocol):
    #: Short stable name ("anthropic"); rate limits and cool-downs are keyed by it.
    name: str
    prices: PriceTable

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
        """Answer one prompt, or the last user message of a conversation.
        ``effort`` and ``fallbacks`` are hints a provider maps to its own
        controls, or ignores when it has none. Raises
        ``ProviderError``; an unknown model raises ``InvalidInputError`` before
        any request is sent."""
        ...

    async def count_tokens(self, *, model: str, prompt: str, system: str = "") -> int:
        """Input tokens the prompt would use."""
        ...

    async def embed(self, *, model: str, texts: Sequence[str]) -> Embeddings:
        """Vectors for ``texts``; ``ErrorKind.UNSUPPORTED`` where the provider has none."""
        ...

    def price(self, model: str, tokens: TokenCounts, *, batch: bool = False) -> float:
        """USD for one request's tokens on ``model``."""
        ...

    def classify_error(self, exc: BaseException) -> ProviderError:
        """The ``ProviderError`` an exception from this provider's transport means."""
        ...

    async def aclose(self) -> None: ...
