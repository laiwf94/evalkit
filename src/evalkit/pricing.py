"""Versioned per-provider token prices, cost computation and token estimates.

Prices are USD per million tokens, from each provider's published list. Every
provider has its own ``PriceList``, named by the provider and versioned by the
date it was last checked against that list:

* ``anthropic``: the Claude models, used by the Claude client.
* ``openai``: OpenAI's models, used by the built-in ``openai`` provider.
* ``gemini``: Google's Gemini API models, used by the built-in ``gemini`` provider.

A server from ``EVALKIT_PROVIDERS_FILE`` starts from no list (a self-hosted
model has no published price) unless it names one with ``price_list``, for
example a proxy in front of OpenAI.

A prompt target or a judge must name a model its provider's table prices: an
unknown model would run with no cost accounting, and a typo would otherwise
only show up as a 404 halfway through a run. Operators override or extend the
tables with ``EVALKIT_PRICES_FILE``. A ``[prices.<model>]`` table applies to
every provider; a ``[prices.<provider>.<model>]`` table applies to that one
provider and wins over the first kind.

Every table has a version: the list's own (``anthropic@2026-11-09``) when no
override changes it, and the list's version plus a hash of the overrides
(``anthropic@2026-11-09+local.1a2b3c4d``) when one does. A run records the
prices of the models it uses with their version (``Runner.create_run``), and a
process whose prices for those models differ will not execute it, so every
cost in one run comes from one set of prices.

Providers without a token counting endpoint (OpenAI-compatible servers) get an
estimate from the text's length (``estimate_tokens``), which ``evalkit
estimate`` labels as an estimate.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evalkit.errors import InvalidInputError

CACHE_WRITE_MULTIPLIER = 1.25  # five-minute cache writes
BATCH_DISCOUNT = 0.5


class ModelPrice(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    # Cached prompt tokens; a tenth of ``input`` when not given.
    cache_read: float | None = Field(default=None, ge=0)
    # Some models charge more once the prompt passes a size threshold.
    long_context_threshold: int | None = Field(default=None, gt=0)
    long_input: float | None = Field(default=None, ge=0)
    long_output: float | None = Field(default=None, ge=0)
    long_cache_read: float | None = Field(default=None, ge=0)

    def record(self) -> dict[str, Any]:
        """The fields that are set, as stored with a run."""
        return self.model_dump(mode="json", exclude_none=True)


@dataclass(frozen=True)
class PriceList:
    """One provider's published prices as of ``version`` (the date they were checked)."""

    name: str
    version: str
    source: str
    prices: Mapping[str, ModelPrice] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "prices", MappingProxyType(dict(self.prices)))


ANTHROPIC_PRICES = PriceList(
    "anthropic",
    "2026-11-09",
    "https://www.anthropic.com/pricing",
    {
        "claude-fable-5-1": ModelPrice(input=10.0, output=50.0, cache_read=0.25),
        "claude-fable-5": ModelPrice(input=10.0, output=50.0),
        "claude-opus-5-5": ModelPrice(input=4.0, output=20.0, cache_read=0.20),
        "claude-opus-5": ModelPrice(input=5.0, output=25.0),
        "claude-opus-4-8": ModelPrice(input=5.0, output=25.0),
        "claude-opus-4-7": ModelPrice(input=5.0, output=25.0),
        "claude-opus-4-6": ModelPrice(input=5.0, output=25.0),
        "claude-sonnet-5-5": ModelPrice(input=2.0, output=10.0, cache_read=0.20),
        "claude-sonnet-5": ModelPrice(input=2.0, output=10.0),
        "claude-sonnet-4-6": ModelPrice(input=3.0, output=15.0),
        "claude-haiku-5-5": ModelPrice(
            input=0.10,
            output=0.50,
            long_context_threshold=100_000,
            long_input=0.50,
            long_output=2.50,
        ),
        "claude-haiku-4-5": ModelPrice(input=1.0, output=5.0),
    },
)

# Standard tier. Reasoning tokens are billed as output and come back inside
# ``completion_tokens``; cached prompt tokens are billed at ``cache_read``.
OPENAI_PRICES = PriceList(
    "openai",
    "2026-11-09",
    "https://platform.openai.com/docs/pricing",
    {
        "gpt-5.1": ModelPrice(input=1.25, output=10.0, cache_read=0.125),
        "gpt-5": ModelPrice(input=1.25, output=10.0, cache_read=0.125),
        "gpt-5-mini": ModelPrice(input=0.25, output=2.0, cache_read=0.025),
        "gpt-5-nano": ModelPrice(input=0.05, output=0.40, cache_read=0.005),
        "gpt-4.1": ModelPrice(input=2.0, output=8.0, cache_read=0.50),
        "gpt-4.1-mini": ModelPrice(input=0.40, output=1.60, cache_read=0.10),
        "gpt-4.1-nano": ModelPrice(input=0.10, output=0.40, cache_read=0.025),
        "gpt-4o": ModelPrice(input=2.50, output=10.0, cache_read=1.25),
        "gpt-4o-mini": ModelPrice(input=0.15, output=0.60, cache_read=0.075),
        "o3": ModelPrice(input=2.0, output=8.0, cache_read=0.50),
        "o4-mini": ModelPrice(input=1.10, output=4.40, cache_read=0.275),
        "text-embedding-3-small": ModelPrice(input=0.02, output=0.0),
        "text-embedding-3-large": ModelPrice(input=0.13, output=0.0),
    },
)

# Paid tier, text input. Thinking tokens are billed as output; prompts over
# 200,000 tokens are billed at the long-context rates.
GEMINI_PRICES = PriceList(
    "gemini",
    "2026-11-09",
    "https://ai.google.dev/gemini-api/docs/pricing",
    {
        "gemini-3-pro-preview": ModelPrice(
            input=2.0,
            output=12.0,
            cache_read=0.20,
            long_context_threshold=200_000,
            long_input=4.0,
            long_output=18.0,
            long_cache_read=0.40,
        ),
        "gemini-2.5-pro": ModelPrice(
            input=1.25,
            output=10.0,
            cache_read=0.125,
            long_context_threshold=200_000,
            long_input=2.50,
            long_output=15.0,
            long_cache_read=0.25,
        ),
        "gemini-2.5-flash": ModelPrice(input=0.30, output=2.50, cache_read=0.03),
        "gemini-2.5-flash-lite": ModelPrice(input=0.10, output=0.40, cache_read=0.01),
        "gemini-embedding-001": ModelPrice(input=0.15, output=0.0),
    },
)

# Where a self-hosted server starts: nothing is priced until the operator says so.
NO_PRICES = PriceList("none", "none", "")

PRICE_LISTS: Mapping[str, PriceList] = MappingProxyType(
    {p.name: p for p in (ANTHROPIC_PRICES, OPENAI_PRICES, GEMINI_PRICES)}
)

# Kept for callers that only want the Claude prices.
DEFAULT_PRICES: Mapping[str, ModelPrice] = ANTHROPIC_PRICES.prices


@dataclass(frozen=True)
class TokenCounts:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class PriceTable:
    """The prices one provider uses: a ``PriceList`` with the operator's overrides."""

    def __init__(
        self,
        overrides: Mapping[str, ModelPrice] | None = None,
        *,
        base: PriceList = ANTHROPIC_PRICES,
    ) -> None:
        self.base = base
        changed = {m: p for m, p in (overrides or {}).items() if base.prices.get(m) != p}
        self._prices = {**base.prices, **changed}
        self.version = f"{base.name}@{base.version}"
        if changed:
            canonical = json.dumps(
                {m: p.record() for m, p in sorted(changed.items())}, sort_keys=True
            )
            digest = hashlib.sha256(canonical.encode()).hexdigest()[:8]
            self.version += f"+local.{digest}"

    def __contains__(self, model: str) -> bool:
        return model in self._prices

    def models(self) -> list[str]:
        return sorted(self._prices)

    def get(self, model: str) -> ModelPrice | None:
        return self._prices.get(model)

    def require(self, model: str) -> ModelPrice:
        price = self._prices.get(model)
        if price is None:
            known = ", ".join(self.models()) or "none"
            raise InvalidInputError(
                f"unknown model {model!r} in price table {self.version}; known models: {known}"
                "; add it to EVALKIT_PRICES_FILE"
            )
        return price

    def cost(self, model: str, tokens: TokenCounts, *, batch: bool = False) -> float:
        """USD for one request. Unknown models (e.g. a fallback we have no price
        for) cost 0 here; the caller records the model so the gap is visible."""
        price = self._prices.get(model)
        if price is None:
            return 0.0
        return price_cost(price, tokens, batch=batch)


def price_cost(price: ModelPrice, tokens: TokenCounts, *, batch: bool = False) -> float:
    input_rate, output_rate, cache_rate = price.input, price.output, price.cache_read
    prompt = tokens.input_tokens + tokens.cache_read_tokens + tokens.cache_write_tokens
    if price.long_context_threshold is not None and prompt > price.long_context_threshold:
        input_rate = price.long_input if price.long_input is not None else input_rate
        output_rate = price.long_output if price.long_output is not None else output_rate
        cache_rate = price.long_cache_read if price.long_cache_read is not None else cache_rate
    cache_read_rate = cache_rate if cache_rate is not None else input_rate * 0.1
    dollars = (
        tokens.input_tokens * input_rate
        + tokens.output_tokens * output_rate
        + tokens.cache_read_tokens * cache_read_rate
        + tokens.cache_write_tokens * input_rate * CACHE_WRITE_MULTIPLIER
    ) / 1_000_000
    return dollars * BATCH_DISCOUNT if batch else dollars


Overrides = Mapping[str, "ModelPrice | Mapping[str, ModelPrice]"]


class PriceBook:
    """Every provider's table, built from the built-in lists and ``EVALKIT_PRICES_FILE``.

    ``overrides`` maps a model to its price (every provider) or a provider name
    to its own ``{model: price}`` overrides.
    """

    def __init__(self, overrides: Overrides | None = None) -> None:
        self._shared: dict[str, ModelPrice] = {}
        self._scoped: dict[str, dict[str, ModelPrice]] = {}
        for key, value in (overrides or {}).items():
            if isinstance(value, ModelPrice):
                self._shared[key] = value
            else:
                self._scoped[key] = dict(value)
        self._tables: dict[tuple[str, str | None], PriceTable] = {}

    def table(self, provider: str, price_list: str | None = None) -> PriceTable:
        """The table for ``provider``, starting from the list named ``price_list``
        (default: the provider's own built-in list, or none)."""
        key = (provider, price_list)
        table = self._tables.get(key)
        if table is None:
            name = provider if price_list is None else price_list
            base = PRICE_LISTS.get(name)
            if base is None:
                if price_list is not None:
                    raise InvalidInputError(
                        f"unknown price list {price_list!r}; built in: {', '.join(PRICE_LISTS)}"
                    )
                base = NO_PRICES
            table = PriceTable({**self._shared, **self._scoped.get(provider, {})}, base=base)
            self._tables[key] = table
        return table


def price_record(uses: Iterable[tuple[str, PriceTable, str]]) -> dict[str, Any]:
    """What a run records about its prices, from ``(provider, table, model)`` for
    the target and each judge: ``{provider: {"version": ..., "models": {model:
    price}}}``. Models the table does not price are left out."""
    record: dict[str, Any] = {}
    for provider, table, model in uses:
        entry = record.setdefault(provider, {"version": table.version, "models": {}})
        price = table.get(model)
        if price is not None:
            entry["models"][model] = price.record()
    return record


def price_changes(recorded: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """How ``current`` prices a recorded run's models differently, one line per
    model both price. A new table version that leaves the run's models alone is
    not a change."""
    changes = []
    for provider, now in sorted(current.items()):
        then = recorded.get(provider)
        if then is None:
            continue
        for model, price in sorted(now["models"].items()):
            old = then["models"].get(model)
            if old is not None and old != price:
                changes.append(
                    f"{provider}/{model}: {_describe(old)} in {then['version']}, "
                    f"{_describe(price)} in {now['version']}"
                )
    return changes


def _describe(price: Mapping[str, Any]) -> str:
    return ", ".join(f"{k} {v:g}" for k, v in price.items())


# Bytes per token assumed when a provider cannot count: English prose runs
# about four bytes a token on current tokenizers, code and numbers fewer and
# most non-Latin scripts about one character (two or three bytes) a token, so
# three bytes leans high for the first and is close for the others.
ESTIMATE_BYTES_PER_TOKEN = 3
# The chat format's own tokens around each message (role markers and separators).
ESTIMATE_MESSAGE_OVERHEAD = 8


def estimate_tokens(prompt: str, system: str = "") -> int:
    """An upper-leaning estimate of the input tokens for ``prompt`` and ``system``,
    for providers with no counting endpoint. Never a bill: actual usage is what
    the provider reports."""
    texts = [system, prompt] if system else [prompt]
    return sum(
        math.ceil(len(text.encode("utf-8")) / ESTIMATE_BYTES_PER_TOKEN) + ESTIMATE_MESSAGE_OVERHEAD
        for text in texts
    )
