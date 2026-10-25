"""Per-model token prices and cost computation.

Prices are USD per million tokens, from the provider's published list. A
prompt target must name a model in this table: an unknown model would run with
no cost accounting, and a typo would otherwise only show up as a 404 halfway
through a run. Operators can override or extend the table with
``EVALKIT_PRICES_FILE`` when prices change.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from evalkit.errors import InvalidInputError

CACHE_WRITE_MULTIPLIER = 1.25  # five-minute cache writes
BATCH_DISCOUNT = 0.5


class ModelPrice(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_read: float | None = Field(default=None, ge=0)
    # Some models charge more once the prompt passes a size threshold.
    long_context_threshold: int | None = Field(default=None, gt=0)
    long_input: float | None = Field(default=None, ge=0)
    long_output: float | None = Field(default=None, ge=0)


DEFAULT_PRICES: dict[str, ModelPrice] = {
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
        input=0.10, output=0.50, long_context_threshold=100_000, long_input=0.50, long_output=2.50
    ),
    "claude-haiku-4-5": ModelPrice(input=1.0, output=5.0),
}


@dataclass(frozen=True)
class TokenCounts:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class PriceTable:
    def __init__(self, overrides: Mapping[str, ModelPrice] | None = None) -> None:
        self._prices = {**DEFAULT_PRICES, **(overrides or {})}

    def __contains__(self, model: str) -> bool:
        return model in self._prices

    def models(self) -> list[str]:
        return sorted(self._prices)

    def require(self, model: str) -> ModelPrice:
        price = self._prices.get(model)
        if price is None:
            raise InvalidInputError(
                f"unknown model {model!r}; known models: {', '.join(self.models())}"
            )
        return price

    def cost(self, model: str, tokens: TokenCounts, *, batch: bool = False) -> float:
        """USD for one request. Unknown models (e.g. a fallback we have no price
        for) cost 0 here; the caller records the model so the gap is visible."""
        price = self._prices.get(model)
        if price is None:
            return 0.0
        input_rate, output_rate = price.input, price.output
        prompt = tokens.input_tokens + tokens.cache_read_tokens + tokens.cache_write_tokens
        if price.long_context_threshold is not None and prompt > price.long_context_threshold:
            input_rate = price.long_input if price.long_input is not None else input_rate
            output_rate = price.long_output if price.long_output is not None else output_rate
        cache_read_rate = price.cache_read if price.cache_read is not None else input_rate * 0.1
        dollars = (
            tokens.input_tokens * input_rate
            + tokens.output_tokens * output_rate
            + tokens.cache_read_tokens * cache_read_rate
            + tokens.cache_write_tokens * input_rate * CACHE_WRITE_MULTIPLIER
        ) / 1_000_000
        return dollars * BATCH_DISCOUNT if batch else dollars
