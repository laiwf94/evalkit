"""An in-process embedding provider with meaningful vectors.

``bag_of_words`` hashes each lower-cased word into one of ``dims`` signed
buckets, so texts that share words point the same way and unrelated texts are
close to orthogonal: enough for cosine thresholds to mean something in tests,
without a model. ``FakeEmbedder`` serves those vectors through the
``Provider`` protocol and records every request.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable, Sequence
from typing import Any

import regex

from evalkit.pricing import PriceTable, TokenCounts
from evalkit.providers.base import Completion, Embeddings, ErrorKind, Prompt, ProviderError
from evalkit.targets.base import Usage

WORD = regex.compile(r"\w+")
MODEL = "text-embedding-3-small"


def bag_of_words(text: str, dims: int = 64) -> list[float]:
    vector = [0.0] * dims
    for word in WORD.findall(text.lower()):
        digest = hashlib.sha256(word.encode()).digest()
        bucket = int.from_bytes(digest[:4], "big") % dims
        vector[bucket] += 1.0 if digest[4] % 2 else -1.0
    return vector


class FakeEmbedder:
    """``embed`` answers with ``vectorize`` per text; each request is recorded in
    ``batches``. ``failures`` are raised by the next requests, in order."""

    name = "openai"

    def __init__(
        self,
        prices: PriceTable,
        *,
        vectorize: Callable[[str], list[float]] = bag_of_words,
        delay: float = 0.0,
    ) -> None:
        self.prices = prices
        self.vectorize = vectorize
        self.delay = delay
        self.batches: list[list[str]] = []
        self.failures: list[BaseException] = []
        self.reply: Callable[[list[str]], Embeddings] | None = None

    async def embed(self, *, model: str, texts: Sequence[str]) -> Embeddings:
        self.prices.require(model)
        self.batches.append(list(texts))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.failures:
            raise self.failures.pop(0)
        if self.reply is not None:
            return self.reply(list(texts))
        tokens = sum(len(WORD.findall(t)) for t in texts)
        cost = self.price(model, TokenCounts(input_tokens=tokens))
        vectors = [self.vectorize(t) for t in texts]
        return Embeddings(vectors=vectors, usage=Usage(tokens, cost_usd=cost), model=model)

    @property
    def texts(self) -> list[str]:
        return [t for batch in self.batches for t in batch]

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
        raise NotImplementedError

    async def count_tokens(self, *, model: str, prompt: str, system: str = "") -> int:
        raise NotImplementedError

    def price(self, model: str, tokens: TokenCounts, *, batch: bool = False) -> float:
        return self.prices.cost(model, tokens, batch=batch)

    def classify_error(self, exc: BaseException) -> ProviderError:
        raise NotImplementedError

    async def aclose(self) -> None:
        return


def overloaded() -> ProviderError:
    return ProviderError("busy", kind=ErrorKind.OVERLOADED, provider="openai", retry_after=0)


def refused() -> ProviderError:
    return ProviderError("bad request", kind=ErrorKind.INVALID_REQUEST, provider="openai")
