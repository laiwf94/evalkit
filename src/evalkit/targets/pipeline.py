"""Retrieval then generation as one target.

A pipeline case is a retrieval case (``evalkit.targets.retrieval``): its input
is a query, its expected value may carry graded relevance labels and a
reference answer. The retrieval stage ranks documents for the query through
the retriever's own target, so it has the same protections, the same
retrieval cache and the same in-flight sharing. The top ``contexts`` documents
then fill the generation template as ``{{ contexts }}``, numbered from 1 with
their ids, and the model's answer is the case's output:

    [1] kb-2
    The Eiffel Tower was completed in 1889.

    [2] kb-7
    ...

Both stages come back in ``Generation.stages`` and are stored as
``result_stages`` rows: the ranking as JSON, then the answer, each with its
latency, whether it was cached and what it cost. The generation stage has its
own cache entries, keyed by the generator and the prompt it was sent, so a
pipeline whose retriever changed only pays for the cases whose documents did.

Scoring reads both stages (``scoring_view``): a scorer that grades a ranking
(``Scorer.reads_ranking``: the ranking metrics, and RAG judges with
``contexts_from = "output"``) scores the retrieval stage, and every other
scorer scores the answer with the documents the model was given as the
case's ``contexts``, so the grounding judges check the answer against exactly
what it was written from.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import httpx2

from evalkit.datasets import Case, canonical_json, sha256_hex
from evalkit.providers.base import Provider
from evalkit.ratelimit import RateLimiter
from evalkit.redaction import Redactor
from evalkit.targets.base import Generation, Stage, Target
from evalkit.targets.conversation import TurnStore, call_limited, turn_key
from evalkit.targets.http import HttpEndpoint, Resolver, system_resolver
from evalkit.targets.retrieval import Document, Ranking, RetrievalTarget, parse_ranking, query_of
from evalkit.targets.specs import PipelineTargetSpec
from evalkit.targets.template import render

RETRIEVAL = "retrieval"
GENERATION = "generation"
NO_DOCUMENTS = "(no documents were retrieved)"


def render_contexts(documents: Sequence[Document]) -> str:
    """The documents as the generation template's ``{{ contexts }}``."""
    if not documents:
        return NO_DOCUMENTS
    return "\n\n".join(
        f"[{n}] {d.id}\n{d.text if d.text else '(no text)'}"
        for n, d in enumerate(documents, start=1)
    )


class PipelineTarget(Target):
    # The run's outer limiter only paces cases; each stage takes its own
    # limiter (``attach``): the retriever's endpoint and the model's provider.
    provider = "pipeline"
    caches_cases = False

    def __init__(
        self,
        spec: PipelineTargetSpec,
        endpoint: HttpEndpoint,
        client: Provider,
        *,
        provider_endpoint: str | None = None,
        resolver: Resolver = system_resolver,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        # An unknown model would run with no cost accounting; refuse it up front.
        client.prices.require(spec.generation.model)
        self.spec = spec
        self.client = client
        self.retriever = RetrievalTarget(
            spec.retrieval, endpoint, resolver=resolver, transport=transport
        )
        self.generator_hash = sha256_hex(
            canonical_json(
                {
                    "generator": spec.generation.model_dump(mode="json"),
                    "extra": provider_endpoint,
                }
            )
        )
        self.store: TurnStore | None = None
        self.redactor = Redactor()
        self.fingerprint = self.redactor.config.fingerprint()
        self.limiter: RateLimiter | None = None
        self.max_retry_after = 120.0
        # Answers paid for by this target, for tests and logs.
        self.generations = 0

    def attach(
        self,
        *,
        store: TurnStore | None,
        redactor: Redactor,
        retrieval_limiter: RateLimiter | None = None,
        generation_limiter: RateLimiter | None = None,
        max_retry_after: float = 120.0,
    ) -> None:
        """What the run provides: the cache (None when the run does not use it)
        for both stages, its redaction and each stage's limiter, which only
        calls the cache cannot answer wait on."""
        self.retriever.attach(
            store=store,
            redactor=redactor,
            limiter=retrieval_limiter,
            max_retry_after=max_retry_after,
        )
        self.store = store
        self.redactor = redactor
        self.fingerprint = redactor.config.fingerprint()
        self.limiter = generation_limiter
        self.max_retry_after = max_retry_after

    async def generate(self, case: Case) -> Generation:
        query = query_of(case)
        start = time.perf_counter()
        ranking, retrieval_cached = await self.retriever.retrieve(query)
        retrieval = Stage(
            0,
            RETRIEVAL,
            ranking.to_output(),
            cached=retrieval_cached,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        documents = ranking.documents[: self.spec.context_count]
        prompt = render(
            self.spec.generation.template,
            {
                "query": query.query,
                "filters": query.filters,
                "contexts": render_contexts(documents),
            },
        )
        generation = await self._generate(prompt)
        return Generation(
            text=generation.output,
            stop_reason=generation.stop_reason,
            usage=generation.usage,
            stages=(retrieval, generation),
            cached=retrieval.cached and generation.cached,
        )

    async def _generate(self, prompt: str) -> Stage:
        key = turn_key(GENERATION, self.generator_hash, prompt, self.fingerprint)
        hit = await self.store.get(key) if self.store is not None else None
        if hit is not None:
            return Stage(1, GENERATION, hit.output, hit.stop_reason, cached=True)
        spec = self.spec.generation
        start = time.perf_counter()
        self.generations += 1
        completion = await call_limited(
            self.limiter,
            self.max_retry_after,
            lambda: self.client.complete(
                model=spec.model,
                prompt=prompt,
                system=spec.system,
                max_tokens=spec.max_tokens,
                effort=spec.effort,
                fallbacks=spec.fallbacks,
            ),
        )
        latency = (time.perf_counter() - start) * 1000
        redacted = self.redactor.redact(completion.text)
        # A redaction that timed out left a placeholder: keep it out of the cache.
        if self.store is not None and not redacted.failed:
            await self.store.put(
                key, self.generator_hash, redacted.text, completion.stop_reason, completion.usage
            )
        return Stage(
            1,
            GENERATION,
            redacted.text,
            completion.stop_reason,
            latency_ms=latency,
            usage=completion.usage,
        )

    def scoring_view(self, case: Case, stages: Sequence[Stage]) -> tuple[Case, str | None]:
        """What the scorers see: the case with the documents the model was given
        as its ``contexts``, and the ranking for scorers that grade retrieval
        (None when the case has no retrieval stage)."""
        stored = next((s for s in stages if s.name == RETRIEVAL), None)
        if stored is None:
            return case, None
        ranking = parse_ranking(stored.output)
        return with_contexts(case, ranking, self.spec.context_count), stored.output

    async def aclose(self) -> None:
        await self.retriever.aclose()


def with_contexts(case: Case, ranking: Ranking, count: int) -> Case:
    """``case`` with its query and the top ``count`` documents as ``contexts``,
    which is where the grounding judges look for sources."""
    query = query_of(case)
    value: dict[str, Any] = {"query": query.query}
    if query.filters:
        value["filters"] = query.filters
    value["contexts"] = [{"id": d.id, "text": d.text or ""} for d in ranking.documents[:count]]
    return case.model_copy(update={"input": value})
