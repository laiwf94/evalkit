"""What a run will cost, before spending anything.

Input tokens are counted exactly with the token counting endpoint (free, but
rate limited), for the target prompt and for every judge call. Output cannot
be known in advance, so the estimate gives the worst case: every call uses its
full ``max_tokens``. Cases already in the response cache cost nothing. For a
judge, the response it will read is not known yet either; its input count
includes the prompt without the response, plus the target's ``max_tokens`` as
an upper bound on the response length when the target is a Claude prompt.
The target and each judge count on their own provider; one whose provider has
no counting endpoint gets a note instead of an input cost. A matrix config is
estimated variant by variant (``estimate_matrix``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterable
from dataclasses import dataclass, field

from sqlalchemy.orm import Session, sessionmaker

from evalkit import cache, matrix
from evalkit.config import Settings
from evalkit.datasets import Case, get_dataset, load_cases
from evalkit.errors import InvalidInputError
from evalkit.pricing import TokenCounts
from evalkit.prompts import resolve_config
from evalkit.providers.anthropic import ClaudeClient
from evalkit.providers.base import ErrorKind, Provider, ProviderError
from evalkit.providers.registry import ANTHROPIC, ProviderRegistry
from evalkit.ratelimit import Limiters, RetryStats, call_with_retries
from evalkit.runspec import RunConfig
from evalkit.scorers import build_scorers
from evalkit.scorers.judge import LLMJudge
from evalkit.targets import PromptTargetSpec
from evalkit.targets.factory import identity
from evalkit.targets.template import render


@dataclass
class Estimate:
    cases: int
    cached_cases: int
    batch: bool
    target_input_tokens: int = 0
    target_input_cost_usd: float = 0.0
    target_max_output_cost_usd: float = 0.0
    judge_calls: int = 0
    judge_input_tokens: int = 0
    judge_input_cost_usd: float = 0.0
    judge_max_output_cost_usd: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def max_total_cost_usd(self) -> float:
        return (
            self.target_input_cost_usd
            + self.target_max_output_cost_usd
            + self.judge_input_cost_usd
            + self.judge_max_output_cost_usd
        )


async def estimate(
    sessions: sessionmaker[Session],
    settings: Settings,
    config: RunConfig,
    claude: ClaudeClient,
    *,
    providers: ProviderRegistry | None = None,
    concurrency: int = 8,
) -> Estimate:
    """What running ``config`` would cost at most; a registry prompt is estimated
    at the version it names now."""
    with sessions() as session:
        config, _ = resolve_config(session, config)
    return await _estimate(
        sessions, settings, config, claude, providers=providers, concurrency=concurrency
    )


async def _estimate(
    sessions: sessionmaker[Session],
    settings: Settings,
    config: RunConfig,
    claude: ClaudeClient,
    *,
    providers: ProviderRegistry | None,
    concurrency: int,
) -> Estimate:
    if providers is None:
        providers = ProviderRegistry({}, claude.prices, claude=claude)
    scorers = build_scorers(config.scorers, allow_import_paths=True, providers=providers)
    judges = [s for s in scorers if isinstance(s, LLMJudge)]
    with sessions() as session:
        target_id = identity(config.target, settings.http_endpoints, providers.configs)
        dataset = get_dataset(session, config.dataset.name, config.dataset.version)
        cases = load_cases(session, dataset, config.filter.split, config.filter.tags)
        if not cases:
            raise InvalidInputError("the filter selects no cases from the dataset")
        fingerprint = settings.redaction.fingerprint()
        keys = {c.id: cache.cache_key(target_id, c.input, fingerprint) for c in cases}
        hits = cache.existing(session, list(keys.values())) if config.options.use_cache else set()
    uncached = [c for c in cases if keys[c.id] not in hits]
    batch = config.options.mode == "batch"
    result = Estimate(cases=len(cases), cached_cases=len(cases) - len(uncached), batch=batch)

    limiters = Limiters({ANTHROPIC: config.options.rate_limit})
    gate = asyncio.Semaphore(concurrency)

    async def count(model: str, prompt: str, system: str, client: Provider = claude) -> int:
        async with gate:
            return await call_with_retries(
                lambda: client.count_tokens(model=model, prompt=prompt, system=system),
                limiter=limiters.get(client.name),
                policy=config.options.retry,
                stats=RetryStats(),
            )

    spec = config.target
    response_bound = 0
    if isinstance(spec, PromptTargetSpec):
        response_bound = spec.max_tokens
        target = providers.get(spec.provider)
        prices = target.prices
        prices.require(spec.model)
        prompts = [p for p in (_render(spec, c) for c in uncached) if p]
        skipped = len(uncached) - len(prompts)
        if skipped:
            result.notes.append(f"{skipped} case(s) do not fit the template and will fail")
        result.target_max_output_cost_usd = prices.cost(
            spec.model, TokenCounts(output_tokens=spec.max_tokens * len(prompts)), batch=batch
        )
        tokens = await _total(count(spec.model, p, spec.system, target) for p in prompts)
        if tokens is None:
            result.notes.append(
                f"the target's provider {spec.provider} cannot count tokens, "
                "so its input cost is not included"
            )
        else:
            result.target_input_tokens = tokens
            result.target_input_cost_usd = prices.cost(
                spec.model, TokenCounts(input_tokens=tokens), batch=batch
            )
    else:
        result.notes.append(f"the {spec.type} target's own cost is not estimated")

    for judge in judges:
        # Every case is judged, cached or not, because a rerun scores again.
        prompts = [p for c in cases for p in _judge_prompts(judge, c)]
        result.judge_calls += len(prompts)
        result.judge_max_output_cost_usd += judge.client.prices.cost(
            judge.model, TokenCounts(output_tokens=judge.max_tokens * len(prompts))
        )
        counted = await _total(count(judge.model, p, judge.system, judge.client) for p in prompts)
        if counted is None:
            result.notes.append(
                f"judge {judge.name}: {judge.provider} cannot count tokens, "
                "so its input cost is not included"
            )
            continue
        tokens = counted + response_bound * len(prompts)
        result.judge_input_tokens += tokens
        result.judge_input_cost_usd += judge.client.prices.cost(
            judge.model, TokenCounts(input_tokens=tokens)
        )
    return result


async def _total(counts: Iterable[Awaitable[int]]) -> int | None:
    """The sum of ``counts``, or None when the provider cannot count tokens."""
    try:
        return sum(await asyncio.gather(*counts))
    except ProviderError as exc:
        if exc.kind is not ErrorKind.UNSUPPORTED:
            raise
        return None


def _render(spec: PromptTargetSpec, case: Case) -> str:
    try:
        return render(spec.template, case.input)
    except InvalidInputError:
        return ""


def _judge_prompts(judge: LLMJudge, case: Case) -> list[str]:
    if judge.mode == "pairwise" and case.expected is None:
        return []
    return judge.prompts(case, "")


@dataclass
class VariantEstimate:
    name: str
    estimate: Estimate


@dataclass
class MatrixEstimate:
    """One estimate per variant of a matrix config, in variant order."""

    variants: list[VariantEstimate]

    @property
    def max_total_cost_usd(self) -> float:
        return sum(v.estimate.max_total_cost_usd for v in self.variants)


async def estimate_matrix(
    sessions: sessionmaker[Session],
    settings: Settings,
    config: RunConfig,
    claude: ClaudeClient,
    *,
    providers: ProviderRegistry | None = None,
) -> MatrixEstimate:
    out = []
    with sessions() as session:
        config, _ = resolve_config(session, config)
    for variant in matrix.expand(config):
        found = await _estimate(
            sessions, settings, variant.config, claude, providers=providers, concurrency=8
        )
        out.append(VariantEstimate(variant.name, found))
    return MatrixEstimate(out)
