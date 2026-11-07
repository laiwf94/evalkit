"""What a run will cost, before spending anything.

Input tokens are counted exactly with the token counting endpoint (free, but
rate limited), for the target prompt and for every judge call. Output cannot
be known in advance, so the estimate gives the worst case: every call uses its
full ``max_tokens``. Cases already in the response cache cost nothing. For a
judge, the response it will read is not known yet either; its input count
includes the prompt without the response, plus the target's ``max_tokens`` as
an upper bound on the response length when the target is a Claude prompt.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from sqlalchemy.orm import Session, sessionmaker

from evalkit import cache
from evalkit.config import Settings
from evalkit.datasets import Case, get_dataset, load_cases
from evalkit.errors import InvalidInputError
from evalkit.pricing import TokenCounts
from evalkit.providers.anthropic import ClaudeClient
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
    concurrency: int = 8,
) -> Estimate:
    target_id = identity(config.target, settings.http_endpoints)
    scorers = build_scorers(config.scorers, allow_import_paths=True, claude=claude)
    judges = [s for s in scorers if isinstance(s, LLMJudge)]
    with sessions() as session:
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

    limiter = Limiters({"anthropic": config.options.rate_limit}).get("anthropic")
    gate = asyncio.Semaphore(concurrency)

    async def count(model: str, prompt: str, system: str) -> int:
        async with gate:
            return await call_with_retries(
                lambda: claude.count_tokens(model=model, prompt=prompt, system=system),
                limiter=limiter,
                policy=config.options.retry,
                stats=RetryStats(),
            )

    prices = claude.prices
    spec = config.target
    response_bound = 0
    if isinstance(spec, PromptTargetSpec):
        response_bound = spec.max_tokens
        prompts = [_render(spec, c) for c in uncached]
        counts = await asyncio.gather(*(count(spec.model, p, spec.system) for p in prompts if p))
        tokens = sum(counts)
        result.target_input_tokens = tokens
        result.target_input_cost_usd = prices.cost(
            spec.model, TokenCounts(input_tokens=tokens), batch=batch
        )
        result.target_max_output_cost_usd = prices.cost(
            spec.model, TokenCounts(output_tokens=spec.max_tokens * len(counts)), batch=batch
        )
        skipped = len(prompts) - len(counts)
        if skipped:
            result.notes.append(f"{skipped} case(s) do not fit the template and will fail")
    else:
        result.notes.append(f"the {spec.type} target's own cost is not estimated")

    for judge in judges:
        # Every case is judged, cached or not, because a rerun scores again.
        prompts = [p for c in cases for p in _judge_prompts(judge, c)]
        counts = await asyncio.gather(*(count(judge.model, p, judge.system) for p in prompts))
        tokens = sum(counts) + response_bound * len(counts)
        result.judge_calls += len(counts)
        result.judge_input_tokens += tokens
        result.judge_input_cost_usd += prices.cost(judge.model, TokenCounts(input_tokens=tokens))
        result.judge_max_output_cost_usd += prices.cost(
            judge.model, TokenCounts(output_tokens=judge.max_tokens * len(counts))
        )
    return result


def _render(spec: PromptTargetSpec, case: Case) -> str:
    try:
        return render(spec.template, case.input)
    except InvalidInputError:
        return ""


def _judge_prompts(judge: LLMJudge, case: Case) -> list[str]:
    if judge.mode == "pairwise" and case.expected is None:
        return []
    return judge.prompts(case, "")
