"""Run summaries and run-to-run comparisons, computed from stored results.

Scoring policy: a case whose target call failed scores 0 on every metric (the
system under test did not produce an answer, which is a failure the gate must
see). A case the scorer could not judge (for example, no expected value) is
left out of that metric and counted as ``unscorable``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from evalkit import runqueue
from evalkit.datasets import load_cases
from evalkit.db.models import Dataset, Result, Run, as_utc, utcnow
from evalkit.errors import InvalidInputError, NotFoundError, PreconditionError
from evalkit.stats import DEFAULT_RESAMPLES, Comparison, bootstrap_ci, compare_many


@dataclass(frozen=True)
class MetricSummary:
    name: str
    mean: float | None
    ci_low: float | None
    ci_high: float | None
    n: int
    unscorable: int
    gate: bool


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    label: str
    status: str
    error: str
    dataset: str
    dataset_version: int
    dataset_hash: str
    target_type: str
    target_hash: str
    cases_total: int
    completed: int
    ok: int
    errors: int
    cached: int
    metrics: list[MetricSummary]
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float
    judge_input_tokens: int
    judge_output_tokens: int
    judge_cost_usd: float
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    created_at: datetime
    finished_at: datetime | None
    priority: int = 0
    max_workers: int | None = None
    workers: int = 0  # worker daemons holding a slot on the run now


@dataclass(frozen=True)
class CompareReport:
    baseline: str
    candidate: str
    alpha: float
    min_effect: float
    comparisons: list[Comparison]
    regressions: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.regressions


def get_run(session: Session, run_id: str) -> Run:
    run = session.get(Run, run_id)
    if run is None:
        raise NotFoundError(f"run {run_id} not found")
    return run


def scorer_names(run: Run) -> list[tuple[str, bool]]:
    return [(s.get("name") or s["type"], bool(s.get("gate", True))) for s in run.scorers]


def metric_scores(session: Session, run: Run) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """Per metric, the score of every case that has one; and the unscorable counts."""
    names = [n for n, _ in scorer_names(run)]
    scores: dict[str, dict[str, float]] = {n: {} for n in names}
    unscorable = dict.fromkeys(names, 0)
    rows = session.execute(
        select(Result.case_id, Result.status, Result.scores).where(Result.run_id == run.id)
    )
    for case_id, status, case_scores in rows:
        for name in names:
            if status != "ok":
                scores[name][case_id] = 0.0
                continue
            value = (case_scores.get(name) or {}).get("value")
            if value is None:
                unscorable[name] += 1
            else:
                scores[name][case_id] = float(value)
    return scores, unscorable


def summarize(session: Session, run_id: str, *, resamples: int = DEFAULT_RESAMPLES) -> RunSummary:
    run = get_run(session, run_id)
    dataset = session.get_one(Dataset, run.dataset_id)
    split = run.case_filter.get("split")
    tags = run.case_filter.get("tags") or ()
    total = (
        dataset.case_count
        if not (split or tags)
        else len(load_cases(session, dataset, split, tags))
    )
    rows = list(
        session.execute(
            select(
                Result.status,
                Result.cached,
                Result.latency_ms,
                Result.input_tokens,
                Result.output_tokens,
                Result.cache_read_tokens,
                Result.cache_write_tokens,
                Result.cost_usd,
                Result.judge_input_tokens,
                Result.judge_output_tokens,
                Result.judge_cost_usd,
            ).where(Result.run_id == run.id)
        )
    )
    scores, unscorable = metric_scores(session, run)
    metrics = []
    for name, gate in scorer_names(run):
        values = list(scores[name].values())
        if values:
            ci = bootstrap_ci(values, resamples=resamples, seed=run.seed)
            metrics.append(
                MetricSummary(name, ci.mean, ci.low, ci.high, ci.n, unscorable[name], gate)
            )
        else:
            metrics.append(MetricSummary(name, None, None, None, 0, unscorable[name], gate))
    fresh = [r.latency_ms for r in rows if r.status == "ok" and not r.cached]
    p50, p95 = np.percentile(fresh, [50, 95]).tolist() if fresh else (None, None)

    def total_of(index: int) -> Any:
        return sum(r[index] for r in rows)

    return RunSummary(
        run_id=run.id,
        label=run.label,
        status=run.status,
        error=run.error,
        dataset=dataset.name,
        dataset_version=dataset.version,
        dataset_hash=dataset.content_hash,
        target_type=str(run.target_spec.get("type")),
        target_hash=run.target_hash,
        cases_total=total,
        completed=len(rows),
        ok=sum(1 for r in rows if r.status == "ok"),
        errors=sum(1 for r in rows if r.status != "ok"),
        cached=sum(1 for r in rows if r.cached),
        metrics=metrics,
        input_tokens=total_of(3),
        output_tokens=total_of(4),
        cache_read_tokens=total_of(5),
        cache_write_tokens=total_of(6),
        cost_usd=float(total_of(7)),
        judge_input_tokens=total_of(8),
        judge_output_tokens=total_of(9),
        judge_cost_usd=float(total_of(10)),
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        created_at=as_utc(run.created_at),
        finished_at=as_utc(run.finished_at) if run.finished_at else None,
        priority=run.priority,
        max_workers=run.max_workers,
        workers=runqueue.live_workers(session, run.id, utcnow()),
    )


def compare_runs(
    session: Session,
    baseline_id: str,
    candidate_id: str,
    *,
    alpha: float = 0.05,
    min_effect: float = 0.0,
    metrics: list[str] | None = None,
    resamples: int = DEFAULT_RESAMPLES,
) -> CompareReport:
    """Paired comparison of two completed runs over the same dataset content."""
    base = get_run(session, baseline_id)
    cand = get_run(session, candidate_id)
    for run in (base, cand):
        if run.status != "completed":
            raise PreconditionError(f"run {run.id} is {run.status}, not completed")
    base_hash = session.get_one(Dataset, base.dataset_id).content_hash
    cand_hash = session.get_one(Dataset, cand.dataset_id).content_hash
    if base_hash != cand_hash:
        raise PreconditionError(
            "the runs used different dataset contents; results cannot be paired"
        )
    base_scores, _ = metric_scores(session, base)
    cand_scores, _ = metric_scores(session, cand)
    shared = [n for n, _ in scorer_names(cand) if n in base_scores]
    if metrics:
        unknown = [m for m in metrics if m not in shared]
        if unknown:
            raise InvalidInputError(
                f"metric(s) {', '.join(unknown)} not in both runs; available: {', '.join(shared)}"
            )
        selected = metrics
    else:
        selected = [n for n, gate in scorer_names(cand) if gate and n in base_scores]
    if not selected:
        raise PreconditionError("the runs have no gated metric in common")
    usable = {
        name: (base_scores[name], cand_scores[name])
        for name in selected
        if base_scores[name].keys() & cand_scores[name].keys()
    }
    missing = [n for n in selected if n not in usable]
    if missing:
        raise PreconditionError(f"no case was scored in both runs for: {', '.join(missing)}")
    comparisons = compare_many(
        usable, alpha=alpha, min_effect=min_effect, resamples=resamples, seed=cand.seed
    )
    return CompareReport(
        baseline=base.id,
        candidate=cand.id,
        alpha=alpha,
        min_effect=min_effect,
        comparisons=comparisons,
        regressions=[c.metric for c in comparisons if c.verdict == "worse"],
    )
