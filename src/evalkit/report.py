"""Run summaries and run-to-run comparisons, from per-run aggregates and compact
score arrays (see ``aggregates.py``) rather than the result rows.

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

from evalkit import aggregates, runqueue
from evalkit.db.models import Dataset, Result, Run, as_utc, utcnow
from evalkit.errors import InvalidInputError, NotFoundError, PreconditionError
from evalkit.scorers.judge import identity_changes, judge_identity
from evalkit.stats import (
    DEFAULT_RESAMPLES,
    Comparison,
    compare_aligned,
    mean_interval,
    normal_ci,
)

# A run that is not completed is summarised exactly from its results (bootstrap
# intervals, about 30 ms per metric at this size) up to this many results, and
# from its aggregates alone (normal intervals) above it. A completed run's
# intervals are computed once, when it is compacted.
LIVE_EXACT_MAX = 1_000


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
    judges: dict[str, dict[str, Any]] = field(default_factory=dict)  # by scorer name
    matrix_id: str | None = None  # the matrix this run is a variant of
    variant: str | None = None  # its variant name there
    # The registry prompt the target named: ref, name, version, label, content_hash.
    prompt: dict[str, Any] | None = None
    # The prices it is costed at: {provider: {"version", "models"}} (None: older runs).
    prices: dict[str, Any] | None = None


@dataclass(frozen=True)
class JudgeChange:
    """A compared metric whose judge is not the same in both runs."""

    metric: str
    changes: list[str]


@dataclass(frozen=True)
class CompareReport:
    baseline: str
    candidate: str
    alpha: float
    min_effect: float
    comparisons: list[Comparison]
    regressions: list[str] = field(default_factory=list)
    # Only ever non-empty when the comparison was asked to allow judge changes.
    judge_changes: list[JudgeChange] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.regressions


def get_run(session: Session, run_id: str) -> Run:
    run = session.get(Run, run_id)
    if run is None:
        raise NotFoundError(f"run {run_id} not found")
    return run


def run_judges(run: Run) -> dict[str, dict[str, Any]]:
    """Each judge metric's identity: as recorded, or for runs from before judge
    identities were recorded (all Claude judges), derived from the scorer specs."""
    if run.judges is not None:
        return dict(run.judges)
    out = {}
    for name, spec in zip(aggregates.metric_names(run), run.scorers, strict=True):
        if spec.get("type") == "judge":
            out[name] = judge_identity(
                provider=str(spec.get("provider", "anthropic")),
                endpoint=None,
                model=str(spec.get("model", "claude-opus-5-5")),
                mode=str(spec.get("mode", "rubric")),
                effort=str(spec.get("effort", "medium")),
                rubric=str(spec.get("rubric", "")),
            )
    return out


def judge_changes(base: Run, cand: Run, metrics: list[str]) -> list[JudgeChange]:
    base_judges, cand_judges = run_judges(base), run_judges(cand)
    out = []
    for metric in metrics:
        changes = identity_changes(base_judges.get(metric), cand_judges.get(metric))
        if changes:
            out.append(JudgeChange(metric, changes))
    return out


def scorer_names(run: Run) -> list[tuple[str, bool]]:
    return [
        (name, bool(s.get("gate", True)))
        for name, s in zip(aggregates.metric_names(run), run.scorers, strict=True)
    ]


def metric_scores(session: Session, run: Run) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """Per metric, the score of every case that has one; and the unscorable counts.
    Reads every result: only for runs small enough to summarise exactly."""
    names = aggregates.metric_names(run)
    scores: dict[str, dict[str, float]] = {n: {} for n in names}
    unscorable = dict.fromkeys(names, 0)
    rows = session.execute(
        select(Result.case_id, Result.status, Result.scores).where(Result.run_id == run.id)
    )
    for case_id, status, case_scores in rows:
        for name in names:
            value = aggregates.score_of(status, case_scores, name)
            if value is None:
                unscorable[name] += 1
            else:
                scores[name][case_id] = value
    return scores, unscorable


def _live_metrics(
    session: Session, run: Run, totals: aggregates.Totals, resamples: int
) -> list[MetricSummary]:
    """Metrics of a run that is not completed. Small runs are summarised exactly
    from their results; larger ones from the aggregates alone."""
    exact = totals.completed <= LIVE_EXACT_MAX
    scores = metric_scores(session, run)[0] if exact else {}
    out = []
    for name, gate in scorer_names(run):
        agg = totals.metrics.get(name, aggregates.MetricTotals())
        if not agg.n:
            out.append(MetricSummary(name, None, None, None, 0, agg.unscorable, gate))
            continue
        if exact:
            by_case = scores[name]
            values = [by_case[k] for k in sorted(by_case)]
            ci = mean_interval(values, resamples=resamples, seed=run.seed)
        else:
            ci = normal_ci(agg.n, agg.total, agg.total_sq)
        out.append(MetricSummary(name, ci.mean, ci.low, ci.high, ci.n, agg.unscorable, gate))
    return out


def _completed_metrics(
    session: Session, run: Run, stored: dict[str, aggregates.MetricScores], resamples: int
) -> list[MetricSummary]:
    arrays = (
        aggregates.load_scores(session, run, list(stored)) if resamples != DEFAULT_RESAMPLES else {}
    )
    out = []
    for name, gate in scorer_names(run):
        row = stored[name]
        mean, low, high = row.mean, row.ci_low, row.ci_high
        if arrays and row.n:
            values = arrays[name]
            ci = mean_interval(values[~np.isnan(values)], resamples=resamples, seed=run.seed)
            mean, low, high = ci.mean, ci.low, ci.high
        out.append(MetricSummary(name, mean, low, high, row.n, row.unscorable, gate))
    return out


def _live_latency(
    session: Session, run: Run, totals: aggregates.Totals
) -> tuple[float | None, float | None]:
    """Latency percentiles of a run that is not completed, over all its fresh
    results: exact for a small run, from the aggregates' latency histogram (within
    ``aggregates.LATENCY_ACCURACY``) for a larger one."""
    if totals.completed > LIVE_EXACT_MAX:
        found = aggregates.histogram_percentiles(totals.latency)
        return (None, None) if found is None else (found[0], found[1])
    query = select(Result.latency_ms).where(
        Result.run_id == run.id, Result.status == "ok", Result.cached.is_(False)
    )
    return aggregates.percentiles(list(session.scalars(query)))


def summarize(session: Session, run_id: str, *, resamples: int = DEFAULT_RESAMPLES) -> RunSummary:
    """A run's summary from its aggregates. A completed run reads only its
    aggregates and compacted metrics, so the cost does not grow with its size."""
    run = get_run(session, run_id)
    dataset = session.get_one(Dataset, run.dataset_id)
    total = aggregates.cases_total(session, run)
    totals = aggregates.read(session, run)
    stored = aggregates.stored(session, run)
    if stored is not None:
        metrics = _completed_metrics(session, run, stored, resamples)
        p50, p95 = run.latency_p50_ms, run.latency_p95_ms
    else:
        metrics = _live_metrics(session, run, totals, resamples)
        p50, p95 = _live_latency(session, run, totals)
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
        completed=totals.completed,
        ok=totals.ok,
        errors=totals.errors,
        cached=totals.cached,
        metrics=metrics,
        input_tokens=totals.input_tokens,
        output_tokens=totals.output_tokens,
        cache_read_tokens=totals.cache_read_tokens,
        cache_write_tokens=totals.cache_write_tokens,
        cost_usd=totals.cost_usd,
        judge_input_tokens=totals.judge_input_tokens,
        judge_output_tokens=totals.judge_output_tokens,
        judge_cost_usd=totals.judge_cost_usd,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        created_at=as_utc(run.created_at),
        finished_at=as_utc(run.finished_at) if run.finished_at else None,
        priority=run.priority,
        max_workers=run.max_workers,
        workers=runqueue.live_workers(session, run.id, utcnow()),
        judges=run_judges(run),
        matrix_id=run.matrix_id,
        variant=str(run.variant["name"]) if run.variant else None,
        prompt=dict(run.prompt) if run.prompt else None,
        prices=dict(run.prices) if run.prices is not None else None,
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
    allow_judge_change: bool = False,
) -> CompareReport:
    """Paired comparison of two completed runs over the same dataset content.

    A metric whose judge differs between the runs (provider, endpoint, model,
    mode, effort or rubric) measures the judge as much as the system under test,
    so the comparison is refused unless ``allow_judge_change``; when allowed, the
    report lists what changed."""
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
    base_names = set(aggregates.metric_names(base))
    shared = [n for n, _ in scorer_names(cand) if n in base_names]
    if metrics:
        unknown = [m for m in metrics if m not in shared]
        if unknown:
            raise InvalidInputError(
                f"metric(s) {', '.join(unknown)} not in both runs; available: {', '.join(shared)}"
            )
        selected = metrics
    else:
        selected = [n for n, gate in scorer_names(cand) if gate and n in base_names]
    if not selected:
        raise PreconditionError("the runs have no gated metric in common")
    changed = judge_changes(base, cand, selected)
    if changed and not allow_judge_change:
        detail = "; ".join(f"{c.metric}: {', '.join(c.changes)}" for c in changed)
        raise PreconditionError(
            f"the judge changed between the runs ({detail}), so a score change may come "
            "from the judge; rerun the baseline with the new judge, or pass "
            "--allow-judge-change (API: allow_judge_change) to compare anyway"
        )
    base_scores = aggregates.load_scores(session, base, selected)
    cand_scores = aggregates.load_scores(session, cand, selected)
    usable = {
        name: (base_scores[name], cand_scores[name])
        for name in selected
        if np.any(~np.isnan(base_scores[name]) & ~np.isnan(cand_scores[name]))
    }
    missing = [n for n in selected if n not in usable]
    if missing:
        raise PreconditionError(f"no case was scored in both runs for: {', '.join(missing)}")
    comparisons = compare_aligned(
        usable, alpha=alpha, min_effect=min_effect, resamples=resamples, seed=cand.seed
    )
    return CompareReport(
        baseline=base.id,
        candidate=cand.id,
        alpha=alpha,
        min_effect=min_effect,
        comparisons=comparisons,
        regressions=[c.metric for c in comparisons if c.verdict == "worse"],
        judge_changes=changed,
    )
