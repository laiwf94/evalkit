"""The cost, quality and latency frontier of a model matrix.

Each completed child run of a matrix (``evalkit.matrix``) is a point: the mean
of one quality metric (higher is better), the target's cost per case and its
latency percentile (both lower is better). A variant is *dominated* when
another is at least as good on every dimension and strictly better on one;
the variants nobody dominates form the Pareto frontier, and every other
variant is a choice no one should make, since some variant is cheaper, faster
and no worse.

Cost is the target's own cost per case it actually called, cached cases left
out, so a variant whose cases came from the cache is not made to look free.
Judge cost is reported but is not a dimension: the same judge grades every
variant. A dimension some variant cannot be measured on (latency, when every
one of its cases came from the cache) is left out for all of them, with a note,
rather than compared against a guess.

Point estimates decide dominance; the paired comparisons say which of the
frontier's differences are real. The frontier is ordered from cheapest to
most expensive (fastest to slowest without cost) and each variant is compared
with the one before it, case by case, with the same paired bootstrap and
permutation test as ``evalkit compare``, and Holm's correction across the
steps. Each step answers the question a frontier poses: is what the next
variant costs buying a quality difference this dataset can detect?
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sqlalchemy.orm import Session

from evalkit import aggregates
from evalkit.db.models import Dataset, Run
from evalkit.errors import InvalidInputError, PreconditionError
from evalkit.matrix import child_runs, get_matrix
from evalkit.report import RunSummary, scorer_names, summarize
from evalkit.stats import DEFAULT_RESAMPLES, Comparison, compare_aligned

Latency = Literal["p50", "p95"]
QUALITY, COST, LATENCY = "quality", "cost", "latency"


@dataclass(frozen=True)
class Point:
    """One variant on the dimensions the frontier is drawn over. ``None`` means
    the dimension could not be measured for this variant."""

    key: str
    quality: float
    cost: float | None = None
    latency: float | None = None


def dominates(a: Point, b: Point, dimensions: Sequence[str]) -> bool:
    """Whether ``a`` is at least as good as ``b`` on every dimension and better on one."""
    better = False
    for dim in dimensions:
        if dim == QUALITY:
            diff = a.quality - b.quality  # higher is better
        else:
            x, y = getattr(a, dim), getattr(b, dim)
            if x is None or y is None:
                raise ValueError(f"{dim} is not known for every point")
            diff = y - x  # lower is better
        if diff < 0:
            return False
        better = better or diff > 0
    return better


def pareto(points: Sequence[Point], dimensions: Sequence[str]) -> dict[str, list[str]]:
    """For every point, the keys of the points that dominate it, in input order:
    empty for the points on the frontier. Equal points do not dominate each
    other, so ties are all kept."""
    if len({p.key for p in points}) != len(points):
        raise ValueError("point keys must be unique")
    return {
        p.key: [q.key for q in points if q is not p and dominates(q, p, dimensions)] for p in points
    }


@dataclass(frozen=True)
class VariantRow:
    run_id: str
    label: str
    name: str
    status: str
    provider: str
    model: str
    effort: str
    max_tokens: int
    quality: float | None
    ci_low: float | None
    ci_high: float | None
    n: int
    cases_total: int
    completed: int
    cached: int
    errors: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    judge_cost_usd: float
    # Target cost per case it called (cached cases left out); None if it called none.
    cost_per_case_usd: float | None
    latency_ms: float | None
    on_frontier: bool
    dominated_by: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FrontierStep:
    """A paired comparison of one frontier variant (the candidate) with the
    frontier variant just below it in cost (the baseline)."""

    baseline: str
    candidate: str
    baseline_run: str
    candidate_run: str
    comparison: Comparison


@dataclass(frozen=True)
class MatrixReport:
    matrix_id: str
    label: str
    dataset: str
    dataset_version: int
    metric: str
    latency: Latency
    alpha: float
    dimensions: list[str]
    variants: list[VariantRow]
    steps: list[FrontierStep]
    notes: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return all(v.status == "completed" for v in self.variants)

    @property
    def frontier(self) -> list[str]:
        return [v.name for v in self.variants if v.on_frontier]


def _row(
    summary: RunSummary,
    variant: dict[str, object],
    metric: str,
    latency: Latency,
    dominated_by: list[str] | None,
) -> VariantRow:
    m = next((x for x in summary.metrics if x.name == metric), None)
    fresh = summary.completed - summary.cached
    return VariantRow(
        run_id=summary.run_id,
        label=summary.label,
        name=str(variant["name"]),
        status=summary.status,
        provider=str(variant["provider"]),
        model=str(variant["model"]),
        effort=str(variant["effort"]),
        max_tokens=int(str(variant["max_tokens"])),
        quality=m.mean if m else None,
        ci_low=m.ci_low if m else None,
        ci_high=m.ci_high if m else None,
        n=m.n if m else 0,
        cases_total=summary.cases_total,
        completed=summary.completed,
        cached=summary.cached,
        errors=summary.errors,
        input_tokens=summary.input_tokens,
        output_tokens=summary.output_tokens,
        cost_usd=summary.cost_usd,
        judge_cost_usd=summary.judge_cost_usd,
        cost_per_case_usd=summary.cost_usd / fresh if fresh > 0 else None,
        latency_ms=summary.latency_p50_ms if latency == "p50" else summary.latency_p95_ms,
        on_frontier=dominated_by == [],
        dominated_by=dominated_by or [],
    )


def matrix_report(
    session: Session,
    matrix_id: str,
    *,
    metric: str | None = None,
    latency: Latency = "p50",
    alpha: float = 0.05,
    min_effect: float = 0.0,
    resamples: int = DEFAULT_RESAMPLES,
) -> MatrixReport:
    """Every variant of a matrix with its quality, cost and latency, the frontier
    over the completed ones, and paired comparisons along the frontier."""
    if latency not in ("p50", "p95"):
        raise InvalidInputError("latency must be p50 or p95")
    group = get_matrix(session, matrix_id)
    runs = child_runs(session, group.id)
    if not runs:
        raise PreconditionError(f"matrix {group.id} has no runs")
    gated = [name for name, gate in scorer_names(runs[0]) if gate]
    names = [name for name, _ in scorer_names(runs[0])]
    if metric is None:
        if not gated:
            raise InvalidInputError(f"no gated metric to rank by; give one of: {', '.join(names)}")
        metric = gated[0]
    elif metric not in names:
        raise InvalidInputError(f"unknown metric {metric!r}; available: {', '.join(names)}")
    dataset = session.get_one(Dataset, group.dataset_id)
    summaries = [summarize(session, r.id) for r in runs]
    variants = [dict(r.variant or {}) for r in runs]
    notes: list[str] = []

    provisional = [
        _row(s, v, metric, latency, None) for s, v in zip(summaries, variants, strict=True)
    ]
    eligible = [r for r in provisional if r.status == "completed" and r.quality is not None]
    left_out = [r.name for r in provisional if r not in eligible]
    if left_out:
        notes.append(
            f"not on the frontier because not completed or not scored: {', '.join(left_out)}"
        )
    dimensions = [QUALITY]
    for dim, attr, why in (
        (COST, "cost_per_case_usd", "called no case (all cached or none run)"),
        (LATENCY, "latency_ms", "has no fresh successful case to time"),
    ):
        unknown = [r.name for r in eligible if getattr(r, attr) is None]
        if unknown:
            notes.append(f"{dim} is not a frontier dimension: {', '.join(unknown)} {why}")
        else:
            dimensions.append(dim)
    points = [
        Point(r.name, float(r.quality or 0.0), r.cost_per_case_usd, r.latency_ms) for r in eligible
    ]
    dominated = pareto(points, dimensions)
    rows = [
        _row(s, v, metric, latency, dominated.get(str(v["name"])))
        for s, v in zip(summaries, variants, strict=True)
    ]
    on_frontier = [r for r in rows if r.on_frontier]
    steps = _steps(
        session,
        runs,
        frontier=on_frontier,
        dimensions=dimensions,
        metric=metric,
        alpha=alpha,
        min_effect=min_effect,
        resamples=resamples,
    )
    if len(on_frontier) == 1 and len(eligible) > 1:
        notes.append(f"{on_frontier[0].name} dominates every other variant")
    return MatrixReport(
        matrix_id=group.id,
        label=group.label,
        dataset=dataset.name,
        dataset_version=dataset.version,
        metric=metric,
        latency=latency,
        alpha=alpha,
        dimensions=dimensions,
        variants=rows,
        steps=steps,
        notes=notes,
    )


def _steps(
    session: Session,
    runs: Sequence[Run],
    *,
    frontier: list[VariantRow],
    dimensions: list[str],
    metric: str,
    alpha: float,
    min_effect: float,
    resamples: int,
) -> list[FrontierStep]:
    def order(r: VariantRow) -> tuple[float, float, float]:
        cost = (r.cost_per_case_usd or 0.0) if COST in dimensions else 0.0
        lat = (r.latency_ms or 0.0) if LATENCY in dimensions else 0.0
        return (cost, lat, float(r.quality or 0.0))

    ordered = sorted(frontier, key=order)
    if len(ordered) < 2:
        return []
    by_id = {r.id: r for r in runs}
    scores = {
        row.run_id: aggregates.load_scores(session, by_id[row.run_id], [metric])[metric]
        for row in ordered
    }
    pairs = list(itertools.pairwise(ordered))
    arrays = {}
    for base, cand in pairs:
        b, c = scores[base.run_id], scores[cand.run_id]
        if b.shape != c.shape:
            raise PreconditionError(f"{base.name} and {cand.name} did not score the same cases")
        if not np.any(~np.isnan(b) & ~np.isnan(c)):
            raise PreconditionError(f"no case was scored in both {base.name} and {cand.name}")
        arrays[f"{base.name} -> {cand.name}"] = (b, c)
    seed = by_id[ordered[0].run_id].seed
    comparisons = compare_aligned(
        arrays, alpha=alpha, min_effect=min_effect, resamples=resamples, seed=seed
    )
    return [
        FrontierStep(base.name, cand.name, base.run_id, cand.run_id, comparison)
        for (base, cand), comparison in zip(pairs, comparisons, strict=True)
    ]
