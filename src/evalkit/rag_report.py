"""``evalkit report rag``: why each failing case of a RAG run failed.

A RAG answer goes wrong in one of three places, and each calls for a different
fix: the retriever did not find what the answer needs (a *retrieval miss*),
the generator stated what its sources do not say (an *unsupported answer*), or
the sources had it and the answer left it out or got it wrong anyway (*ignored
context*). The report reads one run's per-case scores, sorts its metrics by the
stage they measure, and attributes every failing case to the earliest stage
that failed.

Metric roles, from each scorer's type:

* ``retrieval``: whether the needed sources came back. ``ranking`` metrics
  other than precision (recall, hit rate, MRR, MAP, nDCG) and
  ``context_recall``.
* ``precision``: how much of what came back is noise. ``ranking`` precision
  and ``context_relevance``. Noise alone is not a failure the answer shows, so
  these never decide a cause unless they are the only metrics a case fails.
* ``grounding``: whether the answer keeps to its sources. ``faithfulness``,
  ``citation_accuracy`` and ``noise_sensitivity``.
* ``answer``: the answer itself, ``answer_completeness`` and every other
  output scorer (``exact``, ``token_f1``, ``judge``, plugins).
* ``other``: trajectory scorers, which say nothing about RAG; ignored.

A case *fails* when it scores below the threshold (``--threshold``, 0.5 by
default, or per metric with ``NAME=VALUE``) on any gated metric, or its target
call failed. Ungated metrics are evidence only: a run can gate on ``exact``
and keep ``faithfulness`` and ``context_recall`` ungated to explain its
failures. Each failing case gets the first cause that applies:

1. ``error``: the target call failed, so nothing was scored.
2. ``retrieval_miss``: a retrieval metric is below its threshold.
3. ``unsupported_answer``: a grounding metric is below its threshold.
4. ``noisy_retrieval``: the only failing metrics are precision metrics.
5. ``ignored_context``: an answer metric failed while a retrieval metric
   passed, and either a grounding metric passed or the failing metric is
   ``answer_completeness`` (which measures leaving things out directly).
6. ``unattributed``: the run does not measure enough to tell, and the case
   says what is missing (no retrieval metric scored, or no grounding one).

The report streams the run's results once, keeping counts and the first
``limit`` failing cases by id, so it costs the same memory at any run size.
"""

from __future__ import annotations

import bisect
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from evalkit import aggregates
from evalkit.db.models import Dataset, Result, Run
from evalkit.errors import InvalidInputError, NotFoundError

Role = Literal["retrieval", "precision", "grounding", "answer", "other"]
Cause = Literal[
    "error",
    "retrieval_miss",
    "unsupported_answer",
    "noisy_retrieval",
    "ignored_context",
    "unattributed",
]
CAUSES: tuple[Cause, ...] = (
    "retrieval_miss",
    "unsupported_answer",
    "ignored_context",
    "noisy_retrieval",
    "unattributed",
    "error",
)
DEFAULT_THRESHOLD = 0.5
DEFAULT_LIMIT = 20
MAX_LIMIT = 10_000
DETAIL_CHARS = 300

_GROUNDING = {"faithfulness", "citation_accuracy", "noise_sensitivity"}
_OTHER = {"tool_calls", "step_efficiency", "forbidden_tools", "trajectory_judge"}


def role_of(spec: Mapping[str, Any]) -> Role:
    """The stage a scorer measures, from its stored spec."""
    kind = spec.get("type")
    if kind == "ranking":
        return "precision" if spec.get("metric") == "precision" else "retrieval"
    if kind == "context_recall":
        return "retrieval"
    if kind == "context_relevance":
        return "precision"
    if kind in _GROUNDING:
        return "grounding"
    if kind in _OTHER:
        return "other"
    return "answer"


@dataclass(frozen=True)
class MetricRole:
    name: str
    type: str
    role: Role
    gate: bool
    threshold: float
    mean: float | None = None
    n: int = 0
    unscorable: int = 0
    # Cases scoring below the threshold.
    below: int = 0


@dataclass(frozen=True)
class FailingCase:
    case_id: str
    cause: Cause
    # Metrics below their threshold, with their scores.
    failing: dict[str, float]
    # Every metric's score on the case (None: unscorable).
    scores: dict[str, float | None]
    reason: str
    # The detail of the metric that decided the cause, clipped.
    detail: str = ""


@dataclass
class RagReport:
    run_id: str
    label: str
    status: str
    dataset: str
    dataset_version: int
    cases: int
    failing: int
    causes: dict[str, int]
    metrics: list[MetricRole]
    shown: list[FailingCase]
    limit: int
    notes: list[str] = field(default_factory=list)


def parse_thresholds(raw: Sequence[str]) -> tuple[float, dict[str, float]]:
    """``--threshold`` values: a number sets the default, ``NAME=VALUE`` one
    metric's. Raises InvalidInputError."""
    default = DEFAULT_THRESHOLD
    per_metric: dict[str, float] = {}
    for item in raw:
        name, sep, value = item.rpartition("=")
        try:
            number = float(value)
        except ValueError:
            raise InvalidInputError(f"threshold {item!r} is not a number or NAME=NUMBER") from None
        if not math.isfinite(number) or not 0 <= number <= 1:
            raise InvalidInputError(f"threshold {item!r} must be between 0 and 1")
        if sep:
            if not name:
                raise InvalidInputError(f"threshold {item!r} names no metric")
            per_metric[name] = number
        else:
            default = number
    return default, per_metric


def _clip(text: str, limit: int = DETAIL_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def attribute(
    status: str,
    scores: Mapping[str, float | None],
    roles: Sequence[MetricRole],
) -> tuple[Cause, list[str], str] | None:
    """Why a case failed: its cause, the metrics that decided it and the reason;
    None when it did not fail."""
    if status != "ok":
        return "error", [], "the target call failed"
    below = {m.name for m in roles if (v := scores.get(m.name)) is not None and v < m.threshold}
    gated_below = [m for m in roles if m.gate and m.role != "other" and m.name in below]
    if not gated_below:
        return None

    def named(role: Role, failed: bool) -> list[MetricRole]:
        return [
            m
            for m in roles
            if m.role == role and scores.get(m.name) is not None and (m.name in below) == failed
        ]

    if missed := named("retrieval", True):
        return "retrieval_miss", [m.name for m in missed], "the needed sources were not retrieved"
    if unsupported := named("grounding", True):
        return (
            "unsupported_answer",
            [m.name for m in unsupported],
            "the answer states what its sources do not",
        )
    failed_answer = [m for m in gated_below if m.role == "answer"]
    if not failed_answer:
        return (
            "noisy_retrieval",
            [m.name for m in gated_below],
            "the retrieved sources are mostly noise",
        )
    names = [m.name for m in failed_answer]
    if not named("retrieval", False):
        return "unattributed", names, "no retrieval metric scored the case"
    omits = any(m.type == "answer_completeness" for m in failed_answer)
    if omits or named("grounding", False):
        return "ignored_context", names, "the sources had it and the answer did not use it"
    return "unattributed", names, "no grounding metric scored the case"


def _roles(run: Run, threshold: float, thresholds: Mapping[str, float]) -> list[MetricRole]:
    names = aggregates.metric_names(run)
    unknown = sorted(set(thresholds) - set(names))
    if unknown:
        raise InvalidInputError(
            f"no metric named {', '.join(unknown)} in run {run.id}; it has {', '.join(names)}"
        )
    return [
        MetricRole(
            name,
            str(spec.get("type")),
            role_of(spec),
            bool(spec.get("gate", True)),
            thresholds.get(name, threshold),
        )
        for name, spec in zip(names, run.scorers, strict=True)
    ]


def _failing_case(
    case_id: str,
    *,
    found: tuple[Cause, list[str], str],
    scores: dict[str, float | None],
    roles: Sequence[MetricRole],
    stored: Mapping[str, Any],
    error: str,
) -> FailingCase:
    cause, deciders, reason = found
    if cause == "error":
        detail = error
    else:
        detail = "; ".join(f"{n}: {(stored.get(n) or {}).get('detail', '')}" for n in deciders)
    below = {m.name: v for m in roles if (v := scores[m.name]) is not None and v < m.threshold}
    return FailingCase(case_id, cause, below, scores, reason, _clip(detail))


def _notes(roles: Sequence[MetricRole]) -> list[str]:
    notes = []
    present = {m.role for m in roles}
    if "retrieval" not in present:
        notes.append(
            "no retrieval metric (ranking recall, hit_rate, mrr, map, ndcg or "
            "context_recall): retrieval misses cannot be told from ignored context"
        )
    if "grounding" not in present:
        notes.append(
            "no grounding metric (faithfulness, citation_accuracy or noise_sensitivity): "
            "unsupported answers cannot be told from ignored context"
        )
    if not any(m.gate and m.role != "other" for m in roles):
        notes.append("no gated RAG or answer metric: no case can fail")
    return notes


def rag_report(
    session: Session,
    run_id: str,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    thresholds: Mapping[str, float] | None = None,
    limit: int = DEFAULT_LIMIT,
) -> RagReport:
    """The run's metrics by role, its failing cases counted by cause, and the
    first ``limit`` of them by case id."""
    run = session.get(Run, run_id)
    if run is None:
        raise NotFoundError(f"run {run_id} not found")
    if not 0 <= limit <= MAX_LIMIT:
        raise InvalidInputError(f"limit must be 0 to {MAX_LIMIT}")
    roles = _roles(run, threshold, thresholds or {})
    sums = dict.fromkeys((m.name for m in roles), 0.0)
    counts = Counter[str]()
    unscorable = Counter[str]()
    below = Counter[str]()
    causes = Counter[str]()
    total = failing = 0
    shown: list[FailingCase] = []
    rows = session.execute(
        select(Result.case_id, Result.status, Result.scores, Result.error)
        .where(Result.run_id == run.id)
        .execution_options(yield_per=5000)
    )
    for case_id, status, raw_scores, error in rows:
        total += 1
        stored = raw_scores or {}
        scores = {m.name: aggregates.score_of(status, stored, m.name) for m in roles}
        for m in roles:
            value = scores[m.name]
            if value is None:
                unscorable[m.name] += 1
                continue
            counts[m.name] += 1
            sums[m.name] += value
            if value < m.threshold:
                below[m.name] += 1
        found = attribute(status, scores, roles)
        if found is None:
            continue
        failing += 1
        causes[found[0]] += 1
        # Keep the first ``limit`` failing cases by id, whatever order rows come in.
        if len(shown) < limit or (limit and case_id < shown[-1].case_id):
            entry = _failing_case(
                case_id, found=found, scores=scores, roles=roles, stored=stored, error=error or ""
            )
            bisect.insort(shown, entry, key=lambda c: c.case_id)
            del shown[limit:]
    metrics = [
        MetricRole(
            m.name,
            m.type,
            m.role,
            m.gate,
            m.threshold,
            sums[m.name] / counts[m.name] if counts[m.name] else None,
            counts[m.name],
            unscorable[m.name],
            below[m.name],
        )
        for m in roles
    ]
    dataset = session.get_one(Dataset, run.dataset_id)
    return RagReport(
        run_id=run.id,
        label=run.label,
        status=run.status,
        dataset=dataset.name,
        dataset_version=dataset.version,
        cases=total,
        failing=failing,
        causes={c: causes[c] for c in CAUSES},
        metrics=metrics,
        shown=shown,
        limit=limit,
        notes=_notes(roles),
    )


CAUSE_TEXT: dict[str, str] = {
    "retrieval_miss": "retrieval miss",
    "unsupported_answer": "unsupported answer",
    "ignored_context": "ignored context",
    "noisy_retrieval": "noisy retrieval",
    "unattributed": "unattributed",
    "error": "target error",
}


def format_rag(r: RagReport) -> str:
    def pct(value: float | None) -> str:
        return "-" if value is None else f"{value * 100:.1f}%"

    width = max([len("metric"), *(len(m.name) for m in r.metrics)]) + 2
    lines = [
        f"rag report  run {r.run_id}  {r.label or '(no label)'}  "
        f"dataset {r.dataset} v{r.dataset_version}  {r.status}",
        f"{r.cases} cases, {r.failing} failing",
        "",
        f"{'metric':<{width}}{'role':<11}{'gate':<6}{'mean':>8}{'below':>8}"
        f"{'threshold':>11}{'unscorable':>12}",
    ]
    for m in r.metrics:
        lines.append(
            f"{m.name:<{width}}{m.role:<11}{'yes' if m.gate else 'no':<6}{pct(m.mean):>8}"
            f"{m.below:>8}{m.threshold:>11.2f}{m.unscorable:>12}"
        )
    lines += ["", f"{'cause':<20}{'cases':>6}{'share':>9}"]
    for cause, n in r.causes.items():
        share = n / r.failing if r.failing else 0.0
        lines.append(f"{CAUSE_TEXT[cause]:<20}{n:>6}{share * 100:>8.1f}%")
    if r.shown:
        more = r.failing - len(r.shown)
        suffix = f" (first {len(r.shown)} by id; --limit for more)" if more > 0 else ""
        lines += ["", f"failing cases{suffix}:"]
        for c in r.shown:
            failed = ", ".join(f"{n} {v:.2f}" for n, v in c.failing.items()) or "-"
            lines.append(f"  {c.case_id}  {CAUSE_TEXT[c.cause]}: {c.reason}  [{failed}]")
            if c.detail:
                lines.append(f"      {c.detail}")
    lines += [f"note: {n}" for n in r.notes]
    return "\n".join(lines) + "\n"
