"""Per-run aggregates written with each result, and compact per-case score arrays.

A run summary used to read every result row. Now:

* The transaction that saves a batch of results also adds that batch's counts,
  token and cost sums, and each metric's count, sum, sum of squares and
  unscorable count to the run's aggregates. Each writing process adds to its own
  ``shard`` row (its owner id), so processes never queue on one hot row, and a
  run's totals are the sum over its few shard rows.
* The same transaction counts the batch's fresh successful results per
  logarithmic latency bucket (``run_latency_buckets``, also per shard), so a
  running run's latency percentiles cost a read of a few hundred bucket rows and
  are within ``LATENCY_ACCURACY`` (1%) of the exact ones over all its results.
* When a run completes, ``compact`` writes each metric's scores as one array
  (``run_scores``), in case-id order over the run's dataset, together with the
  summary interval and the exact latency percentiles. Comparing two runs reads
  one row per metric per run, and summarising a completed run reads no results.

``recompute`` derives the same numbers from the results, which is what
``evalkit run verify`` checks against, and ``rebuild`` replaces the aggregates
with that recomputation (the migration that added aggregates uses it for runs
that already had results, and ``verify --repair`` for a run that drifted).

Scoring policy (also in ``report.py``): a case whose target call failed scores
0 on every metric; a score of None is unscorable and left out of the metric.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
from sqlalchemy import Table, delete, func, select, update
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from evalkit.datasets import load_cases
from evalkit.db.models import (
    Dataset,
    DatasetCase,
    Result,
    Run,
    RunAggregate,
    RunLatencyBucket,
    RunMetricAggregate,
    RunScores,
)
from evalkit.errors import ConflictError, NotFoundError
from evalkit.stats import DEFAULT_RESAMPLES, FloatArray, mean_interval

# The shard holding aggregates recomputed from results (migration 0005, repair).
REBUILD_SHARD = "rebuild"
SCORE_DTYPE = np.dtype("<f8")
# Relative and absolute tolerance when checking stored float sums against a
# recomputation: the two add the same numbers in a different order.
FLOAT_TOLERANCE = 1e-9
_READ_CHUNK = 5000

# Latency histogram buckets. Bucket ``i`` holds latencies in (GAMMA**(i-1),
# GAMMA**i] milliseconds and stands for 2 * GAMMA**i / (GAMMA + 1), which is within
# LATENCY_ACCURACY of every latency in it (relative error, as in DDSketch).
# Latencies at or below the floor share the floor's bucket (an absolute error of
# about a microsecond), and anything above the ceiling the ceiling's. Migration 0007
# keeps a frozen copy of these numbers; changing them needs a new migration that
# rebuilds every histogram.
LATENCY_GAMMA = 1.02
LATENCY_ACCURACY = (LATENCY_GAMMA - 1) / (LATENCY_GAMMA + 1)
LATENCY_FLOOR_MS = 1e-3
LATENCY_CEIL_MS = 1e9
_LOG_GAMMA = math.log(LATENCY_GAMMA)


def latency_bucket(latency_ms: float) -> int:
    """The histogram bucket of one latency (NaN goes to the floor's bucket)."""
    value = min(latency_ms, LATENCY_CEIL_MS) if latency_ms > LATENCY_FLOOR_MS else LATENCY_FLOOR_MS
    return math.ceil(math.log(value) / _LOG_GAMMA)


def bucket_value(bucket: int) -> float:
    """The latency a bucket stands for."""
    return 2 * LATENCY_GAMMA**bucket / (LATENCY_GAMMA + 1)


@dataclass
class MetricTotals:
    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    unscorable: int = 0

    def add(self, other: MetricTotals) -> None:
        self.n += other.n
        self.total += other.total
        self.total_sq += other.total_sq
        self.unscorable += other.unscorable


@dataclass
class Totals:
    completed: int = 0
    ok: int = 0
    errors: int = 0
    cached: int = 0
    fresh: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    judge_cost_usd: float = 0.0
    metrics: dict[str, MetricTotals] = field(default_factory=dict)
    # Fresh successful results per latency bucket.
    latency: dict[int, int] = field(default_factory=dict)

    def counters(self) -> dict[str, int | float]:
        return {name: getattr(self, name) for name in COUNTERS}


COUNTERS = tuple(f.name for f in fields(Totals) if f.name not in ("metrics", "latency"))
SUMMED = COUNTERS[5:]


def metric_names(run: Run) -> list[str]:
    return [s.get("name") or s["type"] for s in run.scorers]


def score_of(status: str, scores: dict[str, Any], name: str) -> float | None:
    """A case's score on one metric under the scoring policy; None if unscorable."""
    if status != "ok":
        return 0.0
    value = (scores.get(name) or {}).get("value")
    return None if value is None else float(value)


def fold(rows: Iterable[Any], metrics: Sequence[str]) -> Totals:
    """Totals of result-shaped rows: anything with ``status``, ``cached``,
    ``latency_ms``, ``scores`` and the token and cost attributes of a result."""
    out = Totals(metrics={name: MetricTotals() for name in metrics})
    latency: Counter[int] = Counter()
    for row in rows:
        ok = row.status == "ok"
        out.completed += 1
        out.ok += ok
        out.errors += not ok
        out.cached += bool(row.cached)
        if ok and not row.cached:
            out.fresh += 1
            latency[latency_bucket(row.latency_ms)] += 1
        for name in SUMMED:
            setattr(out, name, getattr(out, name) + getattr(row, name))
        for name in metrics:
            agg = out.metrics[name]
            value = score_of(row.status, row.scores, name)
            if value is None:
                agg.unscorable += 1
            else:
                agg.n += 1
                agg.total += value
                agg.total_sq += value * value
    out.latency = dict(latency)
    return out


def add(session: Session, run_id: str, shard: str, totals: Totals) -> None:
    """Add ``totals`` to the run's aggregates, in the caller's transaction."""
    if not totals.completed:
        return
    counters = totals.counters()
    _upsert(
        session,
        RunAggregate.__table__,  # type: ignore[arg-type]
        ("run_id", "shard"),
        [{"run_id": run_id, "shard": shard, **counters}],
        list(counters),
    )
    if totals.metrics:
        _upsert(
            session,
            RunMetricAggregate.__table__,  # type: ignore[arg-type]
            ("run_id", "shard", "metric"),
            [
                {
                    "run_id": run_id,
                    "shard": shard,
                    "metric": name,
                    "n": m.n,
                    "total": m.total,
                    "total_sq": m.total_sq,
                    "unscorable": m.unscorable,
                }
                for name, m in totals.metrics.items()
            ],
            ["n", "total", "total_sq", "unscorable"],
        )
    if totals.latency:
        # In bucket order, so two writers to one shard always lock rows in one order.
        _upsert(
            session,
            RunLatencyBucket.__table__,  # type: ignore[arg-type]
            ("run_id", "shard", "bucket"),
            [
                {"run_id": run_id, "shard": shard, "bucket": bucket, "count": count}
                for bucket, count in sorted(totals.latency.items())
            ],
            ["count"],
        )


def _upsert(
    session: Session,
    table: Table,
    keys: tuple[str, ...],
    rows: list[dict[str, Any]],
    increments: list[str],
) -> None:
    dialect = session.get_bind().dialect.name
    insert = postgresql.insert if dialect == "postgresql" else sqlite.insert
    stmt = insert(table).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=list(keys),
        set_={name: table.c[name] + stmt.excluded[name] for name in increments},
    )
    session.execute(stmt)


def read(session: Session, run: Run) -> Totals:
    """The run's totals: its shard rows summed. A few rows per shard (a few hundred
    for the latency histogram), whatever the run's size."""
    sums = session.execute(
        select(*(func.coalesce(func.sum(RunAggregate.__table__.c[n]), 0) for n in COUNTERS)).where(
            RunAggregate.run_id == run.id
        )
    ).one()
    out = Totals()
    for name, value in zip(COUNTERS, sums, strict=True):
        setattr(out, name, _number(name, value))
    out.metrics = {name: MetricTotals() for name in metric_names(run)}
    rows = session.execute(
        select(
            RunMetricAggregate.metric,
            func.sum(RunMetricAggregate.n),
            func.sum(RunMetricAggregate.total),
            func.sum(RunMetricAggregate.total_sq),
            func.sum(RunMetricAggregate.unscorable),
        )
        .where(RunMetricAggregate.run_id == run.id)
        .group_by(RunMetricAggregate.metric)
    )
    for name, n, total, total_sq, unscorable in rows:
        out.metrics[name] = MetricTotals(int(n), float(total), float(total_sq), int(unscorable))
    out.latency = {
        int(bucket): int(count)
        for bucket, count in session.execute(
            select(RunLatencyBucket.bucket, func.sum(RunLatencyBucket.count))
            .where(RunLatencyBucket.run_id == run.id)
            .group_by(RunLatencyBucket.bucket)
        )
    }
    return out


def _number(name: str, value: Any) -> int | float:
    return float(value) if name.endswith("_usd") else int(value)


def _result_rows(session: Session, run_id: str, *columns: Any) -> Iterable[Any]:
    return session.execute(
        select(*columns).where(Result.run_id == run_id).execution_options(yield_per=_READ_CHUNK)
    )


def recompute(session: Session, run: Run) -> Totals:
    """The run's totals computed from its results, in one streaming pass."""
    columns: list[Any] = [Result.status, Result.cached, Result.latency_ms, Result.scores]
    columns += [Result.__table__.c[name] for name in SUMMED]
    return fold(_result_rows(session, run.id, *columns), metric_names(run))


def rebuild(session: Session, run: Run) -> Totals:
    """Replace the run's aggregates with a recomputation from its results. Only
    safe while no process is writing results for the run."""
    totals = recompute(session, run)
    session.execute(delete(RunMetricAggregate).where(RunMetricAggregate.run_id == run.id))
    session.execute(delete(RunLatencyBucket).where(RunLatencyBucket.run_id == run.id))
    session.execute(delete(RunAggregate).where(RunAggregate.run_id == run.id))
    add(session, run.id, REBUILD_SHARD, totals)
    return totals


def cases_total(session: Session, run: Run) -> int:
    """How many cases the run's filter selects, stored on the run after the first
    count (runs created since migration 0005 have it from the start)."""
    if run.cases_total is None:
        dataset = session.get_one(Dataset, run.dataset_id)
        split = run.case_filter.get("split")
        tags = run.case_filter.get("tags") or ()
        run.cases_total = (
            dataset.case_count
            if not (split or tags)
            else len(load_cases(session, dataset, split, tags))
        )
        session.execute(
            update(Run)
            .where(Run.id == run.id)
            .values(cases_total=run.cases_total)
            .execution_options(synchronize_session=False)
        )
        session.commit()
    return run.cases_total


# Compact score arrays -------------------------------------------------------


@dataclass(frozen=True)
class MetricScores:
    """One metric of a completed run, as stored in ``run_scores``."""

    name: str
    case_count: int
    n: int
    unscorable: int
    mean: float | None
    ci_low: float | None
    ci_high: float | None


@dataclass
class Arrays:
    """Every metric's per-case scores in case-id order, built from the results."""

    case_count: int
    scores: dict[str, FloatArray]
    unscorable: dict[str, int]
    latencies: FloatArray  # fresh successful results, in no particular order


def case_order(session: Session, dataset_id: int) -> list[str]:
    """The dataset's case ids in the order score arrays use. Sorted in Python, not
    by the database, so the order never depends on a database collation; it is the
    order a paired comparison has always resampled in."""
    return sorted(
        session.scalars(select(DatasetCase.case_id).where(DatasetCase.dataset_id == dataset_id))
    )


def build_arrays(session: Session, run: Run) -> Arrays:
    names = metric_names(run)
    order = case_order(session, run.dataset_id)
    index = {case_id: i for i, case_id in enumerate(order)}
    scores = {name: np.full(len(order), np.nan, dtype=np.float64) for name in names}
    unscorable = dict.fromkeys(names, 0)
    latencies: list[float] = []
    rows = _result_rows(
        session, run.id, Result.case_id, Result.status, Result.scores, Result.cached,
        Result.latency_ms,
    )  # fmt: skip
    for case_id, status, case_scores, cached, latency in rows:
        i = index[case_id]
        for name in names:
            value = score_of(status, case_scores, name)
            if value is None:
                unscorable[name] += 1
            else:
                scores[name][i] = value
        if status == "ok" and not cached:
            latencies.append(latency)
    return Arrays(len(order), scores, unscorable, np.asarray(latencies, dtype=np.float64))


def percentiles(latencies: FloatArray | Sequence[float]) -> tuple[float | None, float | None]:
    if not len(latencies):
        return None, None
    p50, p95 = np.percentile(np.asarray(latencies, dtype=np.float64), [50, 95]).tolist()
    return float(p50), float(p95)


def histogram_percentiles(
    histogram: Mapping[int, int], qs: Sequence[float] = (50, 95)
) -> list[float] | None:
    """Percentiles of the latencies a histogram counts, the way ``np.percentile``
    interpolates between order statistics, with each order statistic replaced by
    its bucket's value. Each is within ``LATENCY_ACCURACY`` of the exact percentile
    (or about a microsecond off it, for latencies under ``LATENCY_FLOOR_MS``). None for
    an empty histogram."""
    items = sorted((b, c) for b, c in histogram.items() if c > 0)
    if not items:
        return None
    buckets = np.array([b for b, _ in items], dtype=np.int64)
    ends = np.cumsum([c for _, c in items])  # rank one past each bucket's last value
    n = int(ends[-1])
    out = []
    for q in qs:
        rank = (n - 1) * q / 100
        lo, hi = math.floor(rank), math.ceil(rank)
        v_lo, v_hi = (
            bucket_value(int(buckets[np.searchsorted(ends, k, side="right")])) for k in (lo, hi)
        )
        out.append(v_lo + (rank - lo) * (v_hi - v_lo))
    return out


def summary_rows(
    run: Run, arrays: Arrays, *, resamples: int = DEFAULT_RESAMPLES
) -> list[RunScores]:
    rows = []
    for name, values in arrays.scores.items():
        scored = values[~np.isnan(values)]
        ci = mean_interval(scored, resamples=resamples, seed=run.seed) if scored.size else None
        rows.append(
            RunScores(
                run_id=run.id,
                metric=name,
                case_count=arrays.case_count,
                n=int(scored.size),
                unscorable=arrays.unscorable[name],
                mean=None if ci is None else ci.mean,
                ci_low=None if ci is None else ci.low,
                ci_high=None if ci is None else ci.high,
                scores=encode(values),
            )
        )
    return rows


def encode(values: FloatArray) -> bytes:
    return values.astype(SCORE_DTYPE, copy=False).tobytes()


def decode(data: bytes) -> FloatArray:
    return np.frombuffer(data, dtype=SCORE_DTYPE).astype(np.float64)


def compact(session: Session, run: Run) -> None:
    """Write the completed run's score arrays, summary intervals and latency
    percentiles, and commit. Several processes may race to do it; the rows are a
    function of the results, so whichever commits first is kept."""
    arrays = build_arrays(session, run)
    rows = summary_rows(run, arrays)
    p50, p95 = percentiles(arrays.latencies)
    try:
        session.execute(delete(RunScores).where(RunScores.run_id == run.id))
        session.add_all(rows)
        # Flush here: a concurrent compaction's rows surface as a unique violation
        # on this insert (on PostgreSQL), whichever statement would autoflush it.
        session.flush()
        session.execute(
            update(Run)
            .where(Run.id == run.id)
            .values(latency_p50_ms=p50, latency_p95_ms=p95)
            .execution_options(synchronize_session=False)
        )
        session.commit()
    except IntegrityError:
        session.rollback()
        return
    run.latency_p50_ms, run.latency_p95_ms = p50, p95


def stored(session: Session, run: Run) -> dict[str, MetricScores] | None:
    """The compacted metrics of a completed run (without the arrays), compacting it
    first if that has not happened yet. None for a run that is not completed."""
    if run.status != "completed":
        return None
    found = _stored_rows(session, run)
    if set(found) != set(metric_names(run)):
        compact(session, run)
        found = _stored_rows(session, run)
    return found


def _stored_rows(session: Session, run: Run) -> dict[str, MetricScores]:
    rows = session.execute(
        select(
            RunScores.metric,
            RunScores.case_count,
            RunScores.n,
            RunScores.unscorable,
            RunScores.mean,
            RunScores.ci_low,
            RunScores.ci_high,
        ).where(RunScores.run_id == run.id)
    )
    return {row[0]: MetricScores(*row) for row in rows}


def load_scores(session: Session, run: Run, names: Sequence[str]) -> dict[str, FloatArray]:
    """The score arrays of a completed run: one row per metric."""
    stored(session, run)
    rows = session.execute(
        select(RunScores.metric, RunScores.scores).where(
            RunScores.run_id == run.id, RunScores.metric.in_(list(names))
        )
    )
    return {name: decode(data) for name, data in rows}


# Verification ---------------------------------------------------------------


@dataclass(frozen=True)
class Verification:
    run_id: str
    status: str
    results: int
    compacted: bool
    problems: list[str]
    repaired: bool = False

    @property
    def ok(self) -> bool:
        return not self.problems


def _close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE)


def check(session: Session, run: Run) -> tuple[int, bool, list[str]]:
    """Recompute everything the aggregates and score arrays hold from the results
    and list every difference. Returns (results, compacted, problems)."""
    problems: list[str] = []
    have, want = read(session, run), recompute(session, run)
    for name in COUNTERS:
        stored_value, actual = getattr(have, name), getattr(want, name)
        same = _close(stored_value, actual) if name.endswith("_usd") else stored_value == actual
        if not same:
            problems.append(f"{name}: stored {stored_value}, results give {actual}")
    for name, actual_metric in want.metrics.items():
        got = have.metrics.get(name, MetricTotals())
        for attr in ("n", "unscorable"):
            if getattr(got, attr) != getattr(actual_metric, attr):
                problems.append(
                    f"{name}.{attr}: stored {getattr(got, attr)}, "
                    f"results give {getattr(actual_metric, attr)}"
                )
        for attr in ("total", "total_sq"):
            if not _close(getattr(got, attr), getattr(actual_metric, attr)):
                problems.append(
                    f"{name}.{attr}: stored {getattr(got, attr)!r}, "
                    f"results give {getattr(actual_metric, attr)!r}"
                )
    if have.latency != want.latency:
        wrong = sorted(
            b for b in set(have.latency) | set(want.latency)
            if have.latency.get(b, 0) != want.latency.get(b, 0)
        )  # fmt: skip
        problems.append(
            f"latency histogram: {len(wrong)} bucket(s) differ, stored "
            f"{sum(have.latency.values())} fresh results, results give "
            f"{sum(want.latency.values())}"
        )
    if run.cases_total is not None:
        dataset = session.get_one(Dataset, run.dataset_id)
        selected = len(
            load_cases(
                session, dataset, run.case_filter.get("split"), run.case_filter.get("tags") or ()
            )
        )
        if selected != run.cases_total:
            problems.append(f"cases_total: stored {run.cases_total}, the filter selects {selected}")
    compacted = _stored_rows(session, run)
    if compacted:
        problems += _check_compacted(session, run, compacted)
    return want.completed, bool(compacted), problems


def _check_compacted(session: Session, run: Run, compacted: dict[str, MetricScores]) -> list[str]:
    problems = []
    arrays = build_arrays(session, run)
    expected = {row.metric: row for row in summary_rows(run, arrays)}
    stored_arrays = load_scores(session, run, list(compacted))
    for name in sorted(set(expected) | set(compacted)):
        if name not in compacted or name not in expected:
            problems.append(f"{name}: score array {'missing' if name in expected else 'unknown'}")
            continue
        want, have = expected[name], compacted[name]
        if not np.array_equal(arrays.scores[name], stored_arrays[name], equal_nan=True):
            problems.append(f"{name}: score array differs from the results")
        for attr in ("case_count", "n", "unscorable"):
            if getattr(want, attr) != getattr(have, attr):
                problems.append(
                    f"{name}.{attr}: stored {getattr(have, attr)}, results give "
                    f"{getattr(want, attr)}"
                )
        for attr in ("mean", "ci_low", "ci_high"):
            if not _close(getattr(have, attr), getattr(want, attr)):
                problems.append(
                    f"{name}.{attr}: stored {getattr(have, attr)!r}, results give "
                    f"{getattr(want, attr)!r}"
                )
    p50, p95 = percentiles(arrays.latencies)
    for attr, value in (("latency_p50_ms", p50), ("latency_p95_ms", p95)):
        if not _close(getattr(run, attr), value):
            problems.append(f"{attr}: stored {getattr(run, attr)!r}, results give {value!r}")
    return problems


BUSY = ("queued", "running", "cancelling")


def verify(session: Session, run_id: str, *, repair: bool = False) -> Verification:
    """Check a run's aggregates and score arrays against its results; with
    ``repair``, replace whatever differs with the recomputation.

    Call it on a fresh session: on PostgreSQL every read happens in one
    repeatable-read snapshot, so results written meanwhile cannot make the
    stored and recomputed numbers disagree spuriously."""
    if session.get_bind().dialect.name == "postgresql":
        session.connection(execution_options={"isolation_level": "REPEATABLE READ"})
    run = session.get(Run, run_id)
    if run is None:
        raise NotFoundError(f"run {run_id} not found")
    if repair and run.status in BUSY:
        raise ConflictError(
            f"run {run_id} is {run.status}; repair it when no process is executing it"
        )
    results, compacted, problems = check(session, run)
    if not (repair and problems):
        return Verification(run.id, run.status, results, compacted, problems)
    rebuild(session, run)
    session.commit()
    if compacted or run.status == "completed":
        compact(session, run)
    now_compacted = compacted or run.status == "completed"
    return Verification(run.id, run.status, results, now_compacted, problems, repaired=True)
