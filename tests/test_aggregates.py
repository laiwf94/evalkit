"""Per-run aggregates, compact score arrays and ``evalkit run verify``."""

import asyncio
import io
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session, sessionmaker

from evalkit import aggregates, report
from evalkit.cli import main
from evalkit.config import Settings
from evalkit.datasets import Case, import_dataset
from evalkit.db.models import (
    Result,
    Run,
    RunAggregate,
    RunMetricAggregate,
    RunScores,
)
from evalkit.errors import ConflictError, NotFoundError
from evalkit.report import compare_runs, summarize
from evalkit.runner import CaseResult, Runner
from evalkit.runspec import RunConfig
from evalkit.stats import DEFAULT_RESAMPLES, bootstrap_ci, compare_many, normal_ci


def make_cases(n: int = 60) -> list[Case]:
    # Ids deliberately not in insertion order, so case-id order is exercised.
    return [
        Case(
            id=f"k{(i * 37) % n:03d}",
            input={"q": f"question {i}"},
            # Every fifth case has no expected value: unscorable for exact.
            expected=None if i % 5 == 0 else f"answer {i}",
            split="test" if i % 2 else "dev",
        )
        for i in range(n)
    ]


@pytest.fixture
def dataset(sessions: sessionmaker[Session]) -> str:
    with sessions() as s:
        import_dataset(s, "agg", make_cases())
    return "agg"


def config(**target: Any) -> RunConfig:
    run_filter = target.pop("filter", {})
    options: dict[str, Any] = {"concurrency": 4}
    if "error_rate" in target:
        # Transient failures with one attempt: that fraction of cases ends as an error.
        target["failure_rate"] = target.pop("error_rate")
        options["retry"] = {"max_attempts": 1}
    return RunConfig.model_validate(
        {
            "label": "agg",
            "dataset": {"name": "agg"},
            "target": {"type": "fake", **target},
            "scorers": [{"type": "exact"}, {"type": "token_f1", "gate": False}],
            "options": options,
            "filter": run_filter,
        }
    )


def runner(sessions: sessionmaker[Session]) -> Runner:
    return Runner(sessions, Settings.from_env({}), allow_import_paths=True)


def execute(sessions: sessionmaker[Session], cfg: RunConfig) -> str:
    r = runner(sessions)
    run = r.create_run(cfg)
    asyncio.run(r.execute(run.id))
    return run.id


def reference_scores(s: Session, run_id: str) -> dict[str, dict[str, float]]:
    """Per metric, case id to score, straight from the results (the old summary path)."""
    run = s.get_one(Run, run_id)
    return report.metric_scores(s, run)[0]


def test_completed_run_summary_matches_a_recomputation(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    run_id = execute(sessions, config(accuracy=0.7, seed=3, error_rate=0.1))
    with sessions() as s:
        run = s.get_one(Run, run_id)
        summary = summarize(s, run_id)
        scores = reference_scores(s, run_id)
        latencies = list(
            s.scalars(
                select(Result.latency_ms).where(
                    Result.run_id == run_id, Result.status == "ok", Result.cached.is_(False)
                )
            )
        )
        stored = s.scalars(select(RunScores).where(RunScores.run_id == run_id)).all()
        assert aggregates.read(s, run) == aggregates.recompute(s, run)
    assert {r.metric for r in stored} == {"exact", "token_f1"}
    assert summary.completed == summary.cases_total == 60
    assert summary.errors > 0
    for metric in summary.metrics:
        values = [scores[metric.name][k] for k in sorted(scores[metric.name])]
        ci = bootstrap_ci(values, resamples=DEFAULT_RESAMPLES, seed=run.seed)
        assert metric.mean == pytest.approx(ci.mean)
        assert (metric.ci_low, metric.ci_high, metric.n) == (ci.low, ci.high, ci.n)
    exact = summary.metrics[0]
    # Every fifth case has no expected value; those that failed still score 0.
    assert exact.unscorable > 0
    assert exact.n + exact.unscorable == 60
    p50, p95 = np.percentile(latencies, [50, 95])
    assert summary.latency_p50_ms == pytest.approx(p50)
    assert summary.latency_p95_ms == pytest.approx(p95)


def test_score_arrays_are_in_case_id_order_with_nan_for_missing(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    run_id = execute(sessions, config(accuracy=0.6, filter={"split": "test"}))
    with sessions() as s:
        run = s.get_one(Run, run_id)
        arrays = aggregates.load_scores(s, run, ["exact"])
        order = aggregates.case_order(s, run.dataset_id)
        scores = reference_scores(s, run_id)["exact"]
    values = arrays["exact"]
    assert order == sorted(order)
    assert values.shape == (60,)
    for i, case_id in enumerate(order):
        if case_id in scores:
            assert values[i] == scores[case_id]
        else:
            assert math.isnan(values[i])


def test_summary_of_a_completed_run_reads_no_results(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    run_id = execute(sessions, config(accuracy=0.7))
    with sessions() as s:
        before = summarize(s, run_id)
        s.execute(delete(Result).where(Result.run_id == run_id))
        s.commit()
        after = summarize(s, run_id)
    assert before.metrics == after.metrics
    assert (before.completed, before.ok, before.input_tokens) == (
        after.completed,
        after.ok,
        after.input_tokens,
    )
    assert after.latency_p50_ms == before.latency_p50_ms


def test_summary_with_other_resamples_recomputes_from_the_arrays(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    run_id = execute(sessions, config(accuracy=0.7, seed=5))
    with sessions() as s:
        run = s.get_one(Run, run_id)
        summary = summarize(s, run_id, resamples=300)
        scores = reference_scores(s, run_id)["exact"]
    ci = bootstrap_ci([scores[k] for k in sorted(scores)], resamples=300, seed=run.seed)
    assert (summary.metrics[0].ci_low, summary.metrics[0].ci_high) == (ci.low, ci.high)


def test_compare_reads_arrays_and_matches_the_paired_comparison(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    base = execute(sessions, config(accuracy=0.8, seed=1))
    cand = execute(sessions, config(accuracy=0.4, seed=2, filter={"split": "test"}))
    with sessions() as s:
        result = compare_runs(s, base, cand, metrics=["exact", "token_f1"], resamples=500)
        b, c = reference_scores(s, base), reference_scores(s, cand)
        seed = s.get_one(Run, cand).seed
    expected = compare_many(
        {name: (b[name], c[name]) for name in ("exact", "token_f1")}, resamples=500, seed=seed
    )
    assert result.comparisons == expected
    assert result.comparisons[0].n_unpaired > 0


def test_compare_when_no_case_was_scored_in_both(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    base = execute(sessions, config(accuracy=0.8, filter={"split": "dev"}))
    cand = execute(sessions, config(accuracy=0.8, filter={"split": "test"}))
    with sessions() as s, pytest.raises(Exception, match="no case was scored in both"):
        compare_runs(s, base, cand)


def test_missing_score_arrays_are_rebuilt_on_read(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    run_id = execute(sessions, config(accuracy=0.7))
    with sessions() as s:
        first = summarize(s, run_id)
        s.execute(delete(RunScores).where(RunScores.run_id == run_id))
        s.execute(update(Run).where(Run.id == run_id).values(latency_p50_ms=None))
        s.commit()
    with sessions() as s:
        again = summarize(s, run_id)
        assert s.scalars(select(RunScores.metric).where(RunScores.run_id == run_id)).all()
    assert again.metrics == first.metrics
    assert again.latency_p50_ms == first.latency_p50_ms


def test_a_failed_compaction_does_not_fail_the_run(
    sessions: sessionmaker[Session],
    dataset: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    real = aggregates.compact

    def broken(session: Session, run: Run) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(aggregates, "compact", broken)
    with caplog.at_level(logging.ERROR):
        run_id = execute(sessions, config(accuracy=0.7))
    assert "compacting run scores failed" in caplog.text
    monkeypatch.setattr(aggregates, "compact", real)
    with sessions() as s:
        assert s.get_one(Run, run_id).status == "completed"
        assert not s.scalars(select(RunScores).where(RunScores.run_id == run_id)).all()
        assert summarize(s, run_id).metrics[0].n > 0
        assert s.scalars(select(RunScores).where(RunScores.run_id == run_id)).all()


def test_two_processes_write_their_own_shards(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    first, second = runner(sessions), runner(sessions)
    run = first.create_run(config(accuracy=0.7))

    async def both() -> None:
        await asyncio.gather(first.execute(run.id), second.execute(run.id))

    asyncio.run(both())
    with sessions() as s:
        shards = set(s.scalars(select(RunAggregate.shard).where(RunAggregate.run_id == run.id)))
        stored = s.get_one(Run, run.id)
        assert aggregates.read(s, stored) == aggregates.recompute(s, stored)
        assert aggregates.read(s, stored).completed == 60
    assert shards <= {first.owner, second.owner}
    assert shards


def test_running_run_summary_small_and_large(
    sessions: sessionmaker[Session], dataset: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = runner(sessions)
    run = r.create_run(config(accuracy=0.7, seed=4))
    asyncio.run(r.execute(run.id))
    with sessions() as s:
        # Pretend it is still running, so the summary takes the live path.
        s.execute(update(Run).where(Run.id == run.id).values(status="running"))
        s.commit()
        exact = summarize(s, run.id, resamples=400)
        scores = reference_scores(s, run.id)["exact"]
        ci = bootstrap_ci([scores[k] for k in sorted(scores)], resamples=400, seed=run.seed)
        assert (exact.metrics[0].ci_low, exact.metrics[0].ci_high) == (ci.low, ci.high)
        # Above the exact limit, only the aggregates and a latency sample are read.
        monkeypatch.setattr(report, "LIVE_EXACT_MAX", 10)
        monkeypatch.setattr(report, "LATENCY_SAMPLE", 5)
        large = summarize(s, run.id)
        totals = aggregates.recompute(s, s.get_one(Run, run.id)).metrics["exact"]
        expected = normal_ci(totals.n, totals.total, totals.total_sq)
        recent = list(
            s.scalars(
                select(Result.latency_ms)
                .where(Result.run_id == run.id, Result.status == "ok")
                .order_by(Result.id.desc())
                .limit(5)
            )
        )
    got = large.metrics[0]
    assert (got.mean, got.ci_low, got.ci_high) == pytest.approx(
        (expected.mean, expected.low, expected.high)
    )
    assert large.latency_p50_ms == pytest.approx(float(np.percentile(recent, 50)))
    assert exact.metrics[1].n == large.metrics[1].n


def test_live_metric_with_no_scores(sessions: sessionmaker[Session], dataset: str) -> None:
    r = runner(sessions)
    run = r.create_run(config(accuracy=0.7))
    with sessions() as s:
        summary = summarize(s, run.id)
    assert summary.completed == 0
    assert summary.metrics[0].mean is None
    assert summary.latency_p50_ms is None


def test_results_written_one_by_one_are_aggregated(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    r = runner(sessions)
    run = r.create_run(config(accuracy=0.7))
    asyncio.run(r._prepare(run))
    fence = {c.case_id: c.fence for c in r._claim_cases(run.id, 4)}
    first, second = sorted(fence)[:2]
    with sessions() as s:
        # Written outside the lease protocol: the batch falls back to one-by-one writes.
        s.add(Result(run_id=run.id, case_id=first, status="ok", output="old", scores={}))
        s.commit()
    ok = CaseResult(
        case_id=second,
        status="ok",
        output="x",
        input_tokens=7,
        scores={"exact": {"value": 1.0}},
        fence=fence[second],
    )
    dup = CaseResult(case_id=first, status="ok", output="y", fence=fence[first])
    assert r._write(run.id, [dup, ok]) == 1
    with sessions() as s:
        totals = aggregates.read(s, s.get_one(Run, run.id))
    assert (totals.completed, totals.input_tokens) == (1, 7)
    assert totals.metrics["exact"] == aggregates.MetricTotals(1, 1.0, 1.0, 0)
    assert totals.metrics["token_f1"].unscorable == 1


def test_cases_total_is_filled_in_for_older_runs(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    run_id = execute(sessions, config(accuracy=0.7, filter={"split": "dev"}))
    with sessions() as s:
        s.execute(update(Run).where(Run.id == run_id).values(cases_total=None))
        s.commit()
        assert summarize(s, run_id).cases_total == 30
        assert s.get_one(Run, run_id).cases_total == 30
    no_filter = execute(sessions, config(accuracy=0.7))
    with sessions() as s:
        s.execute(update(Run).where(Run.id == no_filter).values(cases_total=None))
        s.commit()
        assert summarize(s, no_filter).cases_total == 60


# Verification ---------------------------------------------------------------


def test_verify_finds_and_repairs_drift(sessions: sessionmaker[Session], dataset: str) -> None:
    run_id = execute(sessions, config(accuracy=0.7, error_rate=0.1))
    with sessions() as s:
        clean = aggregates.verify(s, run_id)
    assert clean.ok
    assert clean.compacted
    assert clean.results == 60
    with sessions() as s:
        s.execute(update(RunAggregate).where(RunAggregate.run_id == run_id).values(ok=0))
        s.execute(update(RunAggregate).where(RunAggregate.run_id == run_id).values(cost_usd=99.0))
        s.execute(
            update(RunMetricAggregate)
            .where(RunMetricAggregate.run_id == run_id, RunMetricAggregate.metric == "exact")
            .values(n=1, total=123.0)
        )
        s.execute(
            update(RunScores)
            .where(RunScores.run_id == run_id, RunScores.metric == "exact")
            .values(scores=aggregates.encode(np.zeros(60)), n=3, mean=0.123)
        )
        s.execute(update(Run).where(Run.id == run_id).values(latency_p95_ms=-1.0, cases_total=59))
        s.commit()
    with sessions() as s:
        found = aggregates.verify(s, run_id)
    joined = "\n".join(found.problems)
    for fragment in (
        "ok: stored 0",
        "cost_usd: stored 99.0",
        "exact.n: stored 1",
        "exact.total: stored 123.0",
        "exact: score array differs",
        "exact.n: stored 3",
        "exact.mean: stored 0.123",
        "latency_p95_ms: stored -1.0",
        "cases_total: stored 59",
    ):
        assert fragment in joined
    with sessions() as s:
        s.execute(update(Run).where(Run.id == run_id).values(cases_total=60))
        s.commit()
        repaired = aggregates.verify(s, run_id, repair=True)
    assert repaired.repaired
    with sessions() as s:
        assert aggregates.verify(s, run_id).ok
        shards = set(s.scalars(select(RunAggregate.shard).where(RunAggregate.run_id == run_id)))
    assert shards == {aggregates.REBUILD_SHARD}


def test_verify_notices_missing_and_unknown_arrays(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    run_id = execute(sessions, config(accuracy=0.7))
    with sessions() as s:
        s.execute(
            update(RunScores)
            .where(RunScores.run_id == run_id, RunScores.metric == "token_f1")
            .values(metric="renamed")
        )
        s.commit()
        problems = aggregates.verify(s, run_id).problems
    assert "renamed: score array unknown" in problems
    assert "token_f1: score array missing" in problems


def test_verify_refuses_to_repair_a_busy_run(sessions: sessionmaker[Session], dataset: str) -> None:
    r = runner(sessions)
    run = r.create_run(config(accuracy=0.7), queued=True)
    with sessions() as s:
        assert aggregates.verify(s, run.id).ok
    with sessions() as s, pytest.raises(ConflictError, match="queued"):
        aggregates.verify(s, run.id, repair=True)
    with sessions() as s, pytest.raises(NotFoundError):
        aggregates.verify(s, "nope")


def test_repairing_an_incomplete_run_leaves_it_uncompacted(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    run_id = execute(sessions, config(accuracy=0.7))
    with sessions() as s:
        s.execute(update(Run).where(Run.id == run_id).values(status="interrupted"))
        s.execute(delete(RunScores).where(RunScores.run_id == run_id))
        s.execute(delete(RunAggregate).where(RunAggregate.run_id == run_id))
        s.commit()
        fixed = aggregates.verify(s, run_id, repair=True)
    assert fixed.repaired
    assert not fixed.compacted
    with sessions() as s:
        assert aggregates.verify(s, run_id).ok


def run_cli(db: str, *args: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(["--database-url", db, *args], out=out)
    return code, out.getvalue()


def test_run_verify_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVALKIT_DATABASE_URL", raising=False)
    db = f"sqlite:///{tmp_path / 'cli.db'}"
    lines = [json.dumps({"id": f"q{i}", "input": {"q": str(i)}, "expected": "x"}) for i in range(9)]
    (tmp_path / "d.jsonl").write_text("\n".join(lines) + "\n")
    cfg = tmp_path / "run.toml"
    cfg.write_text(
        '[dataset]\nname = "d"\npath = "d.jsonl"\n'
        '[target]\ntype = "fake"\naccuracy = 0.5\n'
        '[[scorers]]\ntype = "exact"\n'
    )
    code, out = run_cli(db, "run", str(cfg), "--json")
    assert code == 0
    run_id = json.loads(out)["run_id"]

    code, out = run_cli(db, "run", "verify", run_id)
    assert code == 0
    assert "OK: aggregates match the results" in out
    assert "score arrays checked" in out

    from evalkit.db import make_engine, make_session_factory

    engine = make_engine(db)
    with make_session_factory(engine)() as s:
        s.execute(update(RunAggregate).where(RunAggregate.run_id == run_id).values(completed=1))
        s.commit()
    engine.dispose()
    code, out = run_cli(db, "run", "verify", run_id, "--json")
    assert code == 3
    assert json.loads(out)["problems"] == ["completed: stored 1, results give 9"]

    code, out = run_cli(db, "run", "verify", run_id, "--repair")
    assert code == 0
    assert "repaired: aggregates rewritten from the results" in out
    assert run_cli(db, "run", "verify", run_id)[0] == 0

    assert run_cli(db, "run", "verify")[0] == 2
    assert run_cli(db, "run", "verify", "missing-run")[0] == 2
    assert run_cli(db, "run", str(cfg), "extra")[0] == 2
    assert run_cli(db, "run", str(cfg), "--repair")[0] == 2


# The migration that added aggregates ------------------------------------------


def test_migration_backfills_aggregates_for_existing_runs(tmp_path: Path) -> None:
    """Results stored before migration 0005 get aggregates in the upgrade."""
    from sqlalchemy import text

    from evalkit.db import make_engine, make_session_factory, upgrade

    engine = make_engine(f"sqlite:///{tmp_path / 'old.db'}")
    upgrade(engine, "0004")
    statuses = ["ok", "ok", "error", "ok", "ok", "ok"]
    values: list[float | None] = [1.0, 0.0, None, None, 0.5, 1.0]
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO datasets (id, name, version, content_hash, case_count, description,"
                " created_at) VALUES (1, 'd', 1, :h, 6, '', CURRENT_TIMESTAMP)"
            ),
            {"h": "h" * 64},
        )
        for i in range(6):
            conn.execute(
                text(
                    "INSERT INTO dataset_cases (dataset_id, position, case_id, input, tags,"
                    " metadata) VALUES (1, :i, :c, '{}', '[]', '{}')"
                ),
                {"i": i, "c": f"c{i}"},
            )
        conn.execute(
            text(
                "INSERT INTO runs (id, label, dataset_id, target_spec, target_hash, scorers,"
                " options, case_filter, seed, status, error, leases_seeded, priority,"
                " created_at) VALUES ('old', '', 1, '{\"type\": \"fake\"}', :t,"
                ' \'[{"type": "exact"}, {"type": "regex", "name": "r"}]\','
                " '{}', '{}', 7, 'completed', '', 0, 0, CURRENT_TIMESTAMP)"
            ),
            {"t": "t" * 64},
        )
        for i, (status, value) in enumerate(zip(statuses, values, strict=True)):
            conn.execute(
                text(
                    "INSERT INTO results (run_id, case_id, status, cached, latency_ms, attempts,"
                    " input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,"
                    " cost_usd, judge_input_tokens, judge_output_tokens, judge_cost_usd,"
                    " scores, error, created_at) VALUES ('old', :c, :s, :cached, :lat, 1, 10, 2,"
                    " 0, 0, 0.25, 3, 1, 0.125, :scores, '', CURRENT_TIMESTAMP)"
                ),
                {
                    "c": f"c{i}",
                    "s": status,
                    "cached": i == 5,
                    "lat": 10.0 * (i + 1),
                    "scores": json.dumps({"exact": {"value": value}}),
                },
            )
    upgrade(engine)
    with make_session_factory(engine)() as s:
        run = s.get_one(Run, "old")
        totals = aggregates.read(s, run)
        shards = set(s.scalars(select(RunAggregate.shard).where(RunAggregate.run_id == "old")))
        assert aggregates.verify(s, "old").ok
        summary = summarize(s, "old")
    engine.dispose()
    assert shards == {aggregates.REBUILD_SHARD}
    assert (totals.completed, totals.ok, totals.errors, totals.cached, totals.fresh) == (
        6,
        5,
        1,
        1,
        4,
    )
    assert (totals.input_tokens, totals.judge_input_tokens) == (60, 18)
    assert totals.cost_usd == pytest.approx(1.5)
    assert totals.judge_cost_usd == pytest.approx(0.75)
    # The error scores 0 on both metrics; the scorer named "r" has no scores at all.
    assert totals.metrics["exact"] == aggregates.MetricTotals(5, 2.5, 2.25, 1)
    assert totals.metrics["r"] == aggregates.MetricTotals(1, 0.0, 0.0, 5)
    assert summary.cases_total == 6
    assert summary.metrics[0].mean == pytest.approx(0.5)
    assert summary.latency_p50_ms == pytest.approx(30.0)


# Properties -------------------------------------------------------------------

_score = st.one_of(st.none(), st.floats(0, 1, allow_nan=False))
_result = st.builds(
    CaseResult,
    case_id=st.just("c"),
    status=st.sampled_from(["ok", "ok", "ok", "error"]),
    cached=st.booleans(),
    input_tokens=st.integers(0, 10_000),
    output_tokens=st.integers(0, 10_000),
    cost_usd=st.floats(0, 1, allow_nan=False),
    judge_cost_usd=st.floats(0, 1, allow_nan=False),
    scores=st.fixed_dictionaries(
        {"a": st.fixed_dictionaries({"value": _score}), "b": st.fixed_dictionaries({})}
    ),
)


@settings(
    max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    batches=st.lists(st.tuples(st.sampled_from(["p1", "p2", "p3"]), st.lists(_result)), max_size=8)
)
def test_aggregates_over_shards_equal_one_fold_of_everything(
    sessions: sessionmaker[Session], dataset: str, batches: list[tuple[str, list[CaseResult]]]
) -> None:
    names = ["a", "b"]
    with sessions() as s:
        run = s.scalars(select(Run).limit(1)).first()
        if run is None:
            run = runner(sessions).create_run(config(accuracy=0.5))
            run = s.get_one(Run, run.id)
        run.scorers = [{"type": "exact", "name": "a"}, {"type": "exact", "name": "b"}]
        s.execute(delete(RunMetricAggregate))
        s.execute(delete(RunAggregate))
        for shard, batch in batches:
            aggregates.add(s, run.id, shard, aggregates.fold(batch, names))
        s.flush()
        got = aggregates.read(s, run)
        s.rollback()
    want = aggregates.fold([r for _, batch in batches for r in batch], names)
    for name in aggregates.COUNTERS:
        assert getattr(got, name) == pytest.approx(getattr(want, name), rel=1e-12, abs=1e-12)
    for name in names:
        g, w = got.metrics[name], want.metrics[name]
        assert (g.n, g.unscorable) == (w.n, w.unscorable)
        assert g.total == pytest.approx(w.total, rel=1e-12, abs=1e-12)
        assert g.total_sq == pytest.approx(w.total_sq, rel=1e-12, abs=1e-12)
