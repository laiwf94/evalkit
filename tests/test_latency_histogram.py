"""The latency histogram in the run aggregates: bucketing, percentiles from the
buckets, and migration 0007, which fills it in for runs that already had results."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import inspect, select, text

from evalkit import aggregates, report
from evalkit.aggregates import (
    LATENCY_ACCURACY,
    LATENCY_CEIL_MS,
    LATENCY_FLOOR_MS,
    bucket_value,
    histogram_percentiles,
    latency_bucket,
)
from evalkit.db import make_engine, make_session_factory, upgrade
from evalkit.db.models import Run, RunLatencyBucket
from evalkit.report import summarize

# Float rounding at bucket edges adds a few ulps on top of the stated accuracy.
_SLACK = 1e-9
# Latencies at or under the floor all stand for the floor bucket's value.
_FLOOR_ERROR = bucket_value(latency_bucket(LATENCY_FLOOR_MS))


def _histogram(latencies: list[float]) -> dict[int, int]:
    out: dict[int, int] = {}
    for ms in latencies:
        out[latency_bucket(ms)] = out.get(latency_bucket(ms), 0) + 1
    return out


def _close_enough(estimate: float, exact: float) -> bool:
    return abs(estimate - exact) <= (LATENCY_ACCURACY + _SLACK) * exact + _FLOOR_ERROR


@given(st.floats(LATENCY_FLOOR_MS, LATENCY_CEIL_MS, exclude_min=True))
def test_every_latency_is_within_the_accuracy_of_its_bucket_value(ms: float) -> None:
    assert abs(bucket_value(latency_bucket(ms)) - ms) <= (LATENCY_ACCURACY + _SLACK) * ms


def test_out_of_range_latencies_share_the_edge_buckets() -> None:
    floor = latency_bucket(LATENCY_FLOOR_MS)
    for ms in (0.0, -5.0, LATENCY_FLOOR_MS / 2, math.nan):
        assert latency_bucket(ms) == floor
    assert latency_bucket(math.inf) == latency_bucket(LATENCY_CEIL_MS)
    assert latency_bucket(1e300) == latency_bucket(LATENCY_CEIL_MS)
    # Buckets are monotone and about 2% wide.
    assert latency_bucket(100.0) < latency_bucket(102.5) <= latency_bucket(100.0) + 2
    assert abs(LATENCY_ACCURACY - 0.0099) < 1e-4


@settings(max_examples=300)
@given(
    latencies=st.lists(st.floats(0, 600_000, allow_nan=False), min_size=1, max_size=400),
    q=st.floats(0, 100),
)
def test_percentiles_from_buckets_are_within_the_accuracy_of_the_exact_ones(
    latencies: list[float], q: float
) -> None:
    found = histogram_percentiles(_histogram(latencies), [q, 50, 95])
    assert found is not None
    exact = np.percentile(np.asarray(latencies), [q, 50, 95]).tolist()
    for estimate, want in zip(found, exact, strict=True):
        assert _close_enough(estimate, want), (estimate, want)


def test_percentiles_on_a_skewed_large_sample() -> None:
    rng = np.random.default_rng(3)
    latencies = np.concatenate([rng.lognormal(5, 0.5, 200_000), rng.lognormal(9, 0.2, 2_000)])
    histogram = _histogram(latencies.tolist())
    # A few hundred buckets whatever the sample size: that is what a summary reads.
    assert len(histogram) < 400
    found = histogram_percentiles(histogram, [50, 95, 99.5])
    assert found is not None
    for estimate, want in zip(found, np.percentile(latencies, [50, 95, 99.5]), strict=True):
        assert _close_enough(estimate, float(want))


def test_empty_and_zero_count_histograms() -> None:
    assert histogram_percentiles({}) is None
    assert histogram_percentiles({5: 0}) is None
    only = histogram_percentiles({latency_bucket(80.0): 3, 7: 0})
    assert only == [bucket_value(latency_bucket(80.0))] * 2


def test_migration_0007_fills_in_histograms_and_drops_the_recent_results_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_engine(f"sqlite:///{tmp_path / 'old.db'}")
    upgrade(engine, "0006")
    rng = np.random.default_rng(11)
    latencies = rng.lognormal(4, 0.7, 300).tolist()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO datasets (id, name, version, content_hash, case_count, description,"
                " created_at) VALUES (1, 'd', 1, :h, 300, '', CURRENT_TIMESTAMP)"
            ),
            {"h": "h" * 64},
        )
        conn.execute(
            text(
                "INSERT INTO dataset_cases (dataset_id, position, case_id, input, tags,"
                " metadata) VALUES (1, :i, :c, '{\"q\": 1}', '[]', '{}')"
            ),
            [{"i": i, "c": f"c{i:03d}"} for i in range(300)],
        )
        conn.execute(
            text(
                "INSERT INTO runs (id, label, dataset_id, target_spec, target_hash, scorers,"
                " options, case_filter, seed, status, error, leases_seeded, priority,"
                " cases_total, created_at) VALUES ('live', '', 1, '{\"type\": \"fake\"}', :t,"
                " '[{\"type\": \"exact\"}]', '{}', '{}', 7, 'running', '', 1, 0, 300,"
                " CURRENT_TIMESTAMP)"
            ),
            {"t": "t" * 64},
        )
        conn.execute(
            text(
                "INSERT INTO results (run_id, case_id, status, cached, latency_ms, attempts,"
                " input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,"
                " cost_usd, judge_input_tokens, judge_output_tokens, judge_cost_usd,"
                " scores, error, created_at) VALUES ('live', :c, :s, :cached, :lat, 1, 1, 1,"
                " 0, 0, 0, 0, 0, 0, :scores, '', CURRENT_TIMESTAMP)"
            ),
            [
                {
                    "c": f"c{i:03d}",
                    "s": "error" if i % 10 == 0 else "ok",
                    "cached": i % 7 == 0,
                    "lat": ms,
                    "scores": json.dumps({"exact": {"value": 1.0}}),
                }
                for i, ms in enumerate(latencies)
            ],
        )
    # Aggregates from 0005 on, written as results were saved: the run's other
    # totals, without the histogram.
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO run_aggregates (run_id, shard, completed, ok, errors, cached,"
                " fresh, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,"
                " cost_usd, judge_input_tokens, judge_output_tokens, judge_cost_usd)"
                " VALUES ('live', 'w1', 300, 270, 30, :cached, :fresh, 300, 300, 0, 0, 0, 0,"
                " 0, 0)"
            ),
            {
                "cached": sum(i % 7 == 0 for i in range(300)),
                "fresh": sum(i % 10 != 0 and i % 7 != 0 for i in range(300)),
            },
        )
    upgrade(engine)
    fresh = [ms for i, ms in enumerate(latencies) if i % 10 != 0 and i % 7 != 0]
    indexes = {ix["name"] for ix in inspect(engine).get_indexes("results")}
    with make_session_factory(engine)() as s:
        shards = set(s.scalars(select(RunLatencyBucket.shard)))
        run = s.get_one(Run, "live")
        assert aggregates.read(s, run).latency == _histogram(fresh)
        assert not [p for p in aggregates.check(s, run)[2] if p.startswith("latency")]
        monkeypatch.setattr(report, "LIVE_EXACT_MAX", 100)
        summary = summarize(s, "live")
    engine.dispose()
    assert "ix_results_run_recent" not in indexes
    assert shards == {aggregates.REBUILD_SHARD}
    exact = np.percentile(fresh, [50, 95]).tolist()
    assert summary.latency_p50_ms is not None
    assert summary.latency_p95_ms is not None
    assert _close_enough(summary.latency_p50_ms, exact[0])
    assert _close_enough(summary.latency_p95_ms, exact[1])


def test_migration_0007_downgrade_restores_the_index(tmp_path: Path) -> None:
    from alembic import command

    from evalkit.db.session import alembic_config

    engine = make_engine(f"sqlite:///{tmp_path / 'd.db'}")
    upgrade(engine)
    command.downgrade(alembic_config(engine), "0006")
    names = {ix["name"] for ix in inspect(engine).get_indexes("results")}
    tables = set(inspect(engine).get_table_names())
    engine.dispose()
    assert "ix_results_run_recent" in names
    assert "run_latency_buckets" not in tables
