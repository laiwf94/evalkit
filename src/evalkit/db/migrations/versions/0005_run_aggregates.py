"""Per-run aggregates written with each result, and compact per-case score arrays.

Revision ID: 0005
Revises: 0004
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The shard that holds aggregates computed from results that existed before this
# migration (and from ``evalkit run verify --repair``).
BACKFILL_SHARD = "rebuild"


def upgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.add_column(sa.Column("cases_total", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("latency_p50_ms", sa.Float(), nullable=True))
        batch.add_column(sa.Column("latency_p95_ms", sa.Float(), nullable=True))
    # A running run's latency percentiles come from its most recent results.
    op.create_index("ix_results_run_recent", "results", ["run_id", "id"])
    op.create_table(
        "run_aggregates",
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("shard", sa.String(length=64), nullable=False),
        sa.Column("completed", sa.BigInteger(), nullable=False),
        sa.Column("ok", sa.BigInteger(), nullable=False),
        sa.Column("errors", sa.BigInteger(), nullable=False),
        sa.Column("cached", sa.BigInteger(), nullable=False),
        sa.Column("fresh", sa.BigInteger(), nullable=False),
        sa.Column("input_tokens", sa.BigInteger(), nullable=False),
        sa.Column("output_tokens", sa.BigInteger(), nullable=False),
        sa.Column("cache_read_tokens", sa.BigInteger(), nullable=False),
        sa.Column("cache_write_tokens", sa.BigInteger(), nullable=False),
        sa.Column("cost_usd", sa.Double(), nullable=False),
        sa.Column("judge_input_tokens", sa.BigInteger(), nullable=False),
        sa.Column("judge_output_tokens", sa.BigInteger(), nullable=False),
        sa.Column("judge_cost_usd", sa.Double(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id", "shard"),
    )
    op.create_table(
        "run_metric_aggregates",
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("shard", sa.String(length=64), nullable=False),
        sa.Column("metric", sa.String(length=128), nullable=False),
        sa.Column("n", sa.BigInteger(), nullable=False),
        sa.Column("total", sa.Double(), nullable=False),
        sa.Column("total_sq", sa.Double(), nullable=False),
        sa.Column("unscorable", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id", "shard", "metric"),
    )
    op.create_table(
        "run_scores",
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("metric", sa.String(length=128), nullable=False),
        sa.Column("case_count", sa.Integer(), nullable=False),
        sa.Column("n", sa.Integer(), nullable=False),
        sa.Column("unscorable", sa.Integer(), nullable=False),
        sa.Column("mean", sa.Double(), nullable=True),
        sa.Column("ci_low", sa.Double(), nullable=True),
        sa.Column("ci_high", sa.Double(), nullable=True),
        sa.Column("scores", sa.LargeBinary(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id", "metric"),
    )
    _backfill()


def _backfill() -> None:
    """Aggregates for runs that already have results, in one pass over each run.
    Self-contained (no application models), so later schema changes cannot break it."""
    bind = op.get_bind()
    runs = sa.table("runs", sa.column("id", sa.String), sa.column("scorers", sa.JSON))
    results = sa.table(
        "results",
        sa.column("run_id", sa.String),
        sa.column("status", sa.String),
        sa.column("cached", sa.Boolean),
        sa.column("scores", sa.JSON),
        *(sa.column(name, sa.Float) for name in _SUMMED),
    )
    with_results = sa.select(results.c.run_id).distinct()
    for run_id, scorers in bind.execute(
        sa.select(runs.c.id, runs.c.scorers).where(runs.c.id.in_(with_results))
    ):
        names = [s.get("name") or s["type"] for s in scorers]
        counts = dict.fromkeys(("completed", "ok", "errors", "cached", "fresh"), 0)
        sums = dict.fromkeys(_SUMMED, 0.0)
        per_metric = {name: [0, 0.0, 0.0, 0] for name in names}
        rows = bind.execute(
            sa.select(results.c.status, results.c.cached, results.c.scores, *_columns(results))
            .where(results.c.run_id == run_id)
            .execution_options(yield_per=5000)
        )
        for status, cached, scores, *values in rows:
            ok = status == "ok"
            counts["completed"] += 1
            counts["ok" if ok else "errors"] += 1
            counts["cached"] += bool(cached)
            counts["fresh"] += ok and not cached
            for name, value in zip(_SUMMED, values, strict=True):
                sums[name] += value or 0
            for name in names:
                score = 0.0 if not ok else (scores.get(name) or {}).get("value")
                agg = per_metric[name]
                if score is None:
                    agg[3] += 1
                else:
                    agg[0] += 1
                    agg[1] += float(score)
                    agg[2] += float(score) ** 2
        totals = {
            name: (value if name.endswith("_usd") else round(value)) for name, value in sums.items()
        }
        aggregates = sa.table(
            "run_aggregates",
            *(sa.column(name) for name in ("run_id", "shard", *counts, *_SUMMED)),
        )
        bind.execute(
            sa.insert(aggregates),
            [{"run_id": run_id, "shard": BACKFILL_SHARD, **counts, **totals}],
        )
        metric_rows = [
            {
                "run_id": run_id,
                "shard": BACKFILL_SHARD,
                "metric": name,
                "n": n,
                "total": total,
                "total_sq": total_sq,
                "unscorable": unscorable,
            }
            for name, (n, total, total_sq, unscorable) in per_metric.items()
        ]
        if metric_rows:
            metric_table = sa.table(
                "run_metric_aggregates", *(sa.column(name) for name in metric_rows[0])
            )
            bind.execute(sa.insert(metric_table), metric_rows)


_SUMMED = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "cost_usd",
    "judge_input_tokens",
    "judge_output_tokens",
    "judge_cost_usd",
)


def _columns(table: sa.TableClause) -> list[sa.ColumnClause[float]]:
    return [table.c[name] for name in _SUMMED]


def downgrade() -> None:
    op.drop_table("run_scores")
    op.drop_table("run_metric_aggregates")
    op.drop_table("run_aggregates")
    op.drop_index("ix_results_run_recent", table_name="results")
    with op.batch_alter_table("runs") as batch:
        batch.drop_column("latency_p95_ms")
        batch.drop_column("latency_p50_ms")
        batch.drop_column("cases_total")
