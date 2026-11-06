"""Latency histograms in the run aggregates.

A running run's latency percentiles used to come from its 5,000 most recent
results, which was most of the time of a summary of a large running run. Each
result write now also counts fresh successful results per logarithmic latency
bucket, in the writer's shard. The index that served the recent-results read
has no other reader, so it goes, which saves a write per result.

Revision ID: 0007
Revises: 0006
"""

import math
from collections import Counter
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

BACKFILL_SHARD = "rebuild"
# Frozen copies of the bucketing in ``evalkit.aggregates`` at this revision, so a
# later change there cannot change what this migration writes.
_LOG_GAMMA = math.log(1.02)
_FLOOR_MS = 1e-3
_CEIL_MS = 1e9


def _bucket(latency_ms: float) -> int:
    value = min(latency_ms, _CEIL_MS) if latency_ms > _FLOOR_MS else _FLOOR_MS
    return math.ceil(math.log(value) / _LOG_GAMMA)


def upgrade() -> None:
    op.create_table(
        "run_latency_buckets",
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("shard", sa.String(length=64), nullable=False),
        sa.Column("bucket", sa.Integer(), nullable=False),
        sa.Column("count", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id", "shard", "bucket"),
    )
    _backfill()
    op.drop_index("ix_results_run_recent", table_name="results")


def _backfill() -> None:
    """Histograms for runs that already have results, in the shard that migration
    0005 put their other aggregates in."""
    bind = op.get_bind()
    results = sa.table(
        "results",
        sa.column("run_id", sa.String),
        sa.column("status", sa.String),
        sa.column("cached", sa.Boolean),
        sa.column("latency_ms", sa.Float),
    )
    buckets = sa.table(
        "run_latency_buckets", *(sa.column(n) for n in ("run_id", "shard", "bucket", "count"))
    )
    run_ids = list(bind.scalars(sa.select(results.c.run_id).distinct()))
    for run_id in run_ids:
        counts: Counter[int] = Counter()
        rows = bind.execute(
            sa.select(results.c.latency_ms)
            .where(
                results.c.run_id == run_id,
                results.c.status == "ok",
                results.c.cached.is_(False),
            )
            .execution_options(yield_per=5000)
        )
        for (latency,) in rows:
            counts[_bucket(float(latency or 0.0))] += 1
        if counts:
            bind.execute(
                sa.insert(buckets),
                [
                    {"run_id": run_id, "shard": BACKFILL_SHARD, "bucket": b, "count": n}
                    for b, n in sorted(counts.items())
                ],
            )


def downgrade() -> None:
    op.create_index("ix_results_run_recent", "results", ["run_id", "id"])
    op.drop_table("run_latency_buckets")
