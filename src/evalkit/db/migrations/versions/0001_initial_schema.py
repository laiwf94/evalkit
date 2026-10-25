"""Initial schema.

Revision ID: 0001
Revises:
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("case_count", sa.Integer(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "content_hash", name="uq_datasets_name_hash"),
        sa.UniqueConstraint("name", "version", name="uq_datasets_name_version"),
    )
    op.create_table(
        "judge_calibrations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("judge_hash", sa.String(length=64), nullable=False),
        sa.Column("judge_spec", sa.JSON(), nullable=False),
        sa.Column("n", sa.Integer(), nullable=False),
        sa.Column("kappa", sa.Float(), nullable=False),
        sa.Column("weighted_kappa", sa.Float(), nullable=True),
        sa.Column("agreement", sa.Float(), nullable=False),
        sa.Column("confusion", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_judge_calibrations_judge_hash", "judge_calibrations", ["judge_hash"])

    op.create_table(
        "response_cache",
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("target_hash", sa.String(length=64), nullable=False),
        sa.Column("output", sa.Text(), nullable=False),
        sa.Column("stop_reason", sa.String(length=32), nullable=True),
        sa.Column("usage", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("key"),
    )
    op.create_index("ix_response_cache_target_hash", "response_cache", ["target_hash"])

    op.create_table(
        "dataset_cases",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("case_id", sa.String(length=256), nullable=False),
        sa.Column("input", sa.JSON(), nullable=False),
        sa.Column("expected", sa.JSON(), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("split", sa.String(length=64), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("dataset_id", "case_id", name="uq_dataset_cases_case"),
    )
    op.create_index("ix_dataset_cases_position", "dataset_cases", ["dataset_id", "position"])

    op.create_table(
        "runs",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("label", sa.String(length=128), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("target_spec", sa.JSON(), nullable=False),
        sa.Column("target_hash", sa.String(length=64), nullable=False),
        sa.Column("scorers", sa.JSON(), nullable=False),
        sa.Column("options", sa.JSON(), nullable=False),
        sa.Column("case_filter", sa.JSON(), nullable=False),
        sa.Column("seed", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("error", sa.Text(), nullable=False),
        sa.Column("batch_id", sa.String(length=128), nullable=True),
        sa.Column("lease_owner", sa.String(length=64), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_runs_created_at", "runs", ["created_at"])

    op.create_table(
        "results",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("case_id", sa.String(length=256), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("output", sa.Text(), nullable=True),
        sa.Column("stop_reason", sa.String(length=32), nullable=True),
        sa.Column("cached", sa.Boolean(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("cache_read_tokens", sa.Integer(), nullable=False),
        sa.Column("cache_write_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.Column("judge_input_tokens", sa.Integer(), nullable=False),
        sa.Column("judge_output_tokens", sa.Integer(), nullable=False),
        sa.Column("judge_cost_usd", sa.Float(), nullable=False),
        sa.Column("scores", sa.JSON(), nullable=False),
        sa.Column("error", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "case_id", name="uq_results_run_case"),
    )


def downgrade() -> None:
    op.drop_table("results")
    op.drop_index("ix_runs_created_at", table_name="runs")

    op.drop_table("runs")
    op.drop_index("ix_dataset_cases_position", table_name="dataset_cases")

    op.drop_table("dataset_cases")
    op.drop_index("ix_response_cache_target_hash", table_name="response_cache")

    op.drop_table("response_cache")
    op.drop_index("ix_judge_calibrations_judge_hash", table_name="judge_calibrations")

    op.drop_table("judge_calibrations")
    op.drop_table("datasets")
