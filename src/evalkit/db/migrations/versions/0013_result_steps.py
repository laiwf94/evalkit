"""Agent trajectories: ``result_steps``.

An agent case's result row holds its final answer; each model reply and each
tool call of its trajectory gets a row here, written in the same transaction
as the result. Existing runs have no agent cases.

Revision ID: 0013
Revises: 0012
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "result_steps",
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("case_id", sa.String(length=256), nullable=False),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("stop_reason", sa.String(length=32), nullable=True),
        sa.Column("tool", sa.String(length=64), nullable=True),
        sa.Column("call_id", sa.String(length=128), nullable=True),
        sa.Column("arguments", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("source", sa.String(length=16), nullable=True),
        sa.Column("cached", sa.Boolean(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("cache_read_tokens", sa.Integer(), nullable=False),
        sa.Column("cache_write_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id", "case_id", "step"),
    )


def downgrade() -> None:
    op.drop_table("result_steps")
