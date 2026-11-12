"""Per-turn rows for conversation cases: ``result_turns``.

A conversation case's result row holds the dialogue's outcome; each turn the
assistant or the simulated user generated gets a row here, written in the same
transaction as the result. Existing runs have no conversation cases.

Revision ID: 0012
Revises: 0011
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "result_turns",
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("case_id", sa.String(length=256), nullable=False),
        sa.Column("turn", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("stop_reason", sa.String(length=32), nullable=True),
        sa.Column("cached", sa.Boolean(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("cache_read_tokens", sa.Integer(), nullable=False),
        sa.Column("cache_write_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id", "case_id", "turn"),
    )


def downgrade() -> None:
    op.drop_table("result_turns")
