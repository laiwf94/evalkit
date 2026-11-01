"""A run queue for worker daemons: priorities, per-run worker caps and worker slots.

Revision ID: 0004
Revises: 0003
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.add_column(sa.Column("queued_at", sa.DateTime(timezone=True), nullable=True))
        batch.add_column(
            sa.Column("priority", sa.Integer(), nullable=False, server_default=sa.text("0"))
        )
        batch.add_column(sa.Column("max_workers", sa.Integer(), nullable=True))
    # Workers look for runs by status, highest priority first, oldest first.
    op.create_index("ix_runs_queue", "runs", ["status", "priority", "queued_at"])
    op.create_table(
        "run_workers",
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("worker_id", sa.String(length=64), nullable=False),
        sa.Column("joined_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id", "worker_id"),
    )


def downgrade() -> None:
    op.drop_table("run_workers")
    op.drop_index("ix_runs_queue", table_name="runs")
    with op.batch_alter_table("runs") as batch:
        batch.drop_column("max_workers")
        batch.drop_column("priority")
        batch.drop_column("queued_at")
