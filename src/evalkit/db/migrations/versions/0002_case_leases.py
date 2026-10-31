"""Case leases, so several workers can execute one run.

Revision ID: 0002
Revises: 0001
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.add_column(
            sa.Column("leases_seeded", sa.Boolean(), nullable=False, server_default=sa.false())
        )
    op.create_table(
        "case_leases",
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("case_id", sa.String(length=256), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("owner", sa.String(length=64), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fence", sa.BigInteger(), nullable=False),
        sa.Column("done", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id", "case_id"),
    )
    op.create_index("ix_case_leases_claimable", "case_leases", ["run_id", "done", "position"])


def downgrade() -> None:
    op.drop_index("ix_case_leases_claimable", table_name="case_leases")
    op.drop_table("case_leases")
    with op.batch_alter_table("runs") as batch:
        batch.drop_column("leases_seeded")
