"""Model matrices: a group table and the link from each child run.

A ``[matrix]`` config expands into child runs; ``matrices`` records the group
(its axes and base config) and each child run points at it, with the variant
it executes. Existing runs belong to no matrix.

Revision ID: 0009
Revises: 0008
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "matrices",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("label", sa.String(128), nullable=False),
        sa.Column(
            "dataset_id",
            sa.Integer(),
            sa.ForeignKey("datasets.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("axes", sa.JSON(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_matrices_label", "matrices", ["label", "created_at"])
    # On SQLite, a plain ADD COLUMN with the reference inline. Adding the
    # constraint in batch mode would rebuild ``runs``, and dropping the old table
    # cascades to every result and aggregate row.
    if op.get_context().dialect.name == "sqlite":
        op.execute("ALTER TABLE runs ADD COLUMN matrix_id VARCHAR(32) REFERENCES matrices (id)")
    else:
        op.add_column("runs", sa.Column("matrix_id", sa.String(32), nullable=True))
        op.create_foreign_key("fk_runs_matrix_id", "runs", "matrices", ["matrix_id"], ["id"])
    op.add_column("runs", sa.Column("variant", sa.JSON(), nullable=True))
    op.create_index("ix_runs_matrix", "runs", ["matrix_id"])


def downgrade() -> None:
    op.drop_index("ix_runs_matrix", table_name="runs")
    if op.get_context().dialect.name == "sqlite":
        # SQLite 3.35+ drops a column in place, which keeps the rows that reference runs;
        # the column's reference has no name to drop first.
        op.execute("ALTER TABLE runs DROP COLUMN variant")
        op.execute("ALTER TABLE runs DROP COLUMN matrix_id")
    else:
        op.drop_constraint("fk_runs_matrix_id", "runs", type_="foreignkey")
        op.drop_column("runs", "variant")
        op.drop_column("runs", "matrix_id")
    op.drop_index("ix_matrices_label", table_name="matrices")
    op.drop_table("matrices")
