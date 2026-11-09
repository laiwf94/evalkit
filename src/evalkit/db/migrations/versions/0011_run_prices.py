"""The prices each run is costed at: ``runs.prices``.

For every provider a run calls, the price table's version and the price of
each model the run uses (``evalkit.pricing.price_record``). Existing runs have
none and are costed at whatever prices the executing process has.

Revision ID: 0011
Revises: 0010
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # A plain ADD COLUMN on both dialects: a batch rebuild of ``runs`` on SQLite
    # would cascade-delete its results (see 0009).
    op.add_column("runs", sa.Column("prices", sa.JSON(), nullable=True))


def downgrade() -> None:
    if op.get_context().dialect.name == "sqlite":
        op.execute("ALTER TABLE runs DROP COLUMN prices")
    else:
        op.drop_column("runs", "prices")
