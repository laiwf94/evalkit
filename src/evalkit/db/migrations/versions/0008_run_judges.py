"""The identity of each judge a run uses, recorded when the run is created.

Judges can now run on any provider, so a gate has to know which provider,
endpoint, model, effort and rubric graded each run. Existing runs keep NULL:
their judges were all Claude judges, and their identity is derived from their
scorer specs when they are compared.

Revision ID: 0008
Revises: 0007
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.add_column(sa.Column("judges", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("runs") as batch:
        batch.drop_column("judges")
