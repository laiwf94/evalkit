"""A bounded response cache: last use, size and blob references for each entry.

Revision ID: 0006
Revises: 0005
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("response_cache") as batch:
        batch.add_column(sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True))
        batch.add_column(
            sa.Column("size_bytes", sa.BigInteger(), nullable=False, server_default=sa.text("0"))
        )
        batch.add_column(sa.Column("output_blob", sa.String(length=64), nullable=True))
    # Existing entries count as last used when they were made, and their size is
    # their output in UTF-8 bytes (what a sweep by size adds up).
    octets = (
        "octet_length(output)"
        if op.get_bind().dialect.name == "postgresql"
        else "length(CAST(output AS BLOB))"
    )
    op.execute(f"UPDATE response_cache SET last_used_at = created_at, size_bytes = {octets}")  # noqa: S608
    with op.batch_alter_table("response_cache") as batch:
        batch.alter_column("last_used_at", existing_type=sa.DateTime(timezone=True), nullable=False)
    # A sweep deletes least recently used first; garbage collection asks which
    # blobs are still referenced.
    op.create_index("ix_response_cache_last_used", "response_cache", ["last_used_at", "key"])
    op.create_index("ix_response_cache_output_blob", "response_cache", ["output_blob"])


def downgrade() -> None:
    op.drop_index("ix_response_cache_output_blob", table_name="response_cache")
    op.drop_index("ix_response_cache_last_used", table_name="response_cache")
    with op.batch_alter_table("response_cache") as batch:
        batch.drop_column("output_blob")
        batch.drop_column("size_bytes")
        batch.drop_column("last_used_at")
