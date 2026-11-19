"""Embedding cache: ``embedding_cache``.

One row per (embedder, text): the vector as little-endian float32 bytes and its
dimension count. Swept by age with the response cache.

Revision ID: 0014
Revises: 0013
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "embedding_cache",
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("embedder_hash", sa.String(length=64), nullable=False),
        sa.Column("dims", sa.Integer(), nullable=False),
        sa.Column("vector", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("key"),
    )
    op.create_index(
        "ix_embedding_cache_embedder_hash", "embedding_cache", ["embedder_hash"], unique=False
    )
    op.create_index(
        "ix_embedding_cache_last_used", "embedding_cache", ["last_used_at", "key"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_embedding_cache_last_used", table_name="embedding_cache")
    op.drop_index("ix_embedding_cache_embedder_hash", table_name="embedding_cache")
    op.drop_table("embedding_cache")
