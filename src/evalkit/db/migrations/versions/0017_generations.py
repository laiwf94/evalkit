"""Synthetic generation and drafts: ``generations`` and ``case_drafts``.

A generation records its settings, generator identity, deduplication settings,
counts, failed calls, tokens and cost. Each generated case is a draft that
waits for review: pending, duplicate, approved (with the dataset version it
went into) or rejected.

Revision ID: 0017
Revises: 0016
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0017"
down_revision: str | None = "0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "generations",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("dataset", sa.String(length=128), nullable=False),
        sa.Column("spec", sa.JSON(), nullable=False),
        sa.Column("generator", sa.JSON(), nullable=False),
        sa.Column("dedupe", sa.JSON(), nullable=False),
        sa.Column("requested", sa.Integer(), nullable=False),
        sa.Column("generated", sa.Integer(), nullable=False),
        sa.Column("invalid", sa.Integer(), nullable=False),
        sa.Column("duplicates", sa.Integer(), nullable=False),
        sa.Column("errors", sa.JSON(), nullable=False),
        sa.Column("calls", sa.Integer(), nullable=False),
        sa.Column("cached_calls", sa.Integer(), nullable=False),
        sa.Column("input_tokens", sa.BigInteger(), nullable=False),
        sa.Column("output_tokens", sa.BigInteger(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_generations_dataset", "generations", ["dataset", "created_at"])
    op.create_table(
        "case_drafts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("generation_id", sa.String(length=32), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("case_id", sa.String(length=256), nullable=False),
        sa.Column("case", sa.JSON(), nullable=False),
        sa.Column("source", sa.String(length=512), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("duplicate_of", sa.String(length=256), nullable=True),
        sa.Column("similarity", sa.Float(), nullable=True),
        sa.Column("note", sa.Text(), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dataset_version", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["generation_id"], ["generations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("generation_id", "case_id", name="uq_case_drafts_case"),
    )
    op.create_index("ix_case_drafts_status", "case_drafts", ["generation_id", "status", "position"])


def downgrade() -> None:
    op.drop_index("ix_case_drafts_status", table_name="case_drafts")
    op.drop_table("case_drafts")
    op.drop_index("ix_generations_dataset", table_name="generations")
    op.drop_table("generations")
