"""The prompt registry: prompts, their immutable versions and movable labels.

A run whose target names a registry prompt records the reference and the
version it resolved to in ``runs.prompt``; existing runs have none.

Revision ID: 0010
Revises: 0009
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "prompts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "prompt_versions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "prompt_id",
            sa.Integer(),
            sa.ForeignKey("prompts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("system", sa.Text(), nullable=False),
        sa.Column("template", sa.Text(), nullable=False),
        sa.Column("note", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("prompt_id", "version", name="uq_prompt_versions_version"),
        sa.UniqueConstraint("prompt_id", "content_hash", name="uq_prompt_versions_hash"),
    )
    op.create_table(
        "prompt_labels",
        sa.Column(
            "prompt_id",
            sa.Integer(),
            sa.ForeignKey("prompts.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("label", sa.String(64), primary_key=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    # A plain ADD COLUMN on both dialects: a batch rebuild of ``runs`` on SQLite
    # would cascade-delete its results (see 0009).
    op.add_column("runs", sa.Column("prompt", sa.JSON(), nullable=True))


def downgrade() -> None:
    if op.get_context().dialect.name == "sqlite":
        op.execute("ALTER TABLE runs DROP COLUMN prompt")
    else:
        op.drop_column("runs", "prompt")
    op.drop_table("prompt_labels")
    op.drop_table("prompt_versions")
    op.drop_table("prompts")
