"""Annotation queues: ``annotation_queues``, ``annotation_tasks`` and
``annotation_assignments``.

A queue holds one task per item to label; each task has up to ``overlap``
assignment slots, each held by one annotator until it is submitted or its
claim expires. Unique (task, slot) caps the annotators per item and unique
(task, annotator) stops anyone labelling an item twice.

Revision ID: 0018
Revises: 0017
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0018"
down_revision: str | None = "0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "annotation_queues",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("spec_hash", sa.String(length=64), nullable=False),
        sa.Column("spec", sa.JSON(), nullable=False),
        sa.Column("dataset", sa.String(length=128), nullable=False),
        sa.Column("dataset_version", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=True),
        sa.Column("labels", sa.JSON(), nullable=False),
        sa.Column("level", sa.String(length=16), nullable=False),
        sa.Column("overlap", sa.Integer(), nullable=False),
        sa.Column("claim_seconds", sa.Integer(), nullable=False),
        sa.Column("task_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "annotation_tasks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("queue_id", sa.String(length=32), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("case_id", sa.String(length=256), nullable=False),
        sa.Column("item", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("label", sa.JSON(), nullable=True),
        sa.Column("adjudicator", sa.String(length=128), nullable=True),
        sa.Column("adjudicated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("note", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["queue_id"], ["annotation_queues.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("queue_id", "case_id", name="uq_annotation_tasks_case"),
    )
    op.create_index(
        "ix_annotation_tasks_status", "annotation_tasks", ["queue_id", "status", "position"]
    )
    op.create_table(
        "annotation_assignments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=False),
        sa.Column("queue_id", sa.String(length=32), nullable=False),
        sa.Column("slot", sa.Integer(), nullable=False),
        sa.Column("annotator", sa.String(length=128), nullable=False),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("label", sa.JSON(), nullable=True),
        sa.Column("note", sa.Text(), nullable=False),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["task_id"], ["annotation_tasks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("task_id", "slot", name="uq_annotation_assignments_slot"),
        sa.UniqueConstraint("task_id", "annotator", name="uq_annotation_assignments_annotator"),
    )
    op.create_index(
        "ix_annotation_assignments_annotator",
        "annotation_assignments",
        ["queue_id", "annotator"],
    )


def downgrade() -> None:
    op.drop_index("ix_annotation_assignments_annotator", table_name="annotation_assignments")
    op.drop_table("annotation_assignments")
    op.drop_index("ix_annotation_tasks_status", table_name="annotation_tasks")
    op.drop_table("annotation_tasks")
    op.drop_table("annotation_queues")
