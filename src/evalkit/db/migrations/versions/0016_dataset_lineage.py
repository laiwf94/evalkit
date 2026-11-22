"""Dataset lineage and per-case hashes, for case-level edits and diffs.

``datasets`` gains its parent version, its origin ("import" or "edit"), the
edit's note and how many cases it added, changed and removed relative to the
parent. ``dataset_cases`` gains each case's SHA-256 in canonical JSON, so two
versions can be compared, and runs on them paired, by case id and hash without
reading the cases.

Existing versions of a name are chained in version order as imports, and their
hashes and counts are computed here. Self-contained (no application models), so
later schema changes cannot break it.

Revision ID: 0016
Revises: 0015
"""

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op

revision: str = "0016"
down_revision: str | None = "0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _case_hash(case_id: str, values: Sequence[Any]) -> str:
    """``values`` is (input, expected, tags, split, metadata); the same canonical
    form as evalkit.datasets.Case.canonical."""
    input_, expected, tags, split, meta = values
    text = json.dumps(
        {
            "id": case_id,
            "input": input_,
            "expected": expected,
            "tags": list(tags or []),
            "split": split,
            "metadata": meta or {},
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(text.encode()).hexdigest()


def upgrade() -> None:
    # Plain ADD COLUMN on both databases: a batch rebuild of ``datasets`` on SQLite
    # would drop a table that runs and cases reference. ``parent_id`` has no
    # foreign key for the same reason (on SQLite a column with one cannot be
    # dropped again); versions are never deleted, and lineage stops at a parent
    # that is missing.
    op.add_column("datasets", sa.Column("parent_id", sa.Integer(), nullable=True))
    op.add_column(
        "datasets",
        sa.Column("origin", sa.String(length=16), nullable=False, server_default="import"),
    )
    op.add_column("datasets", sa.Column("note", sa.Text(), nullable=False, server_default=""))
    for name in ("cases_added", "cases_changed", "cases_removed"):
        op.add_column("datasets", sa.Column(name, sa.Integer(), nullable=True))
    op.add_column("dataset_cases", sa.Column("case_hash", sa.String(length=64), nullable=True))
    _backfill()
    # Nothing references dataset_cases, so rebuilding it on SQLite is safe.
    with op.batch_alter_table("dataset_cases") as batch:
        batch.alter_column("case_hash", existing_type=sa.String(length=64), nullable=False)


def _backfill() -> None:
    bind = op.get_bind()
    datasets = sa.table(
        "datasets",
        sa.column("id", sa.Integer),
        sa.column("name", sa.String),
        sa.column("version", sa.Integer),
        sa.column("parent_id", sa.Integer),
        sa.column("cases_added", sa.Integer),
        sa.column("cases_changed", sa.Integer),
        sa.column("cases_removed", sa.Integer),
    )
    cases = sa.table(
        "dataset_cases",
        sa.column("id", sa.Integer),
        sa.column("dataset_id", sa.Integer),
        sa.column("case_id", sa.String),
        sa.column("input", sa.JSON),
        sa.column("expected", sa.JSON),
        sa.column("tags", sa.JSON),
        sa.column("split", sa.String),
        sa.column("metadata", sa.JSON),
        sa.column("case_hash", sa.String),
    )
    previous: dict[str, tuple[int, dict[str, str]]] = {}
    versions = bind.execute(
        sa.select(datasets.c.id, datasets.c.name).order_by(datasets.c.name, datasets.c.version)
    ).all()
    for dataset_id, name in versions:
        hashes: dict[str, str] = {}
        updates = []
        rows = bind.execute(
            sa.select(
                cases.c.id,
                cases.c.case_id,
                cases.c.input,
                cases.c.expected,
                cases.c.tags,
                cases.c.split,
                cases.c.metadata,
            ).where(cases.c.dataset_id == dataset_id)
        ).all()
        for row_id, case_id, *values in rows:
            digest = _case_hash(case_id, values)
            hashes[case_id] = digest
            updates.append({"row_id": row_id, "digest": digest})
        if updates:
            bind.execute(
                cases.update()
                .where(cases.c.id == sa.bindparam("row_id"))
                .values(case_hash=sa.bindparam("digest")),
                updates,
            )
        if name in previous:
            parent_id, before = previous[name]
            changed = sum(1 for k, h in hashes.items() if k in before and before[k] != h)
            bind.execute(
                datasets.update()
                .where(datasets.c.id == dataset_id)
                .values(
                    parent_id=parent_id,
                    cases_added=sum(1 for k in hashes if k not in before),
                    cases_changed=changed,
                    cases_removed=sum(1 for k in before if k not in hashes),
                )
            )
        previous[name] = (dataset_id, hashes)


def downgrade() -> None:
    # Native DROP COLUMN (SQLite 3.35 and later, PostgreSQL), for the same reason.
    op.drop_column("dataset_cases", "case_hash")
    for name in ("cases_removed", "cases_changed", "cases_added", "note", "origin", "parent_id"):
        op.drop_column("datasets", name)
