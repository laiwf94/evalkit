"""Case-level dataset edits, version lineage, field-level diffs, and the pairing
of runs made on two versions of a dataset.

An edit names the version it was made against by content hash and lists cases
to add or replace (``upsert``), fields to change on existing cases
(``update``) and case ids to remove (``delete``). It becomes the next version,
with the edited version as its parent; cases keep their positions and new ones
are appended. Concurrency is optimistic: if the dataset has moved on since the
edited version, the edit is refused with a conflict that names the newest
version, unless the newest version is exactly what this edit produces (a retry
of an edit that was applied but whose reply was lost). Two edits racing from the
same version both try to insert the same version number; the unique constraint
lets one in and the other re-reads and gets the conflict.

Two versions are compared by case id and case hash (``dataset_cases.case_hash``)
without reading the cases; only the cases that changed are loaded for the
field-level diff. Runs on two versions of one dataset are paired on the cases
that are in both versions unchanged (``pair_versions``), so a comparison after
an edit measures the system on the same cases, not the edit.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from evalkit.datasets import (
    MAX_CASES,
    MAX_TAGS,
    Case,
    Label,
    case_hash,
    case_hashes,
    change_counts,
    content_hash,
    get_dataset,
    load_cases,
    new_rows,
    validate_name,
)
from evalkit.db.models import Dataset, DatasetCase, as_utc
from evalkit.errors import ConflictError, InvalidInputError, NotFoundError, PreconditionError

HASH_PATTERN = r"^[0-9a-f]{64}$"
CaseId = Annotated[str, Field(min_length=1, max_length=256, pattern=r"^[^\x00-\x1f\x7f]+$")]
EDITABLE = ("input", "expected", "tags", "split", "metadata")
# Diffs descend into objects this deep; below it a changed value is shown whole.
MAX_DIFF_DEPTH = 16
MAX_LINEAGE = 10_000
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class CaseUpdate(BaseModel):
    """New values for some fields of an existing case. A field that is absent is
    left as it is; ``expected`` and ``split`` may be set to null."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: CaseId
    input: JsonValue = None
    expected: JsonValue = None
    tags: tuple[Label, ...] | None = Field(default=None, max_length=MAX_TAGS)
    split: Label | None = None
    metadata: dict[str, JsonValue] | None = None

    @field_validator("tags")
    @classmethod
    def _unique_tags(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is not None and len(set(value)) != len(value):
            raise ValueError("tags must be unique")
        return value

    @model_validator(mode="after")
    def _changes_something(self) -> CaseUpdate:
        if not self.model_fields_set & set(EDITABLE):
            raise ValueError(f"update of {self.id!r} sets none of {', '.join(EDITABLE)}")
        for name in ("input", "tags", "metadata"):
            if name in self.model_fields_set and getattr(self, name) is None:
                raise ValueError(f"update of {self.id!r}: {name} cannot be null")
        return self

    def apply(self, case: Case) -> Case:
        values = case.model_dump()
        for name in EDITABLE:
            if name in self.model_fields_set:
                values[name] = getattr(self, name)
        return Case.model_validate(values)


class DatasetEdit(BaseModel):
    """The body of ``PATCH /v1/datasets/{name}`` and of ``evalkit dataset edit``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base: str = Field(pattern=HASH_PATTERN)
    upsert: tuple[Case, ...] = Field(default=(), max_length=MAX_CASES)
    update: tuple[CaseUpdate, ...] = Field(default=(), max_length=MAX_CASES)
    delete: tuple[CaseId, ...] = Field(default=(), max_length=MAX_CASES)
    note: str = Field(default="", max_length=2000)

    @field_validator("base", mode="before")
    @classmethod
    def _lower(cls, value: Any) -> Any:
        return value.lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _one_operation_per_case(self) -> DatasetEdit:
        if not (self.upsert or self.update or self.delete):
            raise ValueError("the edit has no upsert, update or delete")
        seen: dict[str, str] = {}
        for op, ids in (
            ("upsert", [c.id for c in self.upsert]),
            ("update", [u.id for u in self.update]),
            ("delete", list(self.delete)),
        ):
            for case_id in ids:
                if case_id in seen:
                    where = (
                        "twice in " + op if seen[case_id] == op else f"in {seen[case_id]} and {op}"
                    )
                    raise ValueError(f"case {case_id!r} appears {where}")
                seen[case_id] = op
        return self


@dataclass(frozen=True)
class EditResult:
    dataset: Dataset
    created: bool
    # Set when the edit's content is already a version (an older one, or the
    # newest when this edit was applied before): that version is returned.
    existing: bool = False


def find_by_hash(session: Session, name: str, digest: str) -> Dataset:
    found = session.scalar(
        select(Dataset).where(Dataset.name == name, Dataset.content_hash == digest.lower())
    )
    if found is None:
        raise NotFoundError(f"dataset {name!r} has no version with content hash {digest[:12]}")
    return found


def apply_cases(cases: Sequence[Case], edit: DatasetEdit) -> list[Case]:
    """The cases after ``edit``: positions kept, new cases appended in edit order."""
    by_id = {c.id: i for i, c in enumerate(cases)}
    out = list(cases)
    missing = [u.id for u in edit.update if u.id not in by_id] + [
        d for d in edit.delete if d not in by_id
    ]
    if missing:
        shown = ", ".join(repr(m) for m in missing[:5])
        more = f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""
        raise InvalidInputError(f"no case {shown}{more} in the edited version")
    appended: list[Case] = []
    for case in edit.upsert:
        if case.id in by_id:
            out[by_id[case.id]] = case
        else:
            appended.append(case)
    # A case id is in one operation only, so an updated case was not upserted.
    for change in edit.update:
        i = by_id[change.id]
        try:
            out[i] = change.apply(out[i])
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(p) for p in first["loc"]) or "case"
            raise InvalidInputError(f"update of {change.id!r}: {where}: {first['msg']}") from None
    deleted = set(edit.delete)
    result = [c for c in out if c.id not in deleted] + appended
    if not result:
        raise InvalidInputError("the edit removes every case; a dataset needs at least one")
    if len(result) > MAX_CASES:
        raise InvalidInputError(f"the edit leaves more than {MAX_CASES} cases")
    return result


def apply_edit(session: Session, name: str, edit: DatasetEdit) -> EditResult:
    """Store ``edit`` as the next version of ``name``; see the module docstring."""
    validate_name(name)
    base = find_by_hash(session, name, edit.base)
    cases = apply_cases(load_cases(session, base), edit)
    digest = content_hash(cases)
    hashes = [case_hash(c) for c in cases]
    for _ in range(3):
        latest = get_dataset(session, name)
        if latest.id != base.id:
            if latest.content_hash == digest:
                return EditResult(latest, created=False, existing=True)
            raise ConflictError(
                f"dataset {name!r} is at version {latest.version} "
                f"({latest.content_hash[:12]}), but the edit was made against version "
                f"{base.version} ({base.content_hash[:12]}); apply it to version "
                f"{latest.version} and send its hash as base"
            )
        if digest == base.content_hash:
            return EditResult(base, created=False)
        existing = session.scalar(
            select(Dataset).where(Dataset.name == name, Dataset.content_hash == digest)
        )
        if existing is not None:
            return EditResult(existing, created=False, existing=True)
        added, changed, removed = change_counts(
            case_hashes(session, base.id), dict(zip((c.id for c in cases), hashes, strict=True))
        )
        dataset = Dataset(
            name=name,
            version=base.version + 1,
            content_hash=digest,
            case_count=len(cases),
            description=base.description,
            parent_id=base.id,
            origin="edit",
            note=edit.note,
            cases_added=added,
            cases_changed=changed,
            cases_removed=removed,
        )
        dataset.cases = new_rows(cases, hashes)
        session.add(dataset)
        try:
            session.commit()
        except IntegrityError:
            # Another writer took this version number (or this content): look again.
            session.rollback()
            continue
        return EditResult(dataset, created=True)
    raise ConflictError(f"could not store the edit of dataset {name!r}; retry")


# Log and lineage --------------------------------------------------------------


@dataclass(frozen=True)
class VersionInfo:
    name: str
    version: int
    content_hash: str
    case_count: int
    origin: str
    parent_version: int | None
    note: str
    description: str
    created_at: datetime
    added: int | None
    changed: int | None
    removed: int | None


def version_info(ds: Dataset, parent_version: int | None) -> VersionInfo:
    return VersionInfo(
        name=ds.name,
        version=ds.version,
        content_hash=ds.content_hash,
        case_count=ds.case_count,
        origin=ds.origin,
        parent_version=parent_version,
        note=ds.note,
        description=ds.description,
        created_at=as_utc(ds.created_at),
        added=ds.cases_added,
        changed=ds.cases_changed,
        removed=ds.cases_removed,
    )


def dataset_log(session: Session, name: str) -> list[VersionInfo]:
    """Every version of ``name``, newest first, with its parent and changes."""
    rows = list(session.scalars(select(Dataset).where(Dataset.name == name)))
    if not rows:
        raise NotFoundError(f"dataset {name!r} not found")
    number = {ds.id: ds.version for ds in rows}
    rows.sort(key=lambda ds: ds.version, reverse=True)
    return [
        version_info(ds, None if ds.parent_id is None else number.get(ds.parent_id)) for ds in rows
    ]


def lineage(session: Session, name: str, version: int | None = None) -> list[VersionInfo]:
    """The version and its ancestors, newest first, back to a version with no parent."""
    ds: Dataset | None = get_dataset(session, name, version)
    chain: list[Dataset] = []
    seen: set[int] = set()
    while ds is not None and ds.id not in seen and len(chain) < MAX_LINEAGE:
        seen.add(ds.id)
        chain.append(ds)
        ds = None if ds.parent_id is None else session.get(Dataset, ds.parent_id)
    return [
        version_info(d, chain[i + 1].version if i + 1 < len(chain) else None)
        for i, d in enumerate(chain)
    ]


# Diffs ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldChange:
    """One changed field of a case. ``path`` is the field and, inside an object, the
    keys down to the value that changed (``input.question``, ``metadata["a b"]``).
    ``kind`` is "added" or "removed" for a key on one side only, else "changed"."""

    path: str
    kind: str
    before: Any = None
    after: Any = None


@dataclass(frozen=True)
class CaseChange:
    id: str
    changes: list[FieldChange]


@dataclass(frozen=True)
class DatasetDiff:
    name: str
    from_version: int
    to_version: int
    from_hash: str
    to_hash: str
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    changed: list[CaseChange] = field(default_factory=list)
    unchanged: int = 0


def _path(prefix: str, key: str) -> str:
    if _IDENTIFIER.fullmatch(key):
        return f"{prefix}.{key}"
    escaped = key.replace("\\", "\\\\").replace('"', '\\"')
    return f'{prefix}["{escaped}"]'


def diff_values(path: str, before: Any, after: Any, depth: int = 0) -> list[FieldChange]:
    """Field changes between two JSON values: objects key by key (recursively),
    anything else whole."""
    if before == after and type(before) is type(after):
        return []
    if isinstance(before, dict) and isinstance(after, dict) and depth < MAX_DIFF_DEPTH:
        out: list[FieldChange] = []
        for key in sorted(before.keys() | after.keys()):
            sub = _path(path, key)
            if key not in after:
                out.append(FieldChange(sub, "removed", before=before[key]))
            elif key not in before:
                out.append(FieldChange(sub, "added", after=after[key]))
            else:
                out.extend(diff_values(sub, before[key], after[key], depth + 1))
        return out
    return [FieldChange(path, "changed", before=before, after=after)]


def diff_cases(before: Case, after: Case) -> list[FieldChange]:
    a, b = before.model_dump(mode="json"), after.model_dump(mode="json")
    out: list[FieldChange] = []
    for name in EDITABLE:
        out.extend(diff_values(name, a[name], b[name]))
    return out


def _cases_by_id(session: Session, dataset: Dataset, ids: set[str]) -> dict[str, Case]:
    if not ids:
        return {}
    out: dict[str, Case] = {}
    wanted = sorted(ids)
    for start in range(0, len(wanted), 500):
        rows = session.scalars(
            select(DatasetCase).where(
                DatasetCase.dataset_id == dataset.id,
                DatasetCase.case_id.in_(wanted[start : start + 500]),
            )
        )
        for row in rows:
            out[row.case_id] = Case(
                id=row.case_id,
                input=row.input,
                expected=row.expected,
                tags=tuple(row.tags),
                split=row.split,
                metadata=row.meta,
            )
    return out


def diff_versions(
    session: Session, name: str, from_version: int, to_version: int | None = None
) -> DatasetDiff:
    """What changed from one version to another, case by case and field by field.
    Added and removed ids, and changed cases, are in case-id order."""
    old = get_dataset(session, name, from_version)
    new = get_dataset(session, name, to_version)
    before, after = case_hashes(session, old.id), case_hashes(session, new.id)
    changed_ids = {k for k, h in after.items() if k in before and before[k] != h}
    old_cases = _cases_by_id(session, old, changed_ids)
    new_cases = _cases_by_id(session, new, changed_ids)
    return DatasetDiff(
        name=name,
        from_version=old.version,
        to_version=new.version,
        from_hash=old.content_hash,
        to_hash=new.content_hash,
        added=sorted(k for k in after if k not in before),
        removed=sorted(k for k in before if k not in after),
        changed=[
            CaseChange(k, diff_cases(old_cases[k], new_cases[k])) for k in sorted(changed_ids)
        ],
        unchanged=sum(1 for k, h in after.items() if before.get(k) == h),
    )


# Pairing runs across versions -------------------------------------------------

IndexArray = npt.NDArray[np.intp]


@dataclass(frozen=True)
class VersionPairing:
    """How two versions of a dataset line up for a paired comparison: ``shared``
    cases are in both unchanged, ``changed`` are in both with different content,
    and the rest are in one version only. Only the shared cases are compared."""

    dataset: str
    baseline_version: int
    candidate_version: int
    shared: int
    changed: int
    only_baseline: int
    only_candidate: int


def pair_versions(
    session: Session, baseline: Dataset, candidate: Dataset
) -> tuple[VersionPairing, IndexArray, IndexArray]:
    """Pair two versions of one dataset on their unchanged cases. Score arrays of
    runs on the versions are in case-id order over each version; the two index
    arrays returned pick, from each, the shared cases in the same order. Refuses
    versions of different datasets and versions that share no unchanged case."""
    if baseline.name != candidate.name:
        raise PreconditionError(
            f"the runs used different datasets ({baseline.name!r} and {candidate.name!r}); "
            "results cannot be paired"
        )
    before = case_hashes(session, baseline.id)
    after = case_hashes(session, candidate.id)
    base_pos = {k: i for i, k in enumerate(sorted(before))}
    cand_pos = {k: i for i, k in enumerate(sorted(after))}
    shared = sorted(k for k, h in after.items() if before.get(k) == h)
    if not shared:
        raise PreconditionError(
            f"versions {baseline.version} and {candidate.version} of dataset "
            f"{baseline.name!r} have no case in common unchanged; results cannot be paired"
        )
    added, changed, removed = change_counts(before, after)
    pairing = VersionPairing(
        dataset=baseline.name,
        baseline_version=baseline.version,
        candidate_version=candidate.version,
        shared=len(shared),
        changed=changed,
        only_baseline=removed,
        only_candidate=added,
    )
    return (
        pairing,
        np.array([base_pos[k] for k in shared], dtype=np.intp),
        np.array([cand_pos[k] for k in shared], dtype=np.intp),
    )
