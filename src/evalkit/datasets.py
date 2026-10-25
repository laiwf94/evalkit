"""Test-case datasets: strict JSONL parsing, content hashing and versioned storage.

A dataset's identity is the SHA-256 of its cases in canonical JSON, sorted by
case id. Importing the same content under the same name returns the existing
version, so CI jobs can import on every run without creating new versions.
Line order is kept for export but is not part of the identity: reordering a
file does not change what is being evaluated.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from evalkit.db.models import Dataset, DatasetCase
from evalkit.errors import InvalidInputError, NotFoundError

MAX_LINE_BYTES = 1024 * 1024
MAX_CASES = 100_000
MAX_TAGS = 32
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

Label = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[^\x00-\x1f\x7f]+$")]


class Case(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=256, pattern=r"^[^\x00-\x1f\x7f]+$")
    input: JsonValue
    expected: JsonValue = None
    tags: tuple[Label, ...] = Field(default=(), max_length=MAX_TAGS)
    split: Label | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("input")
    @classmethod
    def _input_present(cls, value: JsonValue) -> JsonValue:
        if value is None or value in ("", {}):
            raise ValueError("input must not be empty")
        return value

    @field_validator("tags")
    @classmethod
    def _unique_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("tags must be unique")
        return value

    def canonical(self) -> str:
        return canonical_json(self.model_dump(mode="json"))


def canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, no NaN."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def content_hash(cases: Iterable[Case]) -> str:
    lines = sorted(c.canonical() for c in cases)
    return sha256_hex("\n".join(lines))


def parse_jsonl(data: bytes | str, max_cases: int = MAX_CASES) -> list[Case]:
    """Parse JSONL into cases. Errors name the 1-based line."""
    text = data.decode("utf-8") if isinstance(data, bytes) else data
    cases: list[Case] = []
    seen: dict[str, int] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if len(line.encode()) > MAX_LINE_BYTES:
            raise InvalidInputError(f"line {number}: longer than {MAX_LINE_BYTES} bytes")
        if len(cases) >= max_cases:
            raise InvalidInputError(f"line {number}: more than {max_cases} cases")
        try:
            raw = json.loads(line, parse_constant=_reject_constant)
        except (ValueError, RecursionError) as exc:
            raise InvalidInputError(f"line {number}: not valid JSON ({exc})") from None
        if not isinstance(raw, dict):
            raise InvalidInputError(f"line {number}: each line must be a JSON object")
        try:
            case = Case.model_validate(raw)
        except ValidationError as exc:
            raise InvalidInputError(f"line {number}: {_describe(exc)}") from None
        if case.id in seen:
            raise InvalidInputError(
                f"line {number}: duplicate id {case.id!r} (first on line {seen[case.id]})"
            )
        seen[case.id] = number
        cases.append(case)
    if not cases:
        raise InvalidInputError("dataset has no cases")
    return cases


def to_jsonl(cases: Iterable[Case]) -> str:
    return "".join(c.canonical() + "\n" for c in cases)


def _reject_constant(name: str) -> None:
    raise ValueError(f"{name} is not valid JSON")


def _describe(exc: ValidationError) -> str:
    first = exc.errors()[0]
    where = ".".join(str(p) for p in first["loc"]) or "case"
    more = f" (and {exc.error_count() - 1} more)" if exc.error_count() > 1 else ""
    return f"{where}: {first['msg']}{more}"


def validate_name(name: str) -> str:
    if not NAME_PATTERN.fullmatch(name):
        raise InvalidInputError(
            "dataset name must be 1-128 characters of letters, digits, '.', '_' or '-', "
            "starting with a letter or digit"
        )
    return name


def import_dataset(
    session: Session, name: str, cases: Sequence[Case], description: str = ""
) -> tuple[Dataset, bool]:
    """Store ``cases`` as the next version of ``name``. Returns (dataset, created)."""
    validate_name(name)
    if len(description) > 2000:
        raise InvalidInputError("description is longer than 2000 characters")
    digest = content_hash(cases)
    for _ in range(3):
        existing = session.scalar(
            select(Dataset).where(Dataset.name == name, Dataset.content_hash == digest)
        )
        if existing is not None:
            return existing, False
        latest = session.scalar(select(func.max(Dataset.version)).where(Dataset.name == name))
        dataset = Dataset(
            name=name,
            version=(latest or 0) + 1,
            content_hash=digest,
            case_count=len(cases),
            description=description,
        )
        dataset.cases = [
            DatasetCase(
                position=i,
                case_id=c.id,
                input=c.input,
                expected=c.expected,
                tags=list(c.tags),
                split=c.split,
                meta=c.metadata,
            )
            for i, c in enumerate(cases)
        ]
        session.add(dataset)
        try:
            session.commit()
        except IntegrityError:
            # Another importer took this version number or this content; look again.
            session.rollback()
            continue
        return dataset, True
    raise InvalidInputError(f"could not allocate a version for dataset {name!r}; retry")


def get_dataset(session: Session, name: str, version: int | None = None) -> Dataset:
    query = select(Dataset).where(Dataset.name == name)
    if version is None:
        query = query.order_by(Dataset.version.desc()).limit(1)
    else:
        query = query.where(Dataset.version == version)
    dataset = session.scalar(query)
    if dataset is None:
        suffix = "" if version is None else f" version {version}"
        raise NotFoundError(f"dataset {name!r}{suffix} not found")
    return dataset


def list_datasets(session: Session) -> list[Dataset]:
    return list(session.scalars(select(Dataset).order_by(Dataset.name, Dataset.version)))


def load_cases(
    session: Session,
    dataset: Dataset,
    split: str | None = None,
    tags: Sequence[str] = (),
) -> list[Case]:
    """Cases in import order. ``tags`` selects cases carrying every listed tag."""
    rows = session.scalars(
        select(DatasetCase)
        .where(DatasetCase.dataset_id == dataset.id)
        .order_by(DatasetCase.position)
    )
    wanted = set(tags)
    cases = []
    for row in rows:
        if split is not None and row.split != split:
            continue
        if wanted and not wanted.issubset(row.tags):
            continue
        cases.append(
            Case(
                id=row.case_id,
                input=row.input,
                expected=row.expected,
                tags=tuple(row.tags),
                split=row.split,
                metadata=row.meta,
            )
        )
    return cases
