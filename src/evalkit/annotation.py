"""Annotation queues: people label items, several per item, and the labels they
agree on (or an adjudicator settles) become data a judge can be calibrated on.

A queue is created from a dataset version, or from a run, in which case each
item also shows the run's output for its case. Creating it snapshots the items
(input, expected value and output) into one task each, so later dataset edits
or new runs never change what was labelled. Creating the same queue again
returns it; the same name with different settings is a conflict.

Each task has ``overlap`` slots. ``claim_next`` gives an annotator the first
open task, in dataset order, that they have not labelled and that has a free
slot: a slot never taken, or one whose claim expired without a label. A claim
lasts ``claim_timeout``; asking again returns the same task and extends the
claim, so a lost reply costs nothing; that holds for a lapsed claim too, as
long as nobody has taken the slot over. ``submit`` records the label
for a task the annotator holds (a lapsed claim still counts if nobody took the
slot over) and may change it until the task is adjudicated. ``skip`` gives a claimed task
back: the slot is free for someone else and the task is never offered to that
annotator again (the assignment row stays, with a negative slot, as the record
of the skip, which counts towards no task's overlap). When every slot
has a label the task is ``agreed`` if they are all the same, and ``disputed``
otherwise; ``adjudicate`` gives a disputed task its final label.

Concurrency. Unique (task, slot) caps each task at ``overlap`` annotators and
unique (task, annotator) stops anyone labelling a task twice, whatever the
interleaving. Every write to a task starts by updating the task's row: on
PostgreSQL that takes the row lock, and on SQLite it takes the database write
lock before anything is read, so a submit that completes a task always sees
every other submitted label and the status it sets is the right one. A claim
that loses a race for a slot moves on to the next slot or task.

Agreement is Krippendorff's alpha (``evalkit.agreement``) over every submitted
label, at the queue's level of measurement, alongside how often each annotator
matched the final labels.
"""

from __future__ import annotations

import math
import re
import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import ColumnElement, and_, func, insert, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from evalkit.agreement import AlphaResult, Level, coincidences, krippendorff_alpha
from evalkit.config import parse_duration
from evalkit.datasets import (
    MAX_CASES,
    NAME_PATTERN,
    Label,
    canonical_json,
    get_dataset,
    load_cases,
    sha256_hex,
)
from evalkit.db.models import (
    AnnotationAssignment,
    AnnotationQueue,
    AnnotationTask,
    Dataset,
    Result,
    Run,
    as_utc,
    utcnow,
)
from evalkit.errors import ConflictError, InvalidInputError, NotFoundError

STATUSES = ("open", "agreed", "disputed", "adjudicated")
MAX_LABELS = 50
MAX_OVERLAP = 10
MAX_NOTE = 2000
MIN_CLAIM_SECONDS = 60
MAX_CLAIM_SECONDS = 7 * 86_400
ANNOTATOR_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@+-]{0,127}$")
# Candidate tasks read per query while looking for a free slot.
CANDIDATES = 32
INSERT_CHUNK = 1000


class QueueSpec(BaseModel):
    """An ``[annotate]`` table, or the body of ``POST /v1/annotation-queues``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=128)
    # Exactly one of dataset (with an optional version, the newest by default)
    # and run (whose dataset version and outputs the items come from).
    dataset: str | None = Field(default=None, max_length=128)
    version: int | None = Field(default=None, ge=1)
    run: str | None = Field(default=None, min_length=1, max_length=32)
    labels: tuple[Any, ...] = Field(min_length=2, max_length=MAX_LABELS)
    level: Level = "nominal"
    overlap: int = Field(default=3, ge=1, le=MAX_OVERLAP)
    claim_timeout: str = "30m"
    instructions: str = Field(default="", max_length=10_000)
    # Which cases become items: any of these tags, this split, the first ``limit``.
    tags: tuple[Label, ...] = Field(default=(), max_length=32)
    split: Label | None = None
    limit: int | None = Field(default=None, ge=1, le=MAX_CASES)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        if not NAME_PATTERN.fullmatch(value):
            raise ValueError(
                "must be letters, digits, '.', '_' or '-', starting with one of the first two"
            )
        return value

    @field_validator("labels")
    @classmethod
    def _labels(cls, value: tuple[Any, ...]) -> tuple[Any, ...]:
        for label in value:
            if isinstance(label, str):
                if not 1 <= len(label) <= 64 or any(ord(c) < 32 or ord(c) == 127 for c in label):
                    raise ValueError("a label is 1-64 characters with no control characters")
            elif not _is_number(label):
                raise ValueError(f"a label is a string or a finite number, not {label!r}")
        keys = [_key(v) for v in value]
        if len(set(keys)) != len(keys):
            raise ValueError("labels must be unique")
        return value

    @field_validator("claim_timeout")
    @classmethod
    def _timeout(cls, value: str) -> str:
        seconds = parse_duration(value)
        if not MIN_CLAIM_SECONDS <= seconds <= MAX_CLAIM_SECONDS:
            raise ValueError("must be between 1m and 7d")
        return value

    @model_validator(mode="after")
    def _check(self) -> QueueSpec:
        if (self.dataset is None) == (self.run is None):
            raise ValueError("give either dataset or run")
        if self.run is not None and self.version is not None:
            raise ValueError("version comes from the run; give it only with dataset")
        if self.level in ("interval", "ratio"):
            if not all(_is_number(v) for v in self.labels):
                raise ValueError(f"{self.level} labels must be numbers")
            if self.level == "ratio" and any(v < 0 for v in self.labels):
                raise ValueError("ratio labels must not be negative")
        return self

    @property
    def claim_seconds(self) -> int:
        return int(parse_duration(self.claim_timeout))


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _key(value: Any) -> tuple[str, Any]:
    """Labels compare by kind and value: 1 and 1.0 are one label, "1" another."""
    return ("n", float(value)) if _is_number(value) else ("s", value)


def resolve_label(labels: Sequence[Any], value: Any) -> Any:
    """The queue's own label for ``value``. A string also matches a numeric label
    it spells (``"3"`` for 3), so labels typed on a command line work."""
    by_key = {_key(v): v for v in labels}
    if not isinstance(value, bool) and (_is_number(value) or isinstance(value, str)):
        found = by_key.get(_key(value))
        if found is not None:
            return found
    if isinstance(value, str):
        for label in labels:
            if _is_number(label) and str(label) == value.strip():
                return label
        try:
            number = float(value)
        except ValueError:
            number = math.nan
        if math.isfinite(number) and ("n", number) in by_key:
            return by_key[("n", number)]
    shown = ", ".join(map(str, labels))
    raise InvalidInputError(f"label {value!r} is not one of: {shown}")


def validate_annotator(name: str, what: str = "annotator") -> str:
    if not ANNOTATOR_PATTERN.fullmatch(name):
        raise InvalidInputError(
            f"{what} must be 1-128 characters of letters, digits, '.', '_', '@', '+' or '-'"
        )
    return name


def _note(note: str) -> str:
    if len(note) > MAX_NOTE:
        raise InvalidInputError(f"note is longer than {MAX_NOTE} characters")
    return note


# Creating -----------------------------------------------------------------------


@dataclass(frozen=True)
class QueueSummary:
    id: str
    name: str
    dataset: str
    dataset_version: int
    run: str | None
    labels: list[Any]
    level: str
    overlap: int
    claim_timeout: str
    instructions: str
    tasks: int
    created_at: datetime
    statuses: dict[str, int]


def _source(session: Session, spec: QueueSpec) -> tuple[Dataset, Run | None]:
    if spec.dataset is not None:
        return get_dataset(session, spec.dataset, spec.version), None
    run = session.get(Run, str(spec.run))
    if run is None:
        raise NotFoundError(f"run {spec.run} not found")
    return run.dataset, run


def _items(
    session: Session, spec: QueueSpec, ds: Dataset, run: Run | None
) -> list[tuple[str, dict[str, Any]]]:
    cases = load_cases(session, ds)
    if spec.tags:
        wanted = set(spec.tags)
        cases = [c for c in cases if wanted.intersection(c.tags)]
    if spec.split is not None:
        cases = [c for c in cases if c.split == spec.split]
    outputs: dict[str, str] = {}
    if run is not None:
        rows = session.execute(
            select(Result.case_id, Result.output).where(
                Result.run_id == run.id, Result.status == "ok", Result.output.is_not(None)
            )
        ).all()
        outputs = {str(c): str(o) for c, o in rows}
        cases = [c for c in cases if c.id in outputs]
    if spec.limit is not None:
        cases = cases[: spec.limit]
    items = []
    for c in cases:
        item: dict[str, Any] = {"input": c.input, "expected": c.expected}
        if run is not None:
            item["output"] = outputs[c.id]
        items.append((c.id, item))
    return items


def create_queue(session: Session, spec: QueueSpec) -> tuple[QueueSummary, bool]:
    """Create the queue, or return the existing one if it has the same settings.
    The second value says whether it was created."""
    ds, run = _source(session, spec)
    identity = spec.model_dump(mode="json") | {"version": ds.version, "dataset": ds.name}
    spec_hash = sha256_hex(canonical_json(identity))
    existing = session.scalar(select(AnnotationQueue).where(AnnotationQueue.name == spec.name))
    if existing is not None:
        return _same(session, existing, spec_hash), False
    items = _items(session, spec, ds, run)
    if not items:
        raise InvalidInputError("no cases to label: the filter, or the run's results, left none")
    queue = AnnotationQueue(
        id=uuid.uuid4().hex,
        name=spec.name,
        spec_hash=spec_hash,
        spec=identity,
        dataset=ds.name,
        dataset_version=ds.version,
        run_id=None if run is None else run.id,
        labels=list(spec.labels),
        level=spec.level,
        overlap=spec.overlap,
        claim_seconds=spec.claim_seconds,
        task_count=len(items),
        created_at=utcnow(),
    )
    session.add(queue)
    try:
        session.flush()
        for start in range(0, len(items), INSERT_CHUNK):
            session.execute(
                insert(AnnotationTask),
                [
                    {
                        "queue_id": queue.id,
                        "position": start + i,
                        "case_id": case_id,
                        "item": item,
                        "status": "open",
                        "note": "",
                    }
                    for i, (case_id, item) in enumerate(items[start : start + INSERT_CHUNK])
                ],
            )
        session.commit()
    except IntegrityError:
        # Another writer created a queue of this name first.
        session.rollback()
        found = session.scalar(select(AnnotationQueue).where(AnnotationQueue.name == spec.name))
        if found is None:  # pragma: no cover - the conflicting row was deleted again
            raise
        return _same(session, found, spec_hash), False
    return queue_summary(session, queue.id), True


def _same(session: Session, queue: AnnotationQueue, spec_hash: str) -> QueueSummary:
    if queue.spec_hash != spec_hash:
        raise ConflictError(
            f"annotation queue {queue.name!r} exists with other settings "
            f"(over {queue.dataset} version {queue.dataset_version}); choose another name"
        )
    return queue_summary(session, queue.id)


def get_queue(session: Session, ref: str) -> AnnotationQueue:
    """A queue by name or id."""
    found = session.scalar(select(AnnotationQueue).where(AnnotationQueue.name == ref))
    if found is None and len(ref) == 32:
        found = session.get(AnnotationQueue, ref)
    if found is None:
        raise NotFoundError(f"annotation queue {ref!r} not found")
    return found


def queue_summary(session: Session, ref: str) -> QueueSummary:
    q = get_queue(session, ref)
    rows = session.execute(
        select(AnnotationTask.status, func.count())
        .where(AnnotationTask.queue_id == q.id)
        .group_by(AnnotationTask.status)
    ).all()
    return QueueSummary(
        q.id,
        q.name,
        q.dataset,
        q.dataset_version,
        q.run_id,
        list(q.labels),
        q.level,
        q.overlap,
        str(q.spec.get("claim_timeout", "")),
        str(q.spec.get("instructions", "")),
        q.task_count,
        as_utc(q.created_at),
        dict.fromkeys(STATUSES, 0) | {str(s): int(n) for s, n in rows},
    )


def list_queues(session: Session) -> list[QueueSummary]:
    ids = session.scalars(select(AnnotationQueue.id).order_by(AnnotationQueue.created_at))
    return [queue_summary(session, i) for i in ids]


# Claiming and labelling -------------------------------------------------------------


@dataclass(frozen=True)
class Claim:
    queue: str
    case_id: str
    position: int
    item: dict[str, Any]
    slot: int
    expires_at: datetime


def _live(now: datetime) -> ColumnElement[bool]:
    """Assignments that count against a task's overlap: labelled, or claimed and unexpired."""
    return and_(
        AnnotationAssignment.slot >= 0,
        or_(AnnotationAssignment.submitted_at.is_not(None), AnnotationAssignment.expires_at > now),
    )


def _touch(session: Session, task_id: int, *, open_only: bool = False) -> bool:
    """Update the task's row before reading anything else in the transaction: the
    row lock on PostgreSQL, the write lock on SQLite. False when the task is gone,
    or with ``open_only``, no longer open."""
    query = update(AnnotationTask).where(AnnotationTask.id == task_id)
    if open_only:
        query = query.where(AnnotationTask.status == "open")
    result = session.execute(
        query.values(status=AnnotationTask.status).execution_options(synchronize_session=False)
    )
    return bool(result.rowcount)  # type: ignore[attr-defined]


def claim_next(
    session: Session, ref: str, annotator: str, *, now: datetime | None = None
) -> Claim | None:
    """The next task for ``annotator``, or None when nothing is left for them."""
    validate_annotator(annotator)
    now = now or utcnow()
    q = get_queue(session, ref)
    ttl = timedelta(seconds=q.claim_seconds)
    held = session.execute(
        select(AnnotationAssignment, AnnotationTask)
        .join(AnnotationTask, AnnotationTask.id == AnnotationAssignment.task_id)
        .where(
            AnnotationAssignment.queue_id == q.id,
            AnnotationAssignment.annotator == annotator,
            AnnotationAssignment.slot >= 0,
            AnnotationAssignment.submitted_at.is_(None),
            AnnotationTask.status == "open",
        )
        .order_by(AnnotationTask.position)
        .limit(1)
    ).first()
    if held is not None:
        a, task = held
        session.execute(
            update(AnnotationAssignment)
            .where(AnnotationAssignment.id == a.id, AnnotationAssignment.annotator == annotator)
            .values(expires_at=now + ttl)
            .execution_options(synchronize_session=False)
        )
        session.commit()
        return Claim(q.name, task.case_id, task.position, dict(task.item), a.slot, now + ttl)
    after = -1
    while True:
        live = (
            select(func.count())
            .where(AnnotationAssignment.task_id == AnnotationTask.id, _live(now))
            .scalar_subquery()
        )
        mine = (
            select(AnnotationAssignment.id)
            .where(
                AnnotationAssignment.task_id == AnnotationTask.id,
                AnnotationAssignment.annotator == annotator,
            )
            .exists()
        )
        candidates = list(
            session.execute(
                select(AnnotationTask.id, AnnotationTask.position)
                .where(
                    AnnotationTask.queue_id == q.id,
                    AnnotationTask.status == "open",
                    AnnotationTask.position > after,
                    ~mine,
                    live < q.overlap,
                )
                .order_by(AnnotationTask.position)
                .limit(CANDIDATES)
            ).all()
        )
        # End the read so the claim's transaction starts with its write.
        session.rollback()
        if not candidates:
            return None
        for task_id, position in candidates:
            claim = _take_slot(session, q, int(task_id), annotator, now=now, ttl=ttl)
            if claim is not None:
                return claim
            after = int(position)


def _take_slot(
    session: Session,
    q: AnnotationQueue,
    task_id: int,
    annotator: str,
    *,
    now: datetime,
    ttl: timedelta,
) -> Claim | None:
    try:
        if not _touch(session, task_id, open_only=True):
            session.rollback()
            return None
        slots = list(
            session.scalars(
                select(AnnotationAssignment).where(AnnotationAssignment.task_id == task_id)
            )
        )
        if any(a.annotator == annotator for a in slots):
            session.rollback()
            return None
        taken = {a.slot for a in slots if a.slot >= 0}
        free = next((s for s in range(q.overlap) if s not in taken), None)
        if free is not None:
            session.add(
                AnnotationAssignment(
                    task_id=task_id,
                    queue_id=q.id,
                    slot=free,
                    annotator=annotator,
                    claimed_at=now,
                    expires_at=now + ttl,
                    note="",
                )
            )
            slot = free
        else:
            lapsed = [a for a in slots if a.submitted_at is None and as_utc(a.expires_at) <= now]
            if not lapsed:
                session.rollback()
                return None
            slot = min(a.slot for a in lapsed)
            moved = session.execute(
                update(AnnotationAssignment)
                .where(
                    AnnotationAssignment.task_id == task_id,
                    AnnotationAssignment.slot == slot,
                    AnnotationAssignment.submitted_at.is_(None),
                    AnnotationAssignment.expires_at <= now,
                )
                .values(annotator=annotator, claimed_at=now, expires_at=now + ttl, note="")
                .execution_options(synchronize_session=False)
            )
            if not moved.rowcount:  # type: ignore[attr-defined]
                session.rollback()
                return None
        session.commit()
    except IntegrityError:
        session.rollback()
        return None
    task = session.get(AnnotationTask, task_id)
    if task is None:  # pragma: no cover - tasks are deleted only with their queue
        raise NotFoundError("the item was deleted")
    return Claim(q.name, task.case_id, task.position, dict(task.item), slot, now + ttl)


@dataclass(frozen=True)
class Submitted:
    queue: str
    case_id: str
    label: Any
    status: str  # the task's status after this label
    final: Any  # the task's final label, once agreed or adjudicated
    labels: int  # labels the task now has
    overlap: int


def _task_for_write(session: Session, q: AnnotationQueue, case_id: str) -> AnnotationTask:
    task_id = session.scalar(
        select(AnnotationTask.id).where(
            AnnotationTask.queue_id == q.id, AnnotationTask.case_id == case_id
        )
    )
    if task_id is None:
        raise NotFoundError(f"annotation queue {q.name!r} has no item {case_id!r}")
    session.rollback()
    _touch(session, int(task_id))
    task = session.get(AnnotationTask, int(task_id), populate_existing=True)
    if task is None:  # pragma: no cover - tasks are deleted only with their queue
        raise NotFoundError(f"annotation queue {q.name!r} has no item {case_id!r}")
    return task


def _settle(task: AnnotationTask, labels: list[Any], overlap: int) -> None:
    """Set an unadjudicated task's status from its submitted labels."""
    if len(labels) < overlap:
        task.status, task.label = "open", None
    elif len({_key(v) for v in labels}) == 1:
        task.status, task.label = "agreed", labels[0]
    else:
        task.status, task.label = "disputed", None


def submit(
    session: Session,
    ref: str,
    annotator: str,
    case_id: str,
    label: Any,
    *,
    note: str = "",
    now: datetime | None = None,
) -> Submitted:
    """Record ``annotator``'s label for an item they claimed. Submitting again
    changes the label, until the item is adjudicated."""
    validate_annotator(annotator)
    _note(note)
    now = now or utcnow()
    q = get_queue(session, ref)
    value = resolve_label(q.labels, label)
    task = _task_for_write(session, q, case_id)
    if task.status == "adjudicated":
        session.rollback()
        raise ConflictError(f"item {case_id!r} is adjudicated; its labels are final")
    rows = list(
        session.scalars(select(AnnotationAssignment).where(AnnotationAssignment.task_id == task.id))
    )
    mine = next((a for a in rows if a.annotator == annotator and a.slot >= 0), None)
    if mine is None:
        session.rollback()
        raise ConflictError(
            f"{annotator} does not hold item {case_id!r}: claim it with next first "
            "(or the claim lapsed and someone else took it)"
        )
    mine.label, mine.note, mine.submitted_at = value, note, now
    labels = [a.label for a in sorted(rows, key=lambda a: a.slot) if a.submitted_at is not None]
    _settle(task, labels, q.overlap)
    session.commit()
    return Submitted(q.name, case_id, value, task.status, task.label, len(labels), q.overlap)


def skip(
    session: Session, ref: str, annotator: str, case_id: str, *, now: datetime | None = None
) -> None:
    """Give back a claimed, unlabelled item: its slot is freed for someone else,
    and it is not offered to ``annotator`` again. Skipping twice is a no-op."""
    validate_annotator(annotator)
    now = now or utcnow()
    q = get_queue(session, ref)
    task = _task_for_write(session, q, case_id)
    mine = session.scalar(
        select(AnnotationAssignment).where(
            AnnotationAssignment.task_id == task.id, AnnotationAssignment.annotator == annotator
        )
    )
    if mine is None or mine.submitted_at is not None:
        session.rollback()
        what = "has labelled" if mine is not None else "does not hold"
        raise ConflictError(f"{annotator} {what} item {case_id!r}; only a claimed item is skipped")
    if mine.slot >= 0:
        mine.slot, mine.expires_at = -mine.id, now
    session.commit()


def adjudicate(
    session: Session,
    ref: str,
    adjudicator: str,
    case_id: str,
    label: Any,
    *,
    note: str = "",
    now: datetime | None = None,
) -> Submitted:
    """Give a disputed item its final label (or change an adjudicated one's)."""
    validate_annotator(adjudicator, "adjudicator")
    _note(note)
    now = now or utcnow()
    q = get_queue(session, ref)
    value = resolve_label(q.labels, label)
    task = _task_for_write(session, q, case_id)
    if task.status not in ("disputed", "adjudicated"):
        session.rollback()
        raise ConflictError(
            f"item {case_id!r} is {task.status}: only disputed items are adjudicated"
        )
    task.status, task.label = "adjudicated", value
    task.adjudicator, task.adjudicated_at, task.note = adjudicator, now, note
    count = session.scalar(
        select(func.count()).where(
            AnnotationAssignment.task_id == task.id,
            AnnotationAssignment.submitted_at.is_not(None),
        )
    )
    session.commit()
    return Submitted(q.name, case_id, value, "adjudicated", value, int(count or 0), q.overlap)


# Reading ------------------------------------------------------------------------


@dataclass(frozen=True)
class AssignmentInfo:
    annotator: str
    slot: int  # negative once skipped
    skipped: bool
    label: Any
    note: str
    claimed_at: datetime
    expires_at: datetime
    submitted_at: datetime | None


@dataclass(frozen=True)
class TaskInfo:
    case_id: str
    position: int
    status: str
    item: dict[str, Any]
    label: Any
    adjudicator: str | None
    adjudicated_at: datetime | None
    note: str
    assignments: list[AssignmentInfo]


def list_tasks(
    session: Session,
    ref: str,
    status: str | None = None,
    *,
    limit: int = 100,
    offset: int = 0,
) -> list[TaskInfo]:
    q = get_queue(session, ref)
    if status is not None and status not in STATUSES:
        raise InvalidInputError(f"status must be one of {', '.join(STATUSES)}")
    query = select(AnnotationTask).where(AnnotationTask.queue_id == q.id)
    if status is not None:
        query = query.where(AnnotationTask.status == status)
    tasks = list(
        session.scalars(query.order_by(AnnotationTask.position).offset(offset).limit(limit))
    )
    by_task: dict[int, list[AssignmentInfo]] = defaultdict(list)
    if tasks:
        for a in session.scalars(
            select(AnnotationAssignment)
            .where(AnnotationAssignment.task_id.in_([t.id for t in tasks]))
            .order_by(AnnotationAssignment.task_id, AnnotationAssignment.slot)
        ):
            by_task[a.task_id].append(
                AssignmentInfo(
                    a.annotator,
                    a.slot,
                    a.slot < 0,
                    a.label,
                    a.note,
                    as_utc(a.claimed_at),
                    as_utc(a.expires_at),
                    None if a.submitted_at is None else as_utc(a.submitted_at),
                )
            )
    return [
        TaskInfo(
            t.case_id,
            t.position,
            t.status,
            dict(t.item),
            t.label,
            t.adjudicator,
            None if t.adjudicated_at is None else as_utc(t.adjudicated_at),
            t.note,
            by_task[t.id],
        )
        for t in tasks
    ]


@dataclass(frozen=True)
class AnnotatorStats:
    annotator: str
    submitted: int
    # Of this annotator's labels on items with a final label: how many match it.
    compared: int
    matched: int

    @property
    def agreement(self) -> float | None:
        return self.matched / self.compared if self.compared else None


@dataclass(frozen=True)
class Agreement:
    queue: str
    level: str
    labelled_items: int  # items with at least one label
    labels: int
    alpha: AlphaResult
    annotators: list[AnnotatorStats]
    # How often each pair of labels was given to one item by different
    # annotators: the coincidence matrix, rows and columns in label order.
    coincidences: list[list[float]]


def agreement(session: Session, ref: str) -> Agreement:
    """Krippendorff's alpha over every submitted label, and each annotator's
    agreement with the final labels."""
    q = get_queue(session, ref)
    rows = session.execute(
        select(
            AnnotationAssignment.task_id,
            AnnotationAssignment.annotator,
            AnnotationAssignment.label,
            AnnotationTask.label,
            AnnotationTask.status,
        )
        .join(AnnotationTask, AnnotationTask.id == AnnotationAssignment.task_id)
        .where(
            AnnotationAssignment.queue_id == q.id, AnnotationAssignment.submitted_at.is_not(None)
        )
        .order_by(AnnotationAssignment.task_id, AnnotationAssignment.slot)
    ).all()
    labels = list(q.labels)
    canon = {_key(v): v for v in labels}
    units: dict[int, list[Any]] = defaultdict(list)
    submitted: dict[str, int] = defaultdict(int)
    compared: dict[str, int] = defaultdict(int)
    matched: dict[str, int] = defaultdict(int)
    for task_id, who, given, final, status in rows:
        value = canon[_key(given)]
        units[int(task_id)].append(value)
        submitted[who] += 1
        if status in ("agreed", "adjudicated") and final is not None:
            compared[who] += 1
            matched[who] += _key(given) == _key(final)
    unit_lists = list(units.values())
    alpha = krippendorff_alpha(unit_lists, cast(Level, q.level), labels)
    o, _, _ = coincidences(unit_lists, labels)
    stats = [
        AnnotatorStats(who, submitted[who], compared[who], matched[who])
        for who in sorted(submitted)
    ]
    return Agreement(
        q.name,
        q.level,
        len(units),
        len(rows),
        alpha,
        stats,
        [[round(float(x), 6) for x in row] for row in o],
    )


def export_labels(session: Session, ref: str) -> list[dict[str, Any]]:
    """Every item with a final label (agreed or adjudicated), in queue order, as
    labelled items: ``{id, input, expected, output, label}``, the form
    ``evalkit calibrate`` reads (output is present when the queue labels a run)."""
    q = get_queue(session, ref)
    out = []
    for t in session.scalars(
        select(AnnotationTask)
        .where(
            AnnotationTask.queue_id == q.id,
            AnnotationTask.status.in_(("agreed", "adjudicated")),
        )
        .order_by(AnnotationTask.position)
    ):
        out.append({"id": t.case_id, **t.item, "label": t.label})
    return out
