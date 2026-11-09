"""The prompt registry: named prompts with immutable, content-hashed versions and
movable labels.

A version is a system prompt and a template, identified by the SHA-256 of both
in canonical JSON. Versions are only ever inserted: pushing content that a
prompt already has returns that version, so a CI job can push on every commit
without minting versions, and two pushes racing on the same content get the
same version.

A label such as ``production`` points at one version and moves only by
compare-and-set: the caller names the version it expects the label to be at
(or that it must not exist yet), and a move that lost a race is refused with
the version the label is at now, never applied over someone else's move.

A run config names a version as ``name``, ``name@3`` or ``name@production``.
The reference is resolved once, when the run is created: the run stores the
resolved system prompt and template in its target spec, and the reference and
version in ``runs.prompt``. Moving the label afterwards does not change a run
that already exists, and resuming it uses what it recorded.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, NoReturn

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from evalkit.datasets import canonical_json, sha256_hex
from evalkit.db.models import Prompt, PromptLabel, PromptVersion, utcnow
from evalkit.errors import ConflictError, InvalidInputError, NotFoundError
from evalkit.runspec import RunConfig
from evalkit.targets.specs import (
    PROMPT_LABEL,
    PROMPT_NAME,
    PROMPT_REF,
    PromptTargetSpec,
)
from evalkit.targets.template import placeholders

MAX_TEXT = 200_000  # the limit on a prompt target's system prompt and template
MAX_NOTE = 2000
# The newest version; not a label anyone can move.
LATEST = "latest"

_NAME = re.compile(PROMPT_NAME)
_LABEL = re.compile(PROMPT_LABEL)
_REF = re.compile(PROMPT_REF)


@dataclass(frozen=True)
class Resolved:
    """A version found from a reference, and the label it was found by, if any."""

    name: str
    version: PromptVersion
    label: str | None = None

    @property
    def title(self) -> str:
        via = f"@{self.label}" if self.label else ""
        return f"{self.name}{via} (v{self.version.version})"

    def pinned(self, ref: str) -> dict[str, Any]:
        """What a run records about the prompt it was created with."""
        return {
            "ref": ref,
            "name": self.name,
            "version": self.version.version,
            "label": self.label,
            "content_hash": self.version.content_hash,
        }


@dataclass(frozen=True)
class LabelMove:
    name: str
    label: str
    previous: int | None  # None when the label was created
    version: int


@dataclass(frozen=True)
class PromptInfo:
    name: str
    description: str
    created_at: datetime
    latest_version: int
    labels: dict[str, int]


def content_hash(system: str, template: str) -> str:
    return sha256_hex(canonical_json({"system": system, "template": template}))


def validate_prompt_name(name: str) -> str:
    if not _NAME.fullmatch(name):
        raise InvalidInputError(
            "prompt name must be 1-128 characters of letters, digits, '.', '_' or '-', "
            "starting with a letter or digit"
        )
    return name


def validate_label(label: str) -> str:
    if not _LABEL.fullmatch(label) or label == LATEST:
        raise InvalidInputError(
            "a label is 1-64 characters of lower-case letters, digits, '.', '_' or '-', "
            f"starting with a letter, and not {LATEST!r}"
        )
    return label


def parse_ref(ref: str) -> tuple[str, int | str]:
    """``name``, ``name@N`` or ``name@label`` as (name, version number or label).
    A bare name, or ``name@latest``, means the newest version."""
    if not _REF.fullmatch(ref):
        raise InvalidInputError(f"prompt reference {ref!r} is not NAME, NAME@VERSION or NAME@LABEL")
    name, _, selector = ref.partition("@")
    if not selector:
        return name, LATEST
    return name, int(selector) if selector.isdigit() else selector


def _validate_content(system: str, template: str, note: str) -> None:
    if not template:
        raise InvalidInputError("template must not be empty")
    for field, value in (("system", system), ("template", template)):
        if len(value) > MAX_TEXT:
            raise InvalidInputError(f"{field} is longer than {MAX_TEXT} characters")
    if len(note) > MAX_NOTE:
        raise InvalidInputError(f"note is longer than {MAX_NOTE} characters")
    # The same check a target spec makes, so a pushed version can always run.
    placeholders(template)


def push_version(
    session: Session,
    name: str,
    *,
    template: str,
    system: str = "",
    note: str = "",
    description: str = "",
) -> tuple[PromptVersion, bool]:
    """Store ``system`` and ``template`` as the next version of ``name``, creating
    the prompt on its first push. Returns (version, created): content the prompt
    already has returns that version unchanged."""
    validate_prompt_name(name)
    _validate_content(system, template, note)
    if len(description) > MAX_NOTE:
        raise InvalidInputError(f"description is longer than {MAX_NOTE} characters")
    digest = content_hash(system, template)
    for _ in range(5):
        prompt = session.scalar(select(Prompt).where(Prompt.name == name))
        if prompt is None:
            session.add(Prompt(name=name, description=description, created_at=utcnow()))
            try:
                session.commit()
            except IntegrityError:
                # Another first push created it; use theirs.
                session.rollback()
            continue
        existing = session.scalar(
            select(PromptVersion).where(
                PromptVersion.prompt_id == prompt.id, PromptVersion.content_hash == digest
            )
        )
        if existing is not None:
            return existing, False
        latest = session.scalar(
            select(func.max(PromptVersion.version)).where(PromptVersion.prompt_id == prompt.id)
        )
        version = PromptVersion(
            prompt_id=prompt.id,
            version=(latest or 0) + 1,
            content_hash=digest,
            system=system,
            template=template,
            note=note,
            created_at=utcnow(),
        )
        session.add(version)
        try:
            session.commit()
        except IntegrityError:
            # Another push took this version number or this content; look again.
            session.rollback()
            continue
        return version, True
    raise ConflictError(f"could not allocate a version for prompt {name!r}; retry")


def get_prompt(session: Session, name: str) -> Prompt:
    prompt = session.scalar(select(Prompt).where(Prompt.name == name))
    if prompt is None:
        raise NotFoundError(f"prompt {name!r} not found")
    return prompt


def labels_of(session: Session, prompt: Prompt) -> dict[str, int]:
    rows = session.scalars(
        select(PromptLabel).where(PromptLabel.prompt_id == prompt.id).order_by(PromptLabel.label)
    )
    return {row.label: row.version for row in rows}


def versions_of(session: Session, prompt: Prompt) -> list[PromptVersion]:
    return list(
        session.scalars(
            select(PromptVersion)
            .where(PromptVersion.prompt_id == prompt.id)
            .order_by(PromptVersion.version)
        )
    )


def _version(session: Session, prompt: Prompt, number: int) -> PromptVersion:
    found = session.scalar(
        select(PromptVersion).where(
            PromptVersion.prompt_id == prompt.id, PromptVersion.version == number
        )
    )
    if found is None:
        raise NotFoundError(f"prompt {prompt.name!r} has no version {number}")
    return found


def resolve(session: Session, ref: str) -> Resolved:
    """The version ``ref`` names now."""
    name, selector = parse_ref(ref)
    prompt = get_prompt(session, name)
    if isinstance(selector, int):
        return Resolved(name, _version(session, prompt, selector))
    if selector == LATEST:
        newest = session.scalar(
            select(PromptVersion)
            .where(PromptVersion.prompt_id == prompt.id)
            .order_by(PromptVersion.version.desc())
            .limit(1)
        )
        if newest is None:  # pragma: no cover - a prompt is created with its first push
            raise NotFoundError(f"prompt {name!r} has no versions")
        return Resolved(name, newest)
    number = session.scalar(
        select(PromptLabel.version).where(
            PromptLabel.prompt_id == prompt.id, PromptLabel.label == selector
        )
    )
    if number is None:
        known = ", ".join(labels_of(session, prompt)) or "none"
        raise NotFoundError(f"prompt {name!r} has no label {selector!r} (labels: {known})")
    return Resolved(name, _version(session, prompt, number), label=selector)


def move_label(
    session: Session, name: str, label: str, version: int, *, expected: int | None
) -> LabelMove:
    """Point ``label`` at ``version`` if it is at ``expected`` now; ``expected=None``
    creates the label and requires that it does not exist yet. A label that is
    anywhere else is left alone and the move fails with ConflictError."""
    validate_label(label)
    prompt = get_prompt(session, name)
    _version(session, prompt, version)
    now = utcnow()
    if expected is None:
        session.add(PromptLabel(prompt_id=prompt.id, label=label, version=version, updated_at=now))
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            _conflict(session, prompt, label, expected)
        return LabelMove(name, label, None, version)
    moved = session.execute(
        update(PromptLabel)
        .where(
            PromptLabel.prompt_id == prompt.id,
            PromptLabel.label == label,
            PromptLabel.version == expected,
        )
        .values(version=version, updated_at=now)
    )
    session.commit()
    if moved.rowcount != 1:  # type: ignore[attr-defined]
        _conflict(session, prompt, label, expected)
    return LabelMove(name, label, expected, version)


def _conflict(session: Session, prompt: Prompt, label: str, expected: int | None) -> NoReturn:
    current = labels_of(session, prompt).get(label)
    now = "does not exist" if current is None else f"is at version {current}"
    wanted = "not to exist" if expected is None else f"to be at version {expected}"
    raise ConflictError(
        f"label {label!r} of prompt {prompt.name!r} {now}; the move expected it {wanted}"
    )


def list_prompts(session: Session) -> list[PromptInfo]:
    latest: dict[int, int] = dict(
        session.execute(
            select(PromptVersion.prompt_id, func.max(PromptVersion.version)).group_by(
                PromptVersion.prompt_id
            )
        ).all()
    )
    labels: dict[int, dict[str, int]] = {}
    for row in session.scalars(select(PromptLabel).order_by(PromptLabel.label)):
        labels.setdefault(row.prompt_id, {})[row.label] = row.version
    return [
        PromptInfo(p.name, p.description, p.created_at, latest.get(p.id, 0), labels.get(p.id, {}))
        for p in session.scalars(select(Prompt).order_by(Prompt.name))
    ]


def diff(old: Resolved, new: Resolved) -> str:
    """A unified diff of the system prompt and the template; empty when the two
    versions are the same."""
    out: list[str] = []
    for field in ("system", "template"):
        a, b = getattr(old.version, field), getattr(new.version, field)
        out += _unified(a, b, f"{old.title} {field}", f"{new.title} {field}")
    return "".join(out)


def _unified(a: str, b: str, from_name: str, to_name: str) -> Iterable[str]:
    for line in difflib.unified_diff(_lines(a), _lines(b), fromfile=from_name, tofile=to_name):
        yield line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"


def _lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def resolve_config(session: Session, config: RunConfig) -> tuple[RunConfig, dict[str, Any] | None]:
    """``config`` with its target's registry prompt replaced by the version it names
    now, and what the run should record about it (None when the target names none)."""
    target = config.target
    if not isinstance(target, PromptTargetSpec) or target.prompt is None:
        return config, None
    found = resolve(session, target.prompt)
    spec = PromptTargetSpec.model_validate(
        {
            **target.model_dump(),
            "prompt": None,
            "system": found.version.system,
            "template": found.version.template,
        }
    )
    return config.model_copy(update={"target": spec}), found.pinned(target.prompt)
