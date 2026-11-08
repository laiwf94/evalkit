"""Model matrices: one run config expanded into a group of child runs.

A ``[matrix]`` table names axes for a prompt target, model (each with its
provider), effort and max_tokens, and every combination becomes one child run
of the same dataset, case filter, scorers and options:

    [matrix]
    model = ["claude-opus-5-5", {provider = "openai", model = "gpt-5.1"},
             {provider = "gemini", model = "gemini-3-pro", name = "gemini-pro"}]
    effort = ["low", "high"]

The children are ordinary runs: each has its own target hash, cache entries,
leases and aggregates, can be queued for worker daemons, resumed, shown and
gated like any other. The group (``matrices`` table) records the axes and the
order of its children, so ``evalkit report matrix`` can lay them out and find
their cost, quality and latency frontier (``evalkit.frontier``).

Expansion is pure and deterministic: the same config gives the same children
in the same order (model, then effort, then max_tokens, each in the order
given), and a matrix that names one variant twice is refused rather than
paying for the same run twice.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from evalkit.db.models import Matrix, Run
from evalkit.errors import InvalidInputError, NotFoundError
from evalkit.runspec import MAX_MATRIX_VARIANTS, MatrixAxes, MatrixModel, RunConfig
from evalkit.targets.specs import ANTHROPIC, PromptTargetSpec, target_hash

MAX_LABEL = 128


@dataclass(frozen=True)
class Variant:
    """One child of a matrix: its place, its name, and the run config it executes."""

    index: int
    name: str
    config: RunConfig

    @property
    def target(self) -> PromptTargetSpec:
        target = self.config.target
        assert isinstance(target, PromptTargetSpec)  # noqa: S101 - expand only makes these
        return target

    def axes(self) -> dict[str, Any]:
        """What this variant runs: the values every axis resolves to."""
        t = self.target
        return {
            "index": self.index,
            "name": self.name,
            "provider": t.provider,
            "model": t.model,
            "effort": t.effort,
            "max_tokens": t.max_tokens,
        }


def variant_count(axes: MatrixAxes) -> int:
    return max(1, len(axes.model)) * max(1, len(axes.effort)) * max(1, len(axes.max_tokens))


def expand(config: RunConfig) -> list[Variant]:
    """The child run configs of a matrix config, in axis order."""
    axes = config.matrix
    if axes is None:
        raise InvalidInputError("the run config has no [matrix] table")
    base = config.target
    if not isinstance(base, PromptTargetSpec):
        raise InvalidInputError(
            f"a [matrix] expands a prompt target's model, effort and max_tokens; "
            f"this config's target is {base.type}"
        )
    if not (axes.model or axes.effort or axes.max_tokens):
        raise InvalidInputError("[matrix] needs at least one axis: model, effort or max_tokens")
    count = variant_count(axes)
    if count > MAX_MATRIX_VARIANTS:
        raise InvalidInputError(
            f"[matrix] expands into {count} runs; at most {MAX_MATRIX_VARIANTS} are allowed"
        )
    models: list[MatrixModel | None] = [
        m if isinstance(m, MatrixModel) else MatrixModel(model=m) for m in axes.model
    ] or [None]
    efforts: list[Any] = list(axes.effort) or [None]
    limits: list[int | None] = list(axes.max_tokens) or [None]

    variants: list[Variant] = []
    by_target: dict[str, str] = {}
    for index, (model, effort, limit) in enumerate(itertools.product(models, efforts, limits)):
        update: dict[str, Any] = {}
        parts: list[str] = []
        if model is not None:
            provider = model.provider or base.provider
            update.update(model=model.model, provider=provider)
            parts.append(
                model.name
                or (model.model if provider == ANTHROPIC else f"{provider}:{model.model}")
            )
        if effort is not None:
            update["effort"] = effort
            parts.append(f"effort={effort}")
        if limit is not None:
            update["max_tokens"] = limit
            parts.append(f"max_tokens={limit}")
        name = ",".join(parts)
        # Revalidated, so a variant is exactly what the same target written out would be.
        target = PromptTargetSpec.model_validate({**base.model_dump(), **update})
        key = target_hash(target)
        if key in by_target:
            raise InvalidInputError(
                f"[matrix] variants {by_target[key]!r} and {name!r} are the same target"
            )
        if name in by_target.values():
            raise InvalidInputError(f"[matrix] names two variants {name!r}; give them names")
        by_target[key] = name
        label = f"{config.label}[{name}]" if config.label else name
        if len(label) > MAX_LABEL:
            raise InvalidInputError(
                f"variant label {label!r} is longer than {MAX_LABEL} characters; "
                "shorten the label or name the models"
            )
        child = config.model_copy(update={"target": target, "label": label, "matrix": None})
        variants.append(Variant(index, name, child))
    return variants


def get_matrix(session: Session, matrix_id: str) -> Matrix:
    matrix = session.get(Matrix, matrix_id)
    if matrix is None:
        raise NotFoundError(f"matrix {matrix_id} not found")
    return matrix


def resolve_matrix(session: Session, ref: str) -> str:
    """A matrix id, or ``label:NAME`` for the newest matrix with that label."""
    if ref.startswith("label:"):
        label = ref.removeprefix("label:")
        found = session.scalar(
            select(Matrix.id)
            .where(Matrix.label == label)
            .order_by(Matrix.created_at.desc(), Matrix.id.desc())
            .limit(1)
        )
        if found is None:
            raise NotFoundError(f"no matrix labelled {label!r}")
        return found
    return get_matrix(session, ref).id


def child_runs(session: Session, matrix_id: str) -> list[Run]:
    """A matrix's runs, in variant order."""
    runs = list(session.scalars(select(Run).where(Run.matrix_id == matrix_id)))
    return sorted(runs, key=lambda r: int((r.variant or {}).get("index", 0)))
