"""Slices: the same paired comparison as ``evalkit compare``, inside each group of
cases, so that a regression in one category is not averaged away by the rest.

A slice is the set of cases that share a value of the slicing key:

* ``tag``: one slice per tag. A case with several tags is in several slices, and
  a case with none is in no slice.
* ``split``: one slice per split. Cases without a split are in no slice.
* ``metadata.KEY``: one slice per value of that metadata key. A string is its own
  name; a number or boolean is named by its JSON (``3``, ``true``); a list puts
  the case in a slice for each scalar element, as tags do. A missing key, null or
  an object puts the case in no slice.

Every (slice, metric) pair with at least ``min_size`` paired cases is one test:
the paired bootstrap interval and sign-flip permutation test of
``evalkit.stats.compare_aligned``. All of a comparison's slice tests form one
Holm family, so asking about many slices does not make a chance regression more
likely to be reported. The family is separate from the whole-dataset metrics,
which keep the verdicts they have without slices. A pair below the minimum is
reported as too small and left out of the family.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sqlalchemy import select
from sqlalchemy.orm import Session

from evalkit.config import DEFAULT_MIN_SLICE_SIZE, GateRules
from evalkit.datasets import canonical_json
from evalkit.db.models import DatasetCase
from evalkit.errors import InvalidInputError, PreconditionError
from evalkit.stats import DEFAULT_RESAMPLES, Comparison, FloatArray, compare_aligned

IndexArray = npt.NDArray[np.intp]
SliceKind = Literal["tag", "split", "metadata"]

# Slices compared at once. Each costs a bootstrap and a permutation test per
# metric; past a few hundred the key is almost certainly an id, not a category.
MAX_SLICES = 500


@dataclass(frozen=True)
class SliceBy:
    kind: SliceKind
    key: str = ""

    def __str__(self) -> str:
        return f"metadata.{self.key}" if self.kind == "metadata" else self.kind


def parse_by(raw: str) -> SliceBy:
    """``tag``, ``split`` or ``metadata.KEY``; ValueError otherwise."""
    if raw == "tag":
        return SliceBy("tag")
    if raw == "split":
        return SliceBy("split")
    prefix, _, key = raw.partition(".")
    if prefix == "metadata" and key and len(key) <= 128 and key.isprintable():
        return SliceBy("metadata", key)
    raise ValueError(f"cannot slice by {raw!r}: use tag, split or metadata.KEY")


def _scalar_name(value: Any) -> str | None:
    if isinstance(value, str):
        return value or None
    if isinstance(value, bool | int):
        return canonical_json(value)
    if isinstance(value, float) and math.isfinite(value):
        return canonical_json(value)
    return None


def slice_names(by: SliceBy, tags: Sequence[Any], split: str | None, meta: Any) -> list[str]:
    """The slices one case is in, without repeats, in a stable order."""
    if by.kind == "tag":
        found = [str(t) for t in tags]
    elif by.kind == "split":
        found = [] if split is None else [split]
    else:
        value = meta.get(by.key) if isinstance(meta, dict) else None
        items = value if isinstance(value, list) else [value]
        found = [n for n in map(_scalar_name, items) if n is not None]
    return list(dict.fromkeys(found))


@dataclass(frozen=True)
class Membership:
    """Which compared cases each slice holds, as positions in the score arrays."""

    slices: dict[str, IndexArray]
    unsliced: int  # compared cases in no slice


def membership(
    session: Session, dataset_id: int, by: SliceBy, positions: IndexArray | None = None
) -> Membership:
    """Slices over a dataset's cases in score-array order (case-id order). With
    ``positions`` (a version pairing's index into this dataset's arrays) the result
    indexes the compared arrays instead: compared case ``i`` is ``positions[i]``."""
    columns = {
        "tag": (DatasetCase.case_id, DatasetCase.tags),
        "split": (DatasetCase.case_id, DatasetCase.split),
        "metadata": (DatasetCase.case_id, DatasetCase.meta),
    }[by.kind]
    rows = sorted(
        session.execute(select(*columns).where(DatasetCase.dataset_id == dataset_id)),
        key=lambda r: r[0],
    )
    names_at: list[list[str]] = []
    for _, value in rows:
        if by.kind == "tag":
            names_at.append(slice_names(by, value or (), None, None))
        elif by.kind == "split":
            names_at.append(slice_names(by, (), value, None))
        else:
            names_at.append(slice_names(by, (), None, value))
    order = range(len(rows)) if positions is None else positions.tolist()
    members: dict[str, list[int]] = {}
    unsliced = 0
    for i, at in enumerate(order):
        names = names_at[at]
        if not names:
            unsliced += 1
        for name in names:
            members.setdefault(name, []).append(i)
    if len(members) > MAX_SLICES:
        raise PreconditionError(
            f"slicing by {by} gives {len(members):,} slices, more than {MAX_SLICES}; "
            "slice by a key with fewer values"
        )
    return Membership(
        {name: np.asarray(idx, dtype=np.intp) for name, idx in sorted(members.items())},
        unsliced,
    )


@dataclass(frozen=True)
class SliceResult:
    name: str
    cases: int  # compared cases in the slice
    # Metrics tested in this slice, with p-values adjusted over every slice test.
    comparisons: list[Comparison]
    # Metrics with fewer than ``min_size`` cases scored in both runs: not tested.
    too_small: dict[str, int] = field(default_factory=dict)  # metric -> paired cases

    @property
    def regressions(self) -> list[str]:
        return [c.metric for c in self.comparisons if c.verdict == "worse"]


@dataclass(frozen=True)
class SliceReport:
    by: str
    min_size: int
    tests: int  # size of the Holm family
    unsliced: int
    slices: list[SliceResult]
    # Why the gate fails on slices, one line each; empty when the slice rules
    # pass or none were given.
    failures: list[str] = field(default_factory=list)


def compare_slices(
    member: Membership,
    scores: Mapping[str, tuple[FloatArray, FloatArray]],
    *,
    by: str,
    min_size: int = DEFAULT_MIN_SLICE_SIZE,
    alpha: float = 0.05,
    min_effect: float = 0.0,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> SliceReport:
    """Compare every metric inside every slice, with Holm's correction over all the
    (slice, metric) tests that have at least ``min_size`` paired cases."""
    if min_size < 2:
        raise ValueError("min_size must be at least 2")
    family: dict[str, tuple[FloatArray, FloatArray]] = {}
    keys: list[tuple[str, str, str]] = []  # (family key, slice, metric)
    small: dict[str, dict[str, int]] = {}
    for name, idx in member.slices.items():
        for metric, (base, cand) in scores.items():
            b, c = base[idx], cand[idx]
            pairs = int(np.count_nonzero(~np.isnan(b) & ~np.isnan(c)))
            if pairs < min_size:
                small.setdefault(name, {})[metric] = pairs
                continue
            key = f"{len(keys)}"
            family[key] = (b, c)
            keys.append((key, name, metric))
    tested = (
        compare_aligned(family, alpha=alpha, min_effect=min_effect, resamples=resamples, seed=seed)
        if family
        else []
    )
    by_slice: dict[str, list[Comparison]] = {}
    for (_, name, metric), comparison in zip(keys, tested, strict=True):
        by_slice.setdefault(name, []).append(replace(comparison, metric=metric))
    results = [
        SliceResult(name, int(idx.size), by_slice.get(name, []), small.get(name, {}))
        for name, idx in member.slices.items()
    ]
    return SliceReport(by, min_size, len(keys), member.unsliced, results)


def gate_failures(report: SliceReport, rules: GateRules, metrics: Sequence[str]) -> list[str]:
    """The ways the slice rules fail, one line each. A required slice that is not
    among the compared cases, or names a metric that was not compared, is an error
    rather than a failure: it is a mistake in the rules, not news about the run."""
    found = {s.name: s for s in report.slices}
    missing = [r.slice for r in rules.require if r.slice not in found]
    if missing:
        listed = ", ".join(list(found)[:20]) + (", ..." if len(found) > 20 else "")
        raise InvalidInputError(
            f"required slice(s) {', '.join(repr(m) for m in missing)} not found by "
            f"{report.by} in the compared cases; slices: {listed or 'none'}"
        )
    failures: list[str] = []
    for rule in rules.require:
        unknown = [m for m in rule.metrics if m not in metrics]
        if unknown:
            raise InvalidInputError(
                f"slice {rule.slice!r}: metric(s) {', '.join(unknown)} not compared; "
                f"compared: {', '.join(metrics)}"
            )
        result = found[rule.slice]
        wanted = rule.metrics or tuple(metrics)
        verdicts = {c.metric: c for c in result.comparisons}
        for metric in wanted:
            if metric in result.too_small:
                failures.append(
                    f"slice {rule.slice}: {metric} has {result.too_small[metric]} paired "
                    f"case(s), fewer than {report.min_size}, so it cannot be tested"
                )
            elif verdicts[metric].verdict == "worse":
                failures.append(f"slice {rule.slice}: {metric} regressed")
    if rules.any_slice:
        required = {r.slice for r in rules.require}
        for result in report.slices:
            if result.name not in required:
                failures.extend(f"slice {result.name}: {m} regressed" for m in result.regressions)
    return failures


def slice_compare(
    session: Session,
    dataset_id: int,
    positions: IndexArray | None,
    scores: Mapping[str, tuple[FloatArray, FloatArray]],
    rules: GateRules,
    *,
    alpha: float,
    min_effect: float,
    resamples: int,
    seed: int,
) -> SliceReport:
    """Slice the compared cases of a run comparison and apply the gate rules."""
    if rules.by is None:
        raise ValueError("rules.by is required")
    by = parse_by(rules.by)
    member = membership(session, dataset_id, by, positions)
    report = compare_slices(
        member,
        scores,
        by=str(by),
        min_size=rules.min_slice_size,
        alpha=alpha,
        min_effect=min_effect,
        resamples=resamples,
        seed=seed,
    )
    failures = gate_failures(report, rules, list(scores))
    return replace(report, failures=failures)
