"""How far to trust a judge: agreement with human labels, measured as Cohen's kappa.

Raw agreement overstates quality when one label dominates (a judge that always
says 5 agrees 80% of the time with humans who give 5 to 80% of cases). Kappa
subtracts the agreement expected by chance from the two label distributions
(Cohen, 1960). For the ordinal 1-5 rubric scale the quadratic-weighted kappa
also counts how far apart disagreements are, so 4 vs 5 costs less than 1 vs 5
(Cohen, 1968). Rough reading: below 0.4 the judge should not gate anything,
0.6-0.8 is substantial agreement (Landis & Koch, 1977).

Labelled file, one JSON object per line:

    {"id": "c1", "input": ..., "expected": ..., "output": "...", "label": 4}

``label`` is 1-5 for rubric judges, "win", "tie" or "loss" (the output
against the reference) for pairwise judges, and "faithful" or "unfaithful" for
a ``faithfulness`` judge: faithful when every claim of the output is supported
by the item's sources (``input.contexts``, as the scorer reads them), which is
when the scorer gives 1. Agreement is then measured on whole answers, the
decision a gate on faithfulness makes; ``expected`` may carry the claims the
labeller marked, which the judge does not read. A trajectory judge also reads each
item's ``trajectory``: its steps as the trajectory endpoint serves them
(``GET /v1/runs/{id}/results/{case}/trajectory``, the ``steps`` list).

The ``semantic_similarity`` scorer has no labels of its own to agree on, only a
cosine; ``calibrate_threshold`` finds the cosine threshold at which its
pass/fail verdicts agree best with labels of 1 (the output says what the
reference says) and 0. It sweeps every threshold that changes a verdict in one
pass over the sorted cosines, picks the one with the highest kappa, and
reports, beside that in-sample kappa, a k-fold cross-validated one: each fold
is scored at the threshold chosen on the other folds, so the number does not
reward a threshold fitted to the very labels it is measured on.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    StrictStr,
    ValidationError,
)
from sqlalchemy.orm import Session

from evalkit.datasets import Case, sha256_hex
from evalkit.db.models import Calibration
from evalkit.embeddings import cosine
from evalkit.errors import InvalidInputError
from evalkit.scorers.faithfulness import FaithfulnessJudge
from evalkit.scorers.judge import Judgement, LLMJudge
from evalkit.scorers.similarity import SemanticSimilarity
from evalkit.scorers.trajectory import LabelledStep, TrajectoryJudge

RUBRIC_LABELS: tuple[int, ...] = (1, 2, 3, 4, 5)
PAIR_LABELS: tuple[str, ...] = ("loss", "tie", "win")
BINARY_LABELS: tuple[int, ...] = (0, 1)
FAITHFULNESS_LABELS: tuple[str, ...] = ("unfaithful", "faithful")
DEFAULT_FOLDS = 5
MAX_ITEMS = 5_000


class LabelledItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=256)
    input: JsonValue
    expected: JsonValue = None
    output: str = Field(max_length=200_000)
    label: StrictInt | StrictStr
    trajectory: list[LabelledStep] | None = Field(default=None, max_length=10_000)


@dataclass(frozen=True)
class CalibrationReport:
    n: int
    judged: int
    agreement: float
    kappa: float
    weighted_kappa: float | None
    labels: list[str]
    confusion: list[list[int]]  # rows: human label, columns: judge label
    disagreements: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


def cohens_kappa(
    a: Sequence[Hashable],
    b: Sequence[Hashable],
    categories: Sequence[Hashable],
    *,
    quadratic: bool = False,
) -> float:
    """Cohen's kappa between two raters; ``quadratic`` weights by squared distance
    between category positions (for ordinal scales)."""
    if len(a) != len(b):
        raise ValueError("both raters must label the same items")
    if not a:
        raise ValueError("cannot compute kappa for no items")
    index = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b, strict=True):
        if x not in index or y not in index:
            raise ValueError(f"label outside the categories: {x!r} / {y!r}")
        observed[index[x]][index[y]] += 1
    n = len(a)
    rows = [sum(r) / n for r in observed]
    cols = [sum(observed[i][j] for i in range(k)) / n for j in range(k)]

    def weight(i: int, j: int) -> float:
        if quadratic:
            return ((i - j) / (k - 1)) ** 2 if k > 1 else 0.0
        return 0.0 if i == j else 1.0

    disagreement = sum(weight(i, j) * observed[i][j] / n for i in range(k) for j in range(k))
    expected = sum(weight(i, j) * rows[i] * cols[j] for i in range(k) for j in range(k))
    if expected == 0:
        # Both raters used a single category: kappa is undefined; agreement is total.
        return 1.0
    return 1.0 - disagreement / expected


def label_mode(judge: LLMJudge) -> str:
    """Which labels a judge is calibrated against: ``rubric``, ``pairwise`` or
    ``faithfulness``."""
    return "faithfulness" if isinstance(judge, FaithfulnessJudge) else judge.mode


def categories_of(mode: str) -> tuple[Any, ...]:
    """The labels of a mode, in order (the confusion matrix's rows and columns)."""
    match mode:
        case "binary":
            return BINARY_LABELS
        case "rubric":
            return RUBRIC_LABELS
        case "faithfulness":
            return FAITHFULNESS_LABELS
        case _:
            return PAIR_LABELS


def parse_labels(data: bytes | str, mode: str) -> list[LabelledItem]:
    text = data.decode() if isinstance(data, bytes) else data
    allowed = categories_of(mode)
    items: list[LabelledItem] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if len(items) >= MAX_ITEMS:
            raise InvalidInputError(f"line {number}: more than {MAX_ITEMS} labelled items")
        try:
            item = LabelledItem.model_validate_json(line)
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(p) for p in first["loc"][:1]) or "line"
            raise InvalidInputError(f"line {number}: {where}: {first['msg']}") from None
        if item.label not in allowed:
            raise InvalidInputError(
                f"line {number}: label must be one of {', '.join(map(str, allowed))}"
            )
        items.append(item)
    if not items:
        raise InvalidInputError("no labelled items")
    return items


async def calibrate(
    judge: LLMJudge, items: Sequence[LabelledItem], *, concurrency: int = 8
) -> CalibrationReport:
    """Run ``judge`` on every labelled item and measure agreement with the labels."""
    gate = asyncio.Semaphore(concurrency)

    if isinstance(judge, TrajectoryJudge):
        missing = [item.id for item in items if not item.trajectory]
        if missing:
            raise InvalidInputError(
                f"a trajectory judge needs a trajectory on every item; missing on {missing[0]}"
            )

    async def one(item: LabelledItem) -> Judgement:
        async with gate:
            case = Case(id=item.id, input=item.input, expected=item.expected)
            if isinstance(judge, TrajectoryJudge):
                steps = [s.to_step() for s in item.trajectory or ()]
                return await judge.judge_steps(case, item.output, steps)
            return await judge.judge(case, item.output)

    judgements = await asyncio.gather(*(one(item) for item in items))
    human: list[Hashable] = []
    machine: list[Hashable] = []
    disagreements = []
    for item, j in zip(items, judgements, strict=True):
        if j.label is None:
            continue
        human.append(item.label)
        machine.append(j.label)
        if item.label != j.label:
            disagreements.append(item.id)
    if not human:
        raise InvalidInputError("the judge could not grade any labelled item")
    mode = label_mode(judge)
    categories = categories_of(mode)
    index = {c: i for i, c in enumerate(categories)}
    confusion = [[0] * len(categories) for _ in categories]
    for h, m in zip(human, machine, strict=True):
        confusion[index[h]][index[m]] += 1
    return CalibrationReport(
        n=len(items),
        judged=len(human),
        agreement=sum(h == m for h, m in zip(human, machine, strict=True)) / len(human),
        kappa=cohens_kappa(human, machine, categories),
        weighted_kappa=(
            cohens_kappa(human, machine, categories, quadratic=True) if mode == "rubric" else None
        ),
        labels=[str(c) for c in categories],
        confusion=confusion,
        disagreements=disagreements[:50],
        input_tokens=sum(j.input_tokens for j in judgements),
        output_tokens=sum(j.output_tokens for j in judgements),
        cost_usd=sum(j.cost_usd for j in judgements),
    )


def save_calibration(session: Session, judge: LLMJudge, report: CalibrationReport) -> Calibration:
    row = Calibration(
        judge_hash=judge.fingerprint(),
        judge_spec=judge.spec(),
        n=report.judged,
        kappa=report.kappa,
        weighted_kappa=report.weighted_kappa,
        agreement=report.agreement,
        confusion={"labels": report.labels, "matrix": report.confusion},
    )
    session.add(row)
    session.commit()
    return row


@dataclass(frozen=True)
class ThresholdPoint:
    """Verdicts at one threshold: pass when the cosine is at least ``threshold``."""

    threshold: float
    kappa: float
    agreement: float
    precision: float | None  # of the cases passed, the share labelled 1
    recall: float | None  # of the cases labelled 1, the share passed
    confusion: list[list[int]]  # rows: label 0, 1; columns: verdict fail, pass


@dataclass(frozen=True)
class ThresholdReport:
    n: int
    scored: int
    positives: int
    best: ThresholdPoint
    cv_kappa: float | None
    cv_agreement: float | None
    folds: int
    sweep: list[ThresholdPoint]
    min_cosine: float
    max_cosine: float
    disagreements: list[str] = field(default_factory=list)
    input_tokens: int = 0
    cost_usd: float = 0.0


def binary_kappa(tp: int, fp: int, fn: int, tn: int) -> float:
    """Cohen's kappa of two pass/fail raters from their 2x2 table; equal to
    ``cohens_kappa`` on the same verdicts."""
    n = tp + fp + fn + tn
    if n == 0:
        raise ValueError("cannot compute kappa for no items")
    observed = (tp + tn) / n
    chance = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    if chance == 1.0:
        return 1.0  # both raters used one category: agreement is total
    return (observed - chance) / (1.0 - chance)


def _point(threshold: float, tp: int, fp: int, fn: int, tn: int) -> ThresholdPoint:
    n = tp + fp + fn + tn
    return ThresholdPoint(
        threshold=threshold,
        kappa=binary_kappa(tp, fp, fn, tn),
        agreement=(tp + tn) / n,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        confusion=[[tn, fp], [fn, tp]],
    )


def best_threshold(
    cosines: npt.NDArray[np.float64], labels: npt.NDArray[np.bool_]
) -> ThresholdPoint:
    """The threshold whose verdicts agree best with ``labels`` (highest kappa,
    then highest agreement, then the strictest).

    Only thresholds between two neighbouring distinct cosines change a verdict,
    so the candidates are one above the highest cosine (nothing passes) and
    one per distinct cosine, each the midpoint down to the next lower cosine:
    the same verdicts on these items as the cosine itself, with a margin on
    either side for new ones. One sort and two running sums: O(n log n)."""
    if cosines.shape != labels.shape or cosines.size == 0:
        raise ValueError("need one label per cosine, and at least one")
    order = np.argsort(-cosines, kind="stable")
    values = cosines[order]
    passed = labels[order]
    positives = int(passed.sum())
    negatives = int(values.size) - positives
    tp_run = np.cumsum(passed)
    fp_run = np.cumsum(~passed)
    # The last position of each run of equal cosines.
    ends = np.flatnonzero(np.r_[values[1:] != values[:-1], True])
    top = float(values[0])
    best = _point(float(np.nextafter(top, math.inf)), 0, 0, positives, negatives)
    for j, end in enumerate(ends):
        lower = float(values[ends[j + 1]]) if j + 1 < len(ends) else None
        value = float(values[end])
        threshold = value if lower is None else (value + lower) / 2
        tp, fp = int(tp_run[end]), int(fp_run[end])
        point = _point(threshold, tp, fp, positives - tp, negatives - fp)
        if (point.kappa, point.agreement) > (best.kappa, best.agreement):
            best = point
    return best


def _at(
    cosines: npt.NDArray[np.float64], labels: npt.NDArray[np.bool_], t: float
) -> ThresholdPoint:
    passed = cosines >= t
    tp = int((passed & labels).sum())
    fp = int((passed & ~labels).sum())
    fn = int((~passed & labels).sum())
    return _point(t, tp, fp, fn, int(labels.size) - tp - fp - fn)


def cross_validate(
    ids: Sequence[str],
    cosines: npt.NDArray[np.float64],
    labels: npt.NDArray[np.bool_],
    folds: int,
) -> ThresholdPoint | None:
    """Pooled verdicts of ``folds``-fold cross-validation: each fold judged at the
    threshold chosen on the others. Folds come from a hash of each item's id, so
    they do not depend on the file's order. None when there are fewer than two
    items per fold."""
    if folds < 2 or len(ids) < 2 * folds:
        return None
    order = sorted(range(len(ids)), key=lambda i: sha256_hex(ids[i]))
    fold_of = np.empty(len(ids), dtype=np.int64)
    for position, i in enumerate(order):
        fold_of[i] = position % folds
    verdicts = np.zeros(len(ids), dtype=np.bool_)
    for fold in range(folds):
        held = fold_of == fold
        chosen = best_threshold(cosines[~held], labels[~held]).threshold
        verdicts[held] = cosines[held] >= chosen
    tp = int((verdicts & labels).sum())
    fp = int((verdicts & ~labels).sum())
    fn = int((~verdicts & labels).sum())
    return _point(math.nan, tp, fp, fn, int(labels.size) - tp - fp - fn)


async def calibrate_threshold(
    scorer: SemanticSimilarity, items: Sequence[LabelledItem], *, folds: int = DEFAULT_FOLDS
) -> ThresholdReport:
    """Embed every output and reference (batched and cached by the scorer's
    embedder) and find the threshold that agrees best with the 0/1 labels. An
    item without a reference text, or whose vectors have no direction, is left
    out; an empty output has cosine 0, as the scorer gives it. Raises
    ``TargetError`` when an embedding request fails."""
    pairs: list[tuple[LabelledItem, str, str | None]] = []
    for item in items:
        reference = scorer.reference(Case(id=item.id, input=item.input, expected=item.expected))
        if reference is not None:
            pairs.append((item, reference, item.output if item.output.strip() else None))
    # Every distinct text in one call: the embedder batches and caches them.
    texts = list(dict.fromkeys(t for _, ref, out in pairs for t in (ref, out) if t is not None))
    got = await scorer.embedder.embed(texts) if texts else None
    vector = dict(zip(texts, got.vectors, strict=True)) if got is not None else {}
    ids: list[str] = []
    values: list[float] = []
    truth: list[bool] = []
    for item, reference, output in pairs:
        value = 0.0 if output is None else cosine(vector[output], vector[reference])
        if value is None:
            continue
        ids.append(item.id)
        values.append(value)
        truth.append(item.label == 1)
    if not values:
        raise InvalidInputError("no labelled item could be scored (each needs a reference text)")
    cosines = np.asarray(values, dtype=np.float64)
    labels = np.asarray(truth, dtype=np.bool_)
    best = best_threshold(cosines, labels)
    cv = cross_validate(ids, cosines, labels, folds)
    low, high = float(cosines.min()), float(cosines.max())
    grid = [
        _at(cosines, labels, t)
        for t in np.round(np.arange(-1.0, 1.0001, 0.05), 2)
        if low - 0.05 <= t <= high + 0.05
    ]
    verdicts = cosines >= best.threshold
    return ThresholdReport(
        n=len(items),
        scored=len(values),
        positives=int(labels.sum()),
        best=best,
        cv_kappa=None if cv is None else cv.kappa,
        cv_agreement=None if cv is None else cv.agreement,
        folds=folds if cv is not None else 0,
        sweep=grid,
        min_cosine=low,
        max_cosine=high,
        disagreements=[i for i, v, t in zip(ids, verdicts, truth, strict=True) if v != t][:50],
        input_tokens=got.usage.input_tokens if got is not None else 0,
        cost_usd=got.usage.cost_usd if got is not None else 0.0,
    )


def save_threshold(
    session: Session, scorer: SemanticSimilarity, report: ThresholdReport
) -> Calibration:
    """Stored under the embedder's fingerprint: the threshold belongs to the model."""
    best = report.best
    row = Calibration(
        judge_hash=scorer.embedder.fingerprint(),
        judge_spec={"type": "semantic_similarity", **scorer.embedder.identity()},
        n=report.scored,
        kappa=best.kappa,
        weighted_kappa=None,
        agreement=best.agreement,
        confusion={
            "labels": [str(label) for label in BINARY_LABELS],
            "matrix": best.confusion,
            "threshold": best.threshold,
            "cv_kappa": report.cv_kappa,
            "folds": report.folds,
        },
    )
    session.add(row)
    session.commit()
    return row
