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

``label`` is 1-5 for rubric judges and "win", "tie" or "loss" (the output
against the reference) for pairwise judges.
"""

from __future__ import annotations

import asyncio
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

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

from evalkit.datasets import Case
from evalkit.db.models import Calibration
from evalkit.errors import InvalidInputError
from evalkit.scorers.judge import LLMJudge

RUBRIC_LABELS: tuple[int, ...] = (1, 2, 3, 4, 5)
PAIR_LABELS: tuple[str, ...] = ("loss", "tie", "win")
MAX_ITEMS = 5_000


class LabelledItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=256)
    input: JsonValue
    expected: JsonValue = None
    output: str = Field(max_length=200_000)
    label: StrictInt | StrictStr


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


def parse_labels(data: bytes | str, mode: str) -> list[LabelledItem]:
    text = data.decode() if isinstance(data, bytes) else data
    allowed: tuple[Any, ...] = RUBRIC_LABELS if mode == "rubric" else PAIR_LABELS
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

    async def one(item: LabelledItem) -> Any:
        async with gate:
            case = Case(id=item.id, input=item.input, expected=item.expected)
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
    categories: tuple[Any, ...] = RUBRIC_LABELS if judge.mode == "rubric" else PAIR_LABELS
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
            cohens_kappa(human, machine, categories, quadratic=True)
            if judge.mode == "rubric"
            else None
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
