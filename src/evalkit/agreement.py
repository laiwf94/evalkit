"""Agreement between annotators: Krippendorff's alpha.

Alpha measures how far a set of annotators agree beyond what chance would give,
for any number of annotators, with labels missing wherever an annotator did not
label a unit, and for nominal, ordinal, interval or ratio labels (Krippendorff,
*Content Analysis*, 4th ed., 2019, ch. 12; "Computing Krippendorff's
Alpha-Reliability", 2011). It is 1 for perfect agreement, 0 when agreement is
what chance predicts, and negative when annotators disagree systematically.

The computation goes through the coincidence matrix. A unit with ``m`` labels
contributes ``m (m - 1)`` ordered pairs of labels, each weighted ``1 / (m - 1)``,
so every pairable label counts once; a unit with fewer than two labels cannot
be paired and is left out. With ``o[c, k]`` the coincidences, ``n_c`` their row
sums and ``n`` the number of pairable labels,

    alpha = 1 - D_o / D_e
    D_o = sum o[c, k] delta(c, k) / n
    D_e = sum n_c n_k delta(c, k) / (n (n - 1))

where ``delta`` is the squared difference the level defines:

    nominal   0 when c = k, else 1
    ordinal   (sum of n_g for g from c to k - (n_c + n_k) / 2) ** 2, over
              the values in their given order
    interval  (c - k) ** 2
    ratio     ((c - k) / (c + k)) ** 2

The coincidences are one matrix product over a units-by-values count matrix,
so 100,000 units of three labels take about 0.06 s with 5 possible labels
and 0.1 s with 50, most of it reading the labels. Alpha is
undefined when the pairable labels show no variation at all (every one is the
same value, so ``D_e`` is 0) or when no unit has two labels; it is then
reported as ``None`` rather than a number that means nothing.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

Level = Literal["nominal", "ordinal", "interval", "ratio"]
LEVELS: tuple[Level, ...] = ("nominal", "ordinal", "interval", "ratio")


@dataclass(frozen=True)
class AlphaResult:
    level: Level
    # None when undefined: fewer than two pairable labels, or no variation in them.
    alpha: float | None
    observed: float | None  # D_o, the observed disagreement
    expected: float | None  # D_e, the disagreement chance would give
    units: int  # units with at least one label
    pairable_units: int  # units with two or more labels
    pairable_values: int  # n: labels in pairable units


def _deltas(
    level: Level, values: Sequence[Hashable], totals: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    v = len(values)
    if level == "nominal":
        return 1.0 - np.eye(v)
    if level == "ordinal":
        # Ranks are the positions in ``values``; the distance between two ranks
        # is the mass of labels between them, counting each end at half.
        cum = np.cumsum(totals)
        lo = np.minimum.outer(np.arange(v), np.arange(v))
        hi = np.maximum.outer(np.arange(v), np.arange(v))
        between = cum[hi] - cum[lo] + totals[lo]
        return (between - (totals[lo] + totals[hi]) / 2.0) ** 2
    numbers = np.asarray(values, dtype=np.float64)
    diff = np.subtract.outer(numbers, numbers)
    if level == "interval":
        return diff**2
    total = np.add.outer(numbers, numbers)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(total == 0, 0.0, diff / np.where(total == 0, 1.0, total))
    return ratio**2


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and np.isfinite(value)


def coincidences(
    units: Sequence[Sequence[Hashable]], values: Sequence[Hashable]
) -> tuple[npt.NDArray[np.float64], int, int]:
    """The coincidence matrix over ``values`` (in that order), the number of units
    with any label and the number with two or more. Every label in ``units`` must
    be one of ``values``."""
    index = {v: i for i, v in enumerate(values)}
    rows: list[int] = []
    cols: list[int] = []
    for u, labels in enumerate(units):
        for label in labels:
            try:
                cols.append(index[label])
            except KeyError:
                raise ValueError(f"unit {u}: label {label!r} is not one of the values") from None
            rows.append(u)
    v = len(values)
    if not rows:
        return np.zeros((v, v)), 0, 0
    counts = np.zeros((len(units), v), dtype=np.float64)
    np.add.at(counts, (np.asarray(rows), np.asarray(cols)), 1.0)
    m = counts.sum(axis=1)
    labelled = int(np.count_nonzero(m))
    keep = m >= 2
    counts, m = counts[keep], m[keep]
    if not len(m):
        return np.zeros((v, v)), labelled, 0
    weighted = counts / (m - 1.0)[:, None]
    o = weighted.T @ counts
    o[np.diag_indices(v)] -= weighted.sum(axis=0)
    return o, labelled, len(m)


def krippendorff_alpha(
    units: Sequence[Sequence[Hashable]],
    level: Level = "nominal",
    values: Sequence[Hashable] | None = None,
) -> AlphaResult:
    """Krippendorff's alpha over ``units``: for each unit, the labels it was given
    (any number, missing ones simply left out).

    ``values`` lists the possible labels; for ``ordinal`` its order is the scale's
    order, and it defaults to the labels seen, sorted. ``interval`` and ``ratio``
    need numbers (``ratio`` non-negative ones)."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {', '.join(LEVELS)}")
    if values is None:
        seen = {label for labels in units for label in labels}
        try:
            values = sorted(seen)  # type: ignore[type-var]
        except TypeError:
            raise ValueError("labels of mixed types need an explicit order of values") from None
    else:
        values = list(dict.fromkeys(values))
    if level in ("interval", "ratio"):
        bad = [v for v in values if not _is_number(v)]
        if bad:
            raise ValueError(f"{level} labels must be numbers; got {bad[0]!r}")
        if level == "ratio" and any(float(v) < 0 for v in values):  # type: ignore[arg-type]
            raise ValueError("ratio labels must not be negative")
    o, labelled, pairable = coincidences(units, values)
    totals = o.sum(axis=1)
    n = float(totals.sum())
    if n < 2:
        return AlphaResult(level, None, None, None, labelled, pairable, round(n))
    delta = _deltas(level, values, totals)
    observed = float((o * delta).sum() / n)
    expected = float((np.outer(totals, totals) * delta).sum() / (n * (n - 1.0)))
    alpha = None if expected <= 1e-15 else 1.0 - observed / expected
    return AlphaResult(level, alpha, observed, expected, labelled, pairable, round(n))
