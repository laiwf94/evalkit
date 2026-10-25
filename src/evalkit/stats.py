"""Confidence intervals and paired comparisons for evaluation scores.

Scores from the same case in two runs are strongly correlated (an easy case is
easy for both prompts), so comparisons are done on per-case differences. That
removes case difficulty from the noise and is why a paired test detects a
regression on 100 cases that an unpaired test would need several hundred for.

Methods, all seeded so a report can be reproduced exactly:

* percentile bootstrap for the mean of one run (Efron & Tibshirani, 1993);
* paired bootstrap for the mean difference between two runs;
* paired sign-flip permutation test for "is there any difference", exact for
  up to 16 pairs and Monte Carlo above that (Good, 2005);
* Cohen's d_z (mean difference over the SD of differences) as effect size;
* Holm's step-down correction when several metrics are gated together.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

Verdict = Literal["better", "worse", "no detectable change"]

DEFAULT_RESAMPLES = 10_000
EXACT_PERMUTATION_MAX_N = 16
# Cap on resample-matrix elements held at once, so 10,000 resamples of 100,000
# cases is processed in chunks instead of allocating 8 GB.
_CHUNK_ELEMENTS = 2_000_000

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float
    n: int


@dataclass(frozen=True)
class Comparison:
    metric: str
    n_pairs: int
    n_unpaired: int
    baseline_mean: float
    candidate_mean: float
    difference: float
    ci_low: float
    ci_high: float
    p_value: float
    p_adjusted: float
    effect_size: float | None
    verdict: Verdict


def _as_array(values: Sequence[float] | FloatArray) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if arr.size and not np.all(np.isfinite(arr)):
        raise ValueError("values must be finite")
    return arr


def _bootstrap_means(values: FloatArray, resamples: int, rng: np.random.Generator) -> FloatArray:
    n = values.size
    out = np.empty(resamples, dtype=np.float64)
    rows = max(1, _CHUNK_ELEMENTS // n)
    for start in range(0, resamples, rows):
        stop = min(resamples, start + rows)
        idx = rng.integers(0, n, size=(stop - start, n))
        out[start:stop] = values[idx].mean(axis=1)
    return out


def bootstrap_ci(
    values: Sequence[float] | FloatArray,
    *,
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap interval for the mean."""
    arr = _as_array(values)
    if arr.size == 0:
        raise ValueError("cannot compute an interval for no values")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if resamples < 100:
        raise ValueError("use at least 100 resamples")
    mean = float(arr.mean())
    if arr.size == 1 or np.all(arr == arr[0]):
        return Interval(mean, mean, mean, int(arr.size))
    means = _bootstrap_means(arr, resamples, np.random.default_rng(seed))
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return Interval(mean, float(low), float(high), int(arr.size))


def permutation_test(
    diffs: Sequence[float] | FloatArray, *, resamples: int = DEFAULT_RESAMPLES, seed: int = 0
) -> float:
    """Two-sided p-value for "the mean paired difference is zero".

    Under the null each difference is equally likely to have either sign, so
    the reference distribution is the mean over random sign flips.
    """
    d = _as_array(diffs)
    n = d.size
    if n == 0:
        raise ValueError("cannot test no pairs")
    observed = abs(float(d.mean()))
    if observed == 0.0:
        return 1.0
    # Floating-point slack so a flip that reproduces the observed mean counts as
    # "at least as extreme" despite summation order.
    tol = 1e-12 * max(1.0, float(np.abs(d).max()))
    if n <= EXACT_PERMUTATION_MAX_N:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)), dtype=np.float64)
        flipped = np.abs(signs @ d) / n
        return float(np.mean(flipped >= observed - tol))
    rng = np.random.default_rng(seed)
    extreme = 0
    rows = max(1, _CHUNK_ELEMENTS // n)
    for start in range(0, resamples, rows):
        count = min(rows, resamples - start)
        flips = rng.integers(0, 2, size=(count, n)) * 2.0 - 1.0
        extreme += int(np.sum(np.abs(flips @ d) / n >= observed - tol))
    # The +1 counts the observed labelling itself, which keeps the test valid
    # (never p = 0) with a finite number of resamples (Phipson & Smyth, 2010).
    return (extreme + 1) / (resamples + 1)


def cohens_dz(diffs: Sequence[float] | FloatArray) -> float | None:
    """Mean difference in units of the SD of differences; None when undefined."""
    d = _as_array(diffs)
    if d.size < 2:
        return None
    sd = float(d.std(ddof=1))
    if sd == 0.0:
        return 0.0 if float(d.mean()) == 0.0 else None
    return float(d.mean()) / sd


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def compare(
    metric: str,
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    alpha: float = 0.05,
    min_effect: float = 0.0,
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Comparison:
    """Paired comparison of one metric between two runs, keyed by case id."""
    return compare_many(
        {metric: (baseline, candidate)},
        alpha=alpha,
        min_effect=min_effect,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )[0]


def compare_many(
    metrics: Mapping[str, tuple[Mapping[str, float], Mapping[str, float]]],
    *,
    alpha: float = 0.05,
    min_effect: float = 0.0,
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> list[Comparison]:
    """Compare several metrics, adjusting p-values with Holm across all of them.

    A metric is ``better`` or ``worse`` only if its adjusted p-value is below
    ``alpha`` and the mean difference is larger than ``min_effect`` in size.
    Everything else is "no detectable change": the data does not show a
    difference, which is not the same as showing there is none.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if min_effect < 0:
        raise ValueError("min_effect must not be negative")
    rows = []
    for name, (base, cand) in metrics.items():
        shared = sorted(base.keys() & cand.keys())
        if not shared:
            raise ValueError(f"metric {name!r}: the runs share no scored cases")
        b = np.array([base[k] for k in shared], dtype=np.float64)
        c = np.array([cand[k] for k in shared], dtype=np.float64)
        d = c - b
        ci = bootstrap_ci(d, confidence=confidence, resamples=resamples, seed=seed)
        p = permutation_test(d, resamples=resamples, seed=seed)
        unpaired = len(base.keys() ^ cand.keys())
        rows.append((name, len(shared), unpaired, b, c, ci, p, cohens_dz(d)))
    adjusted = holm([r[6] for r in rows])
    out = []
    for (name, n, unpaired, b, c, ci, p, dz), p_adj in zip(rows, adjusted, strict=True):
        verdict: Verdict = "no detectable change"
        if p_adj < alpha and abs(ci.mean) > min_effect:
            verdict = "better" if ci.mean > 0 else "worse"
        out.append(
            Comparison(
                metric=name,
                n_pairs=n,
                n_unpaired=unpaired,
                baseline_mean=float(b.mean()),
                candidate_mean=float(c.mean()),
                difference=ci.mean,
                ci_low=ci.low,
                ci_high=ci.high,
                p_value=p,
                p_adjusted=p_adj,
                effect_size=dz,
                verdict=verdict,
            )
        )
    return out
