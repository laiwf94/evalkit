import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evalkit.stats import (
    BOOTSTRAP_MAX_N,
    Interval,
    bootstrap_ci,
    cohens_dz,
    compare,
    compare_aligned,
    compare_many,
    holm,
    mean_interval,
    normal_ci,
    permutation_test,
)

finite = st.floats(-10, 10, allow_nan=False, allow_infinity=False)
samples = st.lists(finite, min_size=2, max_size=40)


def test_bootstrap_basic() -> None:
    ci = bootstrap_ci([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0], seed=1)
    assert ci.mean == 0.75
    assert ci.n == 8
    assert 0.3 < ci.low < ci.mean < ci.high <= 1.0


def test_bootstrap_degenerate_and_invalid() -> None:
    assert bootstrap_ci([0.4]) == bootstrap_ci([0.4, 0.4]).__class__(0.4, 0.4, 0.4, 1)
    assert bootstrap_ci([1.0, 1.0, 1.0]).low == 1.0
    with pytest.raises(ValueError, match="no values"):
        bootstrap_ci([])
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci([1.0, 2.0], confidence=1.0)
    with pytest.raises(ValueError, match="resamples"):
        bootstrap_ci([1.0, 2.0], resamples=10)
    with pytest.raises(ValueError, match="finite"):
        bootstrap_ci([1.0, float("nan")])
    with pytest.raises(ValueError, match="one-dimensional"):
        bootstrap_ci(np.zeros((2, 2)))


def test_bootstrap_chunks_large_inputs_identically() -> None:
    rng = np.random.default_rng(3)
    values = rng.random(300_000)
    ci = bootstrap_ci(values, resamples=200, seed=5)
    assert ci.low < ci.mean < ci.high
    assert ci.high - ci.low < 0.01


@given(samples, st.integers(0, 2**31), finite)
@settings(max_examples=60, deadline=None)
def test_bootstrap_is_shift_equivariant_and_contains_mean(
    values: list[float], seed: int, shift: float
) -> None:
    a = bootstrap_ci(values, resamples=500, seed=seed)
    b = bootstrap_ci([v + shift for v in values], resamples=500, seed=seed)
    assert a.low - 1e-9 <= a.mean <= a.high + 1e-9
    assert b.low == pytest.approx(a.low + shift, abs=1e-6)
    assert b.high == pytest.approx(a.high + shift, abs=1e-6)


@given(samples, st.integers(0, 2**31))
@settings(max_examples=40, deadline=None)
def test_wider_confidence_gives_wider_interval(values: list[float], seed: int) -> None:
    narrow = bootstrap_ci(values, confidence=0.8, resamples=500, seed=seed)
    wide = bootstrap_ci(values, confidence=0.99, resamples=500, seed=seed)
    assert wide.low <= narrow.low + 1e-12
    assert wide.high >= narrow.high - 1e-12


def test_permutation_exact_small_sample() -> None:
    # All four positive: only the all-plus and all-minus flips are as extreme.
    assert permutation_test([1.0, 1.0, 1.0, 1.0]) == pytest.approx(2 / 16)
    assert permutation_test([0.0, 0.0]) == 1.0
    with pytest.raises(ValueError, match="no pairs"):
        permutation_test([])


def test_permutation_monte_carlo_is_never_zero() -> None:
    p = permutation_test([1.0] * 40, resamples=999, seed=1)
    assert p == pytest.approx(1 / 1000)


@given(st.lists(finite, min_size=1, max_size=30), st.integers(0, 2**31))
@settings(max_examples=60, deadline=None)
def test_permutation_p_is_valid_and_sign_symmetric(diffs: list[float], seed: int) -> None:
    p = permutation_test(diffs, resamples=999, seed=seed)
    assert 0.0 < p <= 1.0
    assert permutation_test([-d for d in diffs], resamples=999, seed=seed) == pytest.approx(p)


def test_cohens_dz() -> None:
    assert cohens_dz([1.0]) is None
    assert cohens_dz([0.0, 0.0]) == 0.0
    assert cohens_dz([1.0, 1.0]) is None
    assert cohens_dz([1.0, 3.0]) == pytest.approx(2.0 / np.std([1.0, 3.0], ddof=1))


def test_holm() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5]) == [0.5]
    assert holm([0.9, 0.8]) == [1.0, 1.0]
    assert holm([]) == []


@given(st.lists(st.floats(0, 1), min_size=1, max_size=10))
def test_holm_is_conservative_and_bounded(ps: list[float]) -> None:
    adj = holm(ps)
    assert all(a >= p - 1e-15 for a, p in zip(adj, ps, strict=True))
    assert all(a <= 1.0 for a in adj)
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    assert all(adj[order[i]] <= adj[order[i + 1]] for i in range(len(ps) - 1))


def _scores(values: list[float]) -> dict[str, float]:
    return {f"c{i}": v for i, v in enumerate(values)}


def test_compare_detects_regression_and_improvement() -> None:
    rng = np.random.default_rng(0)
    base = (rng.random(200) < 0.85).astype(float)
    worse = base.copy()
    worse[rng.choice(200, 40, replace=False)] = 0.0
    result = compare("exact", _scores(list(base)), _scores(list(worse)))
    assert result.verdict == "worse"
    assert result.difference < 0
    assert result.ci_high < 0
    assert result.n_pairs == 200
    flipped = compare("exact", _scores(list(worse)), _scores(list(base)))
    assert flipped.verdict == "better"


def test_compare_identical_runs_and_min_effect() -> None:
    same = _scores([1.0, 0.0, 1.0, 1.0])
    result = compare("m", same, same)
    assert result.verdict == "no detectable change"
    assert result.p_value == 1.0
    assert result.effect_size == 0.0
    base = _scores([0.5] * 300)
    cand = _scores([0.52] * 150 + [0.5] * 150)
    assert compare("m", base, cand).verdict == "better"
    assert compare("m", base, cand, min_effect=0.05).verdict == "no detectable change"


def test_compare_pairs_only_shared_cases() -> None:
    base = {"a": 1.0, "b": 0.0, "c": 1.0}
    cand = {"b": 1.0, "c": 1.0, "d": 0.0}
    result = compare("m", base, cand)
    assert result.n_pairs == 2
    assert result.n_unpaired == 2
    with pytest.raises(ValueError, match="share no scored cases"):
        compare("m", {"a": 1.0}, {"b": 1.0})


def test_compare_many_adjusts_across_metrics() -> None:
    rng = np.random.default_rng(7)
    base = rng.normal(0.5, 0.1, 30)
    cand = base + rng.normal(0.04, 0.1, 30)
    single = compare("m", _scores(list(base)), _scores(list(cand)))
    many = compare_many({f"m{i}": (_scores(list(base)), _scores(list(cand))) for i in range(5)})
    assert all(r.p_adjusted >= single.p_value for r in many)
    assert many[0].p_adjusted == pytest.approx(min(1.0, single.p_value * 5))


def test_compare_validates_arguments() -> None:
    s = _scores([1.0, 0.0])
    with pytest.raises(ValueError, match="alpha"):
        compare("m", s, s, alpha=0.0)
    with pytest.raises(ValueError, match="min_effect"):
        compare("m", s, s, min_effect=-1)


def test_normal_ci_from_sums() -> None:
    values = np.array([0.0, 1.0, 1.0, 0.5, 0.25])
    ci = normal_ci(values.size, float(values.sum()), float(values @ values))
    half = 1.959963984540054 * values.std(ddof=1) / np.sqrt(values.size)
    assert (ci.mean, ci.low, ci.high, ci.n) == pytest.approx(
        (values.mean(), values.mean() - half, values.mean() + half, 5)
    )
    assert normal_ci(1, 0.5, 0.25) == Interval(0.5, 0.5, 0.5, 1)
    # Rounding cannot make the variance negative.
    same = normal_ci(3, 0.3, 0.03 - 1e-12)
    assert same.low == same.high == pytest.approx(0.1)
    with pytest.raises(ValueError, match="no values"):
        normal_ci(0, 0.0, 0.0)
    with pytest.raises(ValueError, match="confidence"):
        normal_ci(2, 1.0, 1.0, confidence=1.0)


def test_mean_interval_switches_to_normal_above_the_bootstrap_limit() -> None:
    rng = np.random.default_rng(0)
    small = rng.integers(0, 2, size=500).astype(float)
    assert mean_interval(small, resamples=200, seed=3) == bootstrap_ci(small, resamples=200, seed=3)
    large = rng.integers(0, 2, size=BOOTSTRAP_MAX_N + 1).astype(float)
    normal = mean_interval(large)
    boot = bootstrap_ci(large, resamples=2000, seed=1)
    width = boot.high - boot.low
    assert normal.mean == pytest.approx(large.mean())
    # The bootstrap at this size agrees with the normal interval to within its own noise.
    assert abs(normal.low - boot.low) < 0.05 * width
    assert abs(normal.high - boot.high) < 0.05 * width


def test_compare_aligned_matches_compare_many_and_checks_shapes() -> None:
    base = {"a": 1.0, "b": 0.0, "c": 1.0, "d": 0.5}
    cand = {"b": 1.0, "c": 0.0, "d": 0.5, "e": 1.0}
    keys = sorted(base.keys() | cand.keys())
    aligned = (
        np.array([base.get(k, np.nan) for k in keys]),
        np.array([cand.get(k, np.nan) for k in keys]),
    )
    assert compare_aligned({"m": aligned}, resamples=200) == compare_many(
        {"m": (base, cand)}, resamples=200
    )
    with pytest.raises(ValueError, match="do not line up"):
        compare_aligned({"m": (np.zeros(3), np.zeros(4))})
    with pytest.raises(ValueError, match="share no scored cases"):
        compare_aligned({"m": (np.array([1.0, np.nan]), np.array([np.nan, 1.0]))})
    with pytest.raises(ValueError, match="finite"):
        compare_many({"m": ({"a": float("nan")}, {"a": 1.0})})
