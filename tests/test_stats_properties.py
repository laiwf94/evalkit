"""Monte Carlo checks that the statistics do what they claim.

Each test simulates many datasets with a known answer and checks the long-run
rate (coverage, false positives, power) against a binomial tolerance band. The
generator is seeded, so the tests are deterministic; the bands make sure the
method is right, not that one random draw was lucky.
"""

import math

import numpy as np

from evalkit.stats import bootstrap_ci, compare, permutation_test

TRIALS = 1000


def band(p: float, trials: int = TRIALS, z: float = 3.29) -> tuple[float, float]:
    half = z * math.sqrt(p * (1 - p) / trials)
    return p - half, p + half


def test_bootstrap_ci_covers_normal_mean() -> None:
    rng = np.random.default_rng(11)
    true_mean = 0.6
    hits = 0
    for i in range(TRIALS):
        sample = rng.normal(true_mean, 0.2, 100)
        ci = bootstrap_ci(sample, resamples=2000, seed=i)
        hits += ci.low <= true_mean <= ci.high
    low, high = band(0.95)
    # The percentile bootstrap runs slightly narrow at n=100; allow a point of slack below.
    assert low - 0.01 <= hits / TRIALS <= high, hits / TRIALS


def test_bootstrap_ci_covers_bernoulli_rate() -> None:
    rng = np.random.default_rng(12)
    p = 0.7
    hits = 0
    for i in range(TRIALS):
        sample = (rng.random(200) < p).astype(float)
        ci = bootstrap_ci(sample, resamples=2000, seed=i)
        hits += ci.low <= p <= ci.high
    low, high = band(0.95)
    assert low - 0.01 <= hits / TRIALS <= high, hits / TRIALS


def test_permutation_false_positive_rate_matches_alpha() -> None:
    rng = np.random.default_rng(13)
    alpha = 0.05
    rejections = 0
    for i in range(TRIALS):
        base = rng.normal(0.5, 0.15, 60)
        cand = rng.normal(0.5, 0.15, 60)
        rejections += permutation_test(cand - base, resamples=999, seed=i) < alpha
    low, high = band(alpha)
    assert low <= rejections / TRIALS <= high, rejections / TRIALS


def test_permutation_false_positive_rate_on_binary_scores() -> None:
    rng = np.random.default_rng(14)
    alpha = 0.05
    rejections = 0
    for i in range(TRIALS):
        base = (rng.random(80) < 0.75).astype(float)
        cand = (rng.random(80) < 0.75).astype(float)
        rejections += permutation_test(cand - base, resamples=999, seed=i) < alpha
    # Discrete differences make the test conservative, never liberal.
    _, high = band(alpha)
    assert rejections / TRIALS <= high, rejections / TRIALS


def test_known_effect_is_detected() -> None:
    rng = np.random.default_rng(15)
    detected = 0
    trials = 300
    for i in range(trials):
        base = rng.normal(0.5, 0.2, 100)
        cand = base + rng.normal(-0.1, 0.2, 100)  # a drop of 0.5 SD of the differences
        scores_b = {str(k): v for k, v in enumerate(base)}
        scores_c = {str(k): v for k, v in enumerate(cand)}
        result = compare("m", scores_b, scores_c, resamples=999, seed=i)
        detected += result.verdict == "worse"
    assert detected / trials >= 0.95, detected / trials
