"""MinHash and LSH near-duplicate detection: shingles, exact and estimated Jaccard
similarity on sets of known overlap, batched signatures, the band layout and
planted near-duplicates among random texts."""

import random

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evalkit import dedupe
from evalkit.dedupe import (
    Deduper,
    LSHIndex,
    MinHasher,
    estimate,
    jaccard,
    lsh_params,
    normalise,
    shingles,
)


def u64(values: list[int] | range) -> np.ndarray:
    return np.unique(np.array(list(values), dtype=np.uint64))


def overlapping(n: int, shared: int) -> tuple[np.ndarray, np.ndarray]:
    """Two sets of n elements sharing ``shared``: Jaccard shared / (2n - shared)."""
    return u64(range(n)), u64(range(n - shared, 2 * n - shared))


def test_normalise_folds_case_width_and_punctuation() -> None:
    assert normalise("  Hello,   WORLD!! ") == "hello world"
    # NFKC: full-width letters and the "fi" ligature become plain letters.
    full_width = "\N{FULLWIDTH LATIN CAPITAL LETTER H}\N{FULLWIDTH LATIN SMALL LETTER I}"
    assert normalise(f"{full_width} \N{LATIN SMALL LIGATURE FI}ne") == "hi fine"
    assert normalise("?!...") == ""


def test_shingles_are_case_and_spacing_insensitive() -> None:
    a = shingles("How do I reset my password?")
    assert np.array_equal(a, shingles("how do i  RESET my password"))
    assert a.dtype == np.uint64
    assert np.all(a < dedupe.PRIME)
    assert np.array_equal(a, np.unique(a))
    # "how do i reset my password" is 26 characters: 22 windows of 5, all distinct.
    assert a.size == 22


def test_short_and_empty_texts() -> None:
    assert shingles("abc").size == 1
    assert not np.array_equal(shingles("abc"), shingles("abd"))
    assert np.array_equal(shingles("!!!"), shingles(""))
    assert jaccard(shingles("..."), shingles("  ")) == 1.0
    with pytest.raises(ValueError, match="at least 1"):
        shingles("x", n=0)


def test_word_shingles() -> None:
    words = shingles("the cat sat on the mat", unit="word", n=2)
    # the cat, cat sat, sat on, on the, the mat: five distinct bigrams.
    assert words.size == 5
    assert np.array_equal(words, shingles("The cat, sat on... the MAT", unit="word", n=2))
    assert shingles("one", unit="word", n=3).size == 1


def test_exact_jaccard_by_hand() -> None:
    assert jaccard(u64([1, 2, 3]), u64([2, 3, 4])) == 0.5
    assert jaccard(u64([1, 2]), u64([3, 4])) == 0.0
    assert jaccard(u64([7]), u64([7])) == 1.0
    assert jaccard(u64([]), u64([])) == 1.0
    a, b = overlapping(100, 50)
    assert jaccard(a, b) == pytest.approx(50 / 150)
    # Character shingles of two questions, counted by hand: 22 and 23 windows, 16 shared.
    left = shingles("How do I reset my password?")
    right = shingles("How can I reset my password?")
    assert (left.size, right.size) == (22, 23)
    assert np.intersect1d(left, right).size == 16
    assert jaccard(left, right) == pytest.approx(16 / 29)


@pytest.mark.parametrize("shared", [0, 20, 50, 100, 150, 180, 200])
def test_minhash_estimates_known_overlaps(shared: int) -> None:
    a, b = overlapping(200, shared)
    exact = jaccard(a, b)
    hasher = MinHasher(256, seed=7)
    got = estimate(hasher.signature(a), hasher.signature(b))
    # Within four standard errors (and exact at 0 and 1).
    assert abs(got - exact) <= 4 * np.sqrt(exact * (1 - exact) / 256) + 1e-12


def test_minhash_is_unbiased_over_many_pairs() -> None:
    rng = random.Random(3)
    hasher = MinHasher(128)
    errors = []
    for _ in range(300):
        base = rng.sample(range(10**7), 150)
        keep = rng.randint(0, 150)
        a = u64(base)
        b = u64(base[:keep] + rng.sample(range(10**7, 2 * 10**7), 150 - keep))
        errors.append(estimate(hasher.signature(a), hasher.signature(b)) - jaccard(a, b))
    mean = float(np.mean(errors))
    # The mean of 300 errors with sd <= 0.045 has sd <= 0.0026.
    assert abs(mean) < 0.01
    assert float(np.std(errors)) < 0.05


def test_signatures_are_reproducible_from_the_seed() -> None:
    s = shingles("a reasonably long sentence about nothing in particular")
    assert np.array_equal(MinHasher(64, seed=5).signature(s), MinHasher(64, seed=5).signature(s))
    assert not np.array_equal(MinHasher(64, seed=5).signature(s), MinHasher(64, 6).signature(s))
    with pytest.raises(ValueError, match="num_perm"):
        MinHasher(0)
    with pytest.raises(ValueError, match="empty"):
        MinHasher(8).signatures([u64([])])


@settings(max_examples=40, deadline=None)
@given(
    st.lists(
        st.lists(st.integers(0, 2**32 - 1), min_size=1, max_size=300).map(u64),
        min_size=1,
        max_size=40,
    )
)
def test_batched_signatures_match_one_at_a_time(sets: list[np.ndarray]) -> None:
    hasher = MinHasher(32, seed=2)
    batch = hasher.signatures(sets)
    for i, values in enumerate(sets):
        assert np.array_equal(batch[i], hasher.signature(values))
        # The minimum, checked directly.
        direct = ((hasher.a * values[np.newaxis, :]) + hasher.b).min(axis=1)
        assert np.array_equal(batch[i], direct)


def test_sets_larger_than_a_block(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dedupe, "_BLOCK", 64)
    hasher = MinHasher(16)
    sets = [u64(range(10)), u64(range(1000)), u64(range(5, 70)), u64(range(3))]
    batch = hasher.signatures(sets)
    for i, values in enumerate(sets):
        direct = ((hasher.a * values[np.newaxis, :]) + hasher.b).min(axis=1)
        assert np.array_equal(batch[i], direct)


def test_band_layout() -> None:
    bands, rows = lsh_params(0.8, 128)
    assert bands * rows <= 128
    assert (bands, rows) == (12, 10)
    # With equal weights the layout is steeper but lets more duplicates through.
    assert lsh_params(0.8, 128, false_positive_weight=0.5) == (9, 13)

    def p(j: float, b: int, r: int) -> float:
        return float(1 - (1 - j**r) ** b)

    # Steep around the threshold: pairs well above it become candidates, well below do not.
    assert p(0.9, bands, rows) > 0.99
    assert p(0.4, bands, rows) < 0.002
    _low_bands, low_rows = lsh_params(0.5, 128)
    assert low_rows < rows
    for t in (0.3, 0.5, 0.7, 0.9):
        b, r = lsh_params(t, 64)
        assert b * r <= 64
        assert p(t + 0.1 if t < 0.9 else 0.95, b, r) > p(t - 0.2, b, r)
    with pytest.raises(ValueError, match="between 0 and 1"):
        lsh_params(1.0, 128)


def test_index_finds_shared_bands() -> None:
    index = LSHIndex(bands=4, rows=2)
    first = np.arange(8, dtype=np.uint64)
    index.add(first)
    same_first_band = first.copy()
    same_first_band[2:] += np.uint64(100)
    assert index.candidates(same_first_band) == {0}
    assert index.candidates(first + np.uint64(1000)) == set()


def _random_texts(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    vocabulary = ["".join(rng.choice("abcdefghijklmnop") for _ in range(6)) for _ in range(3000)]
    return [" ".join(rng.choice(vocabulary) for _ in range(40)) for _ in range(n)]


def _edit(text: str, rng: random.Random) -> str:
    words = text.split()
    words[rng.randrange(len(words))] = "changed"
    return " ".join(words)


def test_planted_near_duplicates_are_found() -> None:
    rng = random.Random(11)
    texts = _random_texts(400, seed=1)
    items = [(f"t{i}", t) for i, t in enumerate(texts)]
    planted = {}
    for i in rng.sample(range(400), 60):
        planted[f"near{i}"] = f"t{i}"
        items.append((f"near{i}", _edit(texts[i], rng)))
    result = Deduper(threshold=0.7).find_duplicates(items)
    found = {d.id: d.of for d in result.duplicates}
    assert found == planted
    assert result.kept == [f"t{i}" for i in range(400)]
    exact = {
        d.id: jaccard(shingles(dict(items)[d.id]), shingles(dict(items)[d.of]))
        for d in result.duplicates
    }
    for d in result.duplicates:
        assert d.similarity >= 0.7
        assert abs(d.similarity - exact[d.id]) < 0.2
    # Random texts share no band: candidate pairs are about the planted ones only.
    assert result.candidate_pairs < 100


def test_matches_against_existing_texts_and_keeps_the_first() -> None:
    texts = _random_texts(3, seed=4)
    rng = random.Random(5)
    result = Deduper().find_duplicates(
        [("new", _edit(texts[0], rng)), ("fresh", texts[1]), ("again", texts[1])],
        against=[("old", texts[0]), ("other", texts[2])],
    )
    assert result.kept == ["fresh"]
    assert {(d.id, d.of) for d in result.duplicates} == {("new", "old"), ("again", "fresh")}
    again = next(d for d in result.duplicates if d.id == "again")
    assert again.similarity == 1.0


def test_a_chain_of_edits_points_at_the_kept_text() -> None:
    base = _random_texts(1, seed=9)[0]
    words = base.split()
    one = " ".join([*words[:-1], "x1"])
    two = " ".join([*words[:-2], "x2", "x1"])
    result = Deduper(threshold=0.7).find_duplicates([("a", base), ("b", one), ("c", two)])
    assert result.kept == ["a"]
    assert [(d.id, d.of) for d in result.duplicates] == [("b", "a"), ("c", "a")]


def test_deduper_settings() -> None:
    assert Deduper().identity() == {
        "threshold": 0.8,
        "num_perm": 128,
        "unit": "char",
        "n": 5,
        "seed": 1,
    }
    with pytest.raises(ValueError, match="between 0 and 1"):
        Deduper(threshold=0).find_duplicates([("a", "b")])
    word = Deduper(threshold=0.5, unit="word", n=1)
    result = word.find_duplicates([("a", "red green blue"), ("b", "blue green red")])
    assert [d.id for d in result.duplicates] == ["b"]
