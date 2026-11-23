"""Near-duplicate detection with MinHash and locality-sensitive hashing, in numpy.

A text is normalised (NFKC, case-folded, runs of non-word characters collapsed
to one space) and cut into overlapping shingles: character n-grams by default,
which also work for short questions and for scripts written without spaces,
or word n-grams. Each shingle is hashed to an integer below the prime
``PRIME`` by a polynomial hash computed for every window at once, so a text
becomes a set of integers without a Python loop over its shingles.

The Jaccard similarity of two shingle sets (shared shingles over all of their
shingles) is estimated from MinHash signatures (Broder, *On the resemblance and
containment of documents*): ``num_perm`` hash functions, and for each one the
smallest value over the set. Two sets share a minimum with probability equal to their Jaccard
similarity, so the fraction of equal positions estimates it with standard error
sqrt(J(1 - J) / num_perm). The hash functions are multiply-add-shift,
h(x) = (a·x + b) mod 2**64 with random 64-bit a (odd) and b, which is strongly
universal in its top 32 bits for 32-bit shingle hashes (Dietzfelbinger,
*Universal hashing and k-wise independent random variables via integer
arithmetic without primes*). The modulus is uint64 wraparound, so there is no
division. The full 64-bit minimum is kept: it is the minimum of the top 32 bits,
with ties broken by the low bits.

Comparing every pair is quadratic. LSH splits a signature into ``bands`` bands
of ``rows`` values; two texts become candidates when any band is identical,
which happens with probability 1 - (1 - J**rows)**bands, an S-curve that is
steep around the threshold. ``lsh_params`` picks the split that minimises the
area under the false positive and false negative parts of that curve, with
false negatives weighted four times as heavily, and each candidate pair is then
checked against the threshold on its full signature.

``find_duplicates`` keeps the first of each group of near-duplicates, in the
order given, after anything in ``against`` (cases already in the dataset), and
says for every other text which kept text it repeats and how similar they are.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import regex

# The largest prime below 2**32: shingle hashes are taken modulo it.
PRIME = 4_294_967_291
_BASE = 1_000_003  # the polynomial hash's multiplier, below PRIME
DEFAULT_PERMUTATIONS = 128
DEFAULT_THRESHOLD = 0.8
DEFAULT_NGRAM = 5
MAX_PERMUTATIONS = 1024
# Signature work is done in blocks of about this many shingles into one reused
# buffer, so memory stays at num_perm * _BLOCK * 8 bytes (8 MiB at 128
# permutations) for any input.
_BLOCK = 8192
_NO_VALUE = np.iinfo(np.uint64).max
# Shingles are taken from the first MAX_TEXT_CHARS characters of a text.
MAX_TEXT_CHARS = 100_000
_SEPARATORS = regex.compile(r"[^\w]+")
_WORDS = regex.compile(r"\w+")

Unit = Literal["char", "word"]
U64 = npt.NDArray[np.uint64]


def normalise(text: str) -> str:
    """NFKC, case-folded, with every run of non-word characters made one space."""
    folded = unicodedata.normalize("NFKC", text[:MAX_TEXT_CHARS]).casefold()
    return _SEPARATORS.sub(" ", folded).strip()


def _symbols(text: str, unit: Unit) -> U64:
    """The normalised text as integers below PRIME: code points, or word hashes."""
    clean = normalise(text)
    if unit == "char":
        return np.frombuffer(clean.encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)
    words = _WORDS.findall(clean)
    return np.array(
        [
            int.from_bytes(hashlib.blake2b(w.encode(), digest_size=4).digest(), "little") % PRIME
            for w in words
        ],
        dtype=np.uint64,
    )


def shingles(text: str, *, unit: Unit = "char", n: int = DEFAULT_NGRAM) -> U64:
    """The sorted, distinct shingle hashes of ``text``.

    A text shorter than ``n`` symbols is one shingle; an empty one (nothing but
    punctuation and space) is the shingle 0, so all empty texts are duplicates
    of each other."""
    if n < 1:
        raise ValueError("n must be at least 1")
    symbols = _symbols(text, unit)
    if symbols.size == 0:
        return np.zeros(1, dtype=np.uint64)
    width = min(n, int(symbols.size))
    count = int(symbols.size) - width + 1
    hashes = np.zeros(count, dtype=np.uint64)
    for j in range(width):
        # h < PRIME < 2**32 and _BASE < 2**20, so h * _BASE + symbol < 2**64.
        hashes = (hashes * np.uint64(_BASE) + symbols[j : j + count]) % np.uint64(PRIME)
    return np.unique(hashes)


def jaccard(a: U64, b: U64) -> float:
    """The exact Jaccard similarity of two shingle sets (sorted, distinct)."""
    union = np.union1d(a, b).size
    if union == 0:
        return 1.0
    return float(np.intersect1d(a, b, assume_unique=True).size / union)


class MinHasher:
    """``num_perm`` multiply-add-shift hash functions drawn from ``seed``."""

    def __init__(self, num_perm: int = DEFAULT_PERMUTATIONS, seed: int = 1) -> None:
        if not 1 <= num_perm <= MAX_PERMUTATIONS:
            raise ValueError(f"num_perm must be 1 to {MAX_PERMUTATIONS}")
        rng = np.random.default_rng(seed)
        top = np.iinfo(np.uint64).max
        self.num_perm = num_perm
        self.seed = seed
        self.a = rng.integers(0, top, size=(num_perm, 1), dtype=np.uint64, endpoint=True)
        self.a |= np.uint64(1)
        self.b = rng.integers(0, top, size=(num_perm, 1), dtype=np.uint64, endpoint=True)

    def signature(self, shingle_set: U64) -> U64:
        """The MinHash signature of one shingle set: ``num_perm`` uint64 values."""
        row: U64 = self.signatures([shingle_set])[0]
        return row

    def signatures(self, sets: Sequence[U64]) -> npt.NDArray[np.uint64]:
        """Signatures of many sets as one (len(sets), num_perm) array.

        Sets are hashed together in blocks of about ``_BLOCK`` shingles, into one
        reused buffer, and each set's minimum is taken with
        ``np.minimum.reduceat``; a set larger than a block is hashed in pieces and
        its minima combined."""
        out = np.full((len(sets), self.num_perm), _NO_VALUE, dtype=np.uint64)
        buffer = np.empty((self.num_perm, _BLOCK), dtype=np.uint64)
        pending: list[int] = []
        size = 0

        def hashed(values: U64) -> U64:
            view = buffer[:, : values.size]
            # uint64 arrays wrap on overflow: that is the mod 2**64.
            np.multiply(self.a, values[np.newaxis, :], out=view)
            np.add(view, self.b, out=view)
            return view

        def flush() -> None:
            nonlocal size
            if not pending:
                return
            values = np.concatenate([sets[i] for i in pending])
            starts = np.cumsum([0] + [sets[i].size for i in pending[:-1]])
            out[pending] = np.minimum.reduceat(hashed(values), starts, axis=1).T
            pending.clear()
            size = 0

        for i, values in enumerate(sets):
            if values.size == 0:
                raise ValueError(f"set {i} is empty; shingles() never gives an empty set")
            if values.size > _BLOCK:
                flush()
                for start in range(0, values.size, _BLOCK):
                    piece = hashed(values[start : start + _BLOCK]).min(axis=1)
                    np.minimum(out[i], piece, out=out[i])
                continue
            if size + values.size > _BLOCK:
                flush()
            pending.append(i)
            size += values.size
        flush()
        return out


def estimate(sig_a: U64, sig_b: U64) -> float:
    """The Jaccard similarity estimated from two signatures of one MinHasher."""
    return float(np.mean(sig_a == sig_b))


def lsh_params(
    threshold: float, num_perm: int, *, false_positive_weight: float = 0.2
) -> tuple[int, int]:
    """(bands, rows), bands * rows <= num_perm, minimising the weighted sum of the
    false positive area below ``threshold`` and the false negative area above it.

    Every candidate is checked on its full signature, so a false positive costs
    one comparison while a false negative is a duplicate let through: false
    negatives weigh four times as much by default."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    weight_fp, weight_fn = false_positive_weight, 1 - false_positive_weight
    below = np.linspace(0.0, threshold, 201)
    above = np.linspace(threshold, 1.0, 201)
    best: tuple[float, int, int] | None = None
    for bands in range(1, num_perm + 1):
        for rows in range(1, num_perm // bands + 1):
            false_pos = np.trapezoid(1 - (1 - below**rows) ** bands, below)
            false_neg = np.trapezoid((1 - above**rows) ** bands, above)
            error = float(weight_fp * false_pos + weight_fn * false_neg)
            if best is None or error < best[0] - 1e-12:
                best = (error, bands, rows)
    if best is None:  # pragma: no cover - num_perm >= 1 always gives a candidate
        raise ValueError("no band layout fits")
    return best[1], best[2]


class LSHIndex:
    """Signatures in ``bands`` hash tables, one per band of ``rows`` values."""

    def __init__(self, bands: int, rows: int) -> None:
        self.bands = bands
        self.rows = rows
        self.tables: list[dict[bytes, list[int]]] = [{} for _ in range(bands)]
        self.signatures: list[U64] = []

    def _keys(self, signature: U64) -> Iterable[tuple[int, bytes]]:
        for band in range(self.bands):
            yield band, signature[band * self.rows : (band + 1) * self.rows].tobytes()

    def add(self, signature: U64) -> int:
        """Index ``signature``; returns its position."""
        position = len(self.signatures)
        self.signatures.append(signature)
        for band, key in self._keys(signature):
            self.tables[band].setdefault(key, []).append(position)
        return position

    def candidates(self, signature: U64) -> set[int]:
        """Positions of indexed signatures that share at least one band with it."""
        found: set[int] = set()
        for band, key in self._keys(signature):
            found.update(self.tables[band].get(key, ()))
        return found


@dataclass(frozen=True)
class Duplicate:
    """A text left out because it repeats ``of`` (a kept text or one in ``against``)."""

    id: str
    of: str
    similarity: float


@dataclass(frozen=True)
class DedupeResult:
    kept: list[str]
    duplicates: list[Duplicate] = field(default_factory=list)
    bands: int = 0
    rows: int = 0
    candidate_pairs: int = 0


@dataclass(frozen=True)
class Deduper:
    """Settings for ``find_duplicates``; ``identity`` is what a stored result records."""

    threshold: float = DEFAULT_THRESHOLD
    num_perm: int = DEFAULT_PERMUTATIONS
    unit: Unit = "char"
    n: int = DEFAULT_NGRAM
    seed: int = 1

    def identity(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "num_perm": self.num_perm,
            "unit": self.unit,
            "n": self.n,
            "seed": self.seed,
        }

    def find_duplicates(
        self,
        items: Sequence[tuple[str, str]],
        against: Sequence[tuple[str, str]] = (),
    ) -> DedupeResult:
        """Which of ``items`` (id, text) repeat an earlier item or one of ``against``.

        Items are taken in order: each is compared with every kept item and every
        text in ``against`` that shares an LSH band with it, and is a duplicate of
        the most similar one at or above the threshold (the first on a tie).
        ``against`` texts are never reported, only matched."""
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        hasher = MinHasher(self.num_perm, self.seed)
        bands, rows = lsh_params(self.threshold, self.num_perm)
        index = LSHIndex(bands, rows)
        names: list[str] = []
        texts = [text for _, text in against] + [text for _, text in items]
        sigs = hasher.signatures([shingles(t, unit=self.unit, n=self.n) for t in texts])
        for (key, _), sig in zip(against, sigs[: len(against)], strict=True):
            index.add(sig)
            names.append(key)
        kept: list[str] = []
        duplicates: list[Duplicate] = []
        pairs = 0
        for (key, _), sig in zip(items, sigs[len(against) :], strict=True):
            found = sorted(index.candidates(sig))
            pairs += len(found)
            best: tuple[float, int] | None = None
            for position in found:
                similarity = estimate(sig, index.signatures[position])
                if similarity >= self.threshold and (best is None or similarity > best[0]):
                    best = (similarity, position)
            if best is None:
                index.add(sig)
                names.append(key)
                kept.append(key)
            else:
                duplicates.append(Duplicate(key, names[best[1]], best[0]))
        return DedupeResult(kept, duplicates, bands, rows, pairs)
