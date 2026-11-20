"""Ranking metrics for a retriever's output against graded relevance labels.

Each case's output is a ranking written by the retrieval target
(``evalkit.targets.retrieval``) and its labels map document ids to grades
(``expected.relevance``, or the key the scorer names). A document is relevant
when its grade is at least ``min_grade``; unlabelled documents are not. With
``R`` the relevant documents and ``r_i`` the i-th returned document (from 1):

* ``recall@k`` = |R in the top k| / |R|
* ``precision@k`` = |R in the top k| / k (a short list does not raise it)
* ``hit_rate@k`` = 1 when any of R is in the top k
* ``mrr`` = 1 / the rank of the first relevant document, within the top k
* ``map@k`` = the sum over relevant ranks i <= k of precision@i, over min(|R|, k)
* ``ndcg@k`` = DCG@k / IDCG@k with DCG@k = sum gain(g(r_i)) / log2(i + 1), and
  IDCG@k the same over the labels sorted best first. ``gain`` is the grade
  (``linear``, as in trec_eval) or 2^grade - 1
  (``exponential``). Grades below ``min_grade`` gain nothing.

A document that appears more than once counts once, at its first rank, as the
retrieval target already reads it; a repeat does not take a rank.

A case with no relevant document labelled cannot be scored by any of these
and is reported as unscorable rather than as a zero.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from evalkit.datasets import Case
from evalkit.scorers.base import Scorer, ScoreResult, expected_value
from evalkit.targets.retrieval import parse_labels, parse_ranking

Metric = Literal["recall", "precision", "hit_rate", "mrr", "map", "ndcg"]
Gain = Literal["linear", "exponential"]
METRICS: tuple[Metric, ...] = ("recall", "precision", "hit_rate", "mrr", "map", "ndcg")


def _top(ranked: Sequence[str], k: int | None) -> list[str]:
    """The first ``k`` distinct documents, in rank order."""
    return list(dict.fromkeys(ranked))[:k]


def _relevant(labels: Mapping[str, int], min_grade: int) -> set[str]:
    return {doc for doc, grade in labels.items() if grade >= min_grade}


def recall_at_k(
    ranked: Sequence[str], labels: Mapping[str, int], k: int, min_grade: int = 1
) -> float:
    relevant = _relevant(labels, min_grade)
    if not relevant:
        raise ValueError("no relevant document is labelled")
    return len(relevant.intersection(_top(ranked, k))) / len(relevant)


def precision_at_k(
    ranked: Sequence[str], labels: Mapping[str, int], k: int, min_grade: int = 1
) -> float:
    relevant = _relevant(labels, min_grade)
    if not relevant:
        raise ValueError("no relevant document is labelled")
    return sum(1 for doc in _top(ranked, k) if doc in relevant) / k


def hit_rate_at_k(
    ranked: Sequence[str], labels: Mapping[str, int], k: int, min_grade: int = 1
) -> float:
    relevant = _relevant(labels, min_grade)
    if not relevant:
        raise ValueError("no relevant document is labelled")
    return 1.0 if relevant.intersection(_top(ranked, k)) else 0.0


def reciprocal_rank(
    ranked: Sequence[str], labels: Mapping[str, int], k: int | None = None, min_grade: int = 1
) -> float:
    relevant = _relevant(labels, min_grade)
    if not relevant:
        raise ValueError("no relevant document is labelled")
    for rank, doc in enumerate(_top(ranked, k), start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(
    ranked: Sequence[str], labels: Mapping[str, int], k: int, min_grade: int = 1
) -> float:
    relevant = _relevant(labels, min_grade)
    if not relevant:
        raise ValueError("no relevant document is labelled")
    found = 0
    total = 0.0
    for rank, doc in enumerate(_top(ranked, k), start=1):
        if doc in relevant:
            found += 1
            total += found / rank
    return total / min(len(relevant), k)


def _gain(grade: int, gain: Gain) -> float:
    return float(grade) if gain == "linear" else float(2**grade - 1)


def ndcg_at_k(
    ranked: Sequence[str],
    labels: Mapping[str, int],
    k: int,
    min_grade: int = 1,
    gain: Gain = "linear",
) -> float:
    def grade(doc: str) -> int:
        g = labels.get(doc, 0)
        return g if g >= min_grade else 0

    ideal_grades = sorted((grade(d) for d in labels), reverse=True)[:k]
    ideal = sum(_gain(g, gain) / math.log2(i + 2) for i, g in enumerate(ideal_grades))
    if ideal == 0:
        raise ValueError("no relevant document is labelled")
    dcg = sum(_gain(grade(d), gain) / math.log2(i + 2) for i, d in enumerate(_top(ranked, k)))
    return dcg / ideal


_AT_K: dict[str, Callable[[Sequence[str], Mapping[str, int], int, int], float]] = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "hit_rate": hit_rate_at_k,
    "map": average_precision_at_k,
}


def default_name(metric: Metric, k: int | None) -> str:
    """``recall_at_10``, ``ndcg_at_5``, or ``mrr`` when MRR has no cut-off."""
    return metric if k is None else f"{metric}_at_{k}"


class RankingMetric(Scorer):
    """One ranking metric per case. ``k`` defaults to 10, except for MRR, which
    looks at the whole ranking unless given one."""

    reads_ranking = True

    def __init__(
        self,
        name: str,
        *,
        metric: Metric,
        k: int | None = None,
        expected_key: str | None = "relevance",
        min_grade: int = 1,
        gain: Gain = "linear",
    ) -> None:
        super().__init__(name)
        if metric not in METRICS:
            raise ValueError(f"unknown ranking metric {metric!r}")
        if k is None and metric != "mrr":
            k = 10
        if k is not None and k < 1:
            raise ValueError("k must be at least 1")
        if min_grade < 1:
            raise ValueError("min_grade must be at least 1")
        self.metric = metric
        self.k = k
        self.expected_key = expected_key
        self.min_grade = min_grade
        self.gain = gain

    def compute(self, ranked: Sequence[str], labels: Mapping[str, int]) -> float:
        if self.metric == "mrr":
            return reciprocal_rank(ranked, labels, self.k, self.min_grade)
        k = self.k or 10
        if self.metric == "ndcg":
            return ndcg_at_k(ranked, labels, k, self.min_grade, self.gain)
        return _AT_K[self.metric](ranked, labels, k, self.min_grade)

    async def score(self, case: Case, output: str) -> ScoreResult:
        raw = expected_value(case, self.expected_key)
        if raw is None:
            where = "expected" if self.expected_key is None else f"expected.{self.expected_key}"
            return ScoreResult.unscorable(f"case has no relevance labels ({where})")
        try:
            labels = parse_labels(raw)
        except ValueError as exc:
            return ScoreResult.unscorable(str(exc))
        try:
            ranked = parse_ranking(output).ids
        except ValueError as exc:
            return ScoreResult.unscorable(str(exc))
        try:
            value = self.compute(ranked, labels)
        except ValueError as exc:
            return ScoreResult.unscorable(str(exc))
        return ScoreResult(value, self._detail(ranked, labels))

    def _detail(self, ranked: Sequence[str], labels: Mapping[str, int]) -> str:
        relevant = _relevant(labels, self.min_grade)
        window = _top(ranked, self.k)
        found = [i for i, doc in enumerate(window, start=1) if doc in relevant]
        cut = f"the top {self.k}" if self.k is not None else f"all {len(window)}"
        first = f"; first at rank {found[0]}" if found else ""
        return f"{len(found)} of {len(relevant)} relevant in {cut}{first}"
