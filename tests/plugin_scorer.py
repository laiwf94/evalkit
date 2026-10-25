"""A plugin scorer used by the tests: passes when the output is short enough."""

from typing import Any

from evalkit.datasets import Case
from evalkit.scorers import Scorer, ScoreResult


class MaxLength(Scorer):
    def __init__(self, name: str, options: dict[str, Any]) -> None:
        super().__init__(name, options)
        self.limit = int(options.get("limit", 10))

    async def score(self, case: Case, output: str) -> ScoreResult:
        return ScoreResult.passed(len(output) <= self.limit)


NOT_A_SCORER = object()


class Exploding(Scorer):
    async def score(self, case: Case, output: str) -> ScoreResult:
        raise RuntimeError("plugin bug")


class OutOfRange(Scorer):
    async def score(self, case: Case, output: str) -> ScoreResult:
        return ScoreResult(float(self.options.get("value", 2.0)))
