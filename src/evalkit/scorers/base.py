"""The scorer interface. Built-in and plugin scorers implement the same class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from evalkit.datasets import Case
from evalkit.targets.base import Step


@dataclass(frozen=True)
class ScoreResult:
    """``value`` is in [0, 1]; ``None`` means the case could not be scored."""

    value: float | None
    detail: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @classmethod
    def passed(cls, ok: bool, detail: str = "") -> ScoreResult:
        return cls(1.0 if ok else 0.0, detail)

    @classmethod
    def unscorable(cls, detail: str) -> ScoreResult:
        return cls(None, detail)


class Scorer(ABC):
    """Scores one output against one case.

    Implementations must be safe to call concurrently from many tasks and must
    not keep per-case state. Plugins receive the options table from the run
    config as ``options``.
    """

    def __init__(self, name: str, options: dict[str, Any] | None = None) -> None:
        self.name = name
        self.options = options or {}

    @abstractmethod
    async def score(self, case: Case, output: str) -> ScoreResult:
        """Return a score for ``output``; raise only for programming errors."""

    async def score_steps(self, case: Case, output: str, steps: Sequence[Step]) -> ScoreResult:
        """Score with the agent's trajectory: every model reply and tool call, in
        order, or nothing when the target is not an agent (or the result came
        from the response cache). The runner calls this; scorers that grade only
        the output keep the default, which ignores ``steps``."""
        return await self.score(case, output)


def expected_value(case: Case, key: str | None) -> Any:
    """The reference value, optionally one key of a structured ``expected``."""
    if key is None:
        return case.expected
    if isinstance(case.expected, dict):
        return case.expected.get(key)
    return None
