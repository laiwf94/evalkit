"""Scorers turn (case, output) into a number in [0, 1]."""

from evalkit.scorers.base import Scorer, ScoreResult
from evalkit.scorers.registry import ScorerSpec, build_scorers

__all__ = ["ScoreResult", "Scorer", "ScorerSpec", "build_scorers"]
