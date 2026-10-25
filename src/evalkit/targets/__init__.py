"""Targets: the systems a run evaluates."""

from evalkit.targets.base import Generation, Target, TargetError, Usage
from evalkit.targets.specs import (
    FakeTargetSpec,
    HttpTargetSpec,
    PromptTargetSpec,
    TargetSpec,
    target_hash,
)

__all__ = [
    "FakeTargetSpec",
    "Generation",
    "HttpTargetSpec",
    "PromptTargetSpec",
    "Target",
    "TargetError",
    "TargetSpec",
    "Usage",
    "target_hash",
]
