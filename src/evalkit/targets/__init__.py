"""Targets: the systems a run evaluates."""

from evalkit.targets.base import Generation, Target, TargetError, Turn, Usage
from evalkit.targets.specs import (
    ConversationTargetSpec,
    FakeTargetSpec,
    HttpTargetSpec,
    PromptTargetSpec,
    TargetSpec,
    target_hash,
)

__all__ = [
    "ConversationTargetSpec",
    "FakeTargetSpec",
    "Generation",
    "HttpTargetSpec",
    "PromptTargetSpec",
    "Target",
    "TargetError",
    "TargetSpec",
    "Turn",
    "Usage",
    "target_hash",
]
