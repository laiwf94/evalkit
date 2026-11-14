"""Targets: the systems a run evaluates."""

from evalkit.targets.base import Generation, Step, Target, TargetError, Turn, Usage
from evalkit.targets.specs import (
    AgentTargetSpec,
    ConversationTargetSpec,
    FakeTargetSpec,
    HttpAgentTargetSpec,
    HttpTargetSpec,
    PromptTargetSpec,
    TargetSpec,
    target_hash,
)

__all__ = [
    "AgentTargetSpec",
    "ConversationTargetSpec",
    "FakeTargetSpec",
    "Generation",
    "HttpAgentTargetSpec",
    "HttpTargetSpec",
    "PromptTargetSpec",
    "Step",
    "Target",
    "TargetError",
    "TargetSpec",
    "Turn",
    "Usage",
    "target_hash",
]
