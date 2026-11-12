"""Target specs as they appear in run configs, and the stable identity of a target."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evalkit.datasets import canonical_json, sha256_hex
from evalkit.errors import InvalidInputError
from evalkit.targets.template import placeholders

# Provider names, shared with ``evalkit.providers.registry``.
ANTHROPIC = "anthropic"
PROVIDER_NAME = r"^[a-z0-9][a-z0-9_-]{0,63}$"

# Registry prompts (``evalkit.prompts``): NAME, NAME@VERSION or NAME@LABEL.
PROMPT_NAME = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
PROMPT_LABEL = r"^[a-z][a-z0-9._-]{0,63}$"
PROMPT_REF = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}(@([1-9][0-9]{0,8}|[a-z][a-z0-9._-]{0,63}))?$"

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class _Spec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PromptTargetSpec(_Spec):
    """A prompt template sent to a model on a named provider (Claude by default)."""

    type: Literal["prompt"]
    # ``anthropic``, or a provider from ``evalkit.providers.registry``.
    provider: str = Field(default=ANTHROPIC, pattern=PROVIDER_NAME)
    model: str = Field(default="claude-opus-5-5", min_length=1, max_length=64)
    # A registry prompt in place of ``system`` and ``template``. It is resolved
    # when the run is created, and the run stores the content it resolved to.
    prompt: str | None = Field(default=None, pattern=PROMPT_REF)
    system: str = Field(default="", max_length=200_000)
    template: str = Field(default="", max_length=200_000)
    max_tokens: int = Field(default=4096, ge=1, le=128_000)
    effort: Effort = "medium"
    fallbacks: bool = True

    @field_validator("template")
    @classmethod
    def _valid_template(cls, value: str) -> str:
        try:
            placeholders(value)
        except InvalidInputError as exc:
            raise ValueError(exc.message) from None
        return value

    @model_validator(mode="after")
    def _one_source(self) -> PromptTargetSpec:
        if self.prompt is None and not self.template:
            raise ValueError("give a template, or a registry prompt as prompt = NAME@LABEL")
        if self.prompt is not None and (self.template or self.system):
            raise ValueError("give either prompt or system and template, not both")
        return self


class HttpTargetSpec(_Spec):
    """The application under test, behind an endpoint named in operator config."""

    type: Literal["http"]
    endpoint: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    response_field: str = Field(default="output", pattern=r"^[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)*$")


class FakeTargetSpec(_Spec):
    """Deterministic stand-in for demos, tests and benchmarks; makes no network calls."""

    type: Literal["fake"]
    accuracy: float = Field(default=1.0, ge=0, le=1)
    latency_ms: float = Field(default=0.0, ge=0, le=60_000)
    jitter_ms: float = Field(default=0.0, ge=0, le=60_000)
    failure_rate: float = Field(default=0.0, ge=0, le=0.95)
    retry_after_s: float | None = Field(default=None, ge=0, le=60)
    seed: int = 0
    wrong_answer: str = Field(default="I don't know.", max_length=10_000)


class LLMSimulatorSpec(_Spec):
    """A simulated user played by a model (``evalkit.simuser.LLMSimulator``)."""

    type: Literal["llm"]
    provider: str = Field(default=ANTHROPIC, pattern=PROVIDER_NAME)
    model: str = Field(default="claude-sonnet-5-5", min_length=1, max_length=64)
    # Added to the built-in role-play instructions, for every case of the run.
    instructions: str = Field(default="", max_length=20_000)
    max_tokens: int = Field(default=1024, ge=16, le=16_000)
    effort: Effort = "low"


class ScriptedSimulatorSpec(_Spec):
    """A simulated user that says each case's ``user_turns`` in order and calls
    the goal reached when a reply contains one of its ``success`` phrases. It
    makes no calls, so its conversations cost only the assistant's turns."""

    type: Literal["scripted"]
    # Compare success phrases ignoring case.
    ignore_case: bool = True


SimulatorSpec = Annotated[LLMSimulatorSpec | ScriptedSimulatorSpec, Field(discriminator="type")]


class ConversationTargetSpec(_Spec):
    """A model holding a conversation with a simulated user, one case per dialogue.

    The assistant is a model on a provider, as in a prompt target; the case input
    is a ``Conversation`` (``evalkit.datasets``)."""

    type: Literal["conversation"]
    provider: str = Field(default=ANTHROPIC, pattern=PROVIDER_NAME)
    model: str = Field(default="claude-opus-5-5", min_length=1, max_length=64)
    system: str = Field(default="", max_length=200_000)
    max_tokens: int = Field(default=4096, ge=1, le=128_000)
    effort: Effort = "medium"
    fallbacks: bool = True
    simulator: SimulatorSpec = Field(default_factory=lambda: ScriptedSimulatorSpec(type="scripted"))

    def assistant_part(self) -> dict[str, Any]:
        """The fields that decide the assistant's replies: its turn cache entries
        do not depend on who plays the user."""
        return self.model_dump(mode="json", exclude={"simulator", "type"})


AnyTargetSpec = PromptTargetSpec | HttpTargetSpec | FakeTargetSpec | ConversationTargetSpec

TargetSpec = Annotated[AnyTargetSpec, Field(discriminator="type")]


def target_hash(spec: AnyTargetSpec, extra: Any = None) -> str:
    """Identity of a target: every field that can change its output, in canonical form.

    ``extra`` carries resolved configuration that is not in the spec, such as
    the URL behind an HTTP endpoint name, or the endpoint of a prompt target's
    provider. A prompt target names no registry prompt by the time it runs, and
    leaves that field out. A Claude prompt target leaves its provider out, so its identity
    (and with it every cache key) is the one it had before targets named a
    provider.
    """
    dumped = spec.model_dump(mode="json")
    if isinstance(spec, PromptTargetSpec):
        if spec.provider == ANTHROPIC:
            del dumped["provider"]
        # Content resolved from the registry hashes like the same content written
        # inline, so the two share cache entries, and older hashes stay as they were.
        if spec.prompt is None:
            del dumped["prompt"]
    return sha256_hex(canonical_json({"spec": dumped, "extra": extra}))
