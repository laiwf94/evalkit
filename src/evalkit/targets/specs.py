"""Target specs as they appear in run configs, and the stable identity of a target."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from evalkit.datasets import canonical_json, sha256_hex
from evalkit.errors import InvalidInputError
from evalkit.targets.template import placeholders

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class _Spec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PromptTargetSpec(_Spec):
    """A prompt template sent to a Claude model."""

    type: Literal["prompt"]
    model: str = Field(default="claude-opus-5-5", min_length=1, max_length=64)
    system: str = Field(default="", max_length=200_000)
    template: str = Field(min_length=1, max_length=200_000)
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


TargetSpec = Annotated[
    PromptTargetSpec | HttpTargetSpec | FakeTargetSpec, Field(discriminator="type")
]


def target_hash(spec: PromptTargetSpec | HttpTargetSpec | FakeTargetSpec, extra: Any = None) -> str:
    """Identity of a target: every field that can change its output, in canonical form.

    ``extra`` carries resolved configuration that is not in the spec, such as
    the URL behind an HTTP endpoint name.
    """
    return sha256_hex(canonical_json({"spec": spec.model_dump(mode="json"), "extra": extra}))
