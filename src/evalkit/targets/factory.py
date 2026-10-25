"""Turn a target spec into a callable target, with the configuration it needs."""

from __future__ import annotations

from evalkit.errors import InvalidInputError
from evalkit.llm import ClaudeClient
from evalkit.targets.base import Target
from evalkit.targets.fake import FakeTarget
from evalkit.targets.http import HttpEndpoint, HttpTarget
from evalkit.targets.prompt import PromptTarget
from evalkit.targets.specs import (
    FakeTargetSpec,
    HttpTargetSpec,
    PromptTargetSpec,
    target_hash,
)

AnySpec = PromptTargetSpec | HttpTargetSpec | FakeTargetSpec


def resolve_endpoint(spec: HttpTargetSpec, endpoints: dict[str, HttpEndpoint]) -> HttpEndpoint:
    endpoint = endpoints.get(spec.endpoint)
    if endpoint is None:
        known = ", ".join(sorted(endpoints)) or "none configured"
        raise InvalidInputError(f"unknown HTTP endpoint {spec.endpoint!r} (known: {known})")
    return endpoint


def identity(spec: AnySpec, endpoints: dict[str, HttpEndpoint]) -> str:
    """The target hash, including the resolved endpoint for HTTP targets."""
    if isinstance(spec, HttpTargetSpec):
        return target_hash(spec, resolve_endpoint(spec, endpoints).fingerprint())
    return target_hash(spec)


def build_target(
    spec: AnySpec,
    *,
    endpoints: dict[str, HttpEndpoint],
    claude: ClaudeClient | None,
) -> Target:
    match spec:
        case PromptTargetSpec():
            if claude is None:
                raise InvalidInputError("prompt targets need a Claude client")
            return PromptTarget(spec, claude)
        case HttpTargetSpec():
            return HttpTarget(spec, resolve_endpoint(spec, endpoints))
        case FakeTargetSpec():
            return FakeTarget(spec)
