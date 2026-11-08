"""Turn a target spec into a callable target, with the configuration it needs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from evalkit.errors import InvalidInputError
from evalkit.targets.base import Target
from evalkit.targets.fake import FakeTarget
from evalkit.targets.http import HttpEndpoint, HttpTarget
from evalkit.targets.prompt import PromptTarget
from evalkit.targets.specs import (
    ANTHROPIC,
    FakeTargetSpec,
    HttpTargetSpec,
    PromptTargetSpec,
    target_hash,
)

if TYPE_CHECKING:
    from evalkit.providers.registry import ProviderConfig, ProviderRegistry

AnySpec = PromptTargetSpec | HttpTargetSpec | FakeTargetSpec


def resolve_endpoint(spec: HttpTargetSpec, endpoints: Mapping[str, HttpEndpoint]) -> HttpEndpoint:
    endpoint = endpoints.get(spec.endpoint)
    if endpoint is None:
        known = ", ".join(sorted(endpoints)) or "none configured"
        raise InvalidInputError(f"unknown HTTP endpoint {spec.endpoint!r} (known: {known})")
    return endpoint


def identity(
    spec: AnySpec,
    endpoints: Mapping[str, HttpEndpoint],
    providers: Mapping[str, ProviderConfig] | None = None,
) -> str:
    """The target hash, including the resolved endpoint for HTTP targets and,
    for a prompt target on a provider other than Claude, where that provider is."""
    if isinstance(spec, HttpTargetSpec):
        return target_hash(spec, resolve_endpoint(spec, endpoints).fingerprint())
    if isinstance(spec, PromptTargetSpec) and spec.provider != ANTHROPIC:
        config = (providers or {}).get(spec.provider)
        if config is None:
            known = ", ".join(sorted([ANTHROPIC, *(providers or {})]))
            raise InvalidInputError(f"unknown provider {spec.provider!r}; configured: {known}")
        return target_hash(spec, config.endpoint)
    return target_hash(spec)


def build_target(
    spec: AnySpec,
    *,
    endpoints: Mapping[str, HttpEndpoint],
    providers: ProviderRegistry,
) -> Target:
    match spec:
        case PromptTargetSpec():
            return PromptTarget(spec, providers.get(spec.provider))
        case HttpTargetSpec():
            return HttpTarget(spec, resolve_endpoint(spec, endpoints))
        case FakeTargetSpec():
            return FakeTarget(spec)
