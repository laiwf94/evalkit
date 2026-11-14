"""Turn a target spec into a callable target, with the configuration it needs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from evalkit.errors import InvalidInputError
from evalkit.simuser import Simulator, build_simulator
from evalkit.targets.agent import AgentTarget
from evalkit.targets.base import Target
from evalkit.targets.conversation import ConversationTarget
from evalkit.targets.fake import FakeTarget
from evalkit.targets.http import HttpEndpoint, HttpTarget
from evalkit.targets.http_agent import HttpAgentTarget
from evalkit.targets.prompt import PromptTarget
from evalkit.targets.specs import (
    ANTHROPIC,
    AgentTargetSpec,
    AnyTargetSpec,
    ConversationTargetSpec,
    FakeTargetSpec,
    HttpAgentTargetSpec,
    HttpTargetSpec,
    LLMSimulatorSpec,
    PromptTargetSpec,
    SimulatorSpec,
    target_hash,
)

if TYPE_CHECKING:
    from evalkit.providers.registry import ProviderConfig, ProviderRegistry

AnySpec = AnyTargetSpec


def resolve_endpoint(
    spec: HttpTargetSpec | HttpAgentTargetSpec, endpoints: Mapping[str, HttpEndpoint]
) -> HttpEndpoint:
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
    for a prompt or agent target on a provider other than Claude, where that
    provider is. An HTTP agent's includes its endpoint and where its simulated
    user is."""
    if isinstance(spec, HttpTargetSpec):
        return target_hash(spec, resolve_endpoint(spec, endpoints).fingerprint())
    if isinstance(spec, HttpAgentTargetSpec):
        return target_hash(
            spec,
            {
                "endpoint": resolve_endpoint(spec, endpoints).fingerprint(),
                "simulator": simulator_endpoint(spec.simulator, providers),
            },
        )
    if isinstance(spec, PromptTargetSpec | AgentTargetSpec) and spec.provider != ANTHROPIC:
        return target_hash(spec, provider_endpoint(spec.provider, providers))
    if isinstance(spec, ConversationTargetSpec):
        return target_hash(spec, conversation_endpoints(spec, providers))
    return target_hash(spec)


def provider_endpoint(name: str, providers: Mapping[str, ProviderConfig] | None) -> str | None:
    """Where provider ``name`` sends requests: None for Claude (one fixed API)."""
    if name == ANTHROPIC:
        return None
    config = (providers or {}).get(name)
    if config is None:
        known = ", ".join(sorted([ANTHROPIC, *(providers or {})]))
        raise InvalidInputError(f"unknown provider {name!r}; configured: {known}")
    return config.endpoint


def conversation_endpoints(
    spec: ConversationTargetSpec, providers: Mapping[str, ProviderConfig] | None
) -> dict[str, str | None]:
    """Where a conversation's assistant and simulated user send their requests."""
    return {
        "assistant": provider_endpoint(spec.provider, providers),
        "simulator": simulator_endpoint(spec.simulator, providers),
    }


def simulator_endpoint(
    simulator: SimulatorSpec | None, providers: Mapping[str, ProviderConfig] | None
) -> str | None:
    """Where a simulated user sends its requests: None when it makes none."""
    if isinstance(simulator, LLMSimulatorSpec):
        return provider_endpoint(simulator.provider, providers)
    return None


def simulator_for(spec: SimulatorSpec, providers: ProviderRegistry) -> Simulator:
    client = providers.get(spec.provider) if isinstance(spec, LLMSimulatorSpec) else None
    return build_simulator(spec, client)


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
        case AgentTargetSpec():
            return AgentTarget(
                spec,
                providers.get(spec.provider),
                endpoint=provider_endpoint(spec.provider, providers.configs),
            )
        case HttpAgentTargetSpec():
            return HttpAgentTarget(
                spec,
                resolve_endpoint(spec, endpoints),
                None if spec.simulator is None else simulator_for(spec.simulator, providers),
                simulator_endpoint=simulator_endpoint(spec.simulator, providers.configs),
            )
        case ConversationTargetSpec():
            where = conversation_endpoints(spec, providers.configs)
            return ConversationTarget(
                spec,
                providers.get(spec.provider),
                simulator_for(spec.simulator, providers),
                assistant_endpoint=where["assistant"],
                simulator_endpoint=where["simulator"],
            )
