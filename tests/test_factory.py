import httpx2
import pytest

from evalkit.datasets import canonical_json, sha256_hex
from evalkit.errors import InvalidInputError
from evalkit.pricing import ModelPrice, PriceTable
from evalkit.providers.openai_compat import OpenAICompatClient
from evalkit.providers.registry import ProviderConfig, ProviderRegistry
from evalkit.targets import FakeTargetSpec, HttpTargetSpec, PromptTargetSpec, target_hash
from evalkit.targets.factory import build_target, identity
from evalkit.targets.fake import FakeTarget
from evalkit.targets.http import HttpEndpoint, HttpTarget
from evalkit.targets.prompt import PromptTarget
from tests.fake_anthropic import FakeAnthropic

ENDPOINTS = {"bot": HttpEndpoint(url="https://bot.example.com/a")}
LOCAL = ProviderConfig(kind="openai_compat", base_url="http://llm.test/v1")
MOVED = ProviderConfig(kind="openai_compat", base_url="http://llm2.test/v1")


def registry(claude: bool = True) -> ProviderRegistry:
    return ProviderRegistry(
        {"local": LOCAL},
        PriceTable({"gpt-test": ModelPrice(input=1.0, output=2.0)}),
        claude=FakeAnthropic().client() if claude else None,
        transport=httpx2.MockTransport(lambda r: httpx2.Response(500)),
    )


def test_identity_includes_endpoint_for_http() -> None:
    http = HttpTargetSpec(type="http", endpoint="bot")
    moved = {"bot": HttpEndpoint(url="https://bot2.example.com/a")}
    assert identity(http, ENDPOINTS) != identity(http, moved)
    fake = FakeTargetSpec(type="fake")
    assert identity(fake, {}) == target_hash(fake)
    with pytest.raises(InvalidInputError, match="none configured"):
        identity(http, {})


def test_a_claude_prompt_target_keeps_the_identity_it_had_before_providers() -> None:
    spec = PromptTargetSpec(type="prompt", template="{{q}}")
    # Exactly the hash older versions computed: the spec without a provider field.
    before = spec.model_dump(mode="json")
    del before["provider"]
    old = sha256_hex(canonical_json({"spec": before, "extra": None}))
    assert identity(spec, {}) == old == identity(spec, {}, {"local": LOCAL})


def test_a_prompt_target_on_another_provider_includes_its_endpoint() -> None:
    spec = PromptTargetSpec(type="prompt", provider="local", model="gpt-test", template="{{q}}")
    here = identity(spec, {}, {"local": LOCAL})
    assert here != identity(spec, {}, {"local": MOVED})
    claude = spec.model_copy(update={"provider": "anthropic"})
    assert here != identity(claude, {})
    with pytest.raises(InvalidInputError, match=r"unknown provider 'local'; configured: anthropic"):
        identity(spec, {})


def test_build_each_kind() -> None:
    providers = registry()
    assert isinstance(
        build_target(FakeTargetSpec(type="fake"), endpoints={}, providers=providers), FakeTarget
    )
    http = build_target(
        HttpTargetSpec(type="http", endpoint="bot"), endpoints=ENDPOINTS, providers=providers
    )
    assert isinstance(http, HttpTarget)
    prompt = PromptTargetSpec(type="prompt", template="{{input}}")
    built = build_target(prompt, endpoints={}, providers=providers)
    assert isinstance(built, PromptTarget)
    assert built.provider == "anthropic"
    with pytest.raises(InvalidInputError, match="Claude API is not configured"):
        build_target(prompt, endpoints={}, providers=registry(claude=False))


def test_build_a_prompt_target_on_an_http_provider() -> None:
    providers = registry(claude=False)
    spec = PromptTargetSpec(type="prompt", provider="local", model="gpt-test", template="{{q}}")
    built = build_target(spec, endpoints={}, providers=providers)
    assert isinstance(built, PromptTarget)
    assert isinstance(built.client, OpenAICompatClient)
    assert built.provider == "local"  # its rate limits are keyed by the provider name
    with pytest.raises(InvalidInputError, match="unknown model 'gpt-other'"):
        build_target(
            spec.model_copy(update={"model": "gpt-other"}), endpoints={}, providers=providers
        )
