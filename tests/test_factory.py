import pytest

from evalkit.errors import InvalidInputError
from evalkit.targets import FakeTargetSpec, HttpTargetSpec, PromptTargetSpec, target_hash
from evalkit.targets.factory import build_target, identity
from evalkit.targets.fake import FakeTarget
from evalkit.targets.http import HttpEndpoint, HttpTarget
from evalkit.targets.prompt import PromptTarget
from tests.fake_anthropic import FakeAnthropic

ENDPOINTS = {"bot": HttpEndpoint(url="https://bot.example.com/a")}


def test_identity_includes_endpoint_for_http() -> None:
    http = HttpTargetSpec(type="http", endpoint="bot")
    moved = {"bot": HttpEndpoint(url="https://bot2.example.com/a")}
    assert identity(http, ENDPOINTS) != identity(http, moved)
    fake = FakeTargetSpec(type="fake")
    assert identity(fake, {}) == target_hash(fake)
    with pytest.raises(InvalidInputError, match="none configured"):
        identity(http, {})


def test_build_each_kind() -> None:
    assert isinstance(
        build_target(FakeTargetSpec(type="fake"), endpoints={}, claude=None), FakeTarget
    )
    http = build_target(
        HttpTargetSpec(type="http", endpoint="bot"), endpoints=ENDPOINTS, claude=None
    )
    assert isinstance(http, HttpTarget)
    prompt = PromptTargetSpec(type="prompt", template="{{input}}")
    built = build_target(prompt, endpoints={}, claude=FakeAnthropic().client())
    assert isinstance(built, PromptTarget)
    with pytest.raises(InvalidInputError, match="need a Claude client"):
        build_target(prompt, endpoints={}, claude=None)
