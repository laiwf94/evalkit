import asyncio
import time

import pytest
from pydantic import JsonValue, TypeAdapter, ValidationError

from evalkit.datasets import Case
from evalkit.errors import InvalidInputError
from evalkit.targets import (
    FakeTargetSpec,
    PromptTargetSpec,
    TargetError,
    TargetSpec,
    Usage,
    target_hash,
)
from evalkit.targets.fake import FakeTarget, unit
from evalkit.targets.template import placeholders, render

SPEC = TypeAdapter(TargetSpec)


def test_render_fields_nested_and_whole_input() -> None:
    value = {"question": "2+2?", "ctx": {"lang": "en", "n": 3}}
    assert render("Q: {{question}} ({{ ctx.lang }})", value) == "Q: 2+2? (en)"
    assert render("{{ctx}}", value) == '{"lang": "en", "n": 3}'
    assert render("n={{ctx.n}}", value) == "n=3"
    assert render("{{input}}", "plain text") == "plain text"
    assert render("{{input}}", {"b": 1, "a": "é"}) == '{"a": "é", "b": 1}'


def test_render_missing_field_names_it() -> None:
    with pytest.raises(InvalidInputError, match=r"no field 'ctx\.missing'"):
        render("{{ctx.missing}}", {"ctx": {}})
    with pytest.raises(InvalidInputError, match="no field 'question'"):
        render("{{question}}", "a string input")


@pytest.mark.parametrize("bad", ["{{ question }", "{{1abc}}", "{{ a b }}", "x }} y", "{{}}"])
def test_malformed_placeholders(bad: str) -> None:
    with pytest.raises(InvalidInputError, match="not a valid"):
        placeholders(bad)


def test_placeholders_lists_fields() -> None:
    assert placeholders("{{a}} and {{ b.c }} and {a} single braces") == ["a", "b.c"]


def test_spec_parsing_and_defaults() -> None:
    spec = SPEC.validate_python({"type": "prompt", "template": "{{input}}"})
    assert isinstance(spec, PromptTargetSpec)
    assert spec.model == "claude-opus-5-5"
    assert spec.effort == "medium"
    assert spec.fallbacks
    with pytest.raises(ValidationError, match="not a valid"):
        SPEC.validate_python({"type": "prompt", "template": "{{oops}"})
    with pytest.raises(ValidationError):
        SPEC.validate_python({"type": "prompt", "template": "x", "temperature": 0})
    with pytest.raises(ValidationError):
        SPEC.validate_python({"type": "http", "endpoint": "http://evil"})
    with pytest.raises(ValidationError):
        SPEC.validate_python({"type": "fake", "failure_rate": 1.0})


def test_target_hash_is_stable_and_sensitive() -> None:
    base = {"type": "prompt", "template": "{{q}}", "system": "be brief", "max_tokens": 100}
    h = target_hash(SPEC.validate_python(base))
    reordered = dict(reversed(list(base.items())))
    assert target_hash(SPEC.validate_python(reordered)) == h
    for field, value in [
        ("template", "{{q}}!"),
        ("system", "be long"),
        ("max_tokens", 101),
        ("model", "claude-sonnet-5-5"),
        ("effort", "high"),
        ("fallbacks", False),
    ]:
        assert target_hash(SPEC.validate_python({**base, field: value})) != h, field
    assert target_hash(SPEC.validate_python(base), extra={"url": "x"}) != h


def case(cid: str, expected: JsonValue = "right") -> Case:
    return Case(id=cid, input={"q": cid}, expected=expected)


def test_fake_target_accuracy_is_seeded() -> None:
    cases = [case(str(i)) for i in range(400)]
    t = FakeTarget(FakeTargetSpec(type="fake", accuracy=0.7, seed=3))
    outs = [asyncio.run(t.generate(c)).text for c in cases]
    rate = outs.count("right") / len(outs)
    assert 0.63 < rate < 0.77
    again = FakeTarget(FakeTargetSpec(type="fake", accuracy=0.7, seed=3))
    assert [asyncio.run(again.generate(c)).text for c in cases] == outs
    other = FakeTarget(FakeTargetSpec(type="fake", accuracy=0.7, seed=4))
    assert [asyncio.run(other.generate(c)).text for c in cases] != outs


def test_fake_target_output_and_usage() -> None:
    t = FakeTarget(FakeTargetSpec(type="fake"))
    gen = asyncio.run(t.generate(case("a", {"x": 1})))
    assert gen.text == '{"x": 1}'
    assert gen.stop_reason == "end_turn"
    assert gen.usage.input_tokens >= 1
    assert gen.usage.cost_usd == 0.0
    no_expected = asyncio.run(t.generate(case("b", None)))
    assert no_expected.text == "I don't know."
    string_input = asyncio.run(t.generate(Case(id="s", input="hello there", expected="x")))
    assert string_input.usage.input_tokens == 2
    assert t.calls == 3
    asyncio.run(t.aclose())


def test_fake_target_transient_failures_are_per_attempt() -> None:
    t = FakeTarget(FakeTargetSpec(type="fake", failure_rate=0.5, retry_after_s=1.5, seed=1))
    failures = 0
    seen: list[tuple[bool, float | None]] = []
    for i in range(50):
        c = case(str(i))
        for _ in range(20):
            try:
                asyncio.run(t.generate(c))
                break
            except TargetError as exc:
                seen.append((exc.retryable, exc.retry_after))
                failures += 1
        else:
            pytest.fail("never succeeded")
    assert 20 < failures < 90
    assert set(seen) == {(True, 1.5)}


def test_fake_target_latency() -> None:
    t = FakeTarget(FakeTargetSpec(type="fake", latency_ms=30, jitter_ms=10))
    start = time.perf_counter()
    asyncio.run(t.generate(case("a")))
    assert 0.029 <= time.perf_counter() - start < 0.5


def test_unit_and_usage_helpers() -> None:
    assert unit("a", 1) == unit("a", 1)
    assert 0 <= unit("b") < 1
    total = Usage(1, 2, 3, 4, 0.5) + Usage(1, 1, 1, 1, 0.25)
    assert total == Usage(2, 3, 4, 5, 0.75)
