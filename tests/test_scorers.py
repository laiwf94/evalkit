import asyncio
from importlib.metadata import EntryPoint
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import TypeAdapter, ValidationError

from evalkit.datasets import Case
from evalkit.errors import InvalidInputError
from evalkit.scorers import Scorer, ScorerSpec, build_scorers
from evalkit.scorers.builtin import (
    ExactMatch,
    JsonSchemaMatch,
    NumericMatch,
    RegexMatch,
    TokenF1,
    normalize_answer,
    token_f1,
)
from evalkit.scorers.registry import ENTRY_POINT_GROUP

SPECS = TypeAdapter(list[ScorerSpec])


def case(expected: Any = None, **kw: Any) -> Case:
    return Case(id="c", input="q", expected=expected, **kw)


def run(scorer: Scorer, c: Case, output: str) -> tuple[float | None, str]:
    r = asyncio.run(scorer.score(c, output))
    return r.value, r.detail


def test_exact_match_options() -> None:
    assert run(ExactMatch("e"), case("Paris"), "Paris") == (1.0, "")
    assert run(ExactMatch("e"), case("Paris"), "paris")[0] == 0.0
    assert run(ExactMatch("e", case_sensitive=False), case("Paris"), "PARIS")[0] == 1.0
    loose = ExactMatch("e", normalize_whitespace=True)
    assert run(loose, case("a  b"), " a\nb ")[0] == 1.0
    assert run(ExactMatch("e"), case({"b": 1, "a": 2}), '{"a": 2, "b": 1}')[0] == 1.0
    keyed = ExactMatch("e", expected_key="answer")
    assert run(keyed, case({"answer": "4"}), "4")[0] == 1.0
    assert run(keyed, case("4"), "4") == (None, "case has no expected value")
    assert run(ExactMatch("e"), case(None), "x")[0] is None


def test_regex_match() -> None:
    assert run(RegexMatch("r", pattern=r"\d+"), case(), "abc 42")[0] == 1.0
    assert run(RegexMatch("r", pattern=r"\d+", full_match=True), case(), "abc 42")[0] == 0.0
    assert run(RegexMatch("r", pattern="yes", ignore_case=True), case(), "YES")[0] == 1.0
    assert run(RegexMatch("r"), case(r"^Par"), "Paris")[0] == 1.0
    assert run(RegexMatch("r"), case(None), "x")[1] == "case has no expected pattern"
    assert "invalid" in run(RegexMatch("r"), case("(open"), "x")[1]
    assert "longer than 1000" in run(RegexMatch("r"), case("a" * 1001), "x")[1]


def test_regex_timeout_is_unscorable() -> None:
    scorer = RegexMatch("r", pattern=r"(a|aa)+$", timeout_seconds=0.05)
    assert run(scorer, case(), "a" * 40 + "!") == (None, "pattern took longer than the time limit")


SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0}},
    "required": ["name"],
}


def test_json_schema_match() -> None:
    s = JsonSchemaMatch("j", schema=SCHEMA)
    assert run(s, case(), '{"name": "alice", "age": 3}') == (1.0, "")
    assert run(s, case(), '```json\n{"name": "bob"}\n```')[0] == 1.0
    value, detail = run(s, case(), '{"name": "bob", "age": -1}')
    assert value == 0.0
    assert detail.startswith("age:")
    assert run(s, case(), "{}")[1].startswith("(root):")
    assert run(s, case(), "not json")[1].startswith("not JSON")
    per_case = JsonSchemaMatch("j")
    assert run(per_case, case(SCHEMA), '{"name": "x"}')[0] == 1.0
    assert run(per_case, case("nope"), "{}")[1] == "case has no expected schema"
    assert "invalid" in run(per_case, case({"type": "nonsense"}), "{}")[1]


def test_json_schema_does_not_fetch_remote_refs() -> None:
    s = JsonSchemaMatch("j", schema={"$ref": "https://example.invalid/schema.json"})
    value, detail = run(s, case(), "{}")
    assert value is None
    assert detail.startswith("schema reference cannot be resolved")


def test_numeric_match() -> None:
    assert run(NumericMatch("n"), case(4), "The answer is 4.")[0] == 1.0
    assert run(NumericMatch("n"), case("1,234.5"), "total: 1,234.50")[0] == 1.0
    assert run(NumericMatch("n", extract="first"), case(2), "2 then 3")[0] == 1.0
    assert run(NumericMatch("n"), case(2), "2 then 3") == (0.0, "got 3, expected 2")
    assert run(NumericMatch("n", extract="whole"), case(0.5), " .5 ")[0] == 1.0
    assert run(NumericMatch("n", extract="whole"), case(1), "one")[1] == "no number in output"
    assert run(NumericMatch("n", abs_tol=0.1), case(3.14), "3.2")[0] == 1.0
    assert run(NumericMatch("n", rel_tol=0.01), case(100), "100.9")[0] == 1.0
    assert run(NumericMatch("n"), case(1), "no digits") == (0.0, "no number in output")
    assert run(NumericMatch("n", expected_key="v"), case({"v": 7}), "7")[0] == 1.0
    for bad in (None, "seven", True, float("nan"), [1]):
        assert run(NumericMatch("n"), case(bad), "7")[0] is None
    assert run(NumericMatch("n", extract="whole"), case(1), "inf")[1] == "no number in output"
    with pytest.raises(ValueError, match="extract"):
        NumericMatch("n", extract="middle")


def test_token_f1() -> None:
    assert token_f1("The Eiffel Tower!", "eiffel tower") == 1.0
    assert token_f1("", "") == 1.0
    assert token_f1("x", "") == 0.0
    assert token_f1("red car", "blue bike") == 0.0
    assert token_f1("a red car", "red bike") == pytest.approx(0.5)
    assert normalize_answer("The cat, an owl.") == ["cat", "owl"]
    assert run(TokenF1("f"), case("red bike"), "red car")[0] == pytest.approx(0.5)
    assert run(TokenF1("f"), case(3), "3")[0] is None
    assert run(TokenF1("f", expected_key="a"), case({"a": "x y"}), "x y")[0] == 1.0


words = st.lists(st.sampled_from(["red", "car", "the", "blue", "Bike", "a", "x!"]), max_size=8)


@given(words, words)
def test_token_f1_is_symmetric_and_bounded(a: list[str], b: list[str]) -> None:
    x, y = " ".join(a), " ".join(b)
    assert token_f1(x, y) == pytest.approx(token_f1(y, x))
    assert 0.0 <= token_f1(x, y) <= 1.0
    assert token_f1(x, x) == 1.0


def test_build_from_specs() -> None:
    specs = SPECS.validate_python(
        [
            {"type": "exact", "case_sensitive": False},
            {"type": "regex", "name": "has_number", "pattern": r"\d"},
            {"type": "json_schema", "schema": SCHEMA},
            {"type": "numeric", "abs_tol": 0.5},
            {"type": "token_f1", "gate": False},
        ]
    )
    scorers = build_scorers(specs, allow_import_paths=False)
    assert [s.name for s in scorers] == [
        "exact",
        "has_number",
        "json_schema",
        "numeric",
        "token_f1",
    ]
    assert not specs[4].gate


@pytest.mark.parametrize(
    ("specs", "message"),
    [
        ([], "at least one scorer"),
        ([{"type": "exact"}, {"type": "exact"}], "repeated: exact"),
        ([{"type": "regex", "pattern": "(x"}], "invalid pattern"),
        ([{"type": "json_schema", "schema": {"type": 5}}], "invalid JSON schema"),
    ],
)
def test_build_errors(specs: list[dict[str, Any]], message: str) -> None:
    with pytest.raises(InvalidInputError, match=message):
        build_scorers(SPECS.validate_python(specs), allow_import_paths=True)


def test_spec_validation() -> None:
    with pytest.raises(ValidationError):
        SPECS.validate_python([{"type": "exact", "bogus": 1}])
    with pytest.raises(ValidationError):
        SPECS.validate_python([{"type": "nope"}])
    with pytest.raises(ValidationError):
        SPECS.validate_python([{"type": "exact", "name": "Bad Name"}])


def test_plugin_by_import_path() -> None:
    (scorer,) = build_scorers(
        SPECS.validate_python(
            [
                {
                    "type": "plugin",
                    "name": "short",
                    "entry": "tests.plugin_scorer:MaxLength",
                    "options": {"limit": 3},
                }
            ]
        ),
        allow_import_paths=True,
    )
    assert run(scorer, case(), "abc")[0] == 1.0
    assert run(scorer, case(), "abcd")[0] == 0.0


@pytest.mark.parametrize(
    ("entry", "allow", "message"),
    [
        ("tests.plugin_scorer:MaxLength", False, "import paths are not accepted"),
        ("tests.plugin_scorer:Missing", True, "cannot load"),
        ("no_such_module_xyz:X", True, "cannot load"),
        ("tests.plugin_scorer:NOT_A_SCORER", True, "not a Scorer subclass"),
        ("not-installed", True, "no installed plugin named 'not-installed'"),
    ],
)
def test_plugin_errors(entry: str, allow: bool, message: str) -> None:
    spec = SPECS.validate_python([{"type": "plugin", "entry": entry}])
    with pytest.raises(InvalidInputError, match=message):
        build_scorers(spec, allow_import_paths=allow)


def test_plugin_by_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    ep = EntryPoint(
        name="max_length", value="tests.plugin_scorer:MaxLength", group=ENTRY_POINT_GROUP
    )

    def fake_entry_points(group: str) -> list[EntryPoint]:
        return [ep] if group == ENTRY_POINT_GROUP else []

    monkeypatch.setattr("evalkit.scorers.registry.entry_points", fake_entry_points)
    (scorer,) = build_scorers(
        SPECS.validate_python([{"type": "plugin", "entry": "max_length"}]), allow_import_paths=False
    )
    assert scorer.name == "plugin"
    assert run(scorer, case(), "x" * 10)[0] == 1.0
