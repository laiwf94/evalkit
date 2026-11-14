"""Trajectory scorers: tool call matching in and out of order with exact, subset
and schema arguments, step efficiency, forbidden tools, the trajectory judge with
its redaction and calibration, and all of them on a real agent run."""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from evalkit.config import Settings
from evalkit.datasets import Case, import_dataset
from evalkit.db.models import Result
from evalkit.errors import InvalidInputError
from evalkit.ratelimit import RetryPolicy
from evalkit.redaction import RedactionConfig, Redactor
from evalkit.runner import Runner
from evalkit.runspec import RunConfig
from evalkit.scorers import ScorerSpec, build_scorers
from evalkit.scorers.base import ScoreResult
from evalkit.scorers.builtin import ExactMatch
from evalkit.scorers.calibration import calibrate, parse_labels
from evalkit.scorers.judge import LLMJudge
from evalkit.scorers.trajectory import (
    MAX_TRAJECTORY_CHARS,
    NO_TRAJECTORY,
    ForbiddenTools,
    StepEfficiency,
    ToolCallMatch,
    TrajectoryJudge,
    is_subset,
    longest_ordered_match,
    maximum_matching,
    parse_calls,
    render_trajectory,
    strict_equal,
)
from evalkit.targets import Step
from tests.fake_anthropic import FakeAnthropic, json_reply, message
from tests.test_agent import CLAUDE, SANDBOX, Factory
from tests.test_agent import config as agent_config

SPECS: TypeAdapter[list[Any]] = TypeAdapter(list[ScorerSpec])
CASE = Case(id="c1", input={"date": "2026-11-20"})


def model(index: int, text: str = "", stop: str = "tool_use") -> Step:
    return Step(index=index, kind="model", text=text, stop_reason=stop)


def call(index: int, tool: str, arguments: Any = None, status: str = "ok", **kw: Any) -> Step:
    return Step(index=index, kind="tool", tool=tool, arguments=arguments, status=status, **kw)


def trajectory(*calls: tuple[str, Any] | tuple[str, Any, str]) -> tuple[Step, ...]:
    """A model reply, then each call, then a final reply."""
    steps: list[Step] = [model(0)]
    for item in calls:
        steps.append(call(len(steps), *item))
    steps.append(model(len(steps), "done", "end_turn"))
    return tuple(steps)


def score(scorer: Any, steps: tuple[Step, ...], case: Case = CASE, output: str = "") -> ScoreResult:
    result: ScoreResult = asyncio.run(scorer.score_steps(case, output, steps))
    return result


def with_expected(expected: Any) -> Case:
    return Case(id="c1", input={"date": "2026-11-20"}, expected=expected)


# JSON comparisons


def test_strict_equal_keeps_booleans_apart_from_numbers() -> None:
    assert strict_equal({"a": [1, 2.0]}, {"a": [1.0, 2]})
    assert not strict_equal(True, 1)
    assert not strict_equal(0, False)
    assert strict_equal(False, False)
    assert not strict_equal({"a": 1}, {"a": 1, "b": 2})
    assert not strict_equal([1], [1, 1])
    assert not strict_equal("1", 1)
    assert strict_equal(None, None)


def test_subset_is_recursive_and_lists_match_by_position() -> None:
    assert is_subset({"a": {"b": 1}}, {"a": {"b": 1, "c": 2}, "d": 3})
    assert not is_subset({"a": {"b": 1}}, {"a": {"b": 2}})
    assert not is_subset({"a": 1}, {})
    assert not is_subset({"a": 1}, "a")
    assert is_subset([{"x": 1}], [{"x": 1, "y": 2}])
    assert not is_subset([{"x": 1}], [{"x": 1}, {"x": 2}])
    assert not is_subset([1], "1")
    assert not is_subset(True, 1)


def test_expected_calls_are_parsed_or_explained() -> None:
    calls = parse_calls(["lookup", {"tool": "book", "arguments": {"slot": "19:00"}}])
    assert not isinstance(calls, str)
    assert [c.describe() for c in calls] == ["lookup", 'book({"slot":"19:00"})']
    assert parse_calls("lookup") == "expected tool calls must be a list"
    assert "call 1" in str(parse_calls([{"name": "x"}]))
    assert "call 2" in str(parse_calls(["a", ""]))
    assert "unknown keys: args" in str(parse_calls([{"tool": "a", "args": {}}]))
    assert "more than" in str(parse_calls(["a"] * 201))


# Matching algorithms


def test_ordered_matching_finds_the_longest_subsequence_not_the_first() -> None:
    # Taking B first (as a greedy scan of the expectations would) leaves nothing
    # after it; skipping B matches C and D.
    expected, actual = ["B", "C", "D"], ["C", "D", "B"]
    pairs = longest_ordered_match(expected, actual, lambda i, j: expected[i] == actual[j])
    assert pairs == [(1, 0), (2, 1)]
    assert longest_ordered_match([], actual, lambda i, j: True) == []


def test_unordered_matching_is_maximum_not_greedy() -> None:
    # The loose expectation must leave the only call the strict one can take.
    def loose(a: int) -> bool:
        return True

    def strict(a: int) -> bool:
        return a == 1

    checks, actual = [loose, strict], [1, 2]
    pairs = maximum_matching(checks, actual, lambda i, j: checks[i](actual[j]))
    assert pairs == [(0, 1), (1, 0)]


# tool_calls


def test_ordered_calls_score_the_matched_fraction_and_name_what_is_missing() -> None:
    scorer = ToolCallMatch("calls", calls=["find_slots", "book"])
    good = trajectory(("find_slots", {"date": "d"}), ("book", {"slot": "19:00"}))
    assert score(scorer, good) == ScoreResult(1.0, "2/2 expected calls")
    swapped = trajectory(("book", {"slot": "19:00"}), ("find_slots", {"date": "d"}))
    result = score(scorer, swapped)
    assert result.value == 0.5
    assert result.detail == "1/2 expected calls: missing book"
    skipped = trajectory(("book", {"slot": "19:00"}))
    assert score(scorer, skipped).detail == "1/2 expected calls: missing find_slots"
    unordered = ToolCallMatch("calls", calls=["find_slots", "book"], order="unordered")
    assert score(unordered, swapped).value == 1.0


def test_extra_calls_count_against_the_score_only_when_not_allowed() -> None:
    steps = trajectory(("find_slots", {}), ("find_slots", {}), ("book", {}))
    assert score(ToolCallMatch("c", calls=["find_slots", "book"]), steps).value == 1.0
    strict = ToolCallMatch("c", calls=["find_slots", "book"], allow_extra=False)
    result = score(strict, steps)
    assert result.value == pytest.approx(2 / 3)
    assert result.detail == "2/2 expected calls: unexpected find_slots"
    # No calls expected and none allowed: an agent that answers directly passes.
    none = ToolCallMatch("c", calls=[], allow_extra=False)
    assert score(none, trajectory()).value == 1.0
    assert score(none, trajectory(("book", {}))).value == 0.0


def test_argument_modes() -> None:
    steps = trajectory(("book", {"slot": "19:00", "people": 2, "vip": True}))
    want = {"tool": "book", "arguments": {"slot": "19:00"}}

    def value(mode: str, expected: Any = want) -> float | None:
        scorer = ToolCallMatch("c", calls=[expected], arguments=mode)  # type: ignore[arg-type]
        return score(scorer, steps).value

    assert value("subset") == 1.0
    assert value("exact") == 0.0
    assert value(
        "exact", {"tool": "book", "arguments": {"slot": "19:00", "people": 2, "vip": True}}
    )
    assert (
        value("exact", {"tool": "book", "arguments": {"slot": "19:00", "people": 2, "vip": 1}}) == 0
    )
    assert value("ignore", {"tool": "book", "arguments": {"slot": "nope"}}) == 1.0
    schema = {
        "tool": "book",
        "arguments": {
            "type": "object",
            "required": ["slot", "people"],
            "properties": {"people": {"type": "integer", "minimum": 1, "maximum": 8}},
        },
    }
    assert value("schema", schema) == 1.0
    schema["arguments"]["properties"]["people"]["minimum"] = 3  # type: ignore[index]
    assert value("schema", schema) == 0.0
    # A name alone matches any arguments, in every mode.
    assert value("exact", "book") == 1.0


def test_only_answered_calls_can_count() -> None:
    steps = trajectory(("find_slots", {"date": 5}, "invalid_arguments"), ("book", {}))
    assert score(ToolCallMatch("c", calls=["find_slots", "book"]), steps).value == 1.0
    answered = ToolCallMatch("c", calls=["find_slots", "book"], only_ok=True)
    assert score(answered, steps).value == 0.5


def test_expected_calls_come_from_the_case_by_key_or_whole() -> None:
    steps = trajectory(("find_slots", {"date": "d"}))
    by_key = ToolCallMatch("c")
    assert score(by_key, steps, with_expected({"tool_calls": ["find_slots"]})).value == 1.0
    assert score(by_key, steps, with_expected({"answer": "x"})).detail == (
        "case has no expected tool calls"
    )
    assert score(by_key, steps, with_expected({"tool_calls": "find_slots"})).value is None
    whole = ToolCallMatch("c", expected_key=None)
    assert score(whole, steps, with_expected(["find_slots", "book"])).value == 0.5


def test_bad_references_are_unscorable_and_bad_configs_refused() -> None:
    steps = trajectory(("book", {"slot": "19:00"}))
    schema_mode = ToolCallMatch("c", arguments="schema")
    bad = with_expected({"tool_calls": [{"tool": "book", "arguments": {"type": "nonsense"}}]})
    result = score(schema_mode, steps, bad)
    assert result.value is None
    assert result.detail.startswith("schema for book is invalid")
    remote = with_expected(
        {"tool_calls": [{"tool": "book", "arguments": {"$ref": "https://example.com/s.json"}}]}
    )
    assert "cannot be resolved" in score(schema_mode, steps, remote).detail
    with pytest.raises(InvalidInputError, match="scorer c: invalid JSON schema"):
        build_scorers(
            SPECS.validate_python(
                [
                    {
                        "type": "tool_calls",
                        "name": "c",
                        "arguments": "schema",
                        "calls": [{"tool": "b", "arguments": {"type": 5}}],
                    }
                ]
            ),
            allow_import_paths=False,
        )
    with pytest.raises(InvalidInputError, match="scorer c: expected call 1"):
        build_scorers(
            SPECS.validate_python([{"type": "tool_calls", "name": "c", "calls": [{"x": 1}]}]),
            allow_import_paths=False,
        )


def test_without_a_trajectory_every_trajectory_scorer_is_unscorable() -> None:
    fake = FakeAnthropic()
    scorers = [
        ToolCallMatch("a", calls=["x"]),
        StepEfficiency("b", budget=1),
        ForbiddenTools("c", tools=["x"]),
        TrajectoryJudge("d", client=fake.client(), rubric="r"),
    ]
    for scorer in scorers:
        assert asyncio.run(scorer.score(CASE, "out")) == ScoreResult(None, NO_TRAJECTORY)
        assert score(scorer, ()) == ScoreResult(None, NO_TRAJECTORY)
    assert fake.requests == []


def test_output_scorers_ignore_the_trajectory() -> None:
    exact = ExactMatch("e")
    case = with_expected("done")
    assert score(exact, trajectory(("x", {})), case, "done").value == 1.0


# step_efficiency and forbidden_tools


def test_step_efficiency_against_each_kind_of_budget() -> None:
    three = trajectory(("a", {}), ("b", {}), ("c", {}))
    assert score(StepEfficiency("e", budget=3), three) == ScoreResult(
        1.0, "3 tool calls against a budget of 3"
    )
    assert score(StepEfficiency("e", budget=2), three).value == pytest.approx(2 / 3)
    assert score(StepEfficiency("e", budget=0), three).value == 0.0
    assert score(StepEfficiency("e", budget=0), trajectory()).value == 1.0
    from_case = StepEfficiency("e")
    assert score(from_case, three, with_expected({"optimal_steps": 1})).value == pytest.approx(
        1 / 3
    )
    # Without a budget, the reference calls are the optimum.
    assert score(from_case, three, with_expected({"tool_calls": ["a", "b"]})).value == (
        pytest.approx(2 / 3)
    )
    assert score(from_case, three, with_expected({"tool_calls": 3})).value is None
    for bad in (-1, 1.5, True, "2"):
        result = score(from_case, three, with_expected({"optimal_steps": bad}))
        assert result.value is None
        assert "whole number" in result.detail
    assert score(from_case, three, with_expected("x")).detail == "case has no step budget"
    replies = StepEfficiency("e", count="model_steps", budget=1)
    result = score(replies, three)
    assert result.value == 0.5
    assert result.detail == "2 model replies against a budget of 1"
    # Model replies never fall back to the reference tool calls.
    assert (
        score(
            StepEfficiency("e", count="model_steps"), three, with_expected({"tool_calls": []})
        ).value
        is None
    )


def test_forbidden_tools_by_pattern_from_config_and_case() -> None:
    steps = trajectory(("find_slots", {}), ("delete_booking", {}, "unknown_tool"))
    scorer = ForbiddenTools("f", tools=["delete_*"])
    result = score(scorer, steps)
    assert result == ScoreResult(0.0, "called delete_booking (step 2)")
    assert score(ForbiddenTools("f", tools=["drop"]), steps).value == 1.0
    per_case = ForbiddenTools("f")
    assert score(per_case, steps, with_expected({"forbidden_tools": ["find_*"]})).value == 0.0
    assert score(per_case, steps, with_expected({"forbidden_tools": [""]})).value is None
    assert score(per_case, steps, with_expected({"forbidden_tools": "x"})).value is None
    assert score(per_case, steps).detail == "no forbidden tools are configured for this case"
    # Matching is case-sensitive, as tool names are.
    assert score(ForbiddenTools("f", tools=["DELETE_*"]), steps).value == 1.0


# trajectory_judge


def verdict(score_: int, reasoning: str = "ok") -> Any:
    return json_reply(message(json.dumps({"reasoning": reasoning, "score": score_})))


def make_judge(fake: FakeAnthropic, **kw: Any) -> TrajectoryJudge:
    return TrajectoryJudge(
        "path",
        client=fake.client(),
        rubric="Looks slots up before booking.",
        retry=RetryPolicy(max_attempts=1, base_delay=0),
        **kw,
    )


def test_the_judge_reads_the_whole_trajectory() -> None:
    fake = FakeAnthropic(verdict(4, "looked up first"))
    steps = (
        model(0, "Let me check."),
        call(1, "find_slots", {"date": "d"}, result=["19:00"]),
        model(2),
        call(3, "book", {"slot": "7"}, status="error", error="no such slot"),
        model(4, "Sorry", "end_turn"),
    )
    result = score(make_judge(fake), steps, with_expected("booked"), "Sorry")
    assert (result.value, result.detail) == (0.75, "4/5: looked up first")
    assert result.input_tokens > 0
    body = fake.requests[0]
    assert "how an AI agent worked" in body["system"][0]["text"]
    assert "Looks slots up before booking." in body["system"][0]["text"]
    content = body["messages"][0]["content"]
    assert "<reference>\nbooked\n</reference>" in content
    assert '[1] tool call find_slots({"date": "d"}) -> ok: ["19:00"]' in content
    assert '[3] tool call book({"slot": "7"}) -> error: no such slot' in content
    assert "[0] assistant (tool_use): Let me check." in content
    assert content.endswith("<response>\nSorry\n</response>")


def test_the_judge_prompt_is_redacted_before_it_is_sent() -> None:
    fake = FakeAnthropic(verdict(5))
    judge = make_judge(fake, redactor=Redactor(RedactionConfig()))
    steps = (model(0), call(1, "lookup", {"q": "x"}, result={"email": "jo@example.com"}))
    assert score(judge, steps).value == 1.0
    assert "jo@example.com" not in fake.requests[0]["messages"][0]["content"]

    class Failing(Redactor):
        def redact(self, text: str) -> Any:
            return type("R", (), {"failed": True, "text": ""})()

    judge.redactor = Failing(RedactionConfig())
    assert score(judge, steps).detail == "redaction timed out; the trajectory was not sent"
    assert len(fake.requests) == 1


def test_judge_failures_are_unscorable() -> None:
    fake = FakeAnthropic(json_reply(message("not json")))
    assert score(make_judge(fake), trajectory()).value is None


def test_the_judge_identity_is_not_a_rubric_judge_s() -> None:
    fake = FakeAnthropic()
    path = make_judge(fake)
    plain = LLMJudge("plain", client=fake.client(), rubric=path.rubric)
    assert path.identity()["mode"] == "trajectory"
    assert path.fingerprint() != plain.fingerprint()
    assert path.spec()["mode"] == "trajectory"
    assert {k: v for k, v in path.identity().items() if k not in ("mode", "fingerprint")} == {
        k: v for k, v in plain.identity().items() if k not in ("mode", "fingerprint")
    }


def test_long_trajectories_keep_their_start_and_end() -> None:
    steps = tuple(call(i, "lookup", {"page": i}, result="r" * 1500) for i in range(200))
    text = render_trajectory(steps)
    assert len(text) <= MAX_TRAJECTORY_CHARS
    lines = text.splitlines()
    assert lines[0].startswith("[0] tool call lookup")
    assert lines[-1].startswith("[199] tool call lookup")
    assert any(line.startswith("[... ") and "steps left out" in line for line in lines)
    long_step = render_trajectory((model(0, "x" * 5000),))
    assert len(long_step) <= 2000
    shown = long_step.index("...")
    assert long_step.endswith(
        f"... [{len('[0] assistant (tool_use): ') + 5000 - shown} more characters]"
    )
    # Thirty-one steps of 2,000 characters overflow the budget by one step,
    # which goes from the middle.
    middle = render_trajectory(tuple(model(i, "z" * 2000) for i in range(31)))
    assert "[... 1 steps left out ...]" in middle
    assert middle.splitlines()[-1].startswith("[30] assistant")


def test_calibrating_a_trajectory_judge_on_labelled_trajectories() -> None:
    steps_json = [
        {"step": 0, "kind": "model", "text": "", "stop_reason": "tool_use", "cost_usd": 0.1},
        {
            "step": 1,
            "kind": "tool",
            "tool": "find_slots",
            "arguments": {},
            "status": "ok",
            "result": ["19:00"],
            "cached": False,
        },
    ]
    lines = [
        json.dumps(
            {"id": f"l{i}", "input": "q", "output": "o", "label": lab, "trajectory": steps_json}
        )
        for i, lab in enumerate([5, 1, 4])
    ]
    items = parse_labels("\n".join(lines), "rubric")
    fake = FakeAnthropic(verdict(5), verdict(2), verdict(4))
    report = asyncio.run(calibrate(make_judge(fake), items, concurrency=1))
    assert report.judged == 3
    assert report.agreement == pytest.approx(2 / 3)
    assert report.disagreements == ["l1"]
    assert "[1] tool call find_slots({}) -> ok" in fake.requests[0]["messages"][0]["content"]
    bare = parse_labels(json.dumps({"id": "x", "input": "q", "output": "o", "label": 3}), "rubric")
    with pytest.raises(InvalidInputError, match="missing on x"):
        asyncio.run(calibrate(make_judge(fake), bare))


def test_calibrate_command_takes_a_trajectory_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evalkit import cli
    from tests.test_cli import run_cli

    db = f"sqlite:///{tmp_path / 'cal.db'}"
    judge = tmp_path / "judge.toml"
    judge.write_text('[judge]\ntype = "trajectory_judge"\nrubric = "Efficient."\n')
    labels = tmp_path / "labels.jsonl"
    step = {"step": 0, "kind": "model", "text": "hi"}
    labels.write_text(
        json.dumps({"id": "a", "input": "q", "output": "o", "label": 2, "trajectory": [step]})
    )
    fake = FakeAnthropic(verdict(2))
    monkeypatch.setattr(cli.App, "claude", lambda self: fake.client())
    code, text = run_cli(db, "calibrate", "--judge", str(judge), str(labels))
    assert code == 0, text
    assert "Cohen's kappa" in text
    assert (
        "<trajectory>\n[0] assistant: hi\n</trajectory>"
        in fake.requests[0]["messages"][0]["content"]
    )


# On a real agent run


def test_specs_build_each_trajectory_scorer() -> None:
    fake = FakeAnthropic()
    from evalkit.pricing import PriceBook
    from evalkit.providers.registry import ProviderRegistry

    specs = SPECS.validate_python(
        [
            {"type": "tool_calls", "calls": ["a"], "order": "unordered", "arguments": "exact"},
            {"type": "step_efficiency", "budget": 2, "count": "model_steps"},
            {"type": "forbidden_tools", "tools": ["rm_*"]},
            {"type": "trajectory_judge", "rubric": "r", "effort": "low"},
        ]
    )
    built = build_scorers(
        specs,
        allow_import_paths=False,
        providers=ProviderRegistry({}, PriceBook(), claude=fake.client()),
    )
    assert [type(s).__name__ for s in built] == [
        "ToolCallMatch",
        "StepEfficiency",
        "ForbiddenTools",
        "TrajectoryJudge",
    ]
    assert [s.name for s in built] == [
        "tool_calls",
        "step_efficiency",
        "forbidden_tools",
        "trajectory_judge",
    ]
    judge = built[3]
    assert isinstance(judge, TrajectoryJudge)
    assert (judge.effort, judge.mode) == ("low", "rubric")


@pytest.fixture
def bookings(sessions: sessionmaker[Session]) -> str:
    with sessions() as s:
        import_dataset(
            s,
            "bookings",
            [
                Case(
                    id=f"b{i}",
                    input={"date": f"day {i}"},
                    expected={
                        "tool_calls": [
                            {"tool": "find_slots", "arguments": {"date": "2026-11-20"}},
                            {"tool": "book", "arguments": {"slot": "19:00"}},
                        ],
                        "optimal_steps": 2,
                        "forbidden_tools": ["cancel_*"],
                    },
                )
                for i in range(3)
            ],
        )
    return "bookings"


def test_an_agent_run_scores_each_trajectory(
    sessions: sessionmaker[Session], bookings: str
) -> None:
    fake = FakeAnthropic()
    fake.default = lambda body: verdict(5, "lookup then book")
    from evalkit.pricing import PriceBook
    from evalkit.providers.registry import ProviderRegistry

    settings = Settings(redaction=RedactionConfig())
    factory = Factory()
    runner = Runner(
        sessions,
        settings,
        target_factory=factory,
        providers=ProviderRegistry({}, PriceBook(), claude=fake.client()),
    )
    raw = agent_config(bookings).model_dump(mode="json", by_alias=True)
    raw["scorers"] = [
        {"type": "tool_calls", "arguments": "exact", "allow_extra": False},
        {"type": "tool_calls", "name": "lookup_then_cancel", "calls": ["find_slots", "cancel"]},
        {"type": "step_efficiency"},
        {"type": "step_efficiency", "name": "replies", "count": "model_steps", "budget": 2},
        {"type": "forbidden_tools", "tools": ["book"], "name": "no_booking"},
        {"type": "forbidden_tools"},
        {"type": "trajectory_judge", "rubric": "Looks up before booking."},
    ]
    run = runner.create_run(RunConfig.model_validate(raw))
    report = asyncio.run(runner.execute(run.id))
    assert (report.status, report.errors) == ("completed", 0)
    with sessions() as s:
        rows = list(s.scalars(select(Result).where(Result.run_id == run.id)))
    assert len(rows) == 3
    for row in rows:
        values = {name: v["value"] for name, v in row.scores.items()}
        assert values == {
            "tool_calls": 1.0,
            "lookup_then_cancel": 0.5,
            "step_efficiency": 1.0,
            "replies": pytest.approx(2 / 3),
            "no_booking": 0.0,
            "forbidden_tools": 1.0,
            "trajectory_judge": 1.0,
        }
        assert row.judge_cost_usd > 0
    assert len(fake.requests) == 3
    sent = fake.requests[0]["messages"][0]["content"]
    assert "tool call book" in sent
    assert "BK-19" in sent
    assert run.judges is not None
    assert run.judges["trajectory_judge"]["mode"] == "trajectory"


def test_a_prompt_target_leaves_trajectory_metrics_unscorable(
    sessions: sessionmaker[Session], bookings: str
) -> None:
    from tests.fake_anthropic import FakeAnthropic as Fake

    fake = Fake()
    fake.default = lambda body: json_reply(message("Booked"))
    from evalkit.pricing import PriceBook
    from evalkit.providers.registry import ProviderRegistry

    runner = Runner(
        sessions,
        Settings.from_env({}),
        providers=ProviderRegistry({}, PriceBook(), claude=fake.client()),
    )
    run = runner.create_run(
        RunConfig.model_validate(
            {
                "label": "plain",
                "dataset": {"name": bookings},
                "target": {"type": "prompt", "model": CLAUDE, "template": "{{ date }}"},
                "scorers": [{"type": "tool_calls"}, {"type": "forbidden_tools"}],
            }
        )
    )
    asyncio.run(runner.execute(run.id))
    with sessions() as s:
        rows = list(s.scalars(select(Result).where(Result.run_id == run.id)))
    assert rows
    for row in rows:
        assert all(v == {"value": None, "detail": NO_TRAJECTORY} for v in row.scores.values())
    assert SANDBOX  # the agent fixtures this module shares


def test_remaining_edges() -> None:
    # More expectations than calls: the second cannot take the only call.
    same = ["a", "a"]
    assert maximum_matching(same, ["a"], lambda i, j: True) == [(0, 0)]
    unordered = ToolCallMatch("c", calls=["a", "a"], order="unordered")
    assert score(unordered, trajectory(("a", {}))).detail == "1/2 expected calls: missing a"
    # A config schema check skips calls given by name alone.
    named = ToolCallMatch("c", calls=["a", {"tool": "b", "arguments": {}}], arguments="schema")
    assert score(named, trajectory(("a", 1), ("b", {"x": 1}))).value == 1.0
    # With no case key, only the configured patterns apply.
    config_only = ForbiddenTools("f", tools=["x"], expected_key=None)
    assert (
        score(config_only, trajectory(("y", {})), with_expected({"forbidden_tools": ["y"]})).value
        == 1.0
    )
    # A trajectory judge asked to judge without steps still grades what it has.
    fake = FakeAnthropic(verdict(3))
    judgement = asyncio.run(make_judge(fake).judge(CASE, "out"))
    assert judgement.label == 3
    assert "<trajectory>\n\n</trajectory>" in fake.requests[0]["messages"][0]["content"]
