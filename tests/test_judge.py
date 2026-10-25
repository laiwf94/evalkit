import asyncio
import json
from typing import Any

import httpx2
import pytest
from pydantic import TypeAdapter
from sqlalchemy.orm import Session, sessionmaker

from evalkit.datasets import Case
from evalkit.errors import InvalidInputError
from evalkit.ratelimit import RetryPolicy
from evalkit.scorers import ScorerSpec, build_scorers
from evalkit.scorers.calibration import (
    LabelledItem,
    calibrate,
    cohens_kappa,
    parse_labels,
    save_calibration,
)
from evalkit.scorers.judge import LLMJudge, pair_label
from tests.fake_anthropic import FakeAnthropic, error_reply, json_reply, message

CASE = Case(id="c1", input={"question": "What is the capital of France?"}, expected="Paris")


def reply(payload: dict[str, Any], **kw: Any) -> httpx2.Response:
    return json_reply(message(json.dumps(payload), **kw))


def judge(fake: FakeAnthropic, mode: str = "rubric", **kw: Any) -> LLMJudge:
    return LLMJudge(
        "quality",
        client=fake.client(),
        rubric="The answer is correct and concise.",
        mode=mode,  # type: ignore[arg-type]
        retry=RetryPolicy(max_attempts=2, base_delay=0),
        **kw,
    )


def test_rubric_request_and_score() -> None:
    fake = FakeAnthropic(reply({"reasoning": "correct", "score": 4}))
    result = asyncio.run(judge(fake).score(CASE, "Paris."))
    assert result.value == 0.75
    assert result.detail == "4/5: correct"
    assert result.input_tokens == 100
    assert result.cost_usd > 0
    body = fake.requests[0]
    assert "The answer is correct and concise." in body["system"][0]["text"]
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["output_config"]["format"]["schema"]["required"] == ["reasoning", "score"]
    content = body["messages"][0]["content"]
    assert "<reference>\nParis\n</reference>" in content
    assert "<response>\nParis.\n</response>" in content
    assert "tool_choice" not in body


def test_rubric_without_reference() -> None:
    fake = FakeAnthropic(reply({"reasoning": "ok", "score": 1}))
    result = asyncio.run(judge(fake).score(Case(id="x", input="hi"), "hello"))
    assert result.value == 0.0
    assert "<reference>" not in fake.requests[0]["messages"][0]["content"]


def _answer_by_position(winner_when_candidate_is: str) -> Any:
    """A judge that always picks a position, ignoring content: maximal position bias."""

    def handler(body: dict[str, Any]) -> httpx2.Response:
        return reply({"reasoning": "first one", "winner": winner_when_candidate_is})

    return handler


def test_pairwise_swaps_order_between_calls() -> None:
    fake = FakeAnthropic(
        reply({"reasoning": "candidate better", "winner": "A"}),
        reply({"reasoning": "candidate better", "winner": "B"}),
    )
    result = asyncio.run(judge(fake, "pairwise").score(CASE, "Paris, France."))
    first, second = (r["messages"][0]["content"] for r in fake.requests)
    assert "<response_a>\nParis, France.\n</response_a>" in first
    assert "<response_b>\nParis\n</response_b>" in first
    assert "<response_a>\nParis\n</response_a>" in second
    assert "<response_b>\nParis, France.\n</response_b>" in second
    assert result.value == 1.0
    assert result.detail.startswith("win vs reference: ")
    assert result.input_tokens == 200


def test_position_biased_judge_yields_ties() -> None:
    fake = FakeAnthropic()
    fake.default = _answer_by_position("A")
    result = asyncio.run(judge(fake, "pairwise").score(CASE, "Lyon"))
    assert result.value == 0.5
    assert "verdict changed with order" in result.detail


@pytest.mark.parametrize(
    ("first", "second", "value"),
    [("B", "A", 0.0), ("tie", "tie", 0.5), ("A", "tie", 0.75), ("tie", "A", 0.25)],
)
def test_pairwise_outcomes(first: str, second: str, value: float) -> None:
    fake = FakeAnthropic(
        reply({"reasoning": "r", "winner": first}), reply({"reasoning": "r", "winner": second})
    )
    assert asyncio.run(judge(fake, "pairwise").score(CASE, "x")).value == value


def test_pairwise_needs_reference() -> None:
    result = asyncio.run(judge(FakeAnthropic(), "pairwise").score(Case(id="x", input="q"), "a"))
    assert result.value is None
    assert result.detail == "pairwise judging needs a reference answer"


@pytest.mark.parametrize(
    ("response", "detail"),
    [
        (json_reply(message("", stop_reason="refusal")), "judge declined"),
        (json_reply(message('{"reasoning": "x"', stop_reason="max_tokens")), "ran out of output"),
        (json_reply(message('{"reasoning": "x", "score": 9}')), "does not match the schema"),
        (error_reply(400, "invalid_request_error"), "judge call failed"),
    ],
)
def test_rubric_failures_are_unscorable(response: httpx2.Response, detail: str) -> None:
    result = asyncio.run(judge(FakeAnthropic(response)).score(CASE, "x"))
    assert result.value is None
    assert detail in result.detail


def test_pairwise_failures_are_unscorable() -> None:
    bad = FakeAnthropic(reply({"reasoning": "r", "winner": "A"}), json_reply(message("not json")))
    assert asyncio.run(judge(bad, "pairwise").score(CASE, "x")).value is None
    failing = FakeAnthropic(
        error_reply(401, "authentication_error"), error_reply(401, "authentication_error")
    )
    result = asyncio.run(judge(failing, "pairwise").score(CASE, "x"))
    assert result.detail.startswith("judge call failed")


def test_judge_retries_transient_errors() -> None:
    fake = FakeAnthropic(
        error_reply(529, "overloaded_error"), reply({"reasoning": "ok", "score": 5})
    )
    assert asyncio.run(judge(fake).score(CASE, "Paris")).value == 1.0


def test_fingerprint_changes_with_rubric() -> None:
    fake = FakeAnthropic()
    a, b = judge(fake), LLMJudge("q", client=fake.client(), rubric="other")
    assert a.fingerprint() != b.fingerprint()
    assert a.spec()["mode"] == "rubric"


def test_judge_spec_builds_and_needs_a_client() -> None:
    specs = TypeAdapter(list[ScorerSpec]).validate_python(
        [{"type": "judge", "rubric": "be right", "mode": "pairwise", "effort": "low"}]
    )
    with pytest.raises(InvalidInputError, match="ANTHROPIC_API_KEY"):
        build_scorers(specs, allow_import_paths=False)
    (scorer,) = build_scorers(specs, allow_import_paths=False, claude=FakeAnthropic().client())
    assert isinstance(scorer, LLMJudge)
    assert (scorer.name, scorer.mode, scorer.effort) == ("judge", "pairwise", "low")


def test_kappa_hand_computed() -> None:
    # Perfect agreement.
    assert cohens_kappa([1, 2, 3], [1, 2, 3], (1, 2, 3)) == 1.0
    # Classic 2x2 example: 20/5/10/15 -> po = 0.7, pe = 0.5, kappa = 0.4.
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohens_kappa(a, b, ("y", "n")) == pytest.approx(0.4)
    # Agreement no better than chance.
    assert cohens_kappa(["y", "y", "n", "n"], ["y", "n", "y", "n"], ("y", "n")) == pytest.approx(
        0.0
    )
    # Systematic disagreement is negative.
    assert cohens_kappa(["y", "n"], ["n", "y"], ("y", "n")) == pytest.approx(-1.0)
    # Only one category used by both raters.
    assert cohens_kappa(["y", "y"], ["y", "y"], ("y", "n")) == 1.0


def test_weighted_kappa_penalises_distance() -> None:
    human = [1, 2, 3, 4, 5, 5, 4, 3]
    near = [2, 2, 3, 4, 4, 5, 4, 3]
    far = [5, 2, 3, 4, 1, 5, 4, 3]
    assert cohens_kappa(human, near, (1, 2, 3, 4, 5), quadratic=True) > cohens_kappa(
        human, far, (1, 2, 3, 4, 5), quadratic=True
    )
    assert cohens_kappa([1], [1], (1,), quadratic=True) == 1.0


def test_kappa_input_errors() -> None:
    with pytest.raises(ValueError, match="same items"):
        cohens_kappa([1], [1, 2], (1, 2))
    with pytest.raises(ValueError, match="no items"):
        cohens_kappa([], [], (1,))
    with pytest.raises(ValueError, match="outside"):
        cohens_kappa([1], [9], (1, 2))


def test_parse_labels() -> None:
    text = (
        '{"id": "a", "input": "q", "output": "o", "label": 4}\n\n'
        '{"id": "b", "input": "q", "output": "o", "label": 1}'
    )
    assert [i.label for i in parse_labels(text, "rubric")] == [4, 1]
    assert (
        parse_labels(b'{"id": "a", "input": "q", "output": "o", "label": "tie"}', "pairwise")[
            0
        ].label
        == "tie"
    )
    for bad, msg in [
        ('{"id": "a", "input": "q", "output": "o", "label": 6}', "label must be one of"),
        ('{"id": "a", "input": "q", "output": "o", "label": true}', "line 1: label: "),
        ('{"id": "a", "input": "q", "output": "o", "label": "win"}', "label must be one of"),
        ("{nope", "line 1"),
        ("", "no labelled items"),
    ]:
        with pytest.raises(InvalidInputError, match=msg):
            parse_labels(bad, "rubric")
    many = "\n".join(
        json.dumps({"id": str(i), "input": "q", "output": "o", "label": 1}) for i in range(5001)
    )
    with pytest.raises(InvalidInputError, match="more than 5000"):
        parse_labels(many, "rubric")


def test_calibration_report_and_storage(sessions: sessionmaker[Session]) -> None:
    labels = [5, 4, 2, 1, 3, 5]
    judged = [5, 4, 3, 1, 3, 4]
    fake = FakeAnthropic(*(reply({"reasoning": "r", "score": s}) for s in judged))
    items = [
        LabelledItem(id=f"i{n}", input="q", output="o", label=lab) for n, lab in enumerate(labels)
    ]
    j = judge(fake)
    report = asyncio.run(calibrate(j, items, concurrency=1))
    assert (report.n, report.judged) == (6, 6)
    assert report.agreement == pytest.approx(4 / 6)
    assert report.kappa == pytest.approx(cohens_kappa(labels, judged, (1, 2, 3, 4, 5)))
    assert report.weighted_kappa is not None
    assert report.weighted_kappa > report.kappa
    assert report.disagreements == ["i2", "i5"]
    assert report.confusion[1][2] == 1  # human 2, judge 3
    assert report.cost_usd > 0
    with sessions() as s:
        row = save_calibration(s, j, report)
        assert row.judge_hash == j.fingerprint()
        assert row.confusion["labels"] == ["1", "2", "3", "4", "5"]


def test_pairwise_calibration_and_unjudgeable_items() -> None:
    fake = FakeAnthropic(
        reply({"reasoning": "r", "winner": "A"}),
        reply({"reasoning": "r", "winner": "B"}),
        json_reply(message("garbage")),
        json_reply(message("garbage")),
    )
    items = [
        LabelledItem(id="w", input="q", expected="ref", output="o", label="win"),
        LabelledItem(id="x", input="q", expected="ref", output="o", label="loss"),
    ]
    report = asyncio.run(calibrate(judge(fake, "pairwise"), items, concurrency=1))
    assert (report.n, report.judged, report.weighted_kappa) == (2, 1, None)
    assert report.labels == ["loss", "tie", "win"]
    nothing = FakeAnthropic(json_reply(message("garbage")))
    with pytest.raises(InvalidInputError, match="could not grade any"):
        asyncio.run(calibrate(judge(nothing), items[:1]))


def test_pair_label() -> None:
    assert [pair_label(v) for v in (0.0, 0.25, 0.5, 0.75, 1.0)] == [
        "loss",
        "loss",
        "tie",
        "win",
        "win",
    ]


class MemoryCache:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.data.get(key)

    async def put(self, key: str, judge_hash: str, text: str, usage: dict[str, Any]) -> None:
        self.data[key] = text


def test_verdicts_are_cached_and_free_on_reuse() -> None:
    fake = FakeAnthropic(
        reply({"reasoning": "ok", "score": 4}),
        json_reply(message("", stop_reason="refusal")),
        json_reply(message("", stop_reason="refusal")),
    )
    j = judge(fake)
    j.cache = MemoryCache()
    first = asyncio.run(j.score(CASE, "Paris"))
    again = asyncio.run(j.score(CASE, "Paris"))
    assert len(fake.requests) == 1
    assert (first.value, again.value) == (0.75, 0.75)
    assert again.cost_usd == 0.0
    # Refusals are not cached: the next run asks again.
    asyncio.run(j.score(CASE, "Lyon"))
    asyncio.run(j.score(CASE, "Lyon"))
    assert len(fake.requests) == 3
