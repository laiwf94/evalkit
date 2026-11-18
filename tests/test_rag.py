"""RAG metrics and the RAG report: context relevance, context recall, answer
completeness and noise sensitivity against a scripted judge; cause attribution
on constructed cases; and a run whose failing cases ``evalkit report rag``
attributes to a retrieval miss, an unsupported answer and ignored context."""

import asyncio
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import pytest
import regex
from pydantic import TypeAdapter, ValidationError
from sqlalchemy.orm import Session, sessionmaker

from evalkit.cli import main
from evalkit.config import Settings
from evalkit.datasets import Case, import_dataset
from evalkit.errors import InvalidInputError, NotFoundError
from evalkit.pricing import PriceBook
from evalkit.providers.registry import ProviderRegistry
from evalkit.rag_report import (
    CAUSES,
    MetricRole,
    attribute,
    format_rag,
    parse_thresholds,
    rag_report,
    role_of,
)
from evalkit.ratelimit import RetryPolicy
from evalkit.runner import Runner
from evalkit.runspec import RunConfig
from evalkit.scorers.faithfulness import EXTRACT_SYSTEM, VERIFY_SYSTEM, FaithfulnessJudge
from evalkit.scorers.judge import LLMJudge
from evalkit.scorers.rag import (
    RELEVANCE_SYSTEM,
    AnswerCompletenessJudge,
    ContextRecallJudge,
    ContextRelevanceJudge,
    NoiseSensitivityJudge,
    reference_of,
)
from evalkit.scorers.registry import ScorerSpec, build_scorers
from tests.fake_anthropic import FakeAnthropic, error_reply, json_reply, message

SOURCES = [
    {"id": "kb-1", "text": "Paris is the capital of France."},
    {"id": "kb-2", "text": "The Eiffel Tower was completed in 1889."},
    {"id": "kb-3", "text": "Bananas are rich in potassium."},
]
REFERENCE = "Paris is the capital of France and its tower was completed in 1889."
CASE = Case(
    id="c1",
    input={"question": "Tell me about Paris.", "contexts": SOURCES},
    expected={"answer": REFERENCE},
)
RETRY = RetryPolicy(max_attempts=2, base_delay=0)
Handler = Callable[[dict[str, Any]], httpx2.Response]


def reply(payload: dict[str, Any]) -> httpx2.Response:
    return json_reply(message(json.dumps(payload)))


def system_of(body: dict[str, Any]) -> str:
    return str(body["system"][0]["text"])


def content_of(body: dict[str, Any]) -> str:
    return str(body["messages"][0]["content"])


def claims_in(content: str) -> list[str]:
    block = content.split("<claims>\n", 1)[1].split("\n</claims>", 1)[0]
    return [regex.sub(r"^\[\d+\] ", "", line) for line in block.splitlines() if line]


def source_ids(content: str) -> list[str]:
    return regex.findall(r'<source(?: number="\d+" rank="\d+")? id="([^"]+)"', content)


def answer_in(content: str) -> str:
    return content.split("<answer>\n", 1)[1].split("\n</answer>", 1)[0]


def scripted(
    claims: Callable[[str], list[str]],
    verdict: Callable[[str, list[str]], tuple[str, list[str]]],
    relevant: Callable[[str], bool] = lambda _id: True,
) -> Handler:
    """A judge that extracts ``claims(answer)`` and gives each claim
    ``verdict(claim, source ids shown)``; ``relevant(id)`` for relevance checks."""

    def handler(body: dict[str, Any]) -> httpx2.Response:
        system, content = system_of(body), content_of(body)
        if system == EXTRACT_SYSTEM:
            return reply({"claims": claims(answer_in(content))})
        if system == RELEVANCE_SYSTEM:
            ids = source_ids(content)
            return reply(
                {
                    "sources": [
                        {"source": n, "reasoning": "r", "relevant": relevant(i)}
                        for n, i in enumerate(ids, start=1)
                    ]
                }
            )
        assert system == VERIFY_SYSTEM
        ids = source_ids(content)
        out = []
        for n, claim in enumerate(claims_in(content), start=1):
            v, named = verdict(claim, ids)
            out.append({"claim": n, "reasoning": "r", "verdict": v, "sources": named})
        return reply({"verdicts": out})

    return handler


def fake_with(handler: Handler) -> FakeAnthropic:
    fake = FakeAnthropic()
    fake.default = handler
    return fake


def judge[J: LLMJudge](cls: type[J], fake: FakeAnthropic, **kw: Any) -> J:
    return cls(cls.__name__.lower(), client=fake.client(), retry=RETRY, **kw)


# ---------------------------------------------------------------- reference


def test_the_reference_is_a_key_or_the_whole_expected_text() -> None:
    assert reference_of(CASE, "answer") == REFERENCE
    assert reference_of(Case(id="a", input="q", expected="text"), "answer") == "text"
    assert reference_of(Case(id="a", input="q", expected="text"), None) == "text"
    assert reference_of(Case(id="a", input="q", expected={"other": "x"}), "answer") is None
    assert reference_of(Case(id="a", input="q", expected="  "), "answer") is None
    assert reference_of(Case(id="a", input="q"), "answer") is None


# ---------------------------------------------------------------- context relevance


def test_context_relevance_names_the_irrelevant_sources_by_rank() -> None:
    fake = fake_with(scripted(lambda _: [], lambda *_: ("supported", []), lambda i: i != "kb-3"))
    j = asyncio.run(judge(ContextRelevanceJudge, fake).judge(CASE, "unused"))
    assert j.value == pytest.approx(2 / 3)
    assert j.detail == "2/3 sources relevant; irrelevant: kb-3 (rank 3)"
    assert j.label == "fail"
    assert len(fake.requests) == 1
    assert "Tell me about Paris." in content_of(fake.requests[0])


def test_context_relevance_checks_sources_in_batches_with_their_ranks() -> None:
    many = [{"id": f"d{i}", "text": f"text {i}"} for i in range(1, 8)]
    case = Case(id="m", input={"question": "q", "contexts": many})
    fake = fake_with(scripted(lambda _: [], lambda *_: ("supported", []), lambda i: i != "d6"))
    j = asyncio.run(judge(ContextRelevanceJudge, fake, batch_size=3).judge(case, ""))
    assert len(fake.requests) == 3
    assert j.value == pytest.approx(6 / 7)
    assert j.detail.endswith("irrelevant: d6 (rank 6)")
    second = content_of(fake.requests[1])
    assert 'number="1" rank="4" id="d4"' in second


def test_context_relevance_reads_a_ranking_from_the_output() -> None:
    ranking = json.dumps({"documents": [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}]})
    fake = fake_with(scripted(lambda _: [], lambda *_: ("supported", []), lambda i: i == "a"))
    scorer = judge(ContextRelevanceJudge, fake, contexts_from="output")
    j = asyncio.run(scorer.judge(Case(id="r", input="a query"), ranking))
    assert (j.value, j.detail) == (0.5, "1/2 sources relevant; irrelevant: b (rank 2)")
    assert "a query" in content_of(fake.requests[0])
    bad = asyncio.run(scorer.judge(Case(id="r", input="q"), "not json"))
    assert bad.value is None
    assert "not a JSON list of sources" in bad.detail


def test_context_relevance_needs_one_verdict_per_source() -> None:
    fake = fake_with(
        lambda _b: reply({"sources": [{"source": 1, "reasoning": "r", "relevant": True}]})
    )
    j = asyncio.run(judge(ContextRelevanceJudge, fake).judge(CASE, ""))
    assert (j.value, j.detail) == (None, "judge did not return exactly one verdict per source")
    failing = FakeAnthropic(error_reply(400, "invalid_request_error"))
    j = asyncio.run(judge(ContextRelevanceJudge, failing).judge(CASE, ""))
    assert j.value is None
    assert j.detail.startswith("judge call failed")
    no_sources = asyncio.run(
        judge(ContextRelevanceJudge, FakeAnthropic()).judge(Case(id="n", input="q"), "")
    )
    assert no_sources.value is None
    assert "no sources under 'contexts'" in no_sources.detail


# ---------------------------------------------------------------- context recall

REF_CLAIMS = ["Paris is the capital of France.", "The tower was completed in 1889."]


def recall_handler(found: set[str]) -> Handler:
    return scripted(
        lambda _: REF_CLAIMS,
        lambda claim, _ids: ("supported", ["kb-1"]) if claim in found else ("unsupported", []),
    )


def test_context_recall_lists_reference_claims_the_sources_lack() -> None:
    fake = fake_with(recall_handler({REF_CLAIMS[0]}))
    j = asyncio.run(judge(ContextRecallJudge, fake).judge(CASE, "whatever the answer says"))
    assert j.value == 0.5
    assert j.detail == (
        '1/2 reference claims in the sources; not retrieved: "The tower was completed in 1889."'
    )
    extract = content_of(fake.requests[0])
    assert REFERENCE in extract
    assert "whatever the answer says" not in extract
    assert source_ids(content_of(fake.requests[1])) == ["kb-1", "kb-2", "kb-3"]


def test_context_recall_of_a_complete_retrieval_is_one() -> None:
    fake = fake_with(recall_handler(set(REF_CLAIMS)))
    j = asyncio.run(judge(ContextRecallJudge, fake).judge(CASE, ""))
    assert (j.value, j.label) == (1.0, "pass")
    assert j.detail == "2/2 reference claims in the sources"
    assert j.input_tokens == 200


def test_context_recall_needs_a_reference_with_claims() -> None:
    fake = fake_with(scripted(lambda _: [], lambda *_: ("supported", [])))
    scorer = judge(ContextRecallJudge, fake)
    no_ref = Case(id="n", input={"question": "q", "contexts": SOURCES})
    j = asyncio.run(scorer.judge(no_ref, ""))
    assert j.value is None
    assert "no reference answer under expected.answer" in j.detail
    assert fake.requests == []
    j = asyncio.run(scorer.judge(CASE, ""))
    assert (j.value, j.detail) == (None, "the reference answer makes no factual claims")
    j = asyncio.run(judge(ContextRecallJudge, fake, reference_key=None).judge(no_ref, ""))
    assert "under expected:" not in j.detail
    assert "no reference answer under expected" in j.detail


def test_context_recall_from_a_ranking_output() -> None:
    ranking = json.dumps({"documents": [{"id": "d9", "text": "Paris is the capital."}]})
    fake = fake_with(recall_handler({REF_CLAIMS[0]}))
    case = Case(id="r", input="paris", expected=REFERENCE)
    j = asyncio.run(judge(ContextRecallJudge, fake, contexts_from="output").judge(case, ranking))
    assert j.value == 0.5
    assert source_ids(content_of(fake.requests[1])) == ["d9"]


def test_context_recall_judge_failures_are_unscorable() -> None:
    fake = fake_with(scripted(lambda _: REF_CLAIMS, lambda *_: ("supported", [])))
    fake.default = lambda b: (
        reply({"claims": REF_CLAIMS})
        if system_of(b) == EXTRACT_SYSTEM
        else reply(
            {"verdicts": [{"claim": 1, "reasoning": "r", "verdict": "supported", "sources": []}]}
        )
    )
    j = asyncio.run(judge(ContextRecallJudge, fake).judge(CASE, ""))
    assert (j.value, j.detail) == (None, "judge did not return exactly one verdict per claim")
    refusing = FakeAnthropic(json_reply(message("{}", stop_reason="refusal")))
    j = asyncio.run(judge(ContextRecallJudge, refusing).judge(CASE, ""))
    assert (j.value, j.detail) == (None, "judge declined to grade this case")


# ---------------------------------------------------------------- answer completeness


def test_answer_completeness_lists_what_the_answer_leaves_out_or_contradicts() -> None:
    def verdict(claim: str, ids: list[str]) -> tuple[str, list[str]]:
        assert ids == ["answer"]
        if claim == REF_CLAIMS[1]:
            return "contradicted", ["answer"]
        if claim == "France uses the euro.":
            return "unsupported", []
        return "supported", ["answer"]

    fake = fake_with(scripted(lambda _: [*REF_CLAIMS, "France uses the euro."], verdict))
    answer = "Paris is the capital of France; the tower opened in 1925."
    j = asyncio.run(judge(AnswerCompletenessJudge, fake).judge(CASE, answer))
    assert j.value == pytest.approx(1 / 3)
    assert j.detail == (
        '1/3 reference points in the answer; left out: "France uses the euro."; '
        'contradicted: "The tower was completed in 1889."'
    )
    assert answer in content_of(fake.requests[1])
    assert "Bananas" not in content_of(fake.requests[1])


def test_an_empty_answer_covers_nothing_without_a_call() -> None:
    fake = FakeAnthropic()
    j = asyncio.run(judge(AnswerCompletenessJudge, fake).judge(CASE, "  "))
    assert (j.value, j.detail) == (0.0, "the answer is empty")
    j = asyncio.run(judge(AnswerCompletenessJudge, fake).judge(Case(id="x", input="q"), ""))
    assert j.value is None
    assert fake.requests == []


# ---------------------------------------------------------------- noise sensitivity


def test_noise_sensitivity_separates_misled_claims_from_invented_ones() -> None:
    answer_claims = [
        "Paris is the capital of France.",  # true, in kb-1
        "Bananas cure fatigue.",  # wrong, but kb-3 led to it
        "The tower is 900 m tall.",  # wrong, in no source
    ]

    def verdict(claim: str, ids: list[str]) -> tuple[str, list[str]]:
        if ids == ["reference"]:
            ok = claim == answer_claims[0]
            return ("supported", ["reference"]) if ok else ("unsupported", [])
        if claim == answer_claims[0]:
            return "supported", ["kb-1"]
        if claim == answer_claims[1]:
            return "supported", ["kb-3"]
        return "unsupported", []

    fake = fake_with(scripted(lambda _: answer_claims, verdict))
    j = asyncio.run(judge(NoiseSensitivityJudge, fake).judge(CASE, "an answer"))
    assert j.value == pytest.approx(2 / 3)
    assert j.detail == (
        "sensitivity 0.33: 1/3 claims are incorrect and taken from the sources; "
        'misled: "Bananas cure fatigue." (from kb-3); 1 incorrect claim(s) no source states'
    )
    assert len(fake.requests) == 3
    assert REFERENCE in content_of(fake.requests[1]) + content_of(fake.requests[2])


def test_noise_sensitivity_of_an_answer_without_claims_is_one() -> None:
    fake = fake_with(scripted(lambda _: [], lambda *_: ("supported", [])))
    scorer = judge(NoiseSensitivityJudge, fake)
    assert asyncio.run(scorer.judge(CASE, "")).value == 1.0
    assert fake.requests == []
    j = asyncio.run(scorer.judge(CASE, "Hello!"))
    assert (j.value, j.detail) == (1.0, "the answer makes no factual claims")
    no_ref = Case(id="n", input={"question": "q", "contexts": SOURCES})
    assert asyncio.run(scorer.judge(no_ref, "x")).value is None
    no_sources = Case(id="n", input="q", expected="ref")
    assert asyncio.run(scorer.judge(no_sources, "x")).value is None


def test_noise_sensitivity_failures_are_unscorable() -> None:
    def handler(body: dict[str, Any]) -> httpx2.Response:
        if system_of(body) == EXTRACT_SYSTEM:
            return reply({"claims": ["a", "b"]})
        if 'id="reference"' in content_of(body):
            return error_reply(400, "invalid_request_error")
        return reply({"verdicts": []})

    j = asyncio.run(judge(NoiseSensitivityJudge, fake_with(handler)).judge(CASE, "x"))
    assert j.value is None
    assert j.detail.startswith("judge call failed")

    def bad_sources(body: dict[str, Any]) -> httpx2.Response:
        if system_of(body) == EXTRACT_SYSTEM:
            return reply({"claims": ["a"]})
        if 'id="reference"' in content_of(body):
            return reply(
                {
                    "verdicts": [
                        {"claim": 1, "reasoning": "r", "verdict": "supported", "sources": []}
                    ]
                }
            )
        return reply({"verdicts": []})

    j = asyncio.run(judge(NoiseSensitivityJudge, fake_with(bad_sources)).judge(CASE, "x"))
    assert (j.value, j.detail) == (None, "judge did not return exactly one verdict per claim")
    j = asyncio.run(
        judge(
            NoiseSensitivityJudge, FakeAnthropic(error_reply(400, "invalid_request_error"))
        ).judge(CASE, "x")
    )
    assert j.value is None


# ---------------------------------------------------------------- identity, estimates, specs


def test_each_judge_has_its_own_identity_and_settings() -> None:
    fake = FakeAnthropic()
    judges = [
        judge(ContextRelevanceJudge, fake),
        judge(ContextRecallJudge, fake),
        judge(AnswerCompletenessJudge, fake),
        judge(NoiseSensitivityJudge, fake),
        judge(FaithfulnessJudge, fake),
    ]
    modes = [j.identity()["mode"] for j in judges]
    assert modes == [
        "context_relevance",
        "context_recall",
        "answer_completeness",
        "noise_sensitivity",
        "faithfulness",
    ]
    assert len({j.fingerprint() for j in judges}) == 5
    recall = judge(ContextRecallJudge, fake)
    from_output = judge(ContextRecallJudge, fake, contexts_from="output")
    assert recall.fingerprint() != from_output.fingerprint()
    assert recall.fingerprint() != judge(ContextRecallJudge, fake, batch_size=3).fingerprint()
    assert from_output.spec()["contexts_from"] == "output"
    assert recall.spec()["reference_key"] == "answer"
    with pytest.raises(ValueError, match="contexts_from"):
        judge(ContextRecallJudge, fake, contexts_from="nowhere")


def test_prompts_for_estimates() -> None:
    fake = FakeAnthropic()
    assert len(judge(ContextRelevanceJudge, fake, batch_size=2).prompts(CASE, "")) == 2
    assert len(judge(ContextRecallJudge, fake).prompts(CASE, "")) == 2
    assert len(judge(AnswerCompletenessJudge, fake).prompts(CASE, "")) == 2
    assert len(judge(NoiseSensitivityJudge, fake).prompts(CASE, "")) == 3
    bare = Case(id="b", input="q")
    for cls in (ContextRelevanceJudge, ContextRecallJudge, AnswerCompletenessJudge):
        assert judge(cls, fake).prompts(bare, "") == []
    assert judge(NoiseSensitivityJudge, fake).prompts(bare, "") == []
    assert (
        judge(NoiseSensitivityJudge, fake).prompts(Case(id="b", input="q", expected="r"), "") == []
    )
    ref_only = Case(id="b", input="q", expected="r")
    assert judge(ContextRecallJudge, fake).prompts(ref_only, "") == []


def specs(*raw: dict[str, Any]) -> list[ScorerSpec]:
    return TypeAdapter(list[ScorerSpec]).validate_python(list(raw))


def test_specs_build_the_rag_judges() -> None:
    fake = FakeAnthropic()
    built = build_scorers(
        specs(
            {"type": "context_relevance", "contexts_from": "output", "batch_size": 4},
            {"type": "context_recall", "reference_key": "gold", "effort": "low"},
            {"type": "answer_completeness", "name": "complete"},
            {"type": "noise_sensitivity", "contexts_key": "docs"},
        ),
        allow_import_paths=False,
        claude=fake.client(),
    )
    relevance, recall, complete, noise = built
    assert isinstance(relevance, ContextRelevanceJudge)
    assert (relevance.contexts_from, relevance.batch_size) == ("output", 4)
    assert isinstance(recall, ContextRecallJudge)
    assert (recall.reference_key, recall.effort) == ("gold", "low")
    assert isinstance(complete, AnswerCompletenessJudge)
    assert complete.name == "complete"
    assert isinstance(noise, NoiseSensitivityJudge)
    assert noise.contexts_key == "docs"


@pytest.mark.parametrize(
    "raw",
    [
        {"type": "context_relevance", "contexts_from": "elsewhere"},
        {"type": "context_recall", "batch_size": 0},
        {"type": "answer_completeness", "contexts_from": "output"},
        {"type": "noise_sensitivity", "reference_key": ""},
        {"type": "context_recall", "rubric": "not taken"},
    ],
)
def test_bad_rag_specs_are_refused(raw: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        specs(raw)


def test_an_unknown_provider_is_refused_for_rag_judges() -> None:
    with pytest.raises(InvalidInputError, match="scorer context_recall"):
        build_scorers(
            specs({"type": "context_recall", "provider": "nowhere"}),
            allow_import_paths=False,
            claude=FakeAnthropic().client(),
        )


# ---------------------------------------------------------------- attribution


def test_roles_follow_the_scorer_type() -> None:
    assert role_of({"type": "ranking", "metric": "recall"}) == "retrieval"
    assert role_of({"type": "ranking", "metric": "ndcg"}) == "retrieval"
    assert role_of({"type": "ranking", "metric": "precision"}) == "precision"
    assert role_of({"type": "context_recall"}) == "retrieval"
    assert role_of({"type": "context_relevance"}) == "precision"
    for kind in ("faithfulness", "citation_accuracy", "noise_sensitivity"):
        assert role_of({"type": kind}) == "grounding"
    for kind in ("answer_completeness", "exact", "token_f1", "judge", "plugin"):
        assert role_of({"type": kind}) == "answer"
    assert role_of({"type": "tool_calls"}) == "other"


ROLES = [
    MetricRole("recall_at_5", "ranking", "retrieval", False, 0.5),
    MetricRole("precision_at_5", "ranking", "precision", True, 0.2),
    MetricRole("faithfulness", "faithfulness", "grounding", False, 0.8),
    MetricRole("answer_completeness", "answer_completeness", "answer", True, 0.5),
    MetricRole("exact", "exact", "answer", True, 0.5),
    MetricRole("tool_calls", "tool_calls", "other", True, 0.5),
]


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        # Everything above its threshold: not failing.
        ({"recall_at_5": 1, "precision_at_5": 0.4, "faithfulness": 1, "exact": 1}, None),
        # Ungated metrics alone never fail a case.
        ({"recall_at_5": 0, "faithfulness": 0, "exact": 1}, None),
        # Neither do trajectory metrics.
        ({"tool_calls": 0, "exact": 1}, None),
        # Retrieval comes first, whatever the answer did.
        (
            {"recall_at_5": 0.25, "faithfulness": 0.5, "exact": 0},
            ("retrieval_miss", ["recall_at_5"]),
        ),
        # Retrieved, then unfaithful: the answer is unsupported.
        (
            {"recall_at_5": 1, "faithfulness": 0.75, "exact": 0},
            ("unsupported_answer", ["faithfulness"]),
        ),
        # Retrieved and faithful, still wrong: the context was ignored.
        ({"recall_at_5": 1, "faithfulness": 1, "exact": 0}, ("ignored_context", ["exact"])),
        # Completeness measures omission directly, so grounding is not needed.
        (
            {"recall_at_5": 1, "answer_completeness": 0.0},
            ("ignored_context", ["answer_completeness"]),
        ),
        # Without grounding evidence a wrong answer cannot be split.
        ({"recall_at_5": 1, "exact": 0}, ("unattributed", ["exact"])),
        # Without retrieval evidence neither can anything else.
        ({"faithfulness": 1, "exact": 0}, ("unattributed", ["exact"])),
        ({"recall_at_5": None, "faithfulness": 1, "exact": 0}, ("unattributed", ["exact"])),
        # Only precision failed.
        (
            {"recall_at_5": 1, "precision_at_5": 0.0, "exact": 1},
            ("noisy_retrieval", ["precision_at_5"]),
        ),
    ],
)
def test_each_failing_case_gets_the_first_cause_that_applies(
    scores: dict[str, float | None], expected: tuple[str, list[str]] | None
) -> None:
    found = attribute("ok", scores, ROLES)
    assert (found if found is None else found[:2]) == expected


def test_a_failed_target_call_is_an_error() -> None:
    assert attribute("error", {}, ROLES) == ("error", [], "the target call failed")


def test_thresholds() -> None:
    assert parse_thresholds([]) == (0.5, {})
    assert parse_thresholds(["0.7", "faithfulness=0.9", "ndcg_at_5=0"]) == (
        0.7,
        {"faithfulness": 0.9, "ndcg_at_5": 0.0},
    )
    for bad, why in [
        ("x", "not a number"),
        ("faithfulness=high", "not a number"),
        ("1.5", "between 0 and 1"),
        ("nan", "between 0 and 1"),
        ("=0.5", "names no metric"),
    ]:
        with pytest.raises(InvalidInputError, match=why):
            parse_thresholds([bad])


# ---------------------------------------------------------------- a run and its report

QUESTIONS = {
    "ok": "Where is the Louvre?",
    "miss": "When did the tower open?",
    "hallucinated": "How tall is the tower?",
    "ignored": "What currency does France use?",
    "broken": "Is the target up?",
}


def rag_cases() -> list[Case]:
    return [
        Case(
            id=name,
            input={"question": question, "contexts": SOURCES},
            expected={"answer": f"{name} reference"},
        )
        for name, question in QUESTIONS.items()
    ]


def run_handler(body: dict[str, Any]) -> httpx2.Response:
    """Claims name their case and whether they come from the reference or the
    answer; the case decides which check fails."""
    content = content_of(body)
    if system_of(body) == EXTRACT_SYSTEM:
        case = next(n for n, q in QUESTIONS.items() if q in content)
        side = "out" if answer_in(content).startswith("{") else "ref"
        return reply({"claims": [f"{side} {case} 1", f"{side} {case} 2"]})
    ids = source_ids(content)
    out = []
    for n, claim in enumerate(claims_in(content), start=1):
        side, case, _ = claim.split()
        if ids == ["answer"]:
            # Completeness: every wrong answer leaves the reference out.
            bad = case in {"miss", "hallucinated", "ignored"}
        elif side == "ref":
            bad = case == "miss"  # recall: the sources lack the reference
        else:
            bad = case == "hallucinated"  # faithfulness: the answer invents
        verdict = "unsupported" if bad else "supported"
        out.append({"claim": n, "reasoning": "r", "verdict": verdict, "sources": []})
    return reply({"verdicts": out})


def rag_run(sessions: sessionmaker[Session], fake: FakeAnthropic) -> str:
    with sessions() as s:
        import_dataset(s, "rag", rag_cases())
    runner = Runner(
        sessions, Settings(), providers=ProviderRegistry({}, PriceBook(), claude=fake.client())
    )
    config = RunConfig.model_validate(
        {
            "label": "rag-run",
            "dataset": {"name": "rag"},
            "target": {"type": "fake", "accuracy": 1.0},
            "scorers": [
                # Retrieval and grounding explain failures; completeness gates.
                {"type": "context_recall", "gate": False},
                {"type": "faithfulness", "gate": False},
                {"type": "answer_completeness"},
            ],
        }
    )
    run = runner.create_run(config)
    report = asyncio.run(runner.execute(run.id))
    assert report.status == "completed"
    return run.id


def test_report_attributes_each_failing_case(sessions: sessionmaker[Session]) -> None:
    run_id = rag_run(sessions, fake_with(run_handler))
    with sessions() as s:
        report = rag_report(s, run_id)
    assert (report.cases, report.failing) == (5, 3)
    by_case = {c.case_id: c for c in report.shown}
    assert {k: c.cause for k, c in by_case.items()} == {
        "miss": "retrieval_miss",
        "hallucinated": "unsupported_answer",
        "ignored": "ignored_context",
    }
    assert by_case["miss"].failing == {"context_recall": 0.0, "answer_completeness": 0.0}
    assert by_case["miss"].detail.startswith("context_recall: 0/2 reference claims")
    assert by_case["hallucinated"].failing == {"faithfulness": 0.0, "answer_completeness": 0.0}
    assert by_case["hallucinated"].detail.startswith("faithfulness: 0/2 claims supported")
    assert by_case["ignored"].failing == {"answer_completeness": 0.0}
    assert "left out" in by_case["ignored"].detail
    assert report.causes == {
        "retrieval_miss": 1,
        "unsupported_answer": 1,
        "ignored_context": 1,
        "noisy_retrieval": 0,
        "unattributed": 0,
        "error": 0,
    }
    assert list(report.causes) == list(CAUSES)
    metrics = {m.name: m for m in report.metrics}
    assert metrics["context_recall"].mean == pytest.approx(0.8)
    assert (metrics["context_recall"].role, metrics["context_recall"].gate) == ("retrieval", False)
    assert metrics["answer_completeness"].below == 3
    assert metrics["faithfulness"].below == 1
    assert report.notes == []

    text = format_rag(report)
    assert "3 failing" in text
    assert (
        "  miss  retrieval miss: the needed sources were not retrieved  "
        "[context_recall 0.00, answer_completeness 0.00]\n"
    ) in text
    assert "ignored context" in text

    # The cases that pass do so at any threshold; the list is the first by id.
    with sessions() as s:
        strict = rag_report(s, run_id, threshold=1.0, limit=1)
    assert strict.failing == 3
    assert [c.case_id for c in strict.shown] == ["hallucinated"]
    # Without the gate's failure, a grounding failure alone fails nothing.
    with sessions() as s:
        lenient = rag_report(s, run_id, thresholds={"answer_completeness": 0.0})
    assert lenient.failing == 0
    with pytest.raises(InvalidInputError, match="no metric named nope"), sessions() as s:
        rag_report(s, run_id, thresholds={"nope": 0.5})
    with pytest.raises(InvalidInputError, match="limit"), sessions() as s:
        rag_report(s, run_id, limit=-1)
    with pytest.raises(NotFoundError), sessions() as s:
        rag_report(s, "missing")


def test_report_counts_target_errors_and_says_what_a_run_cannot_tell(
    sessions: sessionmaker[Session],
) -> None:
    with sessions() as s:
        import_dataset(s, "plain", [Case(id=f"c{i}", input="q", expected="a") for i in range(4)])
    runner = Runner(sessions, Settings())
    config = RunConfig.model_validate(
        {
            "label": "plain",
            "dataset": {"name": "plain"},
            "target": {"type": "fake", "accuracy": 0.0, "failure_rate": 0.5, "seed": 3},
            "scorers": [{"type": "exact"}],
            "options": {"retry": {"max_attempts": 1}},
        }
    )
    run = runner.create_run(config)
    asyncio.run(runner.execute(run.id))
    with sessions() as s:
        report = rag_report(s, run.id, limit=0)
    assert report.failing == 4
    assert report.shown == []
    assert report.causes["error"] + report.causes["unattributed"] == 4
    assert report.causes["error"] >= 1
    assert len(report.notes) == 2
    assert "no retrieval metric" in report.notes[0]
    assert "no grounding metric" in report.notes[1]
    with sessions() as s:
        shown = rag_report(s, run.id).shown
    errors = [c for c in shown if c.cause == "error"]
    assert errors
    assert "fake transient failure" in errors[0].detail
    assert "target error" in format_rag(rag_report_in(sessions, run.id))


def rag_report_in(sessions: sessionmaker[Session], run_id: str) -> Any:
    with sessions() as s:
        return rag_report(s, run_id)


def test_cli_report_rag(sessions: sessionmaker[Session], tmp_path: Path) -> None:
    run_id = rag_run(sessions, fake_with(run_handler))
    db = f"sqlite:///{tmp_path / 'test.db'}"

    def cli(*args: str) -> tuple[int, str]:
        out = io.StringIO()
        return main(["--database-url", db, "report", "rag", *args], out=out), out.getvalue()

    code, text = cli("label:rag-run")
    assert code == 0
    assert text.startswith(f"rag report  run {run_id}  rag-run  dataset rag v1  completed\n")
    assert "retrieval miss" in text
    code, raw = cli(run_id, "--json", "--threshold", "answer_completeness=1", "--limit", "2")
    assert code == 0
    body = json.loads(raw)
    assert body["failing"] == 3
    assert len(body["shown"]) == 2
    assert body["causes"]["ignored_context"] == 1
    assert cli(run_id, "--threshold", "2")[0] == 2
    assert cli("label:none")[0] == 2
