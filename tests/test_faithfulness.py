"""Claim-level faithfulness and citation accuracy: source and citation parsing,
batched support checks against a scripted judge, failure modes, caching, the
scorer specs and a run that scores both."""

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
import regex
from hypothesis import given
from hypothesis import strategies as st
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from evalkit.config import Settings
from evalkit.datasets import Case, import_dataset
from evalkit.db.models import Result
from evalkit.errors import InvalidInputError
from evalkit.pricing import PriceBook
from evalkit.providers.registry import ProviderRegistry
from evalkit.ratelimit import RetryPolicy
from evalkit.runner import Runner
from evalkit.runspec import RunConfig
from evalkit.scorers.faithfulness import (
    CITATION_SYSTEM,
    EXTRACT_SYSTEM,
    MAX_CLAIMS,
    MAX_SOURCE_CHARS,
    MAX_SOURCES,
    MAX_SOURCES_CHARS,
    VERIFY_SYSTEM,
    CitationAccuracyJudge,
    FaithfulnessJudge,
    Source,
    Statement,
    parse_citations,
    parse_sources,
    question_of,
    render_sources,
    sources_of,
)
from evalkit.scorers.judge import LLMJudge
from evalkit.scorers.registry import ScorerSpec, build_scorers
from tests.fake_anthropic import FakeAnthropic, error_reply, json_reply, message

SOURCES = [
    {"id": "kb-1", "text": "Paris is the capital of France."},
    {"id": "kb-2", "text": "The Eiffel Tower was completed in 1889."},
    {"id": "kb-3", "text": "France uses the euro."},
]
CASE = Case(id="c1", input={"question": "Tell me about Paris.", "contexts": SOURCES})
RETRY = RetryPolicy(max_attempts=2, base_delay=0)

Handler = Callable[[dict[str, Any]], httpx2.Response]


def reply(payload: dict[str, Any]) -> httpx2.Response:
    return json_reply(message(json.dumps(payload)))


def system_of(body: dict[str, Any]) -> str:
    return str(body["system"][0]["text"])


def content_of(body: dict[str, Any]) -> str:
    return str(body["messages"][0]["content"])


def numbered(content: str, tag: str) -> list[str]:
    """The numbered lines inside ``<tag>`` of a judge prompt."""
    block = content.split(f"<{tag}>\n", 1)[1].split(f"\n</{tag}>", 1)[0]
    return [regex.sub(r"^\[\d+\] ", "", line) for line in block.splitlines() if line]


def scripted(
    claims: list[str], verdicts: dict[str, tuple[str, list[str]]] | None = None
) -> Handler:
    """A judge that extracts ``claims`` and gives each claim its scripted verdict
    (supported by kb-1 when not listed)."""
    verdicts = verdicts or {}

    def handler(body: dict[str, Any]) -> httpx2.Response:
        if system_of(body) == EXTRACT_SYSTEM:
            return reply({"claims": claims})
        assert system_of(body) == VERIFY_SYSTEM
        out = []
        for number, claim in enumerate(numbered(content_of(body), "claims"), start=1):
            verdict, sources = verdicts.get(claim, ("supported", ["kb-1"]))
            out.append(
                {"claim": number, "reasoning": f"r{number}", "verdict": verdict, "sources": sources}
            )
        return reply({"verdicts": out})

    return handler


def faithfulness(fake: FakeAnthropic, **kw: Any) -> FaithfulnessJudge:
    return FaithfulnessJudge("faithfulness", client=fake.client(), retry=RETRY, **kw)


def citations(fake: FakeAnthropic, **kw: Any) -> CitationAccuracyJudge:
    return CitationAccuracyJudge("citation_accuracy", client=fake.client(), retry=RETRY, **kw)


# ---------------------------------------------------------------- sources


def test_sources_come_as_texts_objects_or_a_ranking() -> None:
    assert parse_sources(["a", "b"]) == [Source("1", "a"), Source("2", "b")]
    assert parse_sources("only one") == [Source("1", "only one")]
    assert parse_sources([{"id": 7, "text": "x"}, {"text": "y"}]) == [
        Source("7", "x"),
        Source("2", "y"),
    ]
    ranking = {"documents": [{"id": "d1", "score": 0.9, "text": "t1"}]}
    assert parse_sources(ranking) == [Source("d1", "t1")]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (5, "must be a list"),
        ([], "lists no sources"),
        ([1], "source 1 must be a text"),
        ([{"id": "d1"}], "source 1 must be a text"),
        ([{"id": "", "text": "x"}], "id must be 1 to 512"),
        ([{"id": True, "text": "x"}], "id must be 1 to 512"),
        (["a", {"id": "1", "text": "b"}], "'1' is repeated"),
        (["x"] * (MAX_SOURCES + 1), "more than 100 sources"),
        ({"documents": [{"id": "d1"}]}, "source 1 must be a text"),
    ],
)
def test_bad_sources_say_what_is_wrong(raw: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_sources(raw)


def test_sources_are_read_from_the_input_then_the_expected_value() -> None:
    assert sources_of(CASE, "contexts")[0].id == "kb-1"
    in_expected = Case(id="e", input="q", expected={"contexts": ["from expected"]})
    assert sources_of(in_expected, "contexts") == [Source("1", "from expected")]
    with pytest.raises(ValueError, match="no sources under 'docs'"):
        sources_of(CASE, "docs")


def test_the_question_leaves_the_sources_out() -> None:
    assert question_of(CASE, "contexts") == "Tell me about Paris."
    assert question_of(Case(id="q", input={"query": "q?", "contexts": []}), "contexts") == "q?"
    other = Case(id="o", input={"topic": "Paris", "contexts": ["x"]})
    assert question_of(other, "contexts") == '{"topic": "Paris"}'
    assert question_of(Case(id="s", input="plain"), "contexts") == "plain"


def test_sources_are_tagged_escaped_and_cut() -> None:
    text = render_sources([Source('a"<b', "hello"), Source("big", "x" * (MAX_SOURCE_CHARS + 50))])
    assert '<source id="a&quot;&lt;b">\nhello\n</source>' in text
    assert "... [80 more characters]" in text
    many = [Source(str(i), "y" * MAX_SOURCE_CHARS) for i in range(15)]
    rendered = render_sources(many)
    assert len(rendered) < MAX_SOURCES_CHARS + 2_000
    assert "[... 5 more sources left out ...]" in rendered


# ---------------------------------------------------------------- citations

S = [Source("kb-1", "a"), Source("kb-2", "b"), Source("kb-3", "c")]


def test_numbers_are_positions_and_ids_are_ids() -> None:
    parsed = parse_citations("Paris is the capital [1]. It has a tower [kb-2, 3].", S)
    assert parsed == [
        Statement("Paris is the capital.", ("kb-1",)),
        Statement("It has a tower.", ("kb-2", "kb-3")),
    ]


def test_a_number_that_is_an_id_names_that_source() -> None:
    sources = [Source("2", "x"), Source("1", "y")]
    assert parse_citations("A [1].", sources) == [Statement("A.", ("1",))]
    texts = parse_sources(["x", "y", "z"])
    assert parse_citations("A [2-3].", texts) == [Statement("A.", ("2", "3"))]


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("A [1-3].", [Statement("A.", ("kb-1", "kb-2", "kb-3"))]),
        ("A [1\N{EN DASH}2].", [Statement("A.", ("kb-1", "kb-2"))]),
        ("A [^2].", [Statement("A.", ("kb-2",))]),
        ("A [1][2] [1].", [Statement("A.", ("kb-1", "kb-2"))]),
        ("A [1; 3].", [Statement("A.", ("kb-1", "kb-3"))]),
        # A marker after the full stop still closes the sentence it follows.
        (
            "A is true. [1] B is true. [2]",
            [Statement("A is true.", ("kb-1",)), Statement("B is true.", ("kb-2",))],
        ),
        ("A is true.\n[1][2]\nB.", [Statement("A is true.", ("kb-1", "kb-2")), Statement("B.")]),
        # Not citations: free text, links, years, a wide range, zero.
        ("He said [sic] so.", [Statement("He said [sic] so.")]),
        ("See [the docs](https://x.example).", [Statement("See [the docs](https://x.example).")]),
        ("Built in [2020].", [Statement("Built in [2020].")]),
        ("A [1-90].", [Statement("A [1-90].")]),
        ("A [3-1].", [Statement("A [3-1].")]),
        ("A [0].", [Statement("A [0].")]),
        ("A [1, sic].", [Statement("A [1, sic].")]),
        ("A [1,].", [Statement("A [1,].")]),
        # A number past the last source is a citation of nothing.
        ("A [2, 7].", [Statement("A.", ("kb-2",), ("7",))]),
        ("", []),
        ("[1]", [Statement("", ("kb-1",))]),
    ],
)
def test_citation_forms(answer: str, expected: list[Statement]) -> None:
    assert parse_citations(answer, S) == expected


@given(st.text(max_size=300))
def test_parsing_never_fails_and_only_names_known_sources(answer: str) -> None:
    ids = {s.id for s in S}
    for statement in parse_citations(answer, S):
        assert set(statement.cited) <= ids
        assert all(m.isdigit() for m in statement.missing)
        assert len(set(statement.cited)) == len(statement.cited)


# ---------------------------------------------------------------- faithfulness

ANSWER = "Paris is the capital of France. The tower opened in 1925. France uses the franc."
CLAIMS = [
    "Paris is the capital of France.",
    "The Eiffel Tower opened in 1925.",
    "France uses the franc.",
]


def test_unsupported_claims_are_listed_and_scored() -> None:
    fake = FakeAnthropic()
    fake.default = scripted(
        CLAIMS,
        {
            CLAIMS[1]: ("contradicted", ["kb-2"]),
            CLAIMS[2]: ("unsupported", ["kb-404"]),
        },
    )
    judge = faithfulness(fake)
    result = asyncio.run(judge.score(CASE, ANSWER))
    assert result.value == pytest.approx(1 / 3)
    assert result.detail == (
        '1/3 claims supported; unsupported: "The Eiffel Tower opened in 1925." '
        '(contradicted by kb-2: r2); "France uses the franc." (not in the sources: r3)'
    )
    assert len(fake.requests) == 2
    extract, verify = fake.requests
    assert "<question>\nTell me about Paris.\n</question>" in content_of(extract)
    assert f"<answer>\n{ANSWER}\n</answer>" in content_of(extract)
    assert "kb-1" not in content_of(extract)
    assert '<source id="kb-2">\nThe Eiffel Tower was completed in 1889.\n</source>' in (
        content_of(verify)
    )
    assert "[2] The Eiffel Tower opened in 1925." in content_of(verify)
    assert verify["output_config"]["format"]["schema"]["required"] == ["verdicts"]
    assert result.input_tokens == 200
    assert result.cost_usd > 0


def test_claims_are_checked_in_concurrent_batches() -> None:
    claims = [f"Claim {i}." for i in range(7)]
    fake = FakeAnthropic()
    fake.default = scripted(claims, {"Claim 5.": ("unsupported", [])})
    judge = faithfulness(fake, batch_size=3)
    result = asyncio.run(judge.check(CASE, "Seven claims."))
    checks = fake.requests[1:]
    assert sorted(len(numbered(content_of(b), "claims")) for b in checks) == [1, 3, 3]
    assert [c.claim for c in result.claims] == claims
    assert [c.claim for c in result.unsupported] == ["Claim 5."]
    assert result.value == pytest.approx(6 / 7)
    assert result.claims[0].sources == ("kb-1",)


def test_claims_are_cleaned_deduplicated_and_capped() -> None:
    raw = ["  A.  ", "", "A.", *[f"C{i}." for i in range(MAX_CLAIMS + 5)]]
    fake = FakeAnthropic()
    fake.default = scripted(raw)
    result = asyncio.run(faithfulness(fake, batch_size=50).check(CASE, "x"))
    assert len(result.claims) == MAX_CLAIMS
    assert result.claims[0].claim == "A."
    assert result.detail().startswith(f"50/50 claims supported (the first 50 of {MAX_CLAIMS + 6}")


def test_an_answer_without_claims_is_faithful() -> None:
    fake = FakeAnthropic()
    fake.default = scripted([])
    result = asyncio.run(faithfulness(fake).score(CASE, "I don't know."))
    assert (result.value, result.detail) == (1.0, "the answer makes no factual claims")
    assert len(fake.requests) == 1
    empty = asyncio.run(faithfulness(FakeAnthropic()).score(CASE, "   "))
    assert empty.value == 1.0


def test_a_case_without_sources_is_unscorable_and_costs_nothing() -> None:
    result = asyncio.run(faithfulness(FakeAnthropic()).score(Case(id="x", input="q"), "A."))
    assert (result.value, result.detail) == (None, "the case has no sources under 'contexts'")


def test_judgement_labels_faithful_answers() -> None:
    fake = FakeAnthropic()
    fake.default = scripted(CLAIMS[:1])
    good = asyncio.run(faithfulness(fake).judge(CASE, "Paris is the capital."))
    fake.default = scripted(CLAIMS, {CLAIMS[2]: ("unsupported", [])})
    bad = asyncio.run(faithfulness(fake).judge(CASE, ANSWER))
    assert (good.label, bad.label) == ("faithful", "unfaithful")


@pytest.mark.parametrize(
    "verdicts",
    [
        [{"claim": 1, "reasoning": "r", "verdict": "supported", "sources": []}],
        [
            {"claim": 1, "reasoning": "r", "verdict": "supported", "sources": []},
            {"claim": 1, "reasoning": "r", "verdict": "supported", "sources": []},
        ],
        [
            {"claim": 1, "reasoning": "r", "verdict": "supported", "sources": []},
            {"claim": 3, "reasoning": "r", "verdict": "supported", "sources": []},
        ],
    ],
)
def test_verdicts_that_do_not_match_the_claims_are_not_guessed_at(
    verdicts: list[dict[str, Any]],
) -> None:
    fake = FakeAnthropic(reply({"claims": ["A.", "B."]}), reply({"verdicts": verdicts}))
    result = asyncio.run(faithfulness(fake).score(CASE, "A. B."))
    assert (result.value, result.detail) == (
        None,
        "judge did not return exactly one verdict per claim",
    )
    assert result.input_tokens == 200


@pytest.mark.parametrize(
    ("replies", "detail"),
    [
        ([json_reply(message("", stop_reason="refusal"))], "judge declined to grade this case"),
        ([json_reply(message("not json"))], "judge returned output that does not match"),
        (
            [reply({"claims": ["A."]}), json_reply(message('{"verd', stop_reason="max_tokens"))],
            "judge ran out of output tokens",
        ),
        ([error_reply(400, "invalid_request_error")], "judge call failed"),
    ],
)
def test_judge_failures_leave_the_case_unscorable(
    replies: list[httpx2.Response], detail: str
) -> None:
    result = asyncio.run(faithfulness(FakeAnthropic(*replies)).score(CASE, "A."))
    assert result.value is None
    assert result.detail.startswith(detail)


class DictCache:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.data.get(key)

    async def put(self, key: str, judge_hash: str, text: str, usage: dict[str, Any]) -> None:
        self.data[key] = text


def test_verdicts_are_cached_per_call() -> None:
    fake = FakeAnthropic()
    fake.default = scripted(CLAIMS, {CLAIMS[2]: ("unsupported", [])})
    cache = DictCache()
    first = asyncio.run(faithfulness(fake, batch_size=2, cache=cache).score(CASE, ANSWER))
    assert len(fake.requests) == 3
    assert len(cache.data) == 3
    again = asyncio.run(faithfulness(fake, batch_size=2, cache=cache).score(CASE, ANSWER))
    assert len(fake.requests) == 3
    assert (again.value, again.detail) == (first.value, first.detail)
    assert again.cost_usd == 0.0


def test_identity_names_the_mode_and_the_settings() -> None:
    fake = FakeAnthropic()
    a = faithfulness(fake).identity()
    assert a["mode"] == "faithfulness"
    assert faithfulness(fake, batch_size=4).identity()["fingerprint"] != a["fingerprint"]
    assert faithfulness(fake, contexts_key="docs").identity() == a
    cite = citations(fake).identity()
    assert cite["mode"] == "citation_accuracy"
    assert (
        citations(fake, require_citations=True).identity()["rubric_sha256"]
        != (cite["rubric_sha256"])
    )
    spec = faithfulness(fake, batch_size=4).spec()
    assert (spec["mode"], spec["batch_size"], spec["contexts_key"]) == (
        "faithfulness",
        4,
        "contexts",
    )
    assert citations(fake).spec()["require_citations"] is False


def test_prompts_for_estimates() -> None:
    judge = faithfulness(FakeAnthropic())
    extract, verify = judge.prompts(CASE, "")
    assert extract.startswith("<question>")
    assert "<claims>\n\n</claims>" in verify
    assert judge.system == VERIFY_SYSTEM
    assert judge.prompts(Case(id="x", input="q"), "") == []
    cite = citations(FakeAnthropic())
    assert cite.prompts(CASE, "") == ["<sources>\n\n</sources>\n\n<statements>\n\n</statements>"]
    assert cite.prompts(Case(id="x", input="q"), "") == []


def test_batch_size_is_bounded() -> None:
    with pytest.raises(ValueError, match="batch_size must be 1 to 50"):
        faithfulness(FakeAnthropic(), batch_size=0)


# ---------------------------------------------------------------- citation accuracy


def citing(support: dict[tuple[str, str], bool], full: set[str] | None = None) -> Handler:
    """A judge that says whether each (statement, source) pair supports, and
    which statements their sources support in full (all supported ones by default)."""

    def handler(body: dict[str, Any]) -> httpx2.Response:
        assert system_of(body) == CITATION_SYSTEM
        out = []
        for number, line in enumerate(numbered(content_of(body), "statements"), start=1):
            text, _, cites = line.rpartition(" (cites: ")
            ids = cites.rstrip(")").split(", ")
            verdicts = [{"id": i, "supports": support.get((text, i), False)} for i in ids]
            whole = text in full if full is not None else all(v["supports"] for v in verdicts)
            out.append(
                {"statement": number, "reasoning": "r", "sources": verdicts, "supported": whole}
            )
        return reply({"statements": out})

    return handler


CITED = (
    "Paris is the capital of France [1]. The tower dates from 1889 [1, 2]. "
    "France uses the euro [9]. It is in Europe."
)


def test_each_citation_is_checked_against_its_source() -> None:
    fake = FakeAnthropic()
    fake.default = citing(
        {
            ("Paris is the capital of France.", "kb-1"): True,
            ("The tower dates from 1889.", "kb-2"): True,
        }
    )
    result = asyncio.run(citations(fake).score(CASE, CITED))
    # Four citations: kb-1 right, kb-1 wrong, kb-2 right, [9] names nothing.
    assert result.value == pytest.approx(2 / 4)
    assert result.detail == (
        "2/4 citations accurate; 1/3 cited statements fully supported; "
        "1/4 statements cite nothing. Problems: "
        '"The tower dates from 1889.": kb-1 does not support it | '
        '"France uses the euro.": [9] names no source'
    )
    (body,) = fake.requests
    content = content_of(body)
    # Only cited sources are shown, and the statement citing nothing real is not sent.
    assert '<source id="kb-3">' not in content
    assert "euro" not in content.split("<statements>")[1]
    assert body["output_config"]["format"]["schema"]["required"] == ["statements"]


def test_a_statement_its_sources_support_only_in_part() -> None:
    fake = FakeAnthropic()
    fake.default = citing({("Paris is the capital and has the tower.", "kb-1"): True}, full=set())
    result = asyncio.run(citations(fake).score(CASE, "Paris is the capital and has the tower [1]."))
    assert result.value == 1.0
    assert result.detail.endswith(
        'Problems: "Paris is the capital and has the tower.": its sources support only part of it'
    )


def test_statements_are_checked_in_batches() -> None:
    answer = " ".join(f"Fact {i} [1]." for i in range(5))
    fake = FakeAnthropic()
    fake.default = citing({(f"Fact {i}.", "kb-1"): i != 3 for i in range(5)})
    result = asyncio.run(citations(fake, batch_size=2).score(CASE, answer))
    assert len(fake.requests) == 3
    assert result.value == pytest.approx(4 / 5)


def test_an_answer_citing_nothing() -> None:
    fake = FakeAnthropic()
    loose = asyncio.run(citations(fake).score(CASE, "Paris is the capital."))
    strict = asyncio.run(citations(fake, require_citations=True).score(CASE, "Paris."))
    assert (loose.value, loose.detail) == (None, "the answer cites no source")
    assert (strict.value, strict.detail) == (0.0, "the answer cites no source")
    assert fake.requests == []


def test_only_missing_sources_need_no_call() -> None:
    fake = FakeAnthropic()
    result = asyncio.run(citations(fake).score(CASE, "Made up [8]."))
    assert result.value == 0.0
    assert fake.requests == []
    no_sources = asyncio.run(citations(fake).score(Case(id="x", input="q"), "A [1]."))
    assert no_sources.value is None


def test_citation_judgement_labels() -> None:
    fake = FakeAnthropic()
    fake.default = citing({("Paris is the capital.", "kb-1"): True})
    good = asyncio.run(citations(fake).judge(CASE, "Paris is the capital [1]."))
    bad = asyncio.run(citations(fake).judge(CASE, "Paris is the capital [2]."))
    assert (good.label, bad.label) == ("accurate", "inaccurate")
    assert (good.value, bad.value) == (1.0, 0.0)


@pytest.mark.parametrize(
    ("statements", "detail"),
    [
        ([], "judge did not return exactly one verdict per statement"),
        (
            [{"statement": 1, "reasoning": "r", "sources": [], "supported": True}],
            "judge did not return a verdict for every cited source",
        ),
        (
            [
                {
                    "statement": 2,
                    "reasoning": "r",
                    "supported": True,
                    "sources": [{"id": "kb-1", "supports": True}],
                },
            ],
            "judge did not return exactly one verdict per statement",
        ),
    ],
)
def test_citation_verdicts_must_cover_every_citation(
    statements: list[dict[str, Any]], detail: str
) -> None:
    fake = FakeAnthropic(reply({"statements": statements}))
    result = asyncio.run(citations(fake).score(CASE, "Paris [1]."))
    assert (result.value, result.detail) == (None, detail)


def test_citation_judge_failures_are_unscorable() -> None:
    fake = FakeAnthropic(json_reply(message("", stop_reason="refusal")))
    result = asyncio.run(citations(fake).score(CASE, "Paris [1]."))
    assert (result.value, result.detail) == (None, "judge declined to grade this case")


# ---------------------------------------------------------------- specs and runs


def specs(*raw: dict[str, Any]) -> list[ScorerSpec]:
    return TypeAdapter(list[ScorerSpec]).validate_python(list(raw))


def test_specs_build_the_judges() -> None:
    fake = FakeAnthropic()
    built = build_scorers(
        specs(
            {"type": "faithfulness", "batch_size": 5, "contexts_key": "docs", "effort": "low"},
            {"type": "citation_accuracy", "require_citations": True},
        ),
        allow_import_paths=False,
        claude=fake.client(),
    )
    faith, cite = built
    assert isinstance(faith, FaithfulnessJudge)
    assert (faith.name, faith.batch_size, faith.contexts_key, faith.effort) == (
        "faithfulness",
        5,
        "docs",
        "low",
    )
    assert isinstance(cite, CitationAccuracyJudge)
    assert (cite.name, cite.require_citations) == ("citation_accuracy", True)
    assert all(isinstance(s, LLMJudge) for s in built)


@pytest.mark.parametrize(
    "raw",
    [
        {"type": "faithfulness", "batch_size": 0},
        {"type": "faithfulness", "batch_size": 51},
        {"type": "citation_accuracy", "contexts_key": ""},
        {"type": "faithfulness", "rubric": "not taken"},
    ],
)
def test_bad_specs_are_refused(raw: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        specs(raw)


def test_an_unknown_provider_is_refused() -> None:
    with pytest.raises(InvalidInputError, match="scorer faithfulness"):
        build_scorers(
            specs({"type": "faithfulness", "provider": "nowhere"}),
            allow_import_paths=False,
            claude=FakeAnthropic().client(),
        )


def test_a_run_scores_faithfulness_and_citations(sessions: sessionmaker[Session]) -> None:
    answers = {
        "good": "Paris is the capital of France [1].",
        "bad": "Paris is the capital of Spain [1].",
    }
    with sessions() as s:
        import_dataset(
            s,
            "rag",
            [
                Case(id=k, input={"question": "Where?", "contexts": SOURCES}, expected=v)
                for k, v in answers.items()
            ],
        )

    def handler(body: dict[str, Any]) -> httpx2.Response:
        content = content_of(body)
        good = "France" in content.split("<answer>" if "<answer>" in content else "</sources>")[1]
        if system_of(body) == EXTRACT_SYSTEM:
            return reply(
                {"claims": ["Paris is the capital of " + ("France." if good else "Spain.")]}
            )
        if system_of(body) == VERIFY_SYSTEM:
            verdict = "supported" if good else "contradicted"
            return reply(
                {
                    "verdicts": [
                        {"claim": 1, "reasoning": "r", "verdict": verdict, "sources": ["kb-1"]}
                    ]
                }
            )
        return reply(
            {
                "statements": [
                    {
                        "statement": 1,
                        "reasoning": "r",
                        "sources": [{"id": "kb-1", "supports": good}],
                        "supported": good,
                    }
                ]
            }
        )

    fake = FakeAnthropic()
    fake.default = handler
    runner = Runner(
        sessions,
        Settings(),
        providers=ProviderRegistry({}, PriceBook(), claude=fake.client()),
    )
    config = RunConfig.model_validate(
        {
            "label": "rag",
            "dataset": {"name": "rag"},
            "target": {"type": "fake", "accuracy": 1.0},
            "scorers": [{"type": "faithfulness"}, {"type": "citation_accuracy"}],
        }
    )
    run = runner.create_run(config)
    report = asyncio.run(runner.execute(run.id))
    assert (report.status, report.errors) == ("completed", 0)
    with sessions() as s:
        rows = {r.case_id: r for r in s.scalars(select(Result).where(Result.run_id == run.id))}
    assert {k: r.scores["faithfulness"]["value"] for k, r in rows.items()} == {
        "good": 1.0,
        "bad": 0.0,
    }
    assert "Spain" in rows["bad"].scores["faithfulness"]["detail"]
    assert {k: r.scores["citation_accuracy"]["value"] for k, r in rows.items()} == {
        "good": 1.0,
        "bad": 0.0,
    }
    assert rows["good"].judge_cost_usd > 0
    assert len(fake.requests) == 6
    assert run.judges is not None
    assert run.judges["faithfulness"]["mode"] == "faithfulness"
    assert run.judges["citation_accuracy"]["mode"] == "citation_accuracy"
