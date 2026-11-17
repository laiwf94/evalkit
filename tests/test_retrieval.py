"""Retrieval: the case schema, the retrieval target and its cache, ranking
metrics against hand-computed values, and a run that gates a retriever change."""

import asyncio
import json
import math
from typing import Any

import httpx2
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import SecretStr, TypeAdapter, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from evalkit.config import Settings
from evalkit.datasets import Case, import_dataset
from evalkit.db.models import Result
from evalkit.errors import InvalidInputError
from evalkit.pricing import PriceBook
from evalkit.providers.registry import ProviderRegistry
from evalkit.redaction import FAILED_PLACEHOLDER, PatternSpec, RedactionConfig, Redactor
from evalkit.report import compare_runs
from evalkit.runner import Runner
from evalkit.runspec import RunConfig
from evalkit.scorers.ranking import (
    METRICS,
    RankingMetric,
    average_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evalkit.scorers.registry import RankingSpec, ScorerSpec, build_scorers
from evalkit.targets import RetrievalTargetSpec, TargetError, TargetSpec, Usage
from evalkit.targets.factory import build_target, identity
from evalkit.targets.http import HttpEndpoint
from evalkit.targets.retrieval import (
    Document,
    Ranking,
    RetrievalTarget,
    parse_labels,
    parse_ranking,
    query_of,
)

ENDPOINT = HttpEndpoint(
    url="https://search.example.com/v1/retrieve",
    secret_headers={"Authorization": SecretStr("Bearer s3cret")},
)
PUBLIC = "93.184.216.34"

# The worked example: five documents returned, four relevant ones labelled.
RANKED = ["d3", "d1", "d7", "d2", "d9"]
LABELS = {"d1": 3, "d2": 2, "d4": 1, "d7": 0, "d8": 1}


async def public(host: str, port: int) -> list[str]:
    return [PUBLIC]


async def private(host: str, port: int) -> list[str]:
    return ["10.0.0.5"]


def spec(**fields: Any) -> RetrievalTargetSpec:
    return RetrievalTargetSpec(type="retrieval", endpoint="search", **fields)


class Index:
    """A fake retriever: each query maps to a ranked list, served as the endpoint."""

    def __init__(self, answer: Any = None, delay: float = 0.0) -> None:
        self.answer = answer or (lambda body: {"results": [{"id": "d1", "score": 0.9}]})
        self.delay = delay
        self.requests: list[httpx2.Request] = []
        self.fail: int | None = None

    async def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail is not None:
            return httpx2.Response(self.fail)
        return httpx2.Response(200, json=self.answer(json.loads(request.content)))


def target(index: Index, resolver: Any = public, **fields: Any) -> RetrievalTarget:
    return RetrievalTarget(
        spec(**fields), ENDPOINT, resolver=resolver, transport=httpx2.MockTransport(index)
    )


class MemoryStore:
    def __init__(self) -> None:
        self.entries: dict[str, Any] = {}

    async def get(self, key: str) -> Any:
        return self.entries.get(key)

    async def put(
        self, key: str, owner_hash: str, text: str, stop_reason: str | None, usage: Usage
    ) -> None:
        self.entries[key] = type("Hit", (), {"output": text, "stop_reason": stop_reason})()


def generate(t: RetrievalTarget, case: Case) -> Any:
    async def main() -> Any:
        try:
            return await t.generate(case)
        finally:
            await t.aclose()

    return asyncio.run(main())


# The case schema


def test_a_query_is_a_string_or_an_object_with_filters() -> None:
    assert query_of(Case(id="q", input="reset password")).query == "reset password"
    query = query_of(Case(id="q", input={"query": "reset", "filters": {"lang": "en"}}))
    assert (query.query, query.filters) == ("reset", {"lang": "en"})
    with pytest.raises(InvalidInputError, match=r"case 'q' is not a retrieval query: query"):
        query_of(Case(id="q", input={"question": "reset"}))


def test_labels_are_graded_and_a_list_of_ids_means_grade_one() -> None:
    assert parse_labels({"a": 3, "b": 0}) == {"a": 3, "b": 0}
    assert parse_labels(["a", "b"]) == {"a": 1, "b": 1}
    for bad, message in [
        ("a", "must be an object"),
        ([1, 2], "must be an object"),
        ({"a": 11}, "integer from 0 to 10"),
        ({"a": -1}, "integer from 0 to 10"),
        ({"a": 1.5}, "integer from 0 to 10"),
        ({"a": True}, "integer from 0 to 10"),
        ({"": 1}, "1 to 512 characters"),
        ({f"d{i}": 1 for i in range(10_001)}, "more than 10000"),
    ]:
        with pytest.raises(ValueError, match=message):
            parse_labels(bad)


# Ranking metrics, by hand on the worked example


def test_recall_precision_and_hit_rate_at_k() -> None:
    assert recall_at_k(RANKED, LABELS, 3) == 1 / 4
    assert recall_at_k(RANKED, LABELS, 5) == 2 / 4
    assert precision_at_k(RANKED, LABELS, 3) == 1 / 3
    assert precision_at_k(RANKED, LABELS, 5) == 2 / 5
    # A short list does not raise precision: k stays the denominator.
    assert precision_at_k(RANKED, LABELS, 10) == 2 / 10
    assert hit_rate_at_k(RANKED, LABELS, 1) == 0.0
    assert hit_rate_at_k(RANKED, LABELS, 2) == 1.0


def test_mrr_and_average_precision() -> None:
    assert reciprocal_rank(RANKED, LABELS) == 1 / 2
    assert reciprocal_rank(RANKED, LABELS, k=1) == 0.0
    assert reciprocal_rank(["x", "y"], LABELS) == 0.0
    # Relevant at ranks 2 and 4: (1/2 + 2/4) / min(4 relevant, k 5).
    assert average_precision_at_k(RANKED, LABELS, 5) == pytest.approx(0.25)
    assert average_precision_at_k(["d1", "d2"], {"d1": 1, "d2": 1}, 5) == 1.0


def test_ndcg_with_linear_and_exponential_gain() -> None:
    log2 = math.log2
    dcg = 3 / log2(3) + 2 / log2(5)
    idcg = 3 + 2 / log2(3) + 1 / 2 + 1 / log2(5)
    assert ndcg_at_k(RANKED, LABELS, 5) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(RANKED, LABELS, 5) == pytest.approx(0.5304040920)
    assert ndcg_at_k(RANKED, LABELS, 5, gain="exponential") == pytest.approx(0.5811124154)
    # With min_grade 2 the grade-1 documents gain nothing, in the ranking or the ideal.
    assert ndcg_at_k(RANKED, LABELS, 5, min_grade=2) == pytest.approx(
        (3 / log2(3) + 2 / log2(5)) / (3 + 2 / log2(3))
    )
    assert ndcg_at_k(["d1", "d2", "d4", "d8"], LABELS, 4) == pytest.approx(1.0)


def test_a_case_without_a_relevant_label_cannot_be_scored() -> None:
    labels = {"d1": 0}
    for function in (recall_at_k, precision_at_k, hit_rate_at_k, average_precision_at_k):
        with pytest.raises(ValueError, match="no relevant document"):
            function(RANKED, labels, 5)
    with pytest.raises(ValueError, match="no relevant document"):
        reciprocal_rank(RANKED, labels)
    with pytest.raises(ValueError, match="no relevant document"):
        ndcg_at_k(RANKED, labels, 5)


doc_ids = st.text(alphabet="abcdefgh", min_size=1, max_size=2)


def test_a_repeated_document_counts_once_at_its_first_rank() -> None:
    labels = {"d1": 3, "d2": 1}
    repeated = ["d1", "d1", "x", "d2", "d1"]
    distinct = ["d1", "x", "d2"]
    assert precision_at_k(repeated, labels, 2) == 1 / 2
    assert precision_at_k(repeated, labels, 3) == 2 / 3
    assert recall_at_k(repeated, labels, 2) == 1 / 2
    assert average_precision_at_k(repeated, labels, 3) == pytest.approx((1 + 2 / 3) / 2)
    assert ndcg_at_k(["d1", "d1"], labels, 2) < 1.0
    assert reciprocal_rank(["x", "x", "d2"], labels) == 1 / 2
    for metric in METRICS:
        scorer = RankingMetric("m", metric=metric, k=3)
        assert scorer.compute(repeated, labels) == scorer.compute(distinct, labels)


@given(
    ranked=st.lists(doc_ids, max_size=12),
    labels=st.dictionaries(doc_ids, st.integers(0, 4), min_size=1, max_size=12),
    k=st.integers(1, 15),
)
def test_a_ranking_with_repeats_scores_as_its_distinct_documents_do(
    ranked: list[str], labels: dict[str, int], k: int
) -> None:
    if not any(g >= 1 for g in labels.values()):
        return
    distinct = list(dict.fromkeys(ranked))
    for metric in METRICS:
        scorer = RankingMetric("m", metric=metric, k=k)
        value = scorer.compute(ranked, labels)
        assert 0.0 <= value <= 1.0 + 1e-12
        assert value == scorer.compute(distinct, labels)


@given(
    ranked=st.lists(doc_ids, unique=True, max_size=12),
    labels=st.dictionaries(doc_ids, st.integers(0, 4), min_size=1, max_size=12),
    k=st.integers(1, 15),
)
def test_every_metric_is_a_fraction_and_the_ideal_ranking_scores_one(
    ranked: list[str], labels: dict[str, int], k: int
) -> None:
    if not any(g >= 1 for g in labels.values()):
        return
    for metric in METRICS:
        value = RankingMetric("m", metric=metric, k=k).compute(ranked, labels)
        assert 0.0 <= value <= 1.0 + 1e-12
    ideal = sorted((d for d in labels if labels[d] >= 1), key=lambda d: -labels[d])
    assert ndcg_at_k(ideal, labels, k) == pytest.approx(1.0)
    assert recall_at_k(ideal, labels, len(ideal)) == 1.0
    assert reciprocal_rank(ideal, labels) == 1.0


# The scorer


def output(*ids: str) -> str:
    return Ranking(documents=tuple(Document(id=i) for i in ids)).to_output()


def score(scorer: RankingMetric, case: Case, out: str) -> Any:
    return asyncio.run(scorer.score(case, out))


def test_the_scorer_reads_the_ranking_and_the_case_labels() -> None:
    case = Case(id="q", input="x", expected={"relevance": LABELS})
    result = score(RankingMetric("r", metric="recall", k=5), case, output(*RANKED))
    assert result.value == 0.5
    assert result.detail == "2 of 4 relevant in the top 5; first at rank 2"
    mrr = score(RankingMetric("m", metric="mrr"), case, output(*RANKED))
    assert (mrr.value, mrr.detail) == (0.5, "2 of 4 relevant in all 5; first at rank 2")
    miss = score(RankingMetric("n", metric="ndcg", k=1), case, output("d3"))
    assert (miss.value, miss.detail) == (0.0, "0 of 4 relevant in the top 1")
    bare = Case(id="q", input="x", expected=["d1"])
    top = score(RankingMetric("p", metric="precision", k=2, expected_key=None), bare, output("d1"))
    assert top.value == 0.5


def test_the_scorer_reports_what_it_cannot_score() -> None:
    scorer = RankingMetric("r", metric="ndcg")
    cases = {
        "no relevance labels (expected.relevance)": Case(id="q", input="x"),
        "must be an object": Case(id="q", input="x", expected={"relevance": "d1"}),
        "no relevant document": Case(id="q", input="x", expected={"relevance": {"d1": 0}}),
    }
    for message, case in cases.items():
        result = score(scorer, case, output("d1"))
        assert result.value is None
        assert message in result.detail
    labelled = Case(id="q", input="x", expected={"relevance": {"d1": 1}})
    assert "not a ranking" in score(scorer, labelled, "d1, d2").detail
    unkeyed = RankingMetric("r", metric="ndcg", expected_key=None)
    assert "(expected)" in score(unkeyed, Case(id="q", input="x"), output("d1")).detail


def test_scorer_specs_name_themselves_and_check_their_options() -> None:
    adapter: TypeAdapter[ScorerSpec] = TypeAdapter(ScorerSpec)
    names = [
        adapter.validate_python({"type": "ranking", **fields}).resolved_name()
        for fields in (
            {"metric": "recall"},
            {"metric": "ndcg", "k": 5},
            {"metric": "mrr"},
            {"metric": "mrr", "k": 3},
            {"metric": "map", "name": "ap"},
        )
    ]
    assert names == ["recall_at_10", "ndcg_at_5", "mrr", "mrr_at_3", "ap"]
    specs = [
        RankingSpec(type="ranking", metric="ndcg", k=5, gain="exponential", min_grade=2),
        RankingSpec(type="ranking", metric="recall"),
    ]
    built = build_scorers(specs, allow_import_paths=False)
    first = built[0]
    assert isinstance(first, RankingMetric)
    assert (first.name, first.k, first.gain, first.min_grade) == ("ndcg_at_5", 5, "exponential", 2)
    assert built[1].name == "recall_at_10"
    for bad in ({"metric": "f1"}, {"metric": "ndcg", "k": 0}, {"metric": "ndcg", "min_grade": 0}):
        with pytest.raises(ValidationError):
            adapter.validate_python({"type": "ranking", **bad})
    for metric, k, min_grade in (("f1", None, 1), ("ndcg", 0, 1), ("ndcg", 5, 0)):
        with pytest.raises(ValueError, match=r"unknown ranking metric|must be at least 1"):
            RankingMetric("x", metric=metric, k=k, min_grade=min_grade)  # type: ignore[arg-type]


# The retrieval target


def test_a_query_is_posted_to_the_pinned_endpoint_and_the_top_k_kept() -> None:
    index = Index(
        lambda body: {
            "data": {
                "hits": [
                    {"doc": "a", "relevance": 0.9, "body": "alpha"},
                    "b",
                    {"doc": "a", "relevance": 0.5},
                    {"doc": 7},
                    {"doc": "d"},
                ]
            }
        }
    )
    t = target(
        index,
        k=3,
        results_field="data.hits",
        id_field="doc",
        score_field="relevance",
        text_field="body",
    )
    generation = generate(t, Case(id="q", input={"query": "alpha", "filters": {"lang": "en"}}))
    (request,) = index.requests
    assert json.loads(request.content) == {"query": "alpha", "k": 3, "filters": {"lang": "en"}}
    assert request.url.host == PUBLIC
    assert request.headers["host"] == "search.example.com"
    assert request.headers["authorization"] == "Bearer s3cret"
    ranking = parse_ranking(generation.text)
    # The repeated "a" keeps its first rank; an integer id is read as text; k cuts at 3.
    assert ranking.ids == ["a", "b", "7"]
    assert ranking.documents[0] == Document(id="a", score=0.9, text="alpha")
    assert (generation.cached, generation.usage) == (False, Usage())


def test_long_texts_are_cut_and_scores_may_be_integers() -> None:
    index = Index(lambda body: {"results": [{"id": "a", "score": 2, "text": "x" * 100_001}]})
    (doc,) = parse_ranking(generate(target(index), Case(id="q", input="x")).text).documents
    assert (doc.score, len(doc.text or "")) == (2.0, 100_000)


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        ({"hits": []}, "no field 'results'"),
        ({"results": {"id": "a"}}, "must be a list"),
        ({"results": [3.5]}, r"results\[0\] must be an id or an object"),
        ({"results": [{"id": ""}]}, r"results\[0\].id must be an id"),
        ({"results": [{"id": True}]}, r"results\[0\].id must be an id"),
        ({"results": [{"id": "a", "score": "high"}]}, "score must be a number"),
        ({"results": [{"id": "a", "score": None}, {"id": "b", "score": True}]}, "score must be"),
        ({"results": [{"id": "a", "text": 3}]}, "text must be a string"),
    ],
)
def test_a_malformed_answer_fails_the_case_without_a_retry(answer: Any, message: str) -> None:
    with pytest.raises(TargetError, match=message) as caught:
        generate(target(Index(lambda body: answer)), Case(id="q", input="x"))
    assert not caught.value.retryable


def test_the_endpoint_gets_the_http_target_protections() -> None:
    with pytest.raises(TargetError, match="non-public address") as caught:
        generate(target(Index(), resolver=private), Case(id="q", input="x"))
    assert not caught.value.retryable
    index = Index()
    index.fail = 503
    with pytest.raises(TargetError, match="returned 503") as failed:
        generate(target(index), Case(id="q", input="x"))
    assert failed.value.retryable


def test_retrieved_texts_are_redacted_before_they_are_cached() -> None:
    index = Index(lambda body: {"results": [{"id": "a", "text": "mail ann@example.com now"}]})
    t = target(index)
    store = MemoryStore()
    t.attach(store=store, redactor=Redactor())
    text = generate(t, Case(id="q", input="x")).text
    assert "ann@example.com" not in text
    (entry,) = store.entries.values()
    assert "ann@example.com" not in entry.output
    assert parse_ranking(entry.output) == parse_ranking(text)


def test_a_text_that_cannot_be_redacted_is_replaced_and_not_cached() -> None:
    slow = RedactionConfig(
        builtins=(), patterns=(PatternSpec(name="BAD", pattern=r"(a|aa)+$"),), timeout_seconds=0.05
    )
    index = Index(lambda body: {"results": [{"id": "a", "text": "a" * 40 + "!"}, {"id": "b"}]})
    t = target(index)
    store = MemoryStore()
    t.attach(store=store, redactor=Redactor(slow))
    ranking = parse_ranking(generate(t, Case(id="q", input="x")).text)
    assert ranking.documents[0].text == FAILED_PLACEHOLDER
    assert ranking.ids == ["a", "b"]
    assert store.entries == {}


def test_a_cached_ranking_costs_no_request_and_another_k_is_another_entry() -> None:
    store = MemoryStore()
    index = Index()

    async def main() -> list[bool]:
        seen = []
        for k in (10, 10, 5):
            t = target(index, k=k)
            t.attach(store=store, redactor=Redactor())
            seen.append((await t.generate(Case(id="q", input="x"))).cached)
            await t.aclose()
        return seen

    assert asyncio.run(main()) == [False, True, False]
    assert len(index.requests) == 2
    assert len(store.entries) == 2


def test_an_unreadable_cache_entry_is_asked_for_again() -> None:
    store = MemoryStore()
    t = target(Index())
    t.attach(store=store, redactor=Redactor())
    generate(t, Case(id="q", input="x"))
    (key,) = store.entries
    asyncio.run(store.put(key, "", "not json", None, Usage()))
    index = Index()
    again = target(index)
    again.attach(store=store, redactor=Redactor())
    assert not generate(again, Case(id="q", input="x")).cached
    assert len(index.requests) == 1


def test_identical_queries_at_the_same_time_make_one_request() -> None:
    index = Index(delay=0.05)
    t = target(index)

    async def main() -> list[Any]:
        try:
            cases = [Case(id=f"q{i}", input="same") for i in range(5)]
            return list(await asyncio.gather(*(t.generate(c) for c in cases)))
        finally:
            await t.aclose()

    generations = asyncio.run(main())
    assert len(index.requests) == 1
    assert sorted(g.cached for g in generations) == [False, True, True, True, True]
    assert len({g.text for g in generations}) == 1


def test_a_failed_shared_request_fails_every_waiter() -> None:
    index = Index(delay=0.05)
    index.fail = 500
    t = target(index)

    async def main() -> list[Any]:
        try:
            cases = [Case(id=f"q{i}", input="same") for i in range(3)]
            return list(
                await asyncio.gather(*(t.generate(c) for c in cases), return_exceptions=True)
            )
        finally:
            await t.aclose()

    outcomes = asyncio.run(main())
    assert len(index.requests) == 1
    assert all(isinstance(o, TargetError) and o.retryable for o in outcomes)


def test_a_waiter_sends_its_own_request_when_the_first_is_cancelled() -> None:
    index = Index(delay=0.05)
    t = target(index)

    async def main() -> Any:
        try:
            first = asyncio.create_task(t.generate(Case(id="a", input="same")))
            await asyncio.sleep(0.01)
            second = asyncio.create_task(t.generate(Case(id="b", input="same")))
            await asyncio.sleep(0.01)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            return await second
        finally:
            await t.aclose()

    generation = asyncio.run(main())
    assert not generation.cached
    assert len(index.requests) == 2


def test_a_cancelled_waiter_leaves_the_shared_request_running() -> None:
    index = Index(delay=0.05)
    t = target(index)

    async def main() -> Any:
        try:
            first = asyncio.create_task(t.generate(Case(id="a", input="same")))
            await asyncio.sleep(0.01)
            second = asyncio.create_task(t.generate(Case(id="b", input="same")))
            await asyncio.sleep(0.01)
            second.cancel()
            with pytest.raises(asyncio.CancelledError):
                await second
            return await first
        finally:
            await t.aclose()

    assert not asyncio.run(main()).cached
    assert len(index.requests) == 1


def test_the_target_identity_covers_the_endpoint_and_what_is_read() -> None:
    adapter: TypeAdapter[TargetSpec] = TypeAdapter(TargetSpec)
    parsed = adapter.validate_python({"type": "retrieval", "endpoint": "search", "k": 5})
    assert isinstance(parsed, RetrievalTargetSpec)
    endpoints = {"search": ENDPOINT}
    moved = {"search": HttpEndpoint(url="https://other.example.com/retrieve")}
    base = identity(parsed, endpoints)
    assert identity(parsed, moved) != base
    assert identity(spec(k=5, text_field="body"), endpoints) != base
    assert identity(spec(k=5), endpoints) == base
    built = build_target(parsed, endpoints=endpoints, providers=ProviderRegistry({}, PriceBook()))
    assert isinstance(built, RetrievalTarget)
    assert built.provider == "http:search"
    asyncio.run(built.aclose())
    with pytest.raises(InvalidInputError, match="unknown HTTP endpoint"):
        identity(parsed, {})
    for bad in ({"k": 0}, {"k": 1001}, {"results_field": "a..b"}, {"id_field": "a.b"}):
        with pytest.raises(ValidationError):
            adapter.validate_python({"type": "retrieval", "endpoint": "search", **bad})


# A whole run, and a gate on a retriever change

DOCS = [f"kb-{i:03d}" for i in range(400)]


def corpus_case(i: int) -> Case:
    # Each query has two highly relevant documents and one partly relevant one.
    labels = {DOCS[3 * i]: 3, DOCS[3 * i + 1]: 2, DOCS[3 * i + 2]: 1}
    return Case(id=f"q{i:03d}", input=f"question {i}", expected={"relevance": labels})


def retriever(regressed: bool) -> Any:
    def answer(body: dict[str, Any]) -> dict[str, Any]:
        i = int(body["query"].split()[1])
        good = [DOCS[3 * i], DOCS[3 * i + 1], DOCS[3 * i + 2]]
        noise = [d for d in (DOCS[(7 * i + j) % 400] for j in range(1, 8)) if d not in good]
        # The regressed retriever loses the best document for two queries in three.
        if regressed and i % 3:
            good = good[1:]
        return {"results": [{"id": d, "score": 1.0 - n / 20} for n, d in enumerate(good + noise)]}

    return answer


class Factory:
    def __init__(self) -> None:
        self.indexes = {"search": Index(retriever(False)), "search-v2": Index(retriever(True))}

    def __call__(self, s: Any) -> RetrievalTarget:
        index = self.indexes[s.endpoint]
        return RetrievalTarget(s, ENDPOINT, resolver=public, transport=httpx2.MockTransport(index))


SETTINGS = Settings.model_validate(
    {
        **Settings.from_env({}).model_dump(),
        "http_endpoints": {"search": ENDPOINT, "search-v2": ENDPOINT},
    }
)


def run_config(endpoint: str = "search", **options: Any) -> RunConfig:
    return RunConfig.model_validate(
        {
            "label": endpoint,
            "dataset": {"name": "kb"},
            "target": {"type": "retrieval", "endpoint": endpoint, "k": 5},
            "scorers": [
                {"type": "ranking", "metric": "ndcg", "k": 5},
                {"type": "ranking", "metric": "recall", "k": 5},
                {"type": "ranking", "metric": "mrr"},
            ],
            "options": {"concurrency": 8, **options},
        }
    )


@pytest.fixture
def dataset(sessions: sessionmaker[Session]) -> str:
    with sessions() as s:
        import_dataset(s, "kb", [corpus_case(i) for i in range(60)])
        import_dataset(s, "chat", [Case(id="c1", input={"messages": []})])
    return "kb"


def scores(sessions: sessionmaker[Session], run_id: str, metric: str) -> list[float]:
    with sessions() as s:
        rows = s.scalars(select(Result).where(Result.run_id == run_id).order_by(Result.case_id))
        return [r.scores[metric]["value"] for r in rows]


def test_a_retriever_regression_fails_the_gate_and_reruns_are_free(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    factory = Factory()
    runner = Runner(sessions, SETTINGS, target_factory=factory)
    base = runner.create_run(run_config())
    report = asyncio.run(runner.execute(base.id))
    assert (report.status, report.errors) == ("completed", 0)
    assert scores(sessions, base.id, "ndcg_at_5") == [pytest.approx(1.0)] * 60
    assert set(scores(sessions, base.id, "mrr")) == {1.0}
    assert len(factory.indexes["search"].requests) == 60

    candidate = runner.create_run(run_config("search-v2"))
    asyncio.run(runner.execute(candidate.id))
    recall = scores(sessions, candidate.id, "recall_at_5")
    assert sorted(recall) == [pytest.approx(2 / 3)] * 40 + [1.0] * 20
    with sessions() as s:
        gate = compare_runs(s, base.id, candidate.id)
    # A relevant document still comes first, so MRR holds while recall and nDCG drop.
    assert sorted(gate.regressions) == ["ndcg_at_5", "recall_at_5"]

    rerun = runner.create_run(run_config())
    asyncio.run(runner.execute(rerun.id))
    assert len(factory.indexes["search"].requests) == 60
    with sessions() as s:
        rows = list(s.scalars(select(Result).where(Result.run_id == rerun.id)))
    assert len(rows) == 60
    assert all(r.cached for r in rows)
    with sessions() as s:
        assert compare_runs(s, base.id, rerun.id).regressions == []
    uncached = runner.create_run(run_config(use_cache=False))
    asyncio.run(runner.execute(uncached.id))
    assert len(factory.indexes["search"].requests) == 120


def test_a_run_refuses_cases_that_are_not_queries(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    runner = Runner(sessions, SETTINGS, target_factory=Factory())
    config = run_config().model_copy(
        update={"dataset": run_config().dataset.model_copy(update={"name": "chat"})}
    )
    with pytest.raises(InvalidInputError, match="not a retrieval query"):
        runner.create_run(config)
