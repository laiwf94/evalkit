"""Pipeline targets: retrieval then generation as one target, both stages
stored per case, ranking metrics on the retrieval stage and grounding judges on
the documents the answer was written from, the stage caches, and a gate that
catches a retriever regression through the whole pipeline."""

import asyncio
import json
from dataclasses import replace
from typing import Any

import httpx2
import pytest
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from evalkit.config import Settings
from evalkit.datasets import Case, import_dataset
from evalkit.db.models import Result, ResultStage
from evalkit.errors import InvalidInputError
from evalkit.pricing import PriceBook, PriceTable
from evalkit.providers.registry import ProviderRegistry
from evalkit.redaction import FAILED_PLACEHOLDER, Redactor
from evalkit.report import compare_runs
from evalkit.runner import Runner
from evalkit.runspec import RunConfig
from evalkit.scorers.faithfulness import EXTRACT_SYSTEM, VERIFY_SYSTEM
from evalkit.targets import PipelineTargetSpec, RetrievalTargetSpec, Stage, TargetSpec
from evalkit.targets.factory import build_target, identity
from evalkit.targets.http import HttpEndpoint
from evalkit.targets.pipeline import (
    NO_DOCUMENTS,
    PipelineTarget,
    render_contexts,
    with_contexts,
)
from evalkit.targets.retrieval import Document, Ranking, RetrievalTarget
from tests.fake_anthropic import FakeAnthropic, error_reply, json_reply, message
from tests.test_api import AUTH, serving
from tests.test_retrieval import ENDPOINT, Index, MemoryStore, public

ANSWER_SYSTEM = "Answer the question from the numbered sources and cite them."
TEMPLATE = "Sources:\n{{ contexts }}\n\nQuestion: {{ query }}"
ADAPTER: TypeAdapter[Any] = TypeAdapter(TargetSpec)


def pipeline_spec(**fields: Any) -> PipelineTargetSpec:
    raw: dict[str, Any] = {
        "type": "pipeline",
        "retrieval": {"endpoint": "search", "k": 3},
        "generation": {"system": ANSWER_SYSTEM, "template": TEMPLATE},
        **fields,
    }
    return PipelineTargetSpec.model_validate(raw)


def first_source(prompt: str) -> str:
    """The text of source [1] in a generation prompt."""
    block = prompt.split("[1] ", 1)[1]
    return block.split("\n", 1)[1].split("\n\n", 1)[0]


def generator(body: dict[str, Any]) -> httpx2.Response:
    """A model that answers with the first source it was given, citing it."""
    prompt = str(body["messages"][0]["content"])
    if NO_DOCUMENTS in prompt:
        return json_reply(message("I could not find it."))
    return json_reply(message(f"{first_source(prompt)} [1]"))


def judge(body: dict[str, Any]) -> httpx2.Response:
    """A judge that takes the answer (without its citation) as its one claim and
    calls it supported when it is the text of a source it was shown."""
    system = str(body["system"][0]["text"])
    content = str(body["messages"][0]["content"])
    if system == EXTRACT_SYSTEM:
        answer = content.split("<answer>\n", 1)[1].split("\n</answer>", 1)[0]
        return json_reply(message(json.dumps({"claims": [answer.removesuffix(" [1]")]})))
    assert system == VERIFY_SYSTEM
    claims = content.split("<claims>\n", 1)[1].split("\n</claims>", 1)[0].splitlines()
    verdicts = []
    for n, line in enumerate(claims, start=1):
        claim = line.split("] ", 1)[1]
        ok = f">\n{claim}\n</source>" in content
        verdict = "supported" if ok else "unsupported"
        verdicts.append({"claim": n, "reasoning": "r", "verdict": verdict, "sources": []})
    return json_reply(message(json.dumps({"verdicts": verdicts})))


def model() -> FakeAnthropic:
    fake = FakeAnthropic()

    def route(body: dict[str, Any]) -> httpx2.Response:
        if str(body["system"][0]["text"]) == ANSWER_SYSTEM:
            return generator(body)
        return judge(body)

    fake.default = route
    return fake


def documents(body: dict[str, Any]) -> dict[str, Any]:
    q = body["query"]
    return {"results": [{"id": f"{q}-{n}", "text": f"fact {n} about {q}"} for n in range(5)]}


def target(
    index: Index, fake: FakeAnthropic, spec: PipelineTargetSpec | None = None
) -> PipelineTarget:
    return PipelineTarget(
        spec or pipeline_spec(),
        ENDPOINT,
        fake.client(),
        resolver=public,
        transport=httpx2.MockTransport(index),
    )


def generate(t: PipelineTarget, case: Case) -> Any:
    async def main() -> Any:
        try:
            return await t.generate(case)
        finally:
            await t.aclose()

    return asyncio.run(main())


# The spec


def test_a_pipeline_spec_parses_from_a_run_config() -> None:
    spec = ADAPTER.validate_python(
        {
            "type": "pipeline",
            "retrieval": {"endpoint": "search", "k": 8},
            "generation": {"template": "{{ contexts }}\n{{ query }} {{ filters.lang }}"},
            "contexts": 3,
        }
    )
    assert isinstance(spec, PipelineTargetSpec)
    assert (spec.retrieval.k, spec.context_count, spec.generation.model) == (
        8,
        3,
        "claude-opus-5-5",
    )
    assert pipeline_spec().context_count == 3


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"generation": {"template": "{{ query }}"}}, "must give the model"),
        ({"generation": {"template": "{{ contexts }} {{ question }}"}}, "not question"),
        ({"generation": {"template": "{{ contexts }} {{ input }}"}}, "not input"),
        ({"generation": {"template": "{{ contexts } }"}}, "not a valid"),
        ({"contexts": 4}, r"contexts \(4\) is more than the retriever returns \(k = 3\)"),
        ({"retrieval": {"type": "retrieval", "endpoint": "search"}}, "Extra inputs"),
        ({"generation": {"template": TEMPLATE, "temperature": 0}}, "Extra inputs"),
    ],
)
def test_bad_pipeline_specs_are_refused(fields: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        pipeline_spec(**fields)


def test_identity_follows_the_retriever_and_the_generator() -> None:
    endpoints = {"search": ENDPOINT}
    base = identity(pipeline_spec(), endpoints)
    assert identity(pipeline_spec(), endpoints) == base
    moved = {"search": HttpEndpoint(url="https://other.example.com/retrieve")}
    assert identity(pipeline_spec(), moved) != base
    assert identity(pipeline_spec(contexts=2), endpoints) != base
    other_model = pipeline_spec(
        generation={"template": TEMPLATE, "system": ANSWER_SYSTEM, "model": "claude-sonnet-5-5"}
    )
    assert identity(other_model, endpoints) != base
    with pytest.raises(InvalidInputError, match="unknown HTTP endpoint 'search'"):
        identity(pipeline_spec(), {})
    elsewhere = pipeline_spec(
        generation={"template": TEMPLATE, "provider": "openai", "model": "gpt-5.1"}
    )
    with pytest.raises(InvalidInputError, match="unknown provider 'openai'"):
        identity(elsewhere, endpoints)


def test_the_factory_builds_a_pipeline_on_the_named_provider() -> None:
    providers = ProviderRegistry({}, PriceBook(), claude=FakeAnthropic().client())
    built = build_target(pipeline_spec(), endpoints={"search": ENDPOINT}, providers=providers)
    assert isinstance(built, PipelineTarget)
    assert (built.provider, built.caches_cases) == ("pipeline", False)
    assert built.retriever.spec.endpoint == "search"


def test_an_unpriced_generator_is_refused() -> None:
    spec = pipeline_spec(generation={"template": TEMPLATE, "model": "claude-unknown-1"})
    with pytest.raises(InvalidInputError, match="claude-unknown-1"):
        PipelineTarget(spec, ENDPOINT, FakeAnthropic().client(PriceTable()))


# Contexts


def test_contexts_are_numbered_with_their_ids() -> None:
    docs = [Document(id="kb-2", text="The tower was completed in 1889."), Document(id="kb-7")]
    assert render_contexts(docs) == (
        "[1] kb-2\nThe tower was completed in 1889.\n\n[2] kb-7\n(no text)"
    )
    assert render_contexts([]) == NO_DOCUMENTS


def test_scorers_see_the_documents_the_model_was_given() -> None:
    ranking = Ranking(
        documents=(Document(id="a", text="A."), Document(id="b"), Document(id="c", text="C."))
    )
    case = Case(id="q", input={"query": "what", "filters": {"lang": "en"}}, expected="A.")
    view = with_contexts(case, ranking, 2)
    assert view.input == {
        "query": "what",
        "filters": {"lang": "en"},
        "contexts": [{"id": "a", "text": "A."}, {"id": "b", "text": ""}],
    }
    assert (view.id, view.expected) == ("q", "A.")
    assert with_contexts(Case(id="q", input="what"), ranking, 1).input == {
        "query": "what",
        "contexts": [{"id": "a", "text": "A."}],
    }
    t = target(Index(documents), FakeAnthropic())
    assert t.scoring_view(case, ()) == (case, None)
    stages = (Stage(0, "retrieval", ranking.to_output()), Stage(1, "generation", "A. [1]"))
    viewed, output = t.scoring_view(case, stages)
    assert output == ranking.to_output()
    assert isinstance(viewed.input, dict)
    # The default spec gives the model all k = 3 documents.
    contexts = viewed.input["contexts"]
    assert isinstance(contexts, list)
    assert [c["id"] for c in contexts if isinstance(c, dict)] == ["a", "b", "c"]
    asyncio.run(t.aclose())


# The target


def test_a_case_runs_both_stages_and_returns_them() -> None:
    index, fake = Index(documents), model()
    t = target(index, fake, pipeline_spec(contexts=2))
    generation = generate(t, Case(id="q", input={"query": "tower", "filters": {"lang": "en"}}))
    assert generation.text == "fact 0 about tower [1]"
    retrieval, answer = generation.stages
    assert (retrieval.index, retrieval.name, retrieval.cached) == (0, "retrieval", False)
    assert [d["id"] for d in json.loads(retrieval.output)["documents"]] == [
        "tower-0",
        "tower-1",
        "tower-2",
    ]
    assert (answer.index, answer.name, answer.output) == (1, "generation", generation.text)
    assert (answer.stop_reason, answer.cached) == ("end_turn", False)
    assert answer.usage.input_tokens == 100
    assert generation.usage == answer.usage
    assert not generation.cached
    # The retriever got the query and its filters; the model got the top two.
    assert json.loads(index.requests[0].content) == {
        "query": "tower",
        "k": 3,
        "filters": {"lang": "en"},
    }
    (sent,) = fake.requests
    assert sent["system"][0]["text"] == ANSWER_SYSTEM
    assert sent["messages"][0]["content"] == (
        "Sources:\n[1] tower-0\nfact 0 about tower\n\n[2] tower-1\nfact 1 about tower"
        "\n\nQuestion: tower"
    )


def test_both_stages_come_from_the_cache_on_a_rerun() -> None:
    store = MemoryStore()
    index, fake = Index(documents), model()

    async def twice() -> list[Any]:
        t = target(index, fake)
        t.attach(store=store, redactor=Redactor())
        try:
            return [await t.generate(Case(id="q", input="tower")) for _ in range(2)]
        finally:
            await t.aclose()

    first, second = asyncio.run(twice())
    assert (len(index.requests), len(fake.requests)) == (1, 1)
    assert second.text == first.text
    assert [s.cached for s in second.stages] == [True, True]
    assert second.cached
    assert second.stages[1].usage.input_tokens == 0
    # A retrieval target with the same retriever settings reads the same entry.
    alone = RetrievalTarget(
        RetrievalTargetSpec(type="retrieval", endpoint="search", k=3),
        ENDPOINT,
        resolver=public,
        transport=httpx2.MockTransport(index),
    )
    alone.attach(store=store, redactor=Redactor())

    async def retrieve() -> Any:
        try:
            return await alone.generate(Case(id="q", input="tower"))
        finally:
            await alone.aclose()

    assert asyncio.run(retrieve()).cached
    assert len(index.requests) == 1


def test_nothing_retrieved_still_asks_the_model() -> None:
    fake = model()
    t = target(Index(lambda body: {"results": []}), fake)
    generation = generate(t, Case(id="q", input="nothing"))
    assert generation.text == "I could not find it."
    assert json.loads(generation.stages[0].output) == {"documents": []}
    assert NO_DOCUMENTS in fake.requests[0]["messages"][0]["content"]


def test_an_answer_whose_redaction_fails_is_not_cached() -> None:
    class Failing(Redactor):
        def redact(self, text: str) -> Any:
            result = super().redact(text)
            if "[1]" in text:
                return replace(result, text=FAILED_PLACEHOLDER, failed=True)
            return result

    store = MemoryStore()
    t = target(Index(lambda body: {"results": ["d1"]}), model())
    t.attach(store=store, redactor=Failing())
    generation = generate(t, Case(id="q", input="x"))
    assert generation.text == FAILED_PLACEHOLDER
    # The ranking has no text to redact and is cached; the answer is not.
    assert len(store.entries) == 1


def test_a_failed_generation_fails_the_case_call() -> None:
    fake = FakeAnthropic(error_reply(400, "invalid_request_error"))
    t = target(Index(documents), fake)
    with pytest.raises(Exception, match="invalid_request_error"):
        generate(t, Case(id="q", input="x"))


# A whole run


SETTINGS = Settings.model_validate(
    {
        **Settings.from_env({}).model_dump(),
        "http_endpoints": {"search": ENDPOINT, "search-v2": ENDPOINT},
    }
)
DOCS = [f"kb-{i:03d}" for i in range(300)]


def text_of(doc: str) -> str:
    return f"Document {doc} says the answer is {int(doc[3:]) * 7}."


def corpus_case(i: int) -> Case:
    labels = {DOCS[3 * i]: 3, DOCS[3 * i + 1]: 2, DOCS[3 * i + 2]: 1}
    return Case(
        id=f"q{i:03d}",
        input=f"question {i}",
        expected={"relevance": labels, "answer": text_of(DOCS[3 * i])},
    )


def retriever(regressed: bool) -> Any:
    def answer(body: dict[str, Any]) -> dict[str, Any]:
        i = int(body["query"].split()[1])
        good = [DOCS[3 * i], DOCS[3 * i + 1], DOCS[3 * i + 2]]
        noise = [d for d in (DOCS[(7 * i + j) % 300] for j in range(1, 6)) if d not in good]
        # The regressed retriever loses the best document for two queries in three.
        if regressed and i % 3:
            good = good[1:]
        ranked = good + noise
        return {
            "results": [
                {"id": d, "score": 1 - n / 20, "text": text_of(d)} for n, d in enumerate(ranked)
            ]
        }

    return answer


class Factory:
    def __init__(self, fake: FakeAnthropic) -> None:
        self.fake = fake
        self.indexes = {"search": Index(retriever(False)), "search-v2": Index(retriever(True))}
        self.built: list[PipelineTarget] = []

    def __call__(self, spec: Any) -> PipelineTarget:
        index = self.indexes[spec.retrieval.endpoint]
        built = PipelineTarget(
            spec,
            ENDPOINT,
            self.fake.client(),
            resolver=public,
            transport=httpx2.MockTransport(index),
        )
        self.built.append(built)
        return built


def run_config(endpoint: str = "search", **options: Any) -> RunConfig:
    return RunConfig.model_validate(
        {
            "label": endpoint,
            "dataset": {"name": "kb"},
            "target": {
                "type": "pipeline",
                "retrieval": {"endpoint": endpoint, "k": 5},
                "generation": {"system": ANSWER_SYSTEM, "template": TEMPLATE},
                "contexts": 2,
            },
            "scorers": [
                {"type": "ranking", "metric": "ndcg", "k": 5},
                {"type": "ranking", "metric": "recall", "k": 5},
                {"type": "faithfulness", "gate": False},
                {"type": "exact", "expected_key": "answer", "name": "answer_match"},
            ],
            "options": {"concurrency": 8, **options},
        }
    )


@pytest.fixture
def kb(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        import_dataset(s, "kb", [corpus_case(i) for i in range(30)])
        import_dataset(s, "chat", [Case(id="c1", input={"messages": []})])


def results(sessions: sessionmaker[Session], run_id: str) -> list[Result]:
    with sessions() as s:
        query = select(Result).where(Result.run_id == run_id).order_by(Result.case_id)
        return list(s.scalars(query))


def stages(sessions: sessionmaker[Session], run_id: str) -> dict[str, list[ResultStage]]:
    out: dict[str, list[ResultStage]] = {}
    with sessions() as s:
        query = (
            select(ResultStage)
            .where(ResultStage.run_id == run_id)
            .order_by(ResultStage.case_id, ResultStage.position)
        )
        for row in s.scalars(query):
            out.setdefault(row.case_id, []).append(row)
    return out


def runner_for(sessions: sessionmaker[Session], fake: FakeAnthropic, factory: Factory) -> Runner:
    providers = ProviderRegistry({}, PriceBook(), claude=fake.client())
    return Runner(sessions, SETTINGS, providers=providers, target_factory=factory)


def test_a_run_stores_both_stages_and_scores_each(
    sessions: sessionmaker[Session], kb: None
) -> None:
    fake = model()
    factory = Factory(fake)
    runner = runner_for(sessions, fake, factory)
    run = runner.create_run(run_config())
    report = asyncio.run(runner.execute(run.id))
    assert (report.status, report.errors) == ("completed", 0)
    rows = results(sessions, run.id)
    assert len(rows) == 30
    by_case = stages(sessions, run.id)
    assert len(by_case) == 30
    for row in rows:
        retrieval, generation = by_case[row.case_id]
        assert (retrieval.position, retrieval.stage) == (0, "retrieval")
        assert (generation.position, generation.stage) == (1, "generation")
        assert len(json.loads(retrieval.output)["documents"]) == 5
        assert generation.output == row.output
        assert generation.stop_reason == "end_turn"
        assert retrieval.cost_usd == 0
        assert generation.cost_usd == pytest.approx(row.cost_usd)
        assert row.cost_usd > 0
        # Ranking metrics graded the retrieval stage, the rest the answer.
        assert row.scores["ndcg_at_5"]["value"] == pytest.approx(1.0)
        assert row.scores["recall_at_5"]["value"] == pytest.approx(1.0)
        assert row.scores["faithfulness"]["value"] == 1.0
        assert row.scores["answer_match"]["value"] == 0.0  # the answer cites [1]
    assert len(factory.indexes["search"].requests) == 30
    generations = [r for r in fake.requests if r["system"][0]["text"] == ANSWER_SYSTEM]
    assert len(generations) == 30

    # A rerun pays for neither stage and stores both as cached.
    rerun = runner.create_run(run_config())
    asyncio.run(runner.execute(rerun.id))
    assert len(factory.indexes["search"].requests) == 30
    generations = [r for r in fake.requests if r["system"][0]["text"] == ANSWER_SYSTEM]
    assert len(generations) == 30
    rerun_rows = results(sessions, rerun.id)
    assert all(r.cached and r.cost_usd == 0 for r in rerun_rows)
    assert all(s.cached for pair in stages(sessions, rerun.id).values() for s in pair)
    assert [r.output for r in rerun_rows] == [r.output for r in rows]


def test_a_retriever_regression_fails_the_gate_through_the_pipeline(
    sessions: sessionmaker[Session], kb: None
) -> None:
    fake = model()
    factory = Factory(fake)
    runner = runner_for(sessions, fake, factory)
    base = runner.create_run(run_config())
    asyncio.run(runner.execute(base.id))
    candidate = runner.create_run(run_config("search-v2"))
    report = asyncio.run(runner.execute(candidate.id))
    assert (report.status, report.errors) == ("completed", 0)
    recall = sorted(r.scores["recall_at_5"]["value"] for r in results(sessions, candidate.id))
    assert recall == [pytest.approx(2 / 3)] * 20 + [1.0] * 10
    with sessions() as s:
        gate = compare_runs(s, base.id, candidate.id)
    assert sorted(gate.regressions) == ["ndcg_at_5", "recall_at_5"]
    # Only the twenty queries whose top two documents changed paid for an answer.
    generations = [r for r in fake.requests if r["system"][0]["text"] == ANSWER_SYSTEM]
    assert len(generations) == 30 + 20
    paid = [pair[1] for pair in stages(sessions, candidate.id).values() if not pair[1].cached]
    assert len(paid) == 20
    # Answers written from the second-best document are still grounded in it.
    assert all(r.scores["faithfulness"]["value"] == 1.0 for r in results(sessions, candidate.id))


def test_a_failed_generation_is_a_case_error_with_no_stages(
    sessions: sessionmaker[Session], kb: None
) -> None:
    fake = FakeAnthropic()
    fake.default = error_reply(400, "invalid_request_error")
    runner = runner_for(sessions, fake, Factory(fake))
    run = runner.create_run(run_config(use_cache=False))
    report = asyncio.run(runner.execute(run.id))
    assert report.errors == 30
    assert all("invalid_request_error" in r.error for r in results(sessions, run.id))
    assert stages(sessions, run.id) == {}


def test_a_pipeline_run_refuses_cases_that_are_not_queries(
    sessions: sessionmaker[Session], kb: None
) -> None:
    fake = model()
    runner = runner_for(sessions, fake, Factory(fake))
    config = run_config().model_copy(
        update={"dataset": run_config().dataset.model_copy(update={"name": "chat"})}
    )
    with pytest.raises(InvalidInputError, match="not a retrieval query"):
        runner.create_run(config)


def test_a_pipeline_run_refuses_an_unpriced_model(
    sessions: sessionmaker[Session], kb: None
) -> None:
    fake = model()
    runner = runner_for(sessions, fake, Factory(fake))
    raw = run_config().model_dump(mode="json", by_alias=True)
    raw["target"]["generation"]["model"] = "claude-unknown-1"
    with pytest.raises(InvalidInputError, match="claude-unknown-1"):
        runner.create_run(RunConfig.model_validate(raw))


@pytest.mark.anyio
async def test_the_trajectory_endpoint_serves_both_stages(
    sessions: sessionmaker[Session], kb: None
) -> None:
    fake = model()
    runner = runner_for(sessions, fake, Factory(fake))
    run = runner.create_run(run_config())
    await runner.execute(run.id)
    async with serving(sessions, EVALKIT_MAX_ACTIVE_RUNS="0") as (c, _):
        response = await c.get(f"/v1/runs/{run.id}/results/q004/trajectory", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert (body["steps"], body["turns"]) == ([], [])
    retrieval, generation = body["stages"]
    assert retrieval["stage"] == "retrieval"
    assert [d["id"] for d in retrieval["ranking"]["documents"]][:3] == DOCS[12:15]
    assert "output" not in retrieval
    assert generation["stage"] == "generation"
    assert generation["output"] == f"{text_of(DOCS[12])} [1]"
    assert generation["stop_reason"] == "end_turn"
    assert generation["input_tokens"] == 100
    assert generation["cached"] is False
