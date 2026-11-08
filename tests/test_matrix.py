"""Model matrices: expansion into child runs, prompt targets on every provider,
the cost/quality/latency frontier, and the CLI and API around them."""

import asyncio
import io
import json
import re
from pathlib import Path
from typing import Any

import httpx2
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, sessionmaker

from evalkit import cli
from evalkit.api import create_app
from evalkit.config import Settings
from evalkit.datasets import import_dataset, parse_jsonl
from evalkit.db.models import Matrix, Run
from evalkit.errors import InvalidInputError, NotFoundError, PreconditionError
from evalkit.estimate import estimate_matrix
from evalkit.frontier import COST, LATENCY, QUALITY, Point, dominates, matrix_report, pareto
from evalkit.matrix import child_runs, expand, resolve_matrix
from evalkit.pricing import ModelPrice, PriceTable
from evalkit.providers.registry import ProviderConfig, ProviderRegistry
from evalkit.report import summarize
from evalkit.runner import Runner
from evalkit.runspec import MAX_MATRIX_VARIANTS, RunConfig
from evalkit.targets.fake import unit
from tests.fake_anthropic import FakeAnthropic, json_reply, message

N = 120
PRICES = {
    "gpt-test": ModelPrice(input=2.0, output=8.0),
    "gemini-test": ModelPrice(input=1.0, output=4.0),
}
# How often each model answers correctly, decided per case by a hash.
ACCURACY = {"claude-sonnet-5-5": 0.95, "gpt-test": 0.6, "gemini-test": 0.85}
LOCAL = ProviderConfig(kind="openai_compat", base_url="http://llm.test/v1")
GEM = ProviderConfig(kind="gemini", base_url="http://gemini.test/v1beta")
QUESTION = re.compile(r"question (\d+)")


def answer(model: str, request_body: bytes) -> str:
    found = QUESTION.search(request_body.decode())
    assert found, request_body
    i = int(found.group(1))
    return f"a{i}" if unit(model, i) < ACCURACY[model] else "no idea"


class Providers:
    """Every provider a matrix calls: the fake Claude API, an OpenAI-compatible
    server and a Gemini server, answering each question by model."""

    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []
        self.claude = FakeAnthropic()
        self.claude.default = self._claude

    def _claude(self, body: dict[str, Any]) -> httpx2.Response:
        if self.claude.paths[-1].endswith("/count_tokens"):
            return json_reply({"input_tokens": 50})
        text = answer(body["model"], json.dumps(body).encode())
        return json_reply(message(text, model=body["model"]))

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if request.url.host == "gemini.test":
            if request.url.path.endswith(":countTokens"):
                return httpx2.Response(200, json={"totalTokens": 50})
            text = answer("gemini-test", request.content)
            return httpx2.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": text}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 10},
                    "modelVersion": "gemini-test",
                },
            )
        text = answer("gpt-test", request.content)
        return httpx2.Response(
            200,
            json={
                "model": "gpt-test",
                "choices": [
                    {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
        )

    def bodies(self, host: str) -> list[Any]:
        return [json.loads(r.content) for r in self.requests if r.url.host == host]

    def registry(self, **configs: ProviderConfig) -> ProviderRegistry:
        return ProviderRegistry(
            configs or {"local": LOCAL, "gem": GEM},
            PriceTable(PRICES),
            claude=self.claude.client(PriceTable(PRICES)),
            transport=httpx2.MockTransport(self.handle),
        )


@pytest.fixture
def fakes() -> Providers:
    return Providers()


@pytest.fixture
def runner(sessions: sessionmaker[Session], fakes: Providers) -> Runner:
    with sessions() as session:
        import_dataset(session, "qa", parse_jsonl(dataset_bytes()))
    return Runner(sessions, Settings(), providers=fakes.registry())


def dataset_bytes(n: int = N) -> bytes:
    return "\n".join(
        json.dumps({"id": f"q{i:03d}", "input": {"q": f"question {i}"}, "expected": f"a{i}"})
        for i in range(n)
    ).encode()


THREE = [
    "claude-sonnet-5-5",
    {"provider": "local", "model": "gpt-test"},
    {"provider": "gem", "model": "gemini-test", "name": "gemini"},
]


def matrix_config(label: str = "models", **extra: Any) -> RunConfig:
    raw: dict[str, Any] = {
        "label": label,
        "dataset": {"name": "qa"},
        "target": {"type": "prompt", "template": "Answer: {{q}}", "max_tokens": 64},
        "scorers": [{"type": "exact"}, {"type": "token_f1", "gate": False}],
        "options": {"concurrency": 4},
        "matrix": {"model": THREE},
    }
    raw.update(extra)
    return RunConfig.model_validate(raw)


def execute_all(runner: Runner, runs: list[Run]) -> None:
    for run in runs:
        assert asyncio.run(runner.execute(run.id)).status == "completed"


def set_latency(sessions: sessionmaker[Session], run_id: str, p50: float, p95: float) -> None:
    with sessions() as session:
        session.execute(
            update(Run).where(Run.id == run_id).values(latency_p50_ms=p50, latency_p95_ms=p95)
        )
        session.commit()


# Expansion


def test_expansion_order_names_and_labels() -> None:
    config = matrix_config(
        matrix={
            "model": ["claude-opus-5-5", {"provider": "local", "model": "gpt-test"}],
            "effort": ["low", "high"],
            "max_tokens": [256],
        }
    )
    variants = expand(config)
    assert [v.name for v in variants] == [
        "claude-opus-5-5,effort=low,max_tokens=256",
        "claude-opus-5-5,effort=high,max_tokens=256",
        "local:gpt-test,effort=low,max_tokens=256",
        "local:gpt-test,effort=high,max_tokens=256",
    ]
    assert [v.index for v in variants] == [0, 1, 2, 3]
    assert variants[0].config.label == "models[claude-opus-5-5,effort=low,max_tokens=256]"
    third = variants[2]
    assert third.axes() == {
        "index": 2,
        "name": "local:gpt-test,effort=low,max_tokens=256",
        "provider": "local",
        "model": "gpt-test",
        "effort": "low",
        "max_tokens": 256,
    }
    # Everything but the target and the label is the base config's.
    assert third.config.matrix is None
    assert third.config.scorers == config.scorers
    assert third.config.options == config.options
    assert third.target.template == "Answer: {{q}}"
    assert expand(config) == variants  # deterministic


def test_axes_left_out_keep_the_base_target() -> None:
    base = {
        "type": "prompt",
        "template": "{{q}}",
        "provider": "gem",
        "model": "gemini-test",
        "effort": "high",
    }
    variants = expand(matrix_config(label="", target=base, matrix={"max_tokens": [100, 200]}))
    assert [v.name for v in variants] == ["max_tokens=100", "max_tokens=200"]
    assert variants[0].config.label == "max_tokens=100"
    assert {(v.target.provider, v.target.model, v.target.effort) for v in variants} == {
        ("gem", "gemini-test", "high")
    }
    # A bare model name keeps the base target's provider.
    on_gem = expand(matrix_config(target=base, matrix={"model": ["gemini-other"]}))
    assert on_gem[0].target.provider == "gem"
    assert on_gem[0].name == "gem:gemini-other"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"target": {"type": "fake"}}, "this config's target is fake"),
        ({"matrix": {}}, "at least one axis"),
        (
            {"matrix": {"model": [f"m{i}" for i in range(9)], "max_tokens": list(range(1, 9))}},
            "expands into 72 runs; at most 64",
        ),
        ({"matrix": {"model": ["claude-opus-5-5", "claude-opus-5-5"]}}, "are the same target"),
        (
            {"matrix": {"model": [{"model": "a", "name": "x"}, {"model": "b", "name": "x"}]}},
            "names two variants 'x'",
        ),
        ({"label": "l" * 120}, "longer than 128 characters"),
    ],
)
def test_bad_matrices_are_refused(change: dict[str, Any], message: str) -> None:
    with pytest.raises(InvalidInputError, match=re.escape(message)):
        expand(matrix_config(**change))


def test_matrix_values_are_validated() -> None:
    with pytest.raises(ValidationError):
        matrix_config(matrix={"max_tokens": [0]})
    with pytest.raises(ValidationError):
        matrix_config(matrix={"effort": ["extreme"]})
    with pytest.raises(ValidationError):
        matrix_config(matrix={"model": [{"model": "m", "provider": "Not A Name"}]})
    with pytest.raises(ValidationError):
        matrix_config(matrix={"model": [f"m{i}" for i in range(MAX_MATRIX_VARIANTS + 1)]})
    with pytest.raises(InvalidInputError, match="no \\[matrix\\] table"):
        expand(matrix_config(matrix=None))


# The frontier


def test_dominance() -> None:
    dims = [QUALITY, COST, LATENCY]
    best = Point("best", 0.9, 1.0, 100)
    assert dominates(best, Point("worse", 0.8, 1.0, 100), dims)
    assert dominates(best, Point("pricier", 0.9, 2.0, 100), dims)
    assert dominates(best, Point("slower", 0.9, 1.0, 200), dims)
    assert not dominates(best, Point("same", 0.9, 1.0, 100), dims)
    assert not dominates(best, Point("cheaper", 0.5, 0.5, 100), dims)
    # Only the dimensions asked for count.
    assert dominates(best, Point("faster", 0.5, 1.0, 50), [QUALITY, COST])
    with pytest.raises(ValueError, match="latency is not known"):
        dominates(best, Point("p", 0.1, 1.0, None), dims)
    with pytest.raises(ValueError, match="unique"):
        pareto([best, best], dims)


def test_pareto_frontier_of_known_points() -> None:
    points = [
        Point("cheap", 0.70, 1.0, 300),
        Point("mid", 0.80, 2.0, 300),
        Point("waste", 0.75, 3.0, 400),  # mid is better, cheaper and as fast
        Point("best", 0.90, 5.0, 900),
        Point("tie", 0.90, 5.0, 900),  # equal points do not dominate each other
    ]
    assert pareto(points, [QUALITY, COST, LATENCY]) == {
        "cheap": [],
        "mid": [],
        "waste": ["mid"],
        "best": [],
        "tie": [],
    }
    assert pareto(points, [QUALITY])["cheap"] == ["mid", "waste", "best", "tie"]


values = st.floats(min_value=0, max_value=1, allow_nan=False)
point_lists = st.lists(st.tuples(values, values, values), min_size=1, max_size=12)


@settings(max_examples=300, deadline=None)
@given(point_lists, st.sampled_from([[QUALITY], [QUALITY, COST], [QUALITY, COST, LATENCY]]))
def test_frontier_properties(raw: list[tuple[float, float, float]], dims: list[str]) -> None:
    points = [Point(f"p{i}", q, c, lat) for i, (q, c, lat) in enumerate(raw)]
    result = pareto(points, dims)
    frontier = [p for p in points if not result[p.key]]
    assert frontier  # a finite set always has an undominated point
    for p in points:
        # Exactly the points that dominate p are listed ...
        assert result[p.key] == [q.key for q in points if q is not p and dominates(q, p, dims)]
        # ... and anything off the frontier is dominated by a point on it.
        if result[p.key]:
            assert any(dominates(f, p, dims) for f in frontier)
    for a in frontier:
        for b in frontier:
            assert not dominates(a, b, dims)


# Three providers, one matrix


@pytest.fixture
def three(runner: Runner, sessions: sessionmaker[Session]) -> tuple[Matrix, list[Run]]:
    """The three-provider matrix, executed."""
    group, runs = runner.create_matrix(matrix_config())
    assert [r.label for r in runs] == [
        "models[claude-sonnet-5-5]",
        "models[local:gpt-test]",
        "models[gemini]",
    ]
    assert len({r.target_hash for r in runs}) == 3
    execute_all(runner, runs)
    # Latency is measured, but fixed here so dominance does not depend on timing noise.
    for run, (p50, p95) in zip(runs, [(800, 1500), (500, 900), (300, 600)], strict=True):
        set_latency(sessions, run.id, p50, p95)
    return group, runs


EXPECTED = {
    name: sum(unit(model, i) < ACCURACY[model] for i in range(N)) / N
    for name, model in [
        ("claude-sonnet-5-5", "claude-sonnet-5-5"),
        ("local:gpt-test", "gpt-test"),
        ("gemini", "gemini-test"),
    ]
}


def test_three_providers_each_called_and_accounted(
    three: tuple[Matrix, list[Run]], sessions: sessionmaker[Session], fakes: Providers
) -> None:
    group, runs = three

    # Each variant called its own provider, with its own model, for every case.
    claude_bodies = fakes.claude.requests
    assert len(claude_bodies) == N
    assert {b["model"] for b in claude_bodies} == {"claude-sonnet-5-5"}
    openai_bodies = fakes.bodies("llm.test")
    assert len(openai_bodies) == N
    assert openai_bodies[0]["model"] == "gpt-test"
    assert openai_bodies[0]["max_completion_tokens"] == 64
    assert {(b["messages"][-1]["role"], b["messages"][-1]["content"]) for b in openai_bodies} == {
        ("user", f"Answer: question {i}") for i in range(N)
    }
    gemini_requests = [r for r in fakes.requests if r.url.host == "gemini.test"]
    assert len(gemini_requests) == N
    assert gemini_requests[0].url.path == "/v1beta/models/gemini-test:generateContent"

    # Tokens and cost are accounted for each provider at its own prices.
    with sessions() as session:
        summaries = [summarize(session, r.id) for r in runs]
    claude, gpt, gemini = summaries
    assert (claude.input_tokens, claude.output_tokens) == (100 * N, 20 * N)
    assert claude.cost_usd == pytest.approx(N * (100 * 2.0 + 20 * 10.0) / 1e6)
    assert (gpt.input_tokens, gpt.output_tokens) == (100 * N, 10 * N)
    assert gpt.cost_usd == pytest.approx(N * (100 * 2.0 + 10 * 8.0) / 1e6)
    assert (gemini.input_tokens, gemini.output_tokens) == (100 * N, 10 * N)
    assert gemini.cost_usd == pytest.approx(N * (100 * 1.0 + 10 * 4.0) / 1e6)
    assert gpt.matrix_id == group.id
    assert gpt.variant == "local:gpt-test"


def test_the_frontier_of_three_providers(
    three: tuple[Matrix, list[Run]], sessions: sessionmaker[Session]
) -> None:
    group, _ = three
    with sessions() as session:
        report = matrix_report(session, group.id)
    assert report.metric == "exact"
    assert report.dimensions == [QUALITY, COST, LATENCY]
    rows = {v.name: v for v in report.variants}
    expected = EXPECTED
    for name, accuracy in expected.items():
        assert rows[name].quality == pytest.approx(accuracy)
    assert expected["gemini"] > expected["local:gpt-test"]
    assert expected["claude-sonnet-5-5"] > expected["gemini"]
    # GPT is worse, dearer and slower than Gemini; the other two trade cost for quality.
    assert rows["local:gpt-test"].dominated_by == ["gemini"]
    assert report.frontier == ["claude-sonnet-5-5", "gemini"]
    assert rows["gemini"].cost_per_case_usd == pytest.approx(140 / 1e6)
    assert rows["claude-sonnet-5-5"].latency_ms == 800
    assert report.complete
    assert report.notes == []

    # One step along the frontier, cheapest first, paired over every case.
    (step,) = report.steps
    assert (step.baseline, step.candidate) == ("gemini", "claude-sonnet-5-5")
    c = step.comparison
    assert c.n_pairs == N
    assert c.difference == pytest.approx(expected["claude-sonnet-5-5"] - expected["gemini"])
    assert c.difference == pytest.approx(0.10)
    assert c.verdict == "better"
    assert c.p_adjusted < 0.05


def test_the_frontier_moves_with_the_latency_percentile(
    three: tuple[Matrix, list[Run]], sessions: sessionmaker[Session]
) -> None:
    group, runs = three
    # At p95 GPT is the fastest, so it is a choice someone could make after all.
    set_latency(sessions, runs[1].id, 500, 500)
    with sessions() as session:
        at_p95 = matrix_report(session, group.id, latency="p95", metric="token_f1")
    assert at_p95.metric == "token_f1"
    assert {v.name: v.latency_ms for v in at_p95.variants}["local:gpt-test"] == 500
    assert {v.name: v.dominated_by for v in at_p95.variants}["local:gpt-test"] == []
    assert at_p95.frontier == ["claude-sonnet-5-5", "local:gpt-test", "gemini"]
    # Cheapest first: gemini, then gpt (dearer, faster), then claude.
    assert [(s.baseline, s.candidate) for s in at_p95.steps] == [
        ("gemini", "local:gpt-test"),
        ("local:gpt-test", "claude-sonnet-5-5"),
    ]
    assert at_p95.steps[0].comparison.difference < 0
    # Holm over the two steps: adjusted p-values are never below the raw ones.
    assert all(s.comparison.p_adjusted >= s.comparison.p_value for s in at_p95.steps)


def test_a_cached_rerun_and_an_unfinished_matrix(
    runner: Runner, sessions: sessionmaker[Session]
) -> None:
    first, runs = runner.create_matrix(matrix_config())
    execute_all(runner, runs)
    # The same matrix again is served from the cache: nothing to price or time.
    again, rerun = runner.create_matrix(matrix_config())
    execute_all(runner, rerun)
    with sessions() as session:
        report = matrix_report(session, again.id)
    assert report.dimensions == [QUALITY]
    assert all(v.cost_per_case_usd is None and v.latency_ms is None for v in report.variants)
    assert any("cost is not a frontier dimension" in n for n in report.notes)
    assert any("latency is not a frontier dimension" in n for n in report.notes)
    assert report.frontier == ["claude-sonnet-5-5"]
    assert "claude-sonnet-5-5 dominates every other variant" in report.notes
    assert report.steps == []

    third, partial = runner.create_matrix(matrix_config(label="partial"))
    execute_all(runner, partial[:1])
    with sessions() as session:
        report = matrix_report(session, third.id)
    assert not report.complete
    assert report.frontier == ["claude-sonnet-5-5"]
    assert [v.status for v in report.variants] == ["completed", "pending", "pending"]
    assert report.notes[0] == (
        "not on the frontier because not completed or not scored: local:gpt-test, gemini"
    )
    with sessions() as session:
        assert resolve_matrix(session, "label:models") == again.id
        assert resolve_matrix(session, first.id) == first.id
        assert [r.id for r in child_runs(session, first.id)] == [r.id for r in runs]
        with pytest.raises(NotFoundError, match="no matrix labelled 'nope'"):
            resolve_matrix(session, "label:nope")
        with pytest.raises(NotFoundError, match="matrix missing not found"):
            resolve_matrix(session, "missing")
        with pytest.raises(InvalidInputError, match="unknown metric 'nope'; available: exact"):
            matrix_report(session, first.id, metric="nope")
        with pytest.raises(InvalidInputError, match="p50 or p95"):
            matrix_report(session, first.id, latency="p99")  # type: ignore[arg-type]


def test_a_matrix_with_no_gated_metric_needs_one_named(
    runner: Runner, sessions: sessionmaker[Session]
) -> None:
    group, _ = runner.create_matrix(matrix_config(scorers=[{"type": "exact", "gate": False}]))
    with sessions() as session:
        with pytest.raises(
            InvalidInputError, match="no gated metric to rank by; give one of: exact"
        ):
            matrix_report(session, group.id)
        report = matrix_report(session, group.id, metric="exact")
    assert report.frontier == []
    assert report.dimensions == [QUALITY, COST, LATENCY]  # nothing eligible, nothing unknown


def test_every_variant_is_checked_before_any_run_is_recorded(
    runner: Runner, sessions: sessionmaker[Session]
) -> None:
    bad_provider = {"model": [*THREE, {"provider": "nowhere", "model": "m"}]}
    with pytest.raises(InvalidInputError, match="unknown provider 'nowhere'"):
        runner.create_matrix(matrix_config(matrix=bad_provider))
    unpriced = {"model": [*THREE, {"provider": "local", "model": "gpt-unpriced"}]}
    with pytest.raises(InvalidInputError, match="unknown model 'gpt-unpriced'"):
        runner.create_matrix(matrix_config(matrix=unpriced))
    batch = matrix_config(options={"mode": "batch"})
    with pytest.raises(InvalidInputError, match="only available for Claude prompt targets"):
        runner.create_matrix(batch)
    with pytest.raises(InvalidInputError, match="POST /v1/matrices"):
        runner.create_run(matrix_config())
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(Matrix)) == 0
        assert session.scalar(select(func.count()).select_from(Run)) == 0
    # A Claude prompt target with an unknown model is refused before it is recorded too.
    single = matrix_config(matrix=None, target={"type": "prompt", "model": "x", "template": "t"})
    with pytest.raises(InvalidInputError, match="unknown model 'x'"):
        runner.create_run(single)


def test_queued_matrix_runs_and_a_moved_provider(
    runner: Runner, sessions: sessionmaker[Session], fakes: Providers
) -> None:
    group, runs = runner.create_matrix(matrix_config(queue={"priority": 5}), queued=True)
    assert {(r.status, r.priority) for r in runs} == {("queued", 5)}
    with sessions() as session:
        stored = session.get_one(Matrix, group.id)
        assert stored.axes["model"][1] == {"model": "gpt-test", "provider": "local", "name": None}
        assert "matrix" not in stored.config
        assert stored.config["label"] == "models"
    # The OpenAI-compatible server moved after the run was recorded: its identity no
    # longer holds, so the run is refused rather than answered by another server.
    moved = Runner(
        sessions,
        Settings(),
        providers=fakes.registry(
            local=ProviderConfig(kind="openai_compat", base_url="http://llm2.test/v1"), gem=GEM
        ),
    )
    with pytest.raises(PreconditionError, match="configuration changed since the run"):
        asyncio.run(moved.execute(runs[1].id))
    assert fakes.bodies("llm.test") == fakes.bodies("llm2.test") == []
    assert asyncio.run(runner.execute(runs[2].id)).status == "completed"


def test_estimate_each_variant(
    runner: Runner, sessions: sessionmaker[Session], fakes: Providers
) -> None:
    claude = fakes.claude.client(PriceTable(PRICES))
    found = asyncio.run(
        estimate_matrix(sessions, Settings(), matrix_config(), claude, providers=runner.providers)
    )
    names = [v.name for v in found.variants]
    assert names == ["claude-sonnet-5-5", "local:gpt-test", "gemini"]
    on_claude, on_local, on_gemini = (v.estimate for v in found.variants)
    assert on_claude.target_input_tokens == 50 * N
    assert on_claude.target_input_cost_usd == pytest.approx(50 * N * 2.0 / 1e6)
    assert on_claude.target_max_output_cost_usd == pytest.approx(64 * N * 10.0 / 1e6)
    # The OpenAI-compatible API has no counting endpoint: a note, not a guess.
    assert on_local.target_input_tokens == 0
    assert on_local.target_max_output_cost_usd == pytest.approx(64 * N * 8.0 / 1e6)
    assert on_local.notes == [
        "the target's provider local cannot count tokens, so its input cost is not included"
    ]
    assert on_gemini.target_input_tokens == 50 * N
    assert on_gemini.target_input_cost_usd == pytest.approx(50 * N * 1.0 / 1e6)
    assert found.max_total_cost_usd == pytest.approx(
        sum(v.estimate.max_total_cost_usd for v in found.variants)
    )


# The CLI and the API


def run_cli(db: str, *args: str) -> tuple[int, str]:
    out = io.StringIO()
    code = cli.main(["--database-url", db, *args], out=out)
    return code, out.getvalue()


def test_cli_matrix_run_report_resume_and_estimate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fakes: Providers
) -> None:
    monkeypatch.delenv("EVALKIT_DATABASE_URL", raising=False)
    monkeypatch.setattr(cli.App, "claude", lambda self: fakes.claude.client(PriceTable(PRICES)))
    db = f"sqlite:///{tmp_path / 'cli.db'}"
    (tmp_path / "qa.jsonl").write_bytes(dataset_bytes(20))
    cfg = tmp_path / "matrix.toml"
    cfg.write_text(
        'label = "sizes"\n'
        '[dataset]\nname = "qa"\npath = "qa.jsonl"\n'
        '[target]\ntype = "prompt"\nmodel = "claude-sonnet-5-5"\ntemplate = "{{q}}"\n'
        '[[scorers]]\ntype = "exact"\n'
        '[matrix]\neffort = ["low", "high"]\nmax_tokens = [64, 128]\n'
    )
    code, text = run_cli(db, "run", str(cfg))
    assert code == 0, text
    assert "frontier over quality, cost, latency" in text
    assert "effort=high,max_tokens=128" in text
    assert len(fakes.claude.requests) == 4 * 20
    assert {(b["max_tokens"], b["output_config"]["effort"]) for b in fakes.claude.requests} == {
        (64, "low"),
        (64, "high"),
        (128, "low"),
        (128, "high"),
    }

    code, raw = run_cli(db, "report", "matrix", "label:sizes", "--json", "--latency", "p95")
    assert code == 0
    body = json.loads(raw)
    assert body["complete"] is True
    assert body["latency"] == "p95"
    assert len(body["variants"]) == 4
    assert body["frontier"] == [v["name"] for v in body["variants"] if v["on_frontier"]]
    matrix_id = body["matrix_id"]

    code, shown = run_cli(db, "show", "label:sizes[effort=low,max_tokens=64]")
    assert code == 0
    assert f"variant effort=low,max_tokens=64 of matrix {matrix_id}" in shown

    # Queued: four runs for workers, and a resume of the matrix executes them.
    code, queued = run_cli(db, "run", str(cfg), "--queue", "--label", "later", "--json")
    assert code == 0
    later = json.loads(queued)
    assert [v["status"] for v in later["variants"]] == ["queued"] * 4
    assert later["complete"] is False
    code, text = run_cli(db, "run", "--resume", later["matrix_id"])
    assert code == 0
    assert "later" in text
    assert len(fakes.claude.requests) == 4 * 20  # every case came from the cache
    code, text = run_cli(db, "report", "matrix", later["matrix_id"], "--metric", "exact")
    assert code == 0
    assert "note: cost is not a frontier dimension" in text

    fakes.claude.requests.clear()
    code, text = run_cli(db, "estimate", str(cfg))
    assert code == 0, text
    assert "variant effort=low,max_tokens=64" in text
    assert "matrix of 4 runs: at most $" in text
    assert run_cli(db, "report", "matrix", "label:none")[0] == 2
    assert run_cli(db, "run", "--resume", "missing")[0] == 2


def test_cli_matrix_exit_code_for_a_failed_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeAnthropic()
    fake.default = lambda body: httpx2.Response(
        400, json={"type": "error", "error": {"type": "invalid_request_error", "message": "no"}}
    )
    monkeypatch.setattr(cli.App, "claude", lambda self: fake.client())
    db = f"sqlite:///{tmp_path / 'cli.db'}"
    (tmp_path / "qa.jsonl").write_bytes(dataset_bytes(3))
    cfg = tmp_path / "m.toml"
    cfg.write_text(
        '[dataset]\nname = "qa"\npath = "qa.jsonl"\n'
        '[target]\ntype = "prompt"\ntemplate = "{{q}}"\n'
        '[[scorers]]\ntype = "exact"\n'
        '[matrix]\neffort = ["low", "high"]\n'
    )
    # Every case fails, so every variant scores 0: completed, all on the frontier.
    code, text = run_cli(db, "run", str(cfg))
    assert code == 0
    assert "latency is not a frontier dimension" in text
    cfg.write_text(cfg.read_text().replace('type = "prompt"', 'type = "fake"'))
    code, _ = run_cli(db, "run", str(cfg))
    assert code == 2


@pytest.mark.anyio
async def test_api_matrices(sessions: sessionmaker[Session], fakes: Providers) -> None:
    token = "test-token-0123456789"
    auth = {"Authorization": f"Bearer {token}"}
    app = create_app(
        Settings.from_env({"EVALKIT_API_TOKENS": token}),
        sessions=sessions,
        claude=fakes.claude.client(PriceTable(PRICES)),
    )
    transport = httpx2.ASGITransport(app=app)
    body = {
        "label": "api",
        "dataset": {"name": "qa"},
        "target": {"type": "prompt", "model": "claude-sonnet-5-5", "template": "{{q}}"},
        "scorers": [{"type": "exact"}],
        "matrix": {"effort": ["low", "high"]},
    }
    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(transport=transport, base_url="http://test") as c,
    ):
        await c.post("/v1/datasets/qa", content=dataset_bytes(12), headers=auth)
        refused = await c.post("/v1/runs", json=body, headers=auth)
        assert refused.status_code == 400
        assert "POST /v1/matrices" in refused.json()["error"]["message"]
        single = {k: v for k, v in body.items() if k != "matrix"}
        assert (await c.post("/v1/matrices", json=single, headers=auth)).status_code == 400

        created = await c.post("/v1/matrices", json=body, headers=auth)
        assert created.status_code == 202
        made = created.json()
        assert [r["label"] for r in made["runs"]] == ["api[effort=low]", "api[effort=high]"]
        for run in made["runs"]:
            for _ in range(1000):
                status = (await c.get(f"/v1/runs/{run['run_id']}", headers=auth)).json()["status"]
                if status == "completed":
                    break
                await asyncio.sleep(0.01)
            assert status == "completed"
        report = await c.get(f"/v1/matrices/{made['matrix_id']}?latency=p95", headers=auth)
        assert report.status_code == 200
        got = report.json()
        assert got["complete"] is True
        assert got["latency"] == "p95"
        assert [v["name"] for v in got["variants"]] == ["effort=low", "effort=high"]
        assert got["frontier"]
        missing = await c.get("/v1/matrices/nope", headers=auth)
        assert missing.status_code == 404
        bad = await c.get(f"/v1/matrices/{made['matrix_id']}?latency=p99", headers=auth)
        assert bad.status_code == 400
