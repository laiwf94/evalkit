"""The semantic_similarity scorer and its threshold calibration: scores, the
hand-checked threshold sweep against brute force, cross-validation, the
``evalkit calibrate --scorer`` command, and runs that share embedding requests,
reuse cached vectors and refuse to compare across embedding models."""

import asyncio
import functools
import io
import json
from pathlib import Path
from typing import Any

import httpx2
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from evalkit import cli
from evalkit.config import Settings
from evalkit.datasets import Case, import_dataset
from evalkit.db.models import CachedEmbedding, Calibration
from evalkit.embeddings import Embedder
from evalkit.errors import InvalidInputError, PreconditionError
from evalkit.pricing import PriceBook
from evalkit.providers.base import Embeddings
from evalkit.providers.registry import ProviderRegistry, builtin_providers
from evalkit.ratelimit import RetryPolicy
from evalkit.report import compare_runs, summarize
from evalkit.runner import Runner
from evalkit.runspec import RunConfig
from evalkit.scorers.calibration import (
    LabelledItem,
    best_threshold,
    binary_kappa,
    calibrate_threshold,
    cohens_kappa,
    cross_validate,
    parse_labels,
)
from evalkit.scorers.registry import ScorerSpec, build_scorers
from evalkit.scorers.similarity import SemanticSimilarity
from evalkit.targets.base import Usage
from tests.fake_embeddings import MODEL, FakeEmbedder, bag_of_words, refused
from tests.fake_openai import FakeOpenAI

PRICES = Settings.from_env({}).price_book().table("openai")
RETRY = RetryPolicy(max_attempts=1)


def scorer(fake: FakeEmbedder, **kw: Any) -> SemanticSimilarity:
    return SemanticSimilarity("sim", embedder=Embedder(fake, MODEL, retry=RETRY), **kw)


def score(s: SemanticSimilarity, case: Case, output: str) -> Any:
    return asyncio.run(s.score(case, output))


CASE = Case(id="c", input="q", expected="Paris is the capital of France")


def test_scores_are_cosines_or_pass_fail_at_the_threshold() -> None:
    fake = FakeEmbedder(PRICES)
    same = score(scorer(fake), CASE, "paris is the capital of france!")
    assert same.value == pytest.approx(1.0)
    assert same.detail.startswith("cosine 1.0000")
    assert same.input_tokens == 12
    assert same.cost_usd == pytest.approx(12 * 0.02 / 1e6)
    near = score(scorer(fake), CASE, "The capital of France is Paris, of course")
    far = score(scorer(fake), CASE, "Bananas are rich in potassium")
    assert near.value is not None
    assert far.value is not None
    assert 0.6 < near.value < 1.0
    assert far.value < 0.5
    strict = scorer(fake, threshold=0.6)
    assert score(strict, CASE, "The capital of France is Paris, of course").value == 1.0
    failed = score(strict, CASE, "Bananas are rich in potassium")
    assert failed.value == 0.0
    assert failed.detail.endswith("< 0.6")
    assert failed.input_tokens > 0


def test_negative_cosines_score_zero_without_a_threshold() -> None:
    fake = FakeEmbedder(PRICES, vectorize=lambda t: [1.0, 0.0] if t == "a" else [-1.0, 0.1])
    s = scorer(fake)
    result = score(s, Case(id="c", input="q", expected="a"), "b")
    assert result.value == 0.0
    assert "cosine -0.99" in result.detail
    lowest = scorer(fake, threshold=-1.0)
    assert score(lowest, Case(id="c", input="q", expected="a"), "b").value == 1.0


def test_cases_that_cannot_be_scored() -> None:
    fake = FakeEmbedder(PRICES)
    s = scorer(fake)
    for case in (
        Case(id="c", input="q"),
        Case(id="c", input="q", expected="   "),
        Case(id="c", input="q", expected={"answer": "x"}),
    ):
        assert score(s, case, "anything").value is None
    keyed = scorer(fake, expected_key="answer")
    keyed_case = Case(id="c", input="q", expected={"answer": "x y"})
    assert score(keyed, keyed_case, "x y").value == pytest.approx(1.0)
    assert score(s, CASE, "  ") == score(s, CASE, "")
    assert score(s, CASE, "").value == 0.0
    assert fake.batches == [["x y"]]
    zero = FakeEmbedder(PRICES, vectorize=lambda t: [0.0, 0.0])
    assert "zero vector" in score(scorer(zero), CASE, "x").detail
    broken = FakeEmbedder(PRICES)
    broken.failures = [refused()]
    failed = score(scorer(broken), CASE, "x")
    assert (failed.value, failed.detail) == (None, "embedding call failed: bad request")
    with pytest.raises(ValueError, match="threshold"):
        scorer(fake, threshold=1.5)


def test_identity_covers_model_endpoint_and_threshold() -> None:
    fake = FakeEmbedder(PRICES)
    base = scorer(fake).identity()
    assert base["mode"] == "semantic_similarity"
    assert base["model"] == MODEL
    assert scorer(fake, threshold=0.8).identity()["fingerprint"] != base["fingerprint"]
    other = SemanticSimilarity(
        "sim", embedder=Embedder(fake, "text-embedding-3-large", endpoint="openai_compat:x")
    )
    assert other.identity()["fingerprint"] != base["fingerprint"]
    s = scorer(fake)
    assert (s.provider, s.model, s.client) == ("openai", MODEL, fake)


def registry(server: FakeOpenAI) -> ProviderRegistry:
    return ProviderRegistry(
        builtin_providers(SecretStr("sk-test"), None),
        PriceBook(),
        transport=httpx2.MockTransport(server.handle),
    )


def test_spec_builds_a_scorer_on_a_named_provider() -> None:
    adapter: Any = pytest.importorskip("pydantic").TypeAdapter(list[ScorerSpec])
    specs = adapter.validate_python(
        [{"type": "semantic_similarity", "threshold": 0.8, "batch_size": 16, "gate": False}]
    )
    (built,) = build_scorers(specs, allow_import_paths=False, providers=registry(FakeOpenAI()))
    assert isinstance(built, SemanticSimilarity)
    assert (built.name, built.threshold) == ("semantic_similarity", 0.8)
    assert built.embedder.batch_size == 16
    assert built.embedder.endpoint == "openai_compat:https://api.openai.com/v1"
    no_key = ProviderRegistry(builtin_providers(None, None), PriceBook())
    with pytest.raises(InvalidInputError, match="scorer semantic_similarity: provider openai"):
        build_scorers(specs, allow_import_paths=False, providers=no_key)
    claude = adapter.validate_python([{"type": "semantic_similarity", "provider": "anthropic"}])
    with pytest.raises(InvalidInputError, match="Claude API is not configured"):
        build_scorers(claude, allow_import_paths=False, providers=no_key)
    with pytest.raises(ValueError, match="less than or equal to 1"):
        adapter.validate_python([{"type": "semantic_similarity", "threshold": 2}])


# --- the threshold sweep --------------------------------------------------------------


def arrays(cosines: list[float], labels: list[int]) -> tuple[Any, Any]:
    return np.asarray(cosines, dtype=np.float64), np.asarray(labels, dtype=np.bool_)


def test_best_threshold_by_hand() -> None:
    perfect = best_threshold(*arrays([0.9, 0.8, 0.7, 0.6], [1, 1, 0, 0]))
    assert perfect.threshold == pytest.approx(0.75)
    assert (perfect.kappa, perfect.agreement, perfect.precision, perfect.recall) == (1, 1, 1, 1)
    assert perfect.confusion == [[2, 0], [0, 2]]
    # Tied cosines move together. Candidates: nothing passes (kappa 0), >= 0.65
    # (tp 1, fp 1, tn 1: kappa (2/3 - 4/9) / (5/9) = 0.4), everything (kappa 0).
    tied = best_threshold(*arrays([0.8, 0.5, 0.8], [1, 0, 0]))
    assert tied.threshold == pytest.approx(0.65)
    assert tied.kappa == pytest.approx(0.4)
    assert tied.confusion == [[1, 1], [0, 1]]
    # Labels against the cosines: the best is to pass nothing, just above the top.
    upside = best_threshold(*arrays([0.9, 0.1], [0, 1]))
    assert upside.threshold > 0.9
    assert upside.confusion == [[1, 0], [1, 0]]
    assert upside.precision is None
    # One label only: every threshold agrees by chance alone; the strictest wins.
    single = best_threshold(*arrays([0.3, 0.2], [1, 1]))
    assert single.threshold == pytest.approx(0.2)
    assert single.agreement == 1.0
    with pytest.raises(ValueError, match="at least one"):
        best_threshold(*arrays([], []))


@given(
    st.lists(
        st.tuples(st.integers(-20, 20).map(lambda x: x / 20), st.booleans()),
        min_size=1,
        max_size=40,
    )
)
def test_best_threshold_matches_brute_force(items: list[tuple[float, bool]]) -> None:
    cosines, labels = arrays([c for c, _ in items], [int(y) for _, y in items])
    best = best_threshold(cosines, labels)
    candidates = sorted({*cosines.tolist(), 2.0})
    brute = max(
        cohens_kappa(labels.tolist(), (cosines >= t).tolist(), [False, True]) for t in candidates
    )
    assert best.kappa == pytest.approx(brute)
    verdicts = cosines >= best.threshold
    assert cohens_kappa(labels.tolist(), verdicts.tolist(), [False, True]) == pytest.approx(
        best.kappa
    )


@given(st.lists(st.tuples(st.booleans(), st.booleans()), min_size=1, max_size=60))
def test_binary_kappa_is_cohens_kappa(pairs: list[tuple[bool, bool]]) -> None:
    tp = sum(h and m for h, m in pairs)
    fp = sum(m and not h for h, m in pairs)
    fn = sum(h and not m for h, m in pairs)
    tn = len(pairs) - tp - fp - fn
    expected = cohens_kappa([h for h, _ in pairs], [m for _, m in pairs], [False, True])
    assert binary_kappa(tp, fp, fn, tn) == pytest.approx(expected)


def test_binary_kappa_needs_items() -> None:
    with pytest.raises(ValueError, match="no items"):
        binary_kappa(0, 0, 0, 0)


def test_cross_validation_is_out_of_sample_and_order_free() -> None:
    rng = np.random.default_rng(7)
    labels = rng.random(60) < 0.5
    # Separable with a margin: every fold's threshold lands in the gap.
    cosines = np.where(labels, 0.8, 0.4) + rng.normal(0, 0.03, 60)
    ids = [f"i{i}" for i in range(60)]
    cv = cross_validate(ids, cosines, labels, 5)
    assert cv is not None
    assert cv.kappa == 1.0
    # Noise only: in-sample kappa is flattered by the fitted threshold; cross-validated is not.
    noise = rng.random(60)
    fitted = best_threshold(noise, labels).kappa
    honest = cross_validate(ids, noise, labels, 5)
    assert honest is not None
    assert honest.kappa < fitted
    shuffled = rng.permutation(60)
    again = cross_validate([ids[i] for i in shuffled], noise[shuffled], labels[shuffled], 5)
    assert again is not None
    assert again.confusion == honest.confusion
    assert cross_validate(ids[:9], noise[:9], labels[:9], 5) is None
    assert cross_validate(ids, noise, labels, 1) is None


LABELS = [
    ("good-1", "The capital of France is Paris", 1),
    ("good-2", "Paris is the capital city of France", 1),
    ("good-3", "paris, the capital of france", 1),
    ("bad-1", "Bananas are rich in potassium", 0),
    ("bad-2", "I don't know", 0),
    ("bad-3", "Berlin is the capital of Germany", 0),
]


def labelled() -> list[LabelledItem]:
    return [
        LabelledItem(id=i, input="q", expected=CASE.expected, output=o, label=y)
        for i, o, y in LABELS
    ]


def test_calibrate_threshold_separates_the_labels() -> None:
    fake = FakeEmbedder(PRICES)
    s = scorer(fake)
    report = asyncio.run(calibrate_threshold(s, labelled(), folds=3))
    # Seven distinct texts, the reference among them once, in one request.
    assert [len(b) for b in fake.batches] == [7]
    assert fake.texts.count(str(CASE.expected)) == 1
    assert (report.n, report.scored, report.positives) == (6, 6, 3)
    assert report.best.kappa == 1.0
    assert report.disagreements == []
    cosines = {o: asyncio.run(s.similarity(CASE, o))[0] for _, o, _ in LABELS}
    passing = [c for o, c in cosines.items() if c is not None and o.lower().count("paris")]
    failing = [c for o, c in cosines.items() if c is not None and "paris" not in o.lower()]
    assert max(failing) < report.best.threshold <= min(passing)
    assert report.cv_kappa is not None
    assert report.folds == 3
    fewer = asyncio.run(calibrate_threshold(s, labelled(), folds=4))
    assert (fewer.cv_kappa, fewer.folds) == (None, 0)  # fewer than two items per fold
    assert report.sweep
    assert all(-1 <= p.threshold <= 1 for p in report.sweep)
    assert report.input_tokens > 0


def test_calibrate_threshold_leaves_out_directionless_vectors() -> None:
    fake = FakeEmbedder(PRICES, vectorize=lambda t: [0.0, 0.0] if t == "blank" else [1.0, 0.0])
    items = [
        *labelled()[:2],
        LabelledItem(id="z", input="q", expected="r", output="blank", label=0),
    ]
    report = asyncio.run(calibrate_threshold(scorer(fake), items))
    assert (report.n, report.scored) == (3, 2)


def test_calibrate_threshold_needs_references() -> None:
    items = [LabelledItem(id="x", input="q", output="o", label=1)]
    with pytest.raises(InvalidInputError, match="could be scored"):
        asyncio.run(calibrate_threshold(scorer(FakeEmbedder(PRICES)), items))


def test_binary_labels_are_0_or_1() -> None:
    line = json.dumps({"id": "a", "input": "q", "expected": "r", "output": "o", "label": 1})
    assert parse_labels(line, "binary")[0].label == 1
    for bad in (2, "pass", True):
        text = json.dumps({"id": "a", "input": "q", "output": "o", "label": bad})
        with pytest.raises(InvalidInputError):
            parse_labels(text, "binary")


# --- the command ----------------------------------------------------------------------


def patch_openai(monkeypatch: pytest.MonkeyPatch, server: FakeOpenAI) -> None:
    monkeypatch.setattr(
        cli,
        "ProviderRegistry",
        functools.partial(ProviderRegistry, transport=httpx2.MockTransport(server.handle)),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


def run_cli(db: str, *args: str) -> tuple[int, str]:
    out = io.StringIO()
    return cli.main(["--database-url", db, *args], out=out), out.getvalue()


def test_calibrate_command_sweeps_and_stores_the_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = FakeOpenAI(embed=bag_of_words)
    patch_openai(monkeypatch, server)
    db = f"sqlite:///{tmp_path / 'cli.db'}"
    spec = tmp_path / "sim.toml"
    spec.write_text('[scorer]\nmodel = "text-embedding-3-small"\nbatch_size = 4\n')
    lines = [
        json.dumps(
            {"id": f"{i}-{n}", "input": "q", "expected": CASE.expected, "output": o, "label": y}
        )
        for n in range(4)
        for i, o, y in LABELS
    ]
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(lines) + "\n")
    code, text = run_cli(db, "calibrate", "--scorer", str(spec), str(labels))
    assert code == 0, text
    assert "labelled 24  scored 24  labelled 1: 12" in text
    assert "Cohen's kappa 1.000" in text
    assert "cross-validated (5 folds): kappa 1.000" in text
    assert "set threshold = " in text
    assert server.rejected == []
    # Seven distinct texts in batches of four: two requests.
    assert [len(b["input"]) for b in server.bodies] == [4, 3]
    code, raw = run_cli(db, "calibrate", "--scorer", str(spec), str(labels), "--json")
    assert code == 0
    report = json.loads(raw)
    assert report["best"]["kappa"] == 1.0
    assert len(server.requests) == 2  # the rerun came from the embedding cache
    engine_rows = _rows(db)
    assert [(r.kappa, r.confusion["folds"], r.judge_spec["model"]) for r in engine_rows] == [
        (1.0, 5, MODEL)
    ] * 2
    assert engine_rows[0].confusion["threshold"] == report["best"]["threshold"]
    code, text = run_cli(db, "calibrate", "--scorer", str(spec), str(labels), "--folds", "0")
    assert "cross-validated: too few items" in text


def _rows(db: str) -> list[Calibration]:
    from evalkit.db import make_engine, make_session_factory

    engine = make_engine(db)
    try:
        with make_session_factory(engine)() as session:
            rows = list(session.scalars(select(Calibration).order_by(Calibration.id)))
            assert session.scalars(select(CachedEmbedding)).all()
            return rows
    finally:
        engine.dispose()


def test_calibrate_command_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeOpenAI(embed=bag_of_words)
    patch_openai(monkeypatch, server)
    db = f"sqlite:///{tmp_path / 'cli.db'}"
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps({"id": "a", "input": "q", "expected": "r", "output": "o", "label": 1}) + "\n"
    )
    spec = tmp_path / "sim.toml"
    for content in ("[other]\n", "[scorer]\nthreshold = 3\n", '[scorer]\nmodel = "nope"\n'):
        spec.write_text(content)
        assert run_cli(db, "calibrate", "--scorer", str(spec), str(labels))[0] == 2, content
    spec.write_text("[scorer]\n")
    judge = tmp_path / "judge.toml"
    judge.write_text('[judge]\nrubric = "r"\n')
    with pytest.raises(SystemExit):
        run_cli(db, "calibrate", "--scorer", str(spec), "--judge", str(judge), str(labels))
    server.models = {"another-model"}
    assert run_cli(db, "calibrate", "--scorer", str(spec), str(labels))[0] == 3


# --- runs -----------------------------------------------------------------------------

ANSWERS = [
    "Paris is the capital of France",
    "Water boils at one hundred degrees",
    "The Pacific is the largest ocean",
    "Light travels faster than sound",
]


def run_config(label: str, accuracy: float, threshold: float = 0.6) -> RunConfig:
    return RunConfig.model_validate(
        {
            "label": label,
            "dataset": {"name": "facts"},
            "target": {"type": "fake", "accuracy": accuracy, "seed": 3},
            "scorers": [{"type": "semantic_similarity", "name": "sim", "threshold": threshold}],
        }
    )


def test_runs_batch_cache_and_gate_on_similarity(sessions: sessionmaker[Session]) -> None:
    with sessions() as session:
        import_dataset(
            session,
            "facts",
            [Case(id=f"f{i}", input=f"q{i}", expected=ANSWERS[i % 4]) for i in range(40)],
        )
    server = FakeOpenAI(embed=bag_of_words)
    runner = Runner(sessions, Settings(), providers=registry(server))
    base = runner.create_run(run_config("base", 1.0))
    report = asyncio.run(runner.execute(base.id))
    assert report.executed == 40
    sent = [t for body in server.bodies for t in body["input"]]
    # Four distinct references, each also the output: four texts in all, each sent once.
    assert sorted(sent) == sorted(ANSWERS)
    assert server.rejected == []
    cand = runner.create_run(run_config("cand", 0.3))
    asyncio.run(runner.execute(cand.id))
    # Only the wrong answer was new.
    assert [t for body in server.bodies for t in body["input"]][len(sent) :] == ["I don't know."]
    with sessions() as session:
        result = compare_runs(session, base.id, cand.id)
        (metric,) = result.comparisons
        assert metric.metric == "sim"
        assert metric.baseline_mean == 1.0
        assert metric.verdict == "worse"
        assert result.regressions == ["sim"]
        summary = cli._format_summary(summarize(session, cand.id))
        assert (
            "embeddings sim: text-embedding-3-small on openai "
            "(openai_compat:https://api.openai.com/v1)\n"
        ) in summary
        judged = session.get_one(type(base), cand.id).judges
        assert judged is not None
        assert judged["sim"]["mode"] == "semantic_similarity"
        prices = session.get_one(type(base), cand.id).prices
        assert prices is not None
        assert MODEL in json.dumps(prices)
    stricter = runner.create_run(run_config("stricter", 1.0, threshold=0.9))
    asyncio.run(runner.execute(stricter.id))
    with sessions() as session, pytest.raises(PreconditionError, match="sim: rubric changed"):
        compare_runs(session, base.id, stricter.id)
    requests = len(server.requests)
    rerun = runner.create_run(run_config("rerun", 1.0))
    asyncio.run(runner.execute(rerun.id))
    assert len(server.requests) == requests  # outputs and vectors both cached


def test_usage_reaches_the_result_rows(sessions: sessionmaker[Session]) -> None:
    with sessions() as session:
        import_dataset(session, "facts", [Case(id="f0", input="q", expected=ANSWERS[0])])
    server = FakeOpenAI(embed=bag_of_words)
    runner = Runner(sessions, Settings(), providers=registry(server))
    run = runner.create_run(run_config("one", 1.0))
    asyncio.run(runner.execute(run.id))
    with sessions() as session:
        from evalkit.db.models import Result

        (row,) = session.scalars(select(Result)).all()
        assert row.judge_input_tokens == 6
        assert row.judge_cost_usd == pytest.approx(6 * 0.02 / 1e6)


def test_embeddings_reply_type_is_the_provider_protocol() -> None:
    fake = FakeEmbedder(PRICES)
    fake.reply = lambda texts: Embeddings([[1.0]] * len(texts), Usage(), MODEL)
    assert score(scorer(fake), CASE, "x").value == 1.0
