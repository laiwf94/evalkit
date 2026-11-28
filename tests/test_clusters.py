"""Failure clustering: spherical k-means against planted groups and a brute-force
silhouette, choosing k, distinctive words and tag names, cluster naming
replies, and runs whose failures are planted in three topics. One topic only
fails in the candidate, one fails in both, and one fails less in the
candidate, and the report must find exactly that, name the clusters, reuse
its caches on a rerun, and save the clusters as dataset tags."""

import asyncio
import functools
import io
import json
import math
import re
from pathlib import Path
from typing import Any

import httpx2
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import SecretStr, ValidationError
from sqlalchemy.orm import Session, sessionmaker

from evalkit import cli
from evalkit import clusters as clusters_module
from evalkit.cache import VerdictStore
from evalkit.clusters import (
    MIN_SILHOUETTE,
    ClusterSpec,
    Namer,
    NamerSpec,
    _change,
    case_text,
    choose_k,
    cluster_failures,
    distinctive_terms,
    fit_clusters,
    format_clusters,
    format_tags,
    kmeans,
    lloyd,
    name_prompt,
    normalize,
    ordered,
    parse_name,
    save_tags,
    silhouette,
    slug,
    unique_tags,
)
from evalkit.clusters import from_spec as clusters_from_spec
from evalkit.config import Settings
from evalkit.dataset_edits import CaseUpdate, DatasetEdit, apply_edit
from evalkit.datasets import MAX_TAGS, Case, get_dataset, import_dataset, load_cases
from evalkit.embeddings import Embedder
from evalkit.errors import ConflictError, InvalidInputError, NotFoundError
from evalkit.pricing import PriceBook
from evalkit.providers.base import Completion
from evalkit.providers.registry import ProviderRegistry, builtin_providers
from evalkit.ratelimit import RetryPolicy
from evalkit.runner import Runner
from evalkit.runspec import RunConfig
from evalkit.targets.base import TargetError, Usage
from evalkit.targets.fake import unit
from tests.fake_embeddings import bag_of_words
from tests.fake_openai import FakeOpenAI, error_response

RETRY = RetryPolicy(max_attempts=1)
NAMER_MODEL = "gpt-5-mini"


# k-means ------------------------------------------------------------------------


def blobs(k: int, n: int, d: int, seed: int, spread: float = 0.3) -> tuple[Any, Any]:
    rng = np.random.default_rng(seed)
    centers = normalize(rng.normal(size=(k, d)))
    labels = rng.integers(k, size=n)
    noise = rng.normal(scale=spread / math.sqrt(d), size=(n, d))
    return normalize(centers[labels] + noise), labels


def same_partition(a: Any, b: Any) -> bool:
    """The two labellings group the items identically, whatever the numbering."""
    pairs = set(zip(np.asarray(a).tolist(), np.asarray(b).tolist(), strict=True))
    return len(pairs) == len(set(np.asarray(a).tolist())) == len(set(np.asarray(b).tolist()))


def test_kmeans_recovers_planted_groups_and_is_seeded() -> None:
    x, truth = blobs(5, 600, 32, seed=3)
    fit = kmeans(x, 5, seed=0)
    assert same_partition(fit.labels, truth)
    again = kmeans(x, 5, seed=0)
    assert np.array_equal(fit.labels, again.labels)
    # Inertia is the summed cosine distance to the centres.
    expected = float((1 - np.einsum("ij,ij->i", x, fit.centers[fit.labels])).sum())
    assert fit.inertia == pytest.approx(expected, rel=1e-5)
    assert np.allclose(np.linalg.norm(fit.centers, axis=1), 1.0, atol=1e-5)


def test_kmeans_with_fewer_distinct_cases_than_k() -> None:
    x = normalize(np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.float32))
    fit = kmeans(x, 4, seed=1)
    assert fit.centers.shape[0] == 2
    assert fit.labels.tolist() in ([0, 0, 1, 1, 0], [1, 1, 0, 0, 1])
    one = kmeans(normalize(np.ones((6, 3))), 3)
    assert one.centers.shape[0] == 1
    assert one.inertia == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError, match="nothing to cluster"):
        kmeans(np.zeros((0, 3), dtype=np.float32), 2)
    with pytest.raises(ValueError, match="at least 1"):
        kmeans(x, 0)
    with pytest.raises(ValueError, match="matrix"):
        normalize(np.ones(3))


def test_an_emptied_cluster_is_reseeded_with_the_farthest_case() -> None:
    x = normalize(np.array([[1, 0.1], [1, -0.1], [1, 0.05], [0.1, 1], [-0.1, 1]], np.float32))
    # The third centre is nearest to nothing, so its cluster starts empty.
    centers = normalize(np.array([[1, 0], [0, 1], [-1, -1]], dtype=np.float32))
    fit = lloyd(x, centers)
    assert len(set(fit.labels.tolist())) == 3
    assert fit.iterations >= 2


def brute_silhouette(x: Any, labels: list[int]) -> float:
    n = len(labels)
    total = 0.0
    for i in range(n):
        own = [j for j in range(n) if labels[j] == labels[i] and j != i]
        if not own:
            continue
        d = [1 - float(np.dot(x[i], x[j])) for j in range(n)]
        a = sum(d[j] for j in own) / len(own)
        b = min(
            sum(d[j] for j in range(n) if labels[j] == c) / labels.count(c)
            for c in set(labels)
            if c != labels[i]
        )
        total += (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return total / n


@settings(max_examples=60, deadline=None)
@given(
    st.integers(4, 14).flatmap(
        lambda n: st.tuples(
            st.lists(
                st.lists(st.floats(-1, 1, allow_nan=False), min_size=3, max_size=3),
                min_size=n,
                max_size=n,
            ),
            st.lists(st.integers(0, 2), min_size=n, max_size=n),
        )
    )
)
def test_silhouette_matches_brute_force(data: tuple[list[list[float]], list[int]]) -> None:
    raw, labels = data
    x = normalize(np.asarray(raw, dtype=np.float32) + np.array([0.01, 0, 0], np.float32))
    got = silhouette(x, np.asarray(labels, dtype=np.intp))
    if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
        assert got == 0.0
    else:
        assert got == pytest.approx(brute_silhouette(x, labels), abs=1e-4)


def test_choose_k() -> None:
    x, _ = blobs(4, 400, 32, seed=7)
    choice = choose_k(x, 10, seed=0)
    assert (choice.k, choice.by) == (4, "silhouette")
    assert set(choice.tried) == set(range(2, 11))
    assert choice.tried[4] == max(choice.tried.values())
    # Points spread evenly over the sphere have no groups to find.
    noise = normalize(np.random.default_rng(1).normal(size=(300, 64)))
    flat = choose_k(noise, 8)
    assert flat.by == "single"
    assert flat.k == 1
    assert flat.silhouette is not None
    assert flat.silhouette < MIN_SILHOUETTE
    assert choose_k(normalize(np.eye(2)), 5).by == "few"
    assert choose_k(normalize(np.ones((5, 2))), 5).k == 1


def test_large_runs_choose_k_on_a_sample() -> None:
    x, truth = blobs(3, 4500, 16, seed=2)
    fit = fit_clusters(list(x), ClusterSpec())
    assert fit.choice.k == 3
    assert same_partition(fit.labels, truth)
    given_k = fit_clusters(list(x), ClusterSpec(k=3))
    assert given_k.choice.silhouette == pytest.approx(fit.choice.silhouette, abs=1e-6)


def test_lloyd_stops_at_max_iter_and_changes_are_named() -> None:
    x, _ = blobs(3, 200, 8, seed=9, spread=2.0)
    fit = lloyd(x, x[:3].copy(), max_iter=1)
    assert fit.iterations == 1
    assert [_change(0, 3), _change(3, 0), _change(2, 5), _change(5, 2), _change(4, 4)] == [
        "new",
        "resolved",
        "grew",
        "shrank",
        "same",
    ]


def test_clusters_are_numbered_by_size() -> None:
    x, _ = blobs(3, 300, 16, seed=5)
    fit = kmeans(x, 3)
    labels, centers = ordered(fit)
    sizes = np.bincount(labels).tolist()
    assert sizes == sorted(sizes, reverse=True)
    assert np.allclose(centers[labels[0]], fit.centers[fit.labels[0]])
    spec = ClusterSpec(k=3)
    given_k = fit_clusters(list(x), spec)
    assert given_k.choice.by == "given"
    assert given_k.choice.silhouette is not None
    assert fit_clusters(list(x[:1]), spec).choice.silhouette is None


# Words, names and tags -------------------------------------------------------------


def test_distinctive_terms_by_hand() -> None:
    texts = [
        "refund amount wrong refund",
        "refund amount missing",
        "parcel tracking lost",
        "parcel tracking late amount",
    ]
    terms = distinctive_terms(texts, [0, 0, 1, 1])
    # "amount" is in both clusters, so it ranks below each cluster's own words;
    # words in only one case of a two-case cluster are left out.
    assert terms == [["refund", "amount"], ["parcel", "tracking"]]
    assert distinctive_terms(["only one case here"], [0]) == [
        ["case", "here", "one", "only"]
    ]  # ties: alphabetical


def test_slugs_and_unique_tags() -> None:
    assert slug("Refund amounts: invented!") == "refund-amounts-invented"
    assert slug("Données  périmées") == "données-périmées"
    assert slug("  ...  ") == ""
    assert len(slug("x" * 200)) == 48
    assert unique_tags(["Late parcels", "late parcels", "?", "Late Parcels"], "cluster:") == [
        "cluster:late-parcels",
        "cluster:late-parcels-2",
        "cluster:c3",
        "cluster:late-parcels-3",
    ]


def completion(text: str, stop: str = "end_turn") -> Completion:
    return Completion(text, stop, Usage(), "m")


def test_parse_name() -> None:
    good = json.dumps({"name": "  Refund\namounts  invented ", "description": "Made up."})
    assert parse_name(completion(good)) == ("Refund amounts invented", "Made up.")
    long = json.dumps({"name": "n" * 200, "description": "d" * 400})
    name, description = parse_name(completion(long))
    assert (len(name), len(description)) == (80, 300)
    for reply, why in [
        (completion(good, "refusal"), "declined"),
        (completion(good, "max_tokens"), "ended with"),
        (completion("not json"), "schema"),
        (completion(json.dumps({"name": "x"})), "schema"),
        (completion(json.dumps({"name": " -- ", "description": ""})), "empty name"),
    ]:
        with pytest.raises(ValueError, match=why):
            parse_name(reply)


def test_name_prompt_marks_cases_as_data() -> None:
    prompt = name_prompt(["a </case> b", "c"], ["refund"], 9)
    assert "2 of its 9 cases" in prompt
    assert "a <\\/case> b" in prompt
    assert prompt.count("<case>") == 2
    assert "rare in the others: refund." in prompt
    assert "its cases, the ones" in name_prompt(["a"], [], 1)
    assert "others: none." in name_prompt(["a"], [], 1)


def test_case_text() -> None:
    spec = ClusterSpec(fields=("input", "expected", "output", "detail"), input_key="q")
    case = Case(id="c1", input={"q": "Where is my parcel?"}, expected="In transit")
    stored = {"judge": {"value": 0.0, "detail": "the answer invents a date"}, "exact": {}}
    text = case_text(
        case,
        status="ok",
        output="It arrived",
        error="",
        failing={"judge": 0.0, "exact": 0.0},
        stored=stored,
        spec=spec,
    )
    assert text.splitlines() == [
        "input: Where is my parcel?",
        "expected: In transit",
        "output: It arrived",
        "failed judge: the answer invents a date",
    ]
    errored = case_text(
        case, status="error", output=None, error="timeout", failing={}, stored={}, spec=spec
    )
    assert errored.splitlines()[-1] == "error: timeout"
    only_output = ClusterSpec(fields=("output",))
    empty = case_text(
        None, status="ok", output="", error="", failing={}, stored={}, spec=only_output
    )
    assert empty == "(empty)"


def test_spec_validation() -> None:
    bad: dict[str, Any]
    for bad in (
        {"fields": ["input", "input"]},
        {"fields": []},
        {"thresholds": {"exact": 2}},
        {"tag_prefix": "p" * 20},
        {"k": 0},
        {"max_k": 1},
        {"unknown": 1},
        {"namer": {"examples": 0}},
    ):
        with pytest.raises(ValidationError):
            ClusterSpec.model_validate(bad)
    assert ClusterSpec.model_validate({"tag_prefix": "fc:"}).tag_prefix == "fc:"


# Planted runs -------------------------------------------------------------------

ACCURACY = 0.5
BASE_SEED, CAND_SEED = 21, 22
TOPICS = {
    "refund": "refund amount charged card invoice money back overcharged credit",
    "shipping": "parcel courier tracking delivery shipment arrived warehouse postage",
    "password": "password login reset account locked sign email verification",
}


def fails(seed: int, case_id: str) -> bool:
    return unit(seed, case_id) >= ACCURACY


def pick(want_base: bool, want_cand: bool, count: int, prefix: str) -> list[str]:
    out: list[str] = []
    n = 0
    while len(out) < count:
        case_id = f"{prefix}-{n:04d}"
        n += 1
        if fails(BASE_SEED, case_id) == want_base and fails(CAND_SEED, case_id) == want_cand:
            out.append(case_id)
    return out


PLAN = {
    # topic: (fail in both, candidate only, baseline only, neither)
    "refund": pick(False, True, 15, "refund"),
    "shipping": pick(True, True, 12, "shipping"),
    "password": [*pick(True, True, 10, "password"), *pick(True, False, 8, "pwfix")],
}
PASSING = pick(False, False, 20, "ok")


def question(topic: str, case_id: str) -> str:
    words = TOPICS[topic].split()
    rng = np.random.default_rng(int(unit(case_id) * 2**32))
    chosen = rng.choice(words, size=6, replace=False)
    return "Help with " + " ".join(chosen.tolist())


def planted_cases() -> list[Case]:
    cases = []
    for topic, ids in PLAN.items():
        for case_id in ids:
            cases.append(
                Case(id=case_id, input={"q": question(topic, case_id)}, expected=f"fix {case_id}")
            )
    for n, case_id in enumerate(PASSING):
        topic = list(TOPICS)[n % 3]
        cases.append(
            Case(id=case_id, input={"q": question(topic, case_id)}, expected=f"fix {case_id}")
        )
    return cases


def run_config(seed: int) -> RunConfig:
    return RunConfig.model_validate(
        {
            "label": f"seed{seed}",
            "dataset": {"name": "support"},
            "target": {"type": "fake", "accuracy": ACCURACY, "seed": seed},
            "scorers": [{"type": "exact"}],
        }
    )


def topic_of(text: str) -> str:
    counts = {t: sum(w in text for w in words.split()) for t, words in TOPICS.items()}
    return max(counts, key=lambda t: counts[t])


NAMES = {
    "refund": ("Refund amounts wrong", "Answers get the refund amount wrong."),
    "shipping": ("Parcel tracking unknown", "Answers cannot say where the parcel is."),
    "password": ("Login help missing", "Answers do not help with locked accounts."),
}


class Server(FakeOpenAI):
    """Embeddings by bag of words; cluster names by the topic its cases are about."""

    def __init__(self) -> None:
        super().__init__(embed=functools.partial(bag_of_words, dims=256), answer=self.name)
        self.broken: set[str] = set()

    def name(self, body: dict[str, Any]) -> str:
        prompt = body["messages"][-1]["content"]
        topic = topic_of(prompt)
        if topic in self.broken:
            return "not json"
        name, description = NAMES[topic]
        return json.dumps({"name": name, "description": description})

    @property
    def embedding_requests(self) -> int:
        return sum(1 for r in self.requests if r.url.path.endswith("/embeddings"))

    @property
    def chat_requests(self) -> int:
        return sum(1 for r in self.requests if r.url.path.endswith("/chat/completions"))


def registry(server: FakeOpenAI) -> ProviderRegistry:
    return ProviderRegistry(
        builtin_providers(SecretStr("sk-test"), None),
        PriceBook(),
        transport=httpx2.MockTransport(server.handle),
    )


SPEC = ClusterSpec.model_validate(
    {
        "fields": ["input", "output"],
        "input_key": "q",
        "namer": {"provider": "openai", "model": NAMER_MODEL},
    }
)


@pytest.fixture
def runs(sessions: sessionmaker[Session]) -> tuple[str, str]:
    with sessions() as s:
        import_dataset(s, "support", planted_cases())
    runner = Runner(sessions, Settings.from_env({}))
    ids = []
    for seed in (BASE_SEED, CAND_SEED):
        run = runner.create_run(run_config(seed))
        asyncio.run(runner.execute(run.id))
        ids.append(run.id)
    return ids[0], ids[1]


def cluster(
    sessions: sessionmaker[Session],
    server: Server,
    run_id: str,
    spec: ClusterSpec = SPEC,
    **kw: Any,
) -> Any:
    embedder, namer = clusters_from_spec(
        spec, registry(server), sessions, cache=VerdictStore(sessions), retry=RETRY
    )
    return asyncio.run(
        cluster_failures(sessions, run_id, spec=spec, embedder=embedder, namer=namer, **kw)
    )


def test_planted_clusters_and_their_diff(
    sessions: sessionmaker[Session], runs: tuple[str, str]
) -> None:
    base, cand = runs
    server = Server()
    report = cluster(sessions, server, cand, baseline_id=base)
    assert server.rejected == []
    assert (report.cases, report.failing, report.baseline_failing) == (65, 37, 30)
    assert report.clustered == 67
    assert (report.k, report.chosen_by) == (3, "silhouette")
    assert report.silhouette is not None
    assert report.silhouette > MIN_SILHOUETTE
    by_tag = {c.tag: c for c in report.clusters}
    assert set(by_tag) == {
        "cluster:refund-amounts-wrong",
        "cluster:parcel-tracking-unknown",
        "cluster:login-help-missing",
    }
    refund = by_tag["cluster:refund-amounts-wrong"]
    shipping = by_tag["cluster:parcel-tracking-unknown"]
    password = by_tag["cluster:login-help-missing"]
    # Every member is from the planted topic, and every planted failure is found.
    assert {m.case_id for m in refund.members} == set(PLAN["refund"])
    assert {m.case_id for m in shipping.members} == set(PLAN["shipping"])
    assert {m.case_id for m in password.members} == set(PLAN["password"])
    assert (refund.baseline, refund.candidate, refund.change) == (0, 15, "new")
    assert (shipping.baseline, shipping.candidate, shipping.change) == (12, 12, "same")
    assert (password.baseline, password.candidate, password.change) == (18, 10, "shrank")
    assert refund.new_failures == sorted(PLAN["refund"])
    assert password.fixed == sorted(PLAN["password"][10:])
    assert shipping.new_failures == shipping.fixed == []
    assert [c.index for c in report.clusters] == [1, 2, 3]
    assert report.clusters[0] is password  # the largest, with both runs' members
    assert refund.named_by == "model"
    assert refund.description == NAMES["refund"][1]
    assert "refund" in refund.terms or "amount" in refund.terms
    assert len(refund.typical) == 5
    # A case failing in both runs is shown to the namer once.
    assert all(len(set(c.typical)) == len(c.typical) == 5 for c in report.clusters)
    sims = [m.similarity for m in refund.members]
    assert sims == sorted(sims, reverse=True)
    assert 0 < refund.cohesion <= 1
    # A case that fails in both runs gives one text, so 45 distinct texts go in
    # one request of up to 64, and there is one naming call per cluster.
    assert (server.embedding_requests, server.chat_requests) == (1, 3)
    assert (report.embedded, report.cached_texts, report.name_calls) == (45, 0, 3)
    assert report.cost_usd > 0
    text = format_clusters(report, limit=3)
    assert "37 of 65 cases failing (baseline 30)" in text
    assert "3 clusters (k by silhouette, silhouette " in text
    assert (
        "Refund amounts wrong  [cluster:refund-amounts-wrong]  15 failing, baseline 0, new" in text
    )
    assert "and 12 more" in text
    assert "   fixed: " in text
    # A rerun is served from the embedding and naming caches.
    again = cluster(sessions, server, cand, baseline_id=base)
    assert (server.embedding_requests, server.chat_requests) == (1, 3)
    assert (again.embedded, again.cached_texts, again.cached_name_calls) == (0, 45, 3)
    assert [c.tag for c in again.clusters] == [c.tag for c in report.clusters]
    assert again.cost_usd == 0


def test_without_a_baseline_and_without_names(
    sessions: sessionmaker[Session], runs: tuple[str, str]
) -> None:
    _, cand = runs
    server = Server()
    spec = SPEC.model_copy(update={"names": False})
    report = cluster(sessions, server, cand, spec=spec)
    assert server.chat_requests == 0
    assert report.namer is None
    assert report.k == 3
    assert all(
        c.named_by == "terms" and c.baseline is None and c.change is None for c in report.clusters
    )
    assert all(c.tag.startswith("cluster:") for c in report.clusters)
    assert {topic_of(" ".join(c.terms)) for c in report.clusters} == set(TOPICS)
    text = format_clusters(report)
    assert "naming calls" not in text
    given = cluster(sessions, server, cand, spec=spec.model_copy(update={"k": 2}))
    assert (given.k, given.chosen_by) == (2, "given")
    assert "2 clusters (k given, silhouette" in format_clusters(given)


def test_a_cluster_the_model_cannot_name_is_named_by_its_words(
    sessions: sessionmaker[Session], runs: tuple[str, str]
) -> None:
    base, cand = runs
    server = Server()
    server.broken = {"shipping"}
    report = cluster(sessions, server, cand, baseline_id=base)
    shipping = next(c for c in report.clusters if c.members[0].case_id.startswith("shipping"))
    assert shipping.named_by == "terms"
    assert shipping.tag == "cluster:" + slug(" ".join(shipping.terms[:3]))
    assert any("does not match the schema; named by its words" in n for n in report.notes)
    # Failed calls are noted too, and the clusters still stand.
    failing = Server()
    failing.answer = lambda body: error_response(400, "bad request")
    report = cluster(
        sessions,
        failing,
        cand,
        spec=SPEC.model_copy(update={"namer": NamerSpec(provider="openai", model="gpt-5-nano")}),
    )
    assert report.k == 3
    assert sum("naming call failed" in n for n in report.notes) == 3
    assert report.name_calls == 0


def test_errors_and_limits(sessions: sessionmaker[Session], runs: tuple[str, str]) -> None:
    base, cand = runs
    server = Server()
    with pytest.raises(NotFoundError):
        cluster(sessions, server, "nope")
    with pytest.raises(NotFoundError):
        cluster(sessions, server, cand, baseline_id="nope")
    with pytest.raises(InvalidInputError, match="the run itself"):
        cluster(sessions, server, cand, baseline_id=cand)
    with pytest.raises(InvalidInputError, match="no metric named judge"):
        cluster(sessions, server, cand, spec=SPEC.model_copy(update={"metrics": ("judge",)}))
    # A threshold of 0 fails nothing.
    none = cluster(sessions, server, cand, spec=SPEC.model_copy(update={"threshold": 0.0}))
    assert (none.k, none.clustered) == (0, 0)
    assert none.notes == ["no case failed: nothing to cluster"]
    assert "nothing to cluster" in format_clusters(none)
    # max_cases keeps the same cases from both runs.
    few = cluster(
        sessions, server, cand, baseline_id=base, spec=SPEC.model_copy(update={"max_cases": 30})
    )
    assert few.left_out == 35
    assert few.cases == 30
    members = [m for c in few.clusters for m in c.members]
    cand_ids = {m.case_id for m in members if m.side == "candidate"}
    base_ids = {m.case_id for m in members if m.side == "baseline"}
    both = set(PLAN["shipping"]) | set(PLAN["password"][:10])
    assert cand_ids & both == base_ids & both
    assert any("35 cases were left out" in n for n in few.notes)
    # Another dataset cannot be a baseline.
    with sessions() as s:
        import_dataset(s, "other", planted_cases()[:10])
    other = Runner(sessions, Settings.from_env({})).create_run(
        RunConfig.model_validate({**run_config(1).model_dump(), "dataset": {"name": "other"}})
    )
    asyncio.run(Runner(sessions, Settings.from_env({})).execute(other.id))
    with pytest.raises(InvalidInputError, match="one dataset"):
        cluster(sessions, server, cand, baseline_id=other.id)

    # A failed embedding request is raised to the caller.
    def fail(request: httpx2.Request) -> httpx2.Response:
        return error_response(400, "bad request")

    broken_registry = ProviderRegistry(
        builtin_providers(SecretStr("sk-test"), None),
        PriceBook(),
        transport=httpx2.MockTransport(fail),
    )
    embedder = Embedder(broken_registry.get("openai"), "text-embedding-3-small", retry=RETRY)
    with pytest.raises(TargetError):
        asyncio.run(cluster_failures(sessions, cand, spec=SPEC, embedder=embedder))


def test_namer_identity_keys_the_cache() -> None:
    server = Server()
    client = registry(server).get("openai")
    a = Namer(NamerSpec(provider="openai", model=NAMER_MODEL), client)
    b = Namer(NamerSpec(provider="openai", model=NAMER_MODEL, effort="high"), client)
    assert a.identity()["fingerprint"] != b.identity()["fingerprint"]
    with pytest.raises(Exception, match="price"):
        Namer(NamerSpec(provider="openai", model="no-such-model"), client)
    spec = ClusterSpec.model_validate({"embed": {"provider": "nowhere"}})
    with pytest.raises(InvalidInputError, match=r"\[clusters.embed\]"):
        clusters_from_spec(spec, registry(server), None)  # type: ignore[arg-type]
    spec = ClusterSpec.model_validate({"namer": {"provider": "nowhere"}})
    with pytest.raises(InvalidInputError, match=r"\[clusters.namer\]"):
        clusters_from_spec(spec, registry(server), None)  # type: ignore[arg-type]


# Tags ---------------------------------------------------------------------------


def test_save_tags_replaces_earlier_cluster_tags(
    sessions: sessionmaker[Session], runs: tuple[str, str]
) -> None:
    base, cand = runs
    server = Server()
    report = cluster(sessions, server, cand, baseline_id=base)
    with sessions() as s:
        result = save_tags(s, report)
        assert (result.version, result.created, result.tagged, result.cleared) == (2, True, 37, 0)
        tagged = {c.id: c.tags for c in load_cases(s, get_dataset(s, "support"))}
        assert all(tagged[i] == ("cluster:refund-amounts-wrong",) for i in PLAN["refund"])
        # Baseline-only failures are not tagged: the tags describe the candidate.
        assert all(tagged[i] == () for i in PLAN["password"][10:])
        assert get_dataset(s, "support").note == f"failure clusters of run {cand}"
        # Saving the same clustering again changes nothing.
        same = save_tags(s, report)
        assert (same.version, same.created, same.tagged, same.cleared) == (2, False, 0, 0)
    # The baseline's clustering takes the candidate's tags off the cases that pass there.
    report_base = cluster(sessions, server, base)
    with sessions() as s:
        result = save_tags(s, report_base)
        # Shipping and the ten password cases keep the tags they have; the eight
        # cases only the baseline fails are tagged, and refund's 15 are cleared.
        assert (result.version, result.tagged, result.cleared) == (3, 8, 15)
        tagged = {c.id: c.tags for c in load_cases(s, get_dataset(s, "support"))}
        assert all(tagged[i] == () for i in PLAN["refund"])
        assert all(t[0].startswith("cluster:") for i, t in tagged.items() if i.startswith("pwfix"))


def test_save_tags_keeps_other_tags_and_reports_what_it_could_not_tag(
    sessions: sessionmaker[Session], runs: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, cand = runs
    server = Server()
    report = cluster(sessions, server, cand)
    full_id, gone_id = PLAN["refund"][0], PLAN["refund"][1]
    with sessions() as s:
        latest = get_dataset(s, "support")
        apply_edit(
            s,
            "support",
            DatasetEdit(
                base=latest.content_hash,
                update=(
                    CaseUpdate(id=full_id, tags=tuple(f"t{i}" for i in range(MAX_TAGS))),
                    CaseUpdate(id=PLAN["refund"][2], tags=("keep", "cluster:old", "also")),
                ),
                delete=(gone_id,),
            ),
        )
        result = save_tags(s, report)
        assert result.full == [full_id]
        assert result.missing == [gone_id]
        text = format_tags(result)
        assert f"not tagged, already at {MAX_TAGS} tags: {full_id}" in text
        assert f"not in the newest version: {gone_id}" in text
        again = save_tags(s, report)
        assert format_tags(again).startswith(
            f"support unchanged at version {again.version}: the tags are already there"
        )
        tags = {c.id: c.tags for c in load_cases(s, get_dataset(s, "support"))}
        assert tags[PLAN["refund"][2]] == ("keep", "also", "cluster:refund-amounts-wrong")
        assert len(tags[full_id]) == MAX_TAGS
        # Another writer moving the dataset on is retried, five times at most.
        real = apply_edit
        calls: list[int] = []

        def racing(*args: Any, **kw: Any) -> Any:
            calls.append(1)
            if len(calls) == 1:
                raise ConflictError("moved on")
            return real(*args, **kw)

        report.clusters[1].tag = "cluster:renamed"
        with monkeypatch.context() as m:
            m.setattr(clusters_module, "apply_edit", racing)
            moved = save_tags(s, report)
        assert (len(calls), moved.created) == (2, True)

        def always(*args: Any, **kw: Any) -> Any:
            raise ConflictError("moved on")

        report.clusters[1].tag = "cluster:renamed-again"
        with monkeypatch.context() as m:
            m.setattr(clusters_module, "apply_edit", always)
            with pytest.raises(ConflictError, match="kept changing"):
                save_tags(s, report)
        report.clusters[0].tag = "elsewhere:x"
        with pytest.raises(InvalidInputError, match="does not start with"):
            save_tags(s, report)


# CLI ----------------------------------------------------------------------------


def run_cli(db: str, *args: str) -> tuple[int, str]:
    out = io.StringIO()
    return cli.main(["--database-url", db, *args], out=out), out.getvalue()


def test_cli_report_clusters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for var in ("ANTHROPIC_API_KEY", "EVALKIT_DATABASE_URL"):
        monkeypatch.delenv(var, raising=False)
    server = Server()
    monkeypatch.setattr(
        cli,
        "ProviderRegistry",
        functools.partial(ProviderRegistry, transport=httpx2.MockTransport(server.handle)),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    db = f"sqlite:///{tmp_path / 'c.db'}"
    data = tmp_path / "support.jsonl"
    data.write_text("".join(c.canonical() + "\n" for c in planted_cases()))
    assert run_cli(db, "dataset", "import", "support", str(data))[0] == 0
    for seed in (BASE_SEED, CAND_SEED):
        cfg = tmp_path / f"seed{seed}.toml"
        cfg.write_text(
            f'label = "seed{seed}"\n[dataset]\nname = "support"\n'
            f'[target]\ntype = "fake"\naccuracy = {ACCURACY}\nseed = {seed}\n'
            '[[scorers]]\ntype = "exact"\n'
        )
        assert run_cli(db, "run", str(cfg))[0] == 0
    config = tmp_path / "clusters.toml"
    config.write_text(
        '[clusters]\nfields = ["input", "output"]\ninput_key = "q"\n'
        f'[clusters.namer]\nprovider = "openai"\nmodel = "{NAMER_MODEL}"\n'
    )
    pair = ("label:seed22", "--baseline", "label:seed21", "--config", str(config))
    code, out = run_cli(db, "report", "clusters", *pair, "--save-tags")
    assert code == 0, out
    assert "3 clusters (k by silhouette" in out
    assert "15 failing, baseline 0, new" in out
    assert "support now at version 2: 37 cases tagged, 0 old cluster tags removed" in out
    code, out = run_cli(db, "report", "clusters", *pair, "--json", "--save-tags")
    assert code == 0
    value = json.loads(out)
    assert value["k"] == 3
    assert value["cached_name_calls"] == 3
    assert value["tags"]["created"] is False
    assert {c["change"] for c in value["clusters"]} == {"new", "same", "shrank"}
    # The saved tags are slices the gate can follow on runs of the new version.
    for seed in (BASE_SEED, CAND_SEED):
        assert run_cli(db, "run", str(tmp_path / f"seed{seed}.toml"))[0] == 0
    code, out = run_cli(
        db, "compare", "label:seed21", "label:seed22", "--by", "tag", "--min-slice-size", "10"
    )
    assert code == 0, out
    # The new failure cluster is a slice that got worse.
    assert re.search(r"cluster:refund-amo\S* +exact +15 +100\.0% +0\.0% .* worse", out), out
    # Runs on two versions are matched by case id, and the report says so.
    code, out = run_cli(
        db, "report", "clusters", "label:seed22", "--baseline", value["baseline_id"],
        "--config", str(config), "--json",
    )  # fmt: skip
    assert code == 0
    assert "the runs are on versions 1 and 2; cases are matched by id" in json.loads(out)["notes"]
    code, out = run_cli(
        db, "report", "clusters", "label:seed22", "--no-names", "--k", "2",
        "--threshold", "0.5", "--threshold", "exact=0.5", "--metric", "exact", "--limit", "0",
    )  # fmt: skip
    assert code == 0, out
    assert "2 clusters (k given" in out
    assert "cases:" not in out
    for bad in (["--config", str(tmp_path / "seed21.toml")], ["--k", "0"], ["--threshold", "x"]):
        assert run_cli(db, "report", "clusters", "label:seed22", *bad)[0] == 2, bad
    assert "has no [clusters] table" in capsys.readouterr().err
    monkeypatch.setattr(
        cli,
        "ProviderRegistry",
        functools.partial(
            ProviderRegistry,
            transport=httpx2.MockTransport(lambda r: error_response(400, "bad request")),
        ),
    )
    inputs = tmp_path / "inputs.toml"
    inputs.write_text('[clusters]\nfields = ["input"]\nnames = false\n')
    code, _ = run_cli(db, "report", "clusters", "label:seed22", "--config", str(inputs))
    assert code == 3
    assert "embedding call failed" in capsys.readouterr().err
