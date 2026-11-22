"""Slices: grouping cases by tag, split or metadata, per-slice paired comparisons
with Holm's correction over every slice test, minimum slice sizes, the gate's
slice rules, and the CLI and API that expose them. The end-to-end runs plant a
regression in one slice that the whole-dataset comparison cannot see."""

import asyncio
import io
import json
import math
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx2
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy.orm import Session, sessionmaker

from evalkit.api import BodyLimit, create_app
from evalkit.cli import main
from evalkit.config import ConfigError, GateRules, Settings, load_gate_rules
from evalkit.dataset_edits import DatasetEdit, apply_edit, pair_versions
from evalkit.datasets import Case, import_dataset
from evalkit.errors import InvalidInputError, PreconditionError
from evalkit.report import compare_runs
from evalkit.runner import Runner
from evalkit.runspec import RunConfig
from evalkit.slices import (
    MAX_SLICES,
    Membership,
    SliceBy,
    compare_slices,
    gate_failures,
    membership,
    parse_by,
    slice_names,
)
from evalkit.stats import holm
from evalkit.targets.fake import unit
from tests.fake_anthropic import FakeAnthropic

RESAMPLES = 2_000


def idx(*positions: int) -> Any:
    return np.asarray(positions, dtype=np.intp)


# Slicing keys -------------------------------------------------------------------


def test_parse_by() -> None:
    assert parse_by("tag") == SliceBy("tag")
    assert parse_by("split") == SliceBy("split")
    assert parse_by("metadata.region") == SliceBy("metadata", "region")
    # Only the first dot separates: the rest is the key.
    assert parse_by("metadata.a.b") == SliceBy("metadata", "a.b")
    assert str(parse_by("metadata.a.b")) == "metadata.a.b"
    for bad in (
        "tags",
        "metadata",
        "metadata.",
        "meta.x",
        "metadata.\x01",
        "metadata." + "k" * 129,
    ):
        with pytest.raises(ValueError, match="cannot slice by"):
            parse_by(bad)


def test_slice_names() -> None:
    tag, split, region = SliceBy("tag"), SliceBy("split"), SliceBy("metadata", "region")
    assert slice_names(tag, ["b", "a"], "dev", {}) == ["b", "a"]
    assert slice_names(tag, [], "dev", {}) == []
    assert slice_names(split, ["a"], "dev", {}) == ["dev"]
    assert slice_names(split, ["a"], None, {}) == []
    for value, names in (
        ("eu", ["eu"]),
        ("", []),
        (3, ["3"]),
        (2.5, ["2.5"]),
        (math.nan, []),
        (True, ["true"]),
        (None, []),
        ({"x": 1}, []),
        (["eu", "us", "eu", 1, None, ["nested"]], ["eu", "us", "1"]),
    ):
        assert slice_names(region, [], None, {"region": value}) == names, value
    assert slice_names(region, [], None, {"other": "eu"}) == []
    assert slice_names(region, [], None, None) == []


def test_membership_in_score_array_order(sessions: sessionmaker[Session]) -> None:
    rows = [
        Case(id="b", input="q", tags=("x", "y"), split="dev", metadata={"r": "eu"}),
        Case(id="a", input="q", tags=("y",), metadata={"r": ["eu", "us"]}),
        Case(id="c", input="q", split="test", metadata={"r": 7}),
        Case(id="d", input="q"),
    ]
    with sessions() as s:
        ds = import_dataset(s, "m", rows)[0]
        # Score arrays are in case-id order: a, b, c, d.
        by_tag = membership(s, ds.id, SliceBy("tag"))
        assert {k: v.tolist() for k, v in by_tag.slices.items()} == {"x": [1], "y": [0, 1]}
        assert by_tag.unsliced == 2
        by_split = membership(s, ds.id, SliceBy("split"))
        assert {k: v.tolist() for k, v in by_split.slices.items()} == {"dev": [1], "test": [2]}
        by_r = membership(s, ds.id, SliceBy("metadata", "r"))
        assert {k: v.tolist() for k, v in by_r.slices.items()} == {
            "7": [2],
            "eu": [0, 1],
            "us": [0],
        }
        assert by_r.unsliced == 1
        # Through a pairing's positions: compared case i is dataset case positions[i].
        picked = membership(s, ds.id, SliceBy("tag"), idx(3, 1))
        assert {k: v.tolist() for k, v in picked.slices.items()} == {"x": [1], "y": [1]}
        assert picked.unsliced == 1


def test_too_many_slices(sessions: sessionmaker[Session]) -> None:
    rows = [Case(id=f"c{i}", input="q", metadata={"id": i}) for i in range(MAX_SLICES + 1)]
    with sessions() as s:
        ds = import_dataset(s, "ids", rows)[0]
        with pytest.raises(PreconditionError, match="slice by a key with fewer values"):
            membership(s, ds.id, SliceBy("metadata", "id"))
        assert len(membership(s, ds.id, SliceBy("tag")).slices) == 0


# Per-slice comparisons ------------------------------------------------------------


def planted(n: int = 120, seed: int = 0) -> tuple[Membership, dict[str, Any]]:
    """Four slices over n cases: ``bad`` (cases 0-29) loses 0.5 on ``acc`` in the
    candidate, ``good`` and ``noise`` do not change beyond noise, ``tiny`` has 5
    cases. ``noise`` overlaps ``bad``."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.4, 0.9, n)
    cand = base + rng.normal(0, 0.02, n)
    cand[:30] -= 0.5
    other = rng.uniform(0, 1, n)
    member = Membership(
        {
            "bad": idx(*range(30)),
            "good": idx(*range(30, 90)),
            "noise": idx(*range(20, 120, 2)),
            "tiny": idx(*range(115, 120)),
        },
        unsliced=0,
    )
    return member, {"acc": (base, cand), "other": (other, other.copy())}


def test_planted_slice_regression_is_found_and_holm_adjusted() -> None:
    member, scores = planted()
    report = compare_slices(member, scores, by="tag", min_size=10, resamples=RESAMPLES, seed=1)
    by_name = {s.name: s for s in report.slices}
    assert list(by_name) == ["bad", "good", "noise", "tiny"]
    assert by_name["bad"].regressions == ["acc"]
    assert by_name["good"].regressions == []
    # ``noise`` holds 5 of the 30 bad cases among 50: diluted, but still worse.
    assert by_name["noise"].comparisons[0].difference == pytest.approx(-0.05, abs=0.01)
    assert by_name["noise"].regressions == ["acc"]
    assert by_name["tiny"].comparisons == []
    assert by_name["tiny"].too_small == {"acc": 5, "other": 5}
    assert by_name["tiny"].cases == 5
    # One Holm family over the 6 tested (slice, metric) pairs.
    assert report.tests == 6
    tested = [c for s in report.slices for c in s.comparisons]
    assert [c.p_adjusted for c in tested] == pytest.approx(holm([c.p_value for c in tested]))
    bad = by_name["bad"].comparisons[0]
    assert bad.metric == "acc"
    assert bad.n_pairs == 30
    assert bad.difference == pytest.approx(
        float(np.mean(scores["acc"][1][:30] - scores["acc"][0][:30]))
    )
    # Identical scores in every slice: no change, p = 1.
    for s in report.slices:
        for c in s.comparisons:
            if c.metric == "other":
                assert (c.difference, c.p_value, c.verdict) == (0.0, 1.0, "no detectable change")


def test_min_size_counts_cases_scored_in_both_runs() -> None:
    base = np.array([1.0] * 12)
    cand = np.array([0.0] * 12)
    cand[:3] = np.nan  # unscorable in the candidate
    member = Membership({"s": idx(*range(12))}, 0)
    small = compare_slices(member, {"m": (base, cand)}, by="tag", min_size=10, resamples=200)
    assert small.slices[0].too_small == {"m": 9}
    assert small.tests == 0
    ok = compare_slices(member, {"m": (base, cand)}, by="tag", min_size=9, resamples=200)
    assert ok.slices[0].comparisons[0].n_pairs == 9
    assert ok.slices[0].comparisons[0].n_unpaired == 3
    with pytest.raises(ValueError, match="at least 2"):
        compare_slices(member, {"m": (base, cand)}, by="tag", min_size=1)


def test_null_slices_rarely_report_a_regression() -> None:
    """40 slices of 30 cases where nothing changed, 20 times over: Holm keeps the
    chance of any false verdict per comparison under alpha."""
    false_reports = 0
    for trial in range(20):
        rng = np.random.default_rng(100 + trial)
        n = 1200
        base = rng.uniform(0, 1, n)
        cand = base + rng.normal(0, 0.1, n)
        member = Membership({f"s{i}": idx(*range(i * 30, i * 30 + 30)) for i in range(40)}, 0)
        report = compare_slices(
            member, {"m": (base, cand)}, by="tag", min_size=10, resamples=500, seed=trial
        )
        assert report.tests == 40
        false_reports += any(
            s.comparisons[0].verdict != "no detectable change" for s in report.slices
        )
    assert false_reports <= 2


@settings(max_examples=40, deadline=None)
@given(
    st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=6, max_size=30),
    st.integers(min_value=2, max_value=8),
)
def test_adjusted_p_is_holm_over_tested_pairs(shifts: list[float], size: int) -> None:
    rng = np.random.default_rng(len(shifts))
    n = len(shifts) * 4
    base = rng.uniform(0, 1, n)
    cand = base + np.repeat(shifts, 4)
    member = Membership(
        {f"s{i}": idx(*range(i * 4, min(n, i * 4 + size))) for i in range(len(shifts))}, 0
    )
    report = compare_slices(member, {"m": (base, cand)}, by="tag", min_size=4, resamples=100)
    tested = [c for s in report.slices for c in s.comparisons]
    assert len(tested) == report.tests == sum(1 for s in report.slices if s.cases >= 4)
    assert [c.p_adjusted for c in tested] == pytest.approx(holm([c.p_value for c in tested]))
    assert all(c.p_adjusted >= c.p_value for c in tested)


# Gate rules ---------------------------------------------------------------------


def test_gate_failures() -> None:
    member, scores = planted()
    report = compare_slices(member, scores, by="tag", min_size=10, resamples=RESAMPLES, seed=1)

    def rules(**kw: Any) -> GateRules:
        return GateRules.model_validate({"by": "tag", **kw})

    assert gate_failures(report, rules(require=["good"]), ["acc", "other"]) == []
    assert gate_failures(report, rules(require=["bad"]), ["acc", "other"]) == [
        "slice bad: acc regressed"
    ]
    assert (
        gate_failures(
            report, rules(require=[{"slice": "bad", "metrics": ["other"]}]), ["acc", "other"]
        )
        == []
    )
    assert gate_failures(report, rules(require=["tiny"]), ["acc", "other"]) == [
        "slice tiny: acc has 5 paired case(s), fewer than 10, so it cannot be tested",
        "slice tiny: other has 5 paired case(s), fewer than 10, so it cannot be tested",
    ]
    # any_slice: a regression anywhere fails, once per slice even if also required.
    # ``noise`` shares 5 of the bad cases, enough to regress on its own.
    both = ["slice bad: acc regressed", "slice noise: acc regressed"]
    assert gate_failures(report, rules(any_slice=True, require=["bad"]), ["acc", "other"]) == both
    assert gate_failures(report, rules(any_slice=True), ["acc", "other"]) == both
    with pytest.raises(InvalidInputError, match=r"required slice\(s\) 'billing' not found by tag"):
        gate_failures(report, rules(require=["billing", "bad"]), ["acc", "other"])
    with pytest.raises(InvalidInputError, match=r"metric\(s\) latency not compared"):
        gate_failures(report, rules(require=[{"slice": "bad", "metrics": ["latency"]}]), ["acc"])


def test_gate_rules_validation(tmp_path: Path) -> None:
    rules = GateRules.model_validate(
        {"by": "metadata.region", "require": ["eu", {"slice": "us", "metrics": ["exact"]}]}
    )
    assert [(r.slice, r.metrics) for r in rules.require] == [("eu", ()), ("us", ("exact",))]
    assert rules.gating
    assert not GateRules(by="tag").gating
    for bad, message in (
        ({"require": ["eu"]}, "slice rules need by"),
        ({"any_slice": True}, "slice rules need by"),
        ({"by": "tag", "require": ["a", "a"]}, "required more than once: a"),
        ({"by": "tags"}, "String should match pattern"),
        ({"by": "tag", "min_slice_size": 1}, "greater than or equal to 2"),
        ({"by": "tag", "require": [""]}, "at least 1 character"),
    ):
        with pytest.raises(ValueError, match=message):
            GateRules.model_validate(bad)

    path = tmp_path / "gate.toml"
    path.write_text(
        '[gate]\nby = "tag"\nmin_slice_size = 15\nany_slice = true\n'
        'require = ["billing", { slice = "refunds", metrics = ["exact"] }]\n'
    )
    loaded = load_gate_rules(str(path))
    assert (loaded.by, loaded.min_slice_size, loaded.any_slice) == ("tag", 15, True)
    assert [r.slice for r in loaded.require] == ["billing", "refunds"]
    path.write_text('[gate]\nrequire = ["billing"]\n')
    with pytest.raises(ConfigError, match=r"gate.toml: \[gate\] gate: Value error, slice rules"):
        load_gate_rules(str(path))
    path.write_text('[gate]\nby = "tag"\nrequire = [{ slice = "a", extra = 1 }]\n')
    with pytest.raises(ConfigError, match=r"require.0.extra: Extra inputs"):
        load_gate_rules(str(path))
    path.write_text("[other]\n")
    with pytest.raises(ConfigError, match=r"has no \[gate\] table"):
        load_gate_rules(str(path))


# End to end -----------------------------------------------------------------------

# Seeds 11 and 12 at accuracy 0.8 lose 73 cases and win 69 of 400: no detectable
# change overall. ``billing`` holds 25 of the lost cases and 15 that both runs get
# right, so it lost 25 of 40 with nothing won back.
BASE_SEED, CAND_SEED, ACCURACY = 11, 12, 0.8
IDS = [f"c{i:03d}" for i in range(400)]


def right(seed: int, case_id: str) -> bool:
    return unit(seed, case_id) < ACCURACY


def planted_cases() -> list[Case]:
    lost = [i for i in IDS if right(BASE_SEED, i) and not right(CAND_SEED, i)]
    kept = [i for i in IDS if right(BASE_SEED, i) and right(CAND_SEED, i)]
    billing = set(lost[:25]) | set(kept[:15])
    out = []
    for n, case_id in enumerate(IDS):
        tags: tuple[str, ...] = (
            ("billing",) if case_id in billing else ("shipping",) if n % 2 else ("account",)
        )
        if n < 6:
            tags = (*tags, "rare")
        out.append(
            Case(
                id=case_id,
                input={"q": case_id},
                expected=f"answer {case_id}",
                tags=tags,
                split="test" if n % 4 else "dev",
                metadata={"region": ["eu", "us"][n % 2], "tier": n % 3},
            )
        )
    return out


def run_config(seed: int, version: int | None = None) -> RunConfig:
    return RunConfig.model_validate(
        {
            "label": f"seed{seed}" if version is None else f"seed{seed}v{version}",
            "dataset": {"name": "support", "version": version},
            "target": {"type": "fake", "accuracy": ACCURACY, "seed": seed},
            "scorers": [{"type": "exact"}],
        }
    )


@pytest.fixture
def two_runs(sessions: sessionmaker[Session]) -> tuple[str, str]:
    with sessions() as s:
        import_dataset(s, "support", planted_cases())
    runner = Runner(sessions, Settings.from_env({}))
    ids = []
    for seed in (BASE_SEED, CAND_SEED):
        run = runner.create_run(run_config(seed))
        asyncio.run(runner.execute(run.id))
        ids.append(run.id)
    return ids[0], ids[1]


def test_slice_regression_hidden_in_the_average(
    sessions: sessionmaker[Session], two_runs: tuple[str, str]
) -> None:
    base, cand = two_runs
    with sessions() as s:
        overall = compare_runs(s, base, cand)
        sliced = compare_runs(s, base, cand, slices=GateRules(by="tag", require=("billing",)))
        by_split = compare_runs(s, base, cand, slices=GateRules(by="split", any_slice=True))
        by_region = compare_runs(s, base, cand, slices=GateRules(by="metadata.region"))
        # The slices change nothing about the whole-dataset verdict.
        assert sliced.comparisons == overall.comparisons
    (c,) = overall.comparisons
    assert (c.n_pairs, c.verdict) == (400, "no detectable change")
    assert c.difference == pytest.approx((69 - 73) / 400)
    assert overall.passed
    assert overall.slices is None

    report = sliced.slices
    assert report is not None
    assert (report.by, report.min_size, report.unsliced) == ("tag", 20, 0)
    found = {s.name: s for s in report.slices}
    assert list(found) == ["account", "billing", "rare", "shipping"]
    billing = found["billing"].comparisons[0]
    assert (billing.n_pairs, billing.baseline_mean, billing.candidate_mean) == (40, 1.0, 15 / 40)
    assert billing.verdict == "worse"
    assert found["rare"].too_small == {"exact": 6}
    assert [s.regressions for s in found.values()] == [[], ["exact"], [], []]
    assert report.tests == 3
    assert report.failures == ["slice billing: exact regressed"]
    assert sliced.regressions == []
    assert not sliced.passed
    assert sliced.failures == ["slice billing: exact regressed"]

    # Splits and regions mix billing into larger slices: diluted below detection.
    assert by_split.slices is not None
    assert [s.name for s in by_split.slices.slices] == ["dev", "test"]
    assert by_split.passed
    assert by_region.slices is not None
    assert [s.name for s in by_region.slices.slices] == ["eu", "us"]


def test_slices_on_two_versions_use_the_shared_cases(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        v1 = import_dataset(s, "support", planted_cases())[0]
        # Version 2 retags ten billing cases (so they are changed, and left out) and
        # adds new cases.
        billing = [c.id for c in planted_cases() if "billing" in c.tags][:10]
        edit = DatasetEdit.model_validate(
            {
                "base": v1.content_hash,
                "update": [{"id": i, "tags": ["billing", "reviewed"]} for i in billing],
                "upsert": [{"id": f"a{i}", "input": "new", "tags": ["billing"]} for i in range(30)],
            }
        )
        v2 = apply_edit(s, "support", edit).dataset
        pairing = pair_versions(s, v1, v2)[0]
        assert (pairing.shared, pairing.changed, pairing.only_candidate) == (390, 10, 30)
    runner = Runner(sessions, Settings.from_env({}))
    ids = []
    for seed, version in ((BASE_SEED, 1), (CAND_SEED, 2)):
        run = runner.create_run(run_config(seed, version))
        asyncio.run(runner.execute(run.id))
        ids.append(run.id)
    with sessions() as s:
        report = compare_runs(s, ids[0], ids[1], slices=GateRules(by="tag", require=("billing",)))
    assert report.slices is not None
    found = {x.name: x for x in report.slices.slices}
    # Neither the retagged cases nor the new ones are compared.
    assert "reviewed" not in found
    assert found["billing"].cases == 30
    assert found["billing"].comparisons[0].n_pairs == 30
    assert sum(x.cases for x in found.values() if x.name != "rare") == 390
    assert report.failures == ["slice billing: exact regressed"]


# CLI and API ----------------------------------------------------------------------


def cli(db: str, *args: str) -> tuple[int, str]:
    out = io.StringIO()
    return main(["--database-url", db, *args], out=out), out.getvalue()


@pytest.fixture
def cli_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    for var in ("ANTHROPIC_API_KEY", "EVALKIT_DATABASE_URL"):
        monkeypatch.delenv(var, raising=False)
    db = f"sqlite:///{tmp_path / 'cli.db'}"
    data = tmp_path / "support.jsonl"
    data.write_text("".join(c.canonical() + "\n" for c in planted_cases()))
    assert cli(db, "dataset", "import", "support", str(data))[0] == 0
    for seed in (BASE_SEED, CAND_SEED):
        cfg = tmp_path / f"seed{seed}.toml"
        cfg.write_text(
            f'label = "seed{seed}"\n[dataset]\nname = "support"\n'
            f'[target]\ntype = "fake"\naccuracy = {ACCURACY}\nseed = {seed}\n'
            '[[scorers]]\ntype = "exact"\n'
        )
        assert cli(db, "run", str(cfg))[0] == 0
    return db


def test_cli_compare_and_gate_by_slice(
    cli_db: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pair = ("label:seed11", "label:seed12")
    code, out = cli(cli_db, "compare", *pair, "--by", "tag")
    assert code == 0
    lines = out.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("slices by tag"))
    assert lines[start] == (
        "slices by tag: 4 slice(s), 3 test(s), p adjusted by Holm across them; "
        "tested from 20 paired cases"
    )
    table = lines[start + 2 : start + 6]
    assert table[0].startswith("account             exact")
    assert table[1].startswith(
        "billing             exact               40  100.0%   37.5%   -62.5%"
    )
    assert table[1].endswith("worse")
    assert table[2].split() == [
        "rare",
        "exact",
        "6",
        "-",
        "-",
        "-",
        "-",
        "too",
        "small",
        "to",
        "test",
    ]
    assert lines[-1] == "PASS: no regression"

    # The plain gate passes; requiring billing fails it with exit 1.
    assert cli(cli_db, "gate", *pair)[0] == 0
    code, out = cli(cli_db, "gate", *pair, "--by", "tag", "--require-slice", "billing")
    assert code == 1
    assert out.splitlines()[-2:] == ["FAIL: slice rules", "  slice billing: exact regressed"]
    assert "slice billing: exact regressed" in capsys.readouterr().err
    code, out = cli(cli_db, "gate", *pair, "--by", "tag", "--any-slice", "--json")
    assert code == 1
    body = json.loads(out)
    assert (body["passed"], body["failures"]) == (False, ["slice billing: exact regressed"])
    assert body["slices"]["tests"] == 3
    # Too small to test is a failure; raising the bar on slice size fails billing too.
    code, _ = cli(cli_db, "gate", *pair, "--by", "tag", "--require-slice", "rare")
    assert code == 1
    assert "slice rare: exact has 6 paired case(s), fewer than 20" in capsys.readouterr().err
    assert cli(cli_db, "gate", *pair, "--by", "tag", "--require-slice", "account")[0] == 0

    rules = tmp_path / "gate.toml"
    rules.write_text('[gate]\nby = "split"\nrequire = ["dev"]\n')
    assert cli(cli_db, "gate", *pair, "--rules", str(rules))[0] == 0
    # Flags go over the file: slice by tag instead, and require billing as well.
    code, out = cli(
        cli_db, "gate", *pair, "--rules", str(rules), "--by", "tag", "--min-slice-size", "5"
    )
    assert code == 2
    assert "required slice(s) 'dev' not found by tag" in capsys.readouterr().err
    rules.write_text('[gate]\nby = "tag"\nmin_slice_size = 5\nrequire = ["rare"]\n')
    assert cli(cli_db, "gate", *pair, "--rules", str(rules))[0] == 0
    assert cli(cli_db, "gate", *pair, "--rules", str(rules), "--require-slice", "billing")[0] == 1

    for args, message in (
        (("--require-slice", "billing"), "slice rules need by"),
        (("--by", "tags"), "String should match pattern"),
        (("--rules", str(tmp_path / "missing.toml")), "cannot read"),
        (("--by", "tag", "--require-slice", "nope"), "'nope' not found by tag"),
    ):
        assert cli(cli_db, "gate", *pair, *args)[0] == 2
        assert message in capsys.readouterr().err


TOKEN = "test-token-0123456789"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
async def client(sessions: sessionmaker[Session]) -> AsyncIterator[httpx2.AsyncClient]:
    app = create_app(
        Settings.from_env({"EVALKIT_API_TOKENS": TOKEN}),
        sessions=sessions,
        claude=FakeAnthropic().client(),
    )
    transport = httpx2.ASGITransport(app=BodyLimit(app, 64 * 1024), raise_app_exceptions=False)
    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(transport=transport, base_url="http://test") as c,
    ):
        yield c


@pytest.mark.anyio
async def test_api_compare_with_slices(
    client: httpx2.AsyncClient, two_runs: tuple[str, str]
) -> None:
    base, cand = two_runs
    plain = await client.post(
        "/v1/compare", json={"baseline": base, "candidate": cand}, headers=AUTH
    )
    assert plain.status_code == 200
    assert (plain.json()["passed"], plain.json()["slices"]) == (True, None)
    sliced = await client.post(
        "/v1/compare",
        json={"baseline": base, "candidate": cand, "slices": {"by": "tag", "require": ["billing"]}},
        headers=AUTH,
    )
    assert sliced.status_code == 200
    body = sliced.json()
    assert (body["passed"], body["failures"]) == (False, ["slice billing: exact regressed"])
    billing = next(s for s in body["slices"]["slices"] if s["name"] == "billing")
    assert billing["comparisons"][0]["verdict"] == "worse"
    for slices, message in (
        ({"require": ["billing"]}, "slice rules need by"),
        ({"by": "tag", "require": ["nope"]}, "'nope' not found by tag"),
    ):
        bad = await client.post(
            "/v1/compare",
            json={"baseline": base, "candidate": cand, "slices": slices},
            headers=AUTH,
        )
        assert bad.status_code == 400, bad.text
        assert message in bad.json()["error"]["message"]
