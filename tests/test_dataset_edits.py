"""Case-level dataset edits: versions with parents, optimistic concurrency on the
version hash, the log and lineage, field-level diffs, comparison of runs on two
versions over their unchanged cases, migration 0016, the CLI and the API."""

import asyncio
import io
import json
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx2
import pytest
from alembic import command
from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from evalkit.api import BodyLimit, create_app
from evalkit.cli import main
from evalkit.config import Settings
from evalkit.dataset_edits import (
    MAX_DIFF_DEPTH,
    CaseUpdate,
    DatasetEdit,
    FieldChange,
    apply_cases,
    apply_edit,
    dataset_log,
    diff_values,
    diff_versions,
    lineage,
    pair_versions,
)
from evalkit.datasets import (
    Case,
    case_hash,
    case_hashes,
    content_hash,
    get_dataset,
    import_dataset,
    load_cases,
)
from evalkit.db import make_engine, make_session_factory, upgrade
from evalkit.db.models import Dataset, DatasetCase
from evalkit.db.session import alembic_config
from evalkit.errors import ConflictError, InvalidInputError, NotFoundError, PreconditionError
from evalkit.report import compare_runs
from evalkit.runner import Runner
from evalkit.runspec import RunConfig
from tests.fake_anthropic import FakeAnthropic


def cases(n: int = 6) -> list[Case]:
    return [
        Case(
            id=f"c{i}",
            input={"question": f"q{i}", "lang": "en"},
            expected=f"a{i}",
            tags=("easy",) if i % 2 else ("hard", "math"),
            split="test",
            metadata={"source": "hand", "rev": 1},
        )
        for i in range(n)
    ]


def edit(base: str, **ops: Any) -> DatasetEdit:
    return DatasetEdit.model_validate({"base": base, **ops})


@pytest.fixture
def v1(sessions: sessionmaker[Session]) -> str:
    with sessions() as s:
        ds, _ = import_dataset(s, "qa", cases(), description="support questions")
        return ds.content_hash


# Applying edits -------------------------------------------------------------------


def test_an_edit_becomes_the_next_version_with_its_parent(
    sessions: sessionmaker[Session], v1: str
) -> None:
    with sessions() as s:
        result = apply_edit(
            s,
            "qa",
            edit(
                v1,
                upsert=[
                    {"id": "c1", "input": "replaced", "expected": "r"},
                    {"id": "new", "input": "added", "expected": "n"},
                ],
                update=[
                    {"id": "c2", "expected": None, "split": None},
                    {"id": "c3", "metadata": {"source": "review"}, "tags": ["easy", "checked"]},
                ],
                delete=["c0"],
                note="fix c1, review c3",
            ),
        )
        assert result.created
        assert not result.existing
        ds = result.dataset
        assert (ds.version, ds.origin, ds.note, ds.description) == (
            2,
            "edit",
            "fix c1, review c3",
            "support questions",
        )
        assert (ds.cases_added, ds.cases_changed, ds.cases_removed) == (1, 3, 1)
        base = get_dataset(s, "qa", 1)
        assert ds.parent_id == base.id
        after = load_cases(s, ds)
        # Positions kept, the deleted case gone, the new one appended.
        assert [c.id for c in after] == ["c1", "c2", "c3", "c4", "c5", "new"]
        by_id = {c.id: c for c in after}
        assert by_id["c1"] == Case(id="c1", input="replaced", expected="r")
        assert by_id["c2"].expected is None
        assert by_id["c2"].split is None
        assert by_id["c2"].input == {"question": "q2", "lang": "en"}
        assert by_id["c3"].metadata == {"source": "review"}
        assert by_id["c3"].tags == ("easy", "checked")
        assert by_id["c4"] == cases()[4]
        assert ds.content_hash == content_hash(after)
        assert ds.case_count == 6
        # Every stored row carries its own hash.
        assert case_hashes(s, ds.id) == {c.id: case_hash(c) for c in after}
        # The edited version is untouched.
        assert load_cases(s, base) == cases()


def test_an_edit_that_changes_nothing_returns_the_base(
    sessions: sessionmaker[Session], v1: str
) -> None:
    with sessions() as s:
        same = apply_edit(s, "qa", edit(v1, upsert=[cases()[2].model_dump()]))
        assert (same.created, same.existing, same.dataset.version) == (False, False, 1)
        unchanged = apply_edit(s, "qa", edit(v1, update=[{"id": "c2", "expected": "a2"}]))
        assert (unchanged.created, unchanged.dataset.version) == (False, 1)


def test_an_edit_back_to_older_content_returns_that_version(
    sessions: sessionmaker[Session], v1: str
) -> None:
    with sessions() as s:
        v2 = apply_edit(s, "qa", edit(v1, delete=["c5"])).dataset.content_hash
        back = apply_edit(s, "qa", edit(v2, upsert=[cases()[5].model_dump()]))
        assert (back.created, back.existing, back.dataset.version) == (False, True, 1)


def test_a_stale_base_is_a_conflict_naming_the_newest_version(
    sessions: sessionmaker[Session], v1: str
) -> None:
    with sessions() as s:
        v2 = apply_edit(s, "qa", edit(v1, delete=["c5"])).dataset
        with pytest.raises(ConflictError) as info:
            apply_edit(s, "qa", edit(v1, delete=["c4"]))
        assert info.value.message == (
            f"dataset 'qa' is at version 2 ({v2.content_hash[:12]}), but the edit was made "
            f"against version 1 ({v1[:12]}); apply it to version 2 and send its hash as base"
        )
        # Retrying the edit that made version 2 (a lost reply) is not a conflict.
        retry = apply_edit(s, "qa", edit(v1.upper(), delete=["c5"]))
        assert (retry.created, retry.existing, retry.dataset.version) == (False, True, 2)
        assert len(dataset_log(s, "qa")) == 2


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({}, "the edit has no upsert, update or delete"),
        ({"delete": ["c1", "c1"]}, "case 'c1' appears twice in delete"),
        (
            {"upsert": [{"id": "c1", "input": "x"}], "delete": ["c1"]},
            "case 'c1' appears in upsert and delete",
        ),
        ({"update": [{"id": "c1"}]}, "update of 'c1' sets none of"),
        ({"update": [{"id": "c1", "input": None}]}, "input cannot be null"),
        ({"update": [{"id": "c1", "tags": None}]}, "tags cannot be null"),
        ({"update": [{"id": "c1", "colour": 1}]}, "Extra inputs"),
        ({"delete": [""]}, "at least 1 character"),
        ({"update": [{"id": "c1", "tags": ["a", "a"]}]}, "tags must be unique"),
    ],
)
def test_malformed_edits_are_refused(body: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        edit("a" * 64, **body)


def test_bad_bases_and_ids(sessions: sessionmaker[Session], v1: str) -> None:
    with pytest.raises(ValueError, match="base"):
        edit("not-a-hash", delete=["c1"])
    with sessions() as s:
        with pytest.raises(NotFoundError, match="no version with content hash bbbbbbbbbbbb"):
            apply_edit(s, "qa", edit("b" * 64, delete=["c1"]))
        with pytest.raises(NotFoundError):
            apply_edit(s, "nope", edit(v1, delete=["c1"]))
        with pytest.raises(InvalidInputError, match="dataset name"):
            apply_edit(s, "-bad", edit(v1, delete=["c1"]))
        with pytest.raises(InvalidInputError, match=r"no case 'x', 'y' in the edited version"):
            apply_edit(s, "qa", edit(v1, update=[{"id": "x", "expected": 1}], delete=["y"]))
        many = [f"m{i}" for i in range(7)]
        with pytest.raises(InvalidInputError, match=r"\(and 2 more\)"):
            apply_edit(s, "qa", edit(v1, delete=many))
        with pytest.raises(InvalidInputError, match="removes every case"):
            apply_edit(s, "qa", edit(v1, delete=[c.id for c in cases()]))
        with pytest.raises(InvalidInputError, match=r"update of 'c1'.*input must not be empty"):
            apply_edit(s, "qa", edit(v1, update=[{"id": "c1", "input": ""}]))
        assert len(dataset_log(s, "qa")) == 1


def test_case_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    import evalkit.dataset_edits as module

    monkeypatch.setattr(module, "MAX_CASES", 6)
    with pytest.raises(InvalidInputError, match="more than 6 cases"):
        apply_cases(cases(), edit("a" * 64, upsert=[{"id": "x", "input": "y"}]))


def test_update_keeps_unset_fields() -> None:
    before = cases()[0]
    change = CaseUpdate.model_validate({"id": "c0", "split": "dev"})
    assert change.apply(before) == before.model_copy(update={"split": "dev"})


# Concurrency -------------------------------------------------------------------

WRITERS = 8


def race(sessions: sessionmaker[Session], bodies: list[DatasetEdit]) -> list[Any]:
    barrier = threading.Barrier(len(bodies))
    out: list[Any] = [None] * len(bodies)

    def work(i: int) -> None:
        barrier.wait()
        try:
            with sessions() as s:
                result = apply_edit(s, "qa", bodies[i])
                out[i] = (result.created, result.existing, result.dataset.version)
        except Exception as exc:
            out[i] = exc

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(bodies))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    return out


def test_concurrent_edits_from_one_version_let_exactly_one_in(
    sessions: sessionmaker[Session], v1: str
) -> None:
    bodies = [edit(v1, update=[{"id": "c0", "expected": f"writer {i}"}]) for i in range(WRITERS)]
    out = race(sessions, bodies)
    created = [o for o in out if isinstance(o, tuple) and o[0]]
    conflicts = [o for o in out if isinstance(o, ConflictError)]
    assert created == [(True, False, 2)]
    assert len(conflicts) == WRITERS - 1, out
    with sessions() as s:
        assert [v.version for v in dataset_log(s, "qa")] == [2, 1]
        winner = load_cases(s, get_dataset(s, "qa"))[0].expected
        assert out.index(created[0]) == int(str(winner).split()[1])


def test_concurrent_identical_edits_mint_one_version(
    sessions: sessionmaker[Session], v1: str
) -> None:
    out = race(sessions, [edit(v1, delete=["c3"]) for _ in range(WRITERS)])
    assert sorted(out) == [(False, True, 2)] * (WRITERS - 1) + [(True, False, 2)]


# Imports, log and lineage ---------------------------------------------------------


def test_imports_record_their_parent_and_changes(sessions: sessionmaker[Session], v1: str) -> None:
    changed = cases()
    changed[0] = changed[0].model_copy(update={"expected": "other"})
    with sessions() as s:
        ds, created = import_dataset(s, "qa", [*changed[:5], Case(id="z", input="z")])
        assert created
        assert (ds.origin, ds.cases_added, ds.cases_changed, ds.cases_removed) == (
            "import",
            1,
            1,
            1,
        )
        v3 = apply_edit(s, "qa", edit(ds.content_hash, delete=["z"], note="drop z")).dataset
        log = dataset_log(s, "qa")
        assert [(v.version, v.origin, v.parent_version, v.note) for v in log] == [
            (3, "edit", 2, "drop z"),
            (2, "import", 1, ""),
            (1, "import", None, ""),
        ]
        assert (log[2].added, log[2].changed, log[2].removed) == (None, None, None)
        assert log[0].content_hash == v3.content_hash
        assert [v.version for v in lineage(s, "qa")] == [3, 2, 1]
        assert [v.version for v in lineage(s, "qa", 2)] == [2, 1]
        assert lineage(s, "qa", 1)[0].parent_version is None
        with pytest.raises(NotFoundError):
            dataset_log(s, "nope")


def test_lineage_stops_at_a_missing_parent(sessions: sessionmaker[Session], v1: str) -> None:
    with sessions() as s:
        v2 = apply_edit(s, "qa", edit(v1, delete=["c1"])).dataset
        v2.parent_id = 999_999
        s.commit()
        assert [(v.version, v.parent_version) for v in lineage(s, "qa")] == [(2, None)]
        assert dataset_log(s, "qa")[0].parent_version is None


# Diffs --------------------------------------------------------------------------


def test_diff_lists_field_level_changes(sessions: sessionmaker[Session], v1: str) -> None:
    with sessions() as s:
        apply_edit(
            s,
            "qa",
            edit(
                v1,
                update=[
                    {"id": "c1", "input": {"question": "q1 reworded", "lang": "en"}},
                    {
                        "id": "c2",
                        "metadata": {"source": "hand", "reviewed by": "ann", "rev": 2},
                        "tags": ["hard", "geo"],
                    },
                    {"id": "c3", "expected": {"answer": "a3"}, "split": None},
                ],
                upsert=[{"id": "a-new", "input": "x"}],
                delete=["c5"],
            ),
        )
        d = diff_versions(s, "qa", 1)
    assert (d.from_version, d.to_version, d.added, d.removed, d.unchanged) == (
        1,
        2,
        ["a-new"],
        ["c5"],
        2,
    )
    assert {c.id: c.changes for c in d.changed} == {
        "c1": [FieldChange("input.question", "changed", "q1", "q1 reworded")],
        "c2": [
            FieldChange("tags", "changed", ["hard", "math"], ["hard", "geo"]),
            FieldChange("metadata.rev", "changed", 1, 2),
            FieldChange('metadata["reviewed by"]', "added", None, "ann"),
        ],
        "c3": [
            FieldChange("expected", "changed", "a3", {"answer": "a3"}),
            FieldChange("split", "changed", "test", None),
        ],
    }
    with sessions() as s:
        backwards = diff_versions(s, "qa", 2, 1)
        assert (backwards.added, backwards.removed) == (["c5"], ["a-new"])
        assert backwards.changed[1].changes[2] == FieldChange(
            'metadata["reviewed by"]', "removed", "ann", None
        )
        assert diff_versions(s, "qa", 1, 1).unchanged == 6
        with pytest.raises(NotFoundError):
            diff_versions(s, "qa", 7)


def test_diff_values_edge_cases() -> None:
    assert diff_values("x", 1, 1) == []
    # Equal in Python but not the same JSON value.
    assert diff_values("x", 1, 1.0) == [FieldChange("x", "changed", 1, 1.0)]
    assert diff_values("x", True, 1) == [FieldChange("x", "changed", True, 1)]
    assert diff_values("m", {'a"b\\': 1}, {}) == [FieldChange('m["a\\"b\\\\"]', "removed", 1)]
    assert diff_values("x", [1, 2], [2, 1]) == [FieldChange("x", "changed", [1, 2], [2, 1])]
    deep_a: Any = "a"
    deep_b: Any = "b"
    for _ in range(MAX_DIFF_DEPTH + 3):
        deep_a, deep_b = {"k": deep_a}, {"k": deep_b}
    (only,) = diff_values("v", deep_a, deep_b)
    assert only.path == "v" + ".k" * MAX_DIFF_DEPTH
    assert only.kind == "changed"


json_values = st.recursive(
    st.none() | st.booleans() | st.integers(-5, 5) | st.text(max_size=3),
    lambda inner: (
        st.lists(inner, max_size=3)
        | st.dictionaries(st.sampled_from(["a", "b", "c d", "e"]), inner, max_size=3)
    ),
    max_leaves=12,
)


def _set(root: Any, keys: list[str], kind: str, value: Any) -> Any:
    if not keys:
        return value
    out = dict(root)
    if len(keys) == 1 and kind == "removed":
        del out[keys[0]]
    else:
        out[keys[0]] = _set(root.get(keys[0]), keys[1:], kind, value)
    return out


def _keys(path: str) -> list[str]:
    """The object keys of a diff path ``v.a["c d"]`` (inverse of ``_path``)."""
    rest, keys = path[1:], []
    while rest:
        if rest.startswith("."):
            name = rest[1:].split(".")[0].split("[")[0]
            keys.append(name)
            rest = rest[1 + len(name) :]
        else:
            end = rest.index('"]')
            keys.append(json.loads(rest[1 : end + 1]))
            rest = rest[end + 2 :]
    return keys


@given(json_values, json_values)
@settings(max_examples=200, deadline=None)
def test_applying_a_diff_reconstructs_the_new_value(before: Any, after: Any) -> None:
    rebuilt = before
    for change in diff_values("v", before, after):
        rebuilt = _set(rebuilt, _keys(change.path), change.kind, change.after)
    assert json.dumps(rebuilt, sort_keys=True) == json.dumps(after, sort_keys=True)
    assert (diff_values("v", before, after) == []) == (
        json.dumps(before, sort_keys=True) == json.dumps(after, sort_keys=True)
    )


# Pairing runs on two versions -----------------------------------------------------


def run_config(version: int, **target: Any) -> RunConfig:
    return RunConfig.model_validate(
        {
            "label": f"v{version}",
            "dataset": {"name": "pairs", "version": version},
            "target": {"type": "fake", **target},
            "scorers": [{"type": "exact"}],
            "options": {"concurrency": 8},
        }
    )


def many(n: int) -> list[Case]:
    return [Case(id=f"k{i:03d}", input={"q": i}, expected=f"a{i}") for i in range(n)]


def test_runs_on_two_versions_pair_on_their_unchanged_cases(
    sessions: sessionmaker[Session],
) -> None:
    with sessions() as s:
        base = import_dataset(s, "pairs", many(120))[0].content_hash
        # Ids that sort before and among the old ones shift every array position.
        v2 = apply_edit(
            s,
            "pairs",
            edit(
                base,
                upsert=[{"id": f"a{i}", "input": "new", "expected": "x"} for i in range(15)]
                + [{"id": f"k{i:03d}x", "input": "new", "expected": "x"} for i in range(5)],
                update=[{"id": f"k{i:03d}", "expected": f"changed {i}"} for i in range(0, 30, 3)],
                delete=[f"k{i:03d}" for i in range(1, 30, 3)],
            ),
        ).dataset
        assert v2.version == 2
    runner = Runner(sessions, Settings.from_env({}))
    ids = {}
    for label, version, accuracy in (("base", 1, 0.5), ("same", 2, 0.5), ("worse", 2, 0.2)):
        run = runner.create_run(run_config(version, accuracy=accuracy, seed=3))
        asyncio.run(runner.execute(run.id))
        ids[label] = run.id
    with sessions() as s:
        same = compare_runs(s, ids["base"], ids["same"])
        worse = compare_runs(s, ids["base"], ids["worse"])
        flipped = compare_runs(s, ids["same"], ids["base"])
    assert same.dataset_change is not None
    assert (
        same.dataset_change.shared,
        same.dataset_change.changed,
        same.dataset_change.only_baseline,
        same.dataset_change.only_candidate,
    ) == (100, 10, 10, 20)
    # The fake answers each case id the same way at the same accuracy and seed, so
    # correctly paired cases agree exactly; a misaligned pairing would not.
    (c,) = same.comparisons
    assert (c.n_pairs, c.difference, c.verdict) == (100, 0.0, "no detectable change")
    assert c.baseline_mean == c.candidate_mean
    assert worse.regressions == ["exact"]
    assert worse.comparisons[0].n_pairs == 100
    assert flipped.dataset_change is not None
    assert (flipped.dataset_change.only_baseline, flipped.dataset_change.only_candidate) == (
        20,
        10,
    )


def test_versions_with_nothing_unchanged_cannot_be_paired(
    sessions: sessionmaker[Session],
) -> None:
    with sessions() as s:
        one = import_dataset(s, "pairs", many(3))[0]
        two = apply_edit(
            s,
            "pairs",
            edit(one.content_hash, update=[{"id": c.id, "expected": "z"} for c in many(3)]),
        ).dataset
        with pytest.raises(PreconditionError, match="have no case in common unchanged"):
            pair_versions(s, one, two)
        other = import_dataset(s, "elsewhere", many(3))[0]
        with pytest.raises(PreconditionError, match="different datasets"):
            pair_versions(s, one, other)


# Migration ------------------------------------------------------------------------


def test_migration_0016_backfills_hashes_and_lineage(tmp_path: Path) -> None:
    engine = make_engine(f"sqlite:///{tmp_path / 'old.db'}")
    upgrade(engine, "0015")
    rows: dict[int, list[tuple[Any, ...]]] = {
        1: [("a", {"q": 1}, "x", ["t"], "dev", {"m": 1}), ("b", "two", None, [], None, {})],
        2: [("a", {"q": 1}, "y", ["t"], "dev", {"m": 1}), ("c", "three", "z", [], None, {})],
        3: [("q", "other", None, [], None, {})],
    }
    with engine.begin() as conn:
        for dataset_id, name, version in ((1, "d", 1), (2, "d", 2), (3, "e", 1)):
            conn.execute(
                text(
                    "INSERT INTO datasets (id, name, version, content_hash, case_count,"
                    " description, created_at) VALUES (:i, :n, :v, :h, 2, '', CURRENT_TIMESTAMP)"
                ),
                {"i": dataset_id, "n": name, "v": version, "h": str(dataset_id) * 64},
            )
            for position, (case_id, input_, expected, tags, split, meta) in enumerate(
                rows[dataset_id]
            ):
                conn.execute(
                    text(
                        "INSERT INTO dataset_cases (dataset_id, position, case_id, input,"
                        " expected, tags, split, metadata) VALUES (:d, :p, :c, :i, :e, :t,"
                        " :s, :m)"
                    ),
                    {
                        "d": dataset_id,
                        "p": position,
                        "c": case_id,
                        "i": json.dumps(input_),
                        "e": json.dumps(expected),
                        "t": json.dumps(tags),
                        "s": split,
                        "m": json.dumps(meta),
                    },
                )
    upgrade(engine)
    sessions = make_session_factory(engine)
    with sessions() as s:
        for dataset_id in (1, 2, 3):
            ds = s.get_one(Dataset, dataset_id)
            assert case_hashes(s, ds.id) == {c.id: case_hash(c) for c in load_cases(s, ds)}
        log = dataset_log(s, "d")
        assert [(v.version, v.origin, v.parent_version) for v in log] == [
            (2, "import", 1),
            (1, "import", None),
        ]
        assert (log[0].added, log[0].changed, log[0].removed) == (1, 1, 1)
        assert dataset_log(s, "e")[0].parent_version is None
        assert [c.id for c in diff_versions(s, "d", 1).changed] == ["a"]
    command.downgrade(alembic_config(engine), "0015")
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM dataset_cases")).scalar() == 5
    upgrade(engine)
    with sessions() as s:
        assert s.scalar(select(DatasetCase.case_hash).where(DatasetCase.case_id == "q"))
    engine.dispose()


# CLI ----------------------------------------------------------------------------


def cli(db: str, *args: str) -> tuple[int, str]:
    out = io.StringIO()
    return main(["--database-url", db, *args], out=out), out.getvalue()


@pytest.fixture
def cli_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A database with dataset ``pairs``: version 1 (60 cases) and version 2, an
    edit of it (k001 changed, k002 removed, k100 added)."""
    for var in ("ANTHROPIC_API_KEY", "EVALKIT_DATABASE_URL"):
        monkeypatch.delenv(var, raising=False)
    db = f"sqlite:///{tmp_path / 'cli.db'}"
    data = tmp_path / "qa.jsonl"
    data.write_text("".join(c.canonical() + "\n" for c in many(60)))
    assert cli(db, "dataset", "import", "pairs", str(data))[0] == 0
    (tmp_path / "edit.json").write_text(json.dumps(CLI_EDIT))
    code, out = cli(db, "dataset", "edit", "pairs", str(tmp_path / "edit.json"), "--base", "1")
    assert code == 0
    assert out.startswith("pairs edited: version 2 (")
    assert out.endswith("+1 added, ~1 changed, -1 removed\n")
    return db


CLI_EDIT = {
    "update": [{"id": "k001", "expected": "fixed", "tags": ["reviewed"]}],
    "upsert": [{"id": "k100", "input": {"q": 100}, "expected": "a100"}],
    "delete": ["k002"],
    "note": "review pass",
}


def test_cli_edit(cli_db: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    change = tmp_path / "edit.json"
    # The same edit again, against version 1: it already made version 2.
    code, out = cli(cli_db, "dataset", "edit", "pairs", str(change), "--base", "v1")
    assert (code, out.split(":")[0]) == (0, "pairs unchanged")
    assert "which already exists" in out
    # A different edit against version 1 is a conflict: exit 3.
    change.write_text(json.dumps({"delete": ["k003"]}))
    assert cli(cli_db, "dataset", "edit", "pairs", str(change), "--base", "1")[0] == 3
    assert "is at version 2" in capsys.readouterr().err
    engine = make_engine(cli_db)
    with Session(engine) as s:
        v2 = get_dataset(s, "pairs").content_hash
    engine.dispose()
    change.write_text(json.dumps({"base": v2, "update": [{"id": "k004", "expected": "a4"}]}))
    assert cli(cli_db, "dataset", "edit", "pairs", str(change)) == (
        0,
        "pairs unchanged: the edit changes nothing\n",
    )
    for content, message in (
        ("{oops", "is not valid JSON"),
        ("[1]", "must hold a JSON object"),
        ('{"base": "x", "delete": ["k1"]}', "base"),
    ):
        change.write_text(content)
        assert cli(cli_db, "dataset", "edit", "pairs", str(change))[0] == 2
        assert message in capsys.readouterr().err


def test_cli_diff_log_and_gate(cli_db: str, tmp_path: Path) -> None:
    code, out = cli(cli_db, "dataset", "diff", "pairs", "1")
    assert code == 0
    lines = out.splitlines()
    assert lines[0].endswith("1 added, 1 changed, 1 removed, 58 unchanged")
    assert lines[1:] == [
        "+ k100",
        "- k002",
        "~ k001",
        '    expected: "a1" -> "fixed"',
        "    tags: +reviewed",
    ]
    code, out = cli(cli_db, "dataset", "diff", "pairs", "1", "2", "--limit", "1")
    assert out.splitlines()[-1] == "... and 2 more (--limit, or --json for all)"
    code, out = cli(cli_db, "dataset", "diff", "pairs", "1", "--json")
    assert json.loads(out)["changed"][0]["changes"][0]["path"] == "expected"

    code, out = cli(cli_db, "dataset", "log", "pairs")
    assert code == 0
    first, second = out.splitlines()
    assert first.startswith("v2\t")
    assert "\tedit\tfrom v1\t60 cases\t+1 added, ~1 changed, -1 removed\t" in first
    assert first.endswith("  review pass")
    assert "\timport\tfrom -\t60 cases\tfirst version\t" in second
    code, out = cli(cli_db, "dataset", "log", "pairs", "--version", "1", "--json")
    assert [v["version"] for v in json.loads(out)] == [1]

    for label, version in (("one", 1), ("two", 2)):
        cfg = tmp_path / f"{label}.toml"
        cfg.write_text(
            f'label = "{label}"\n[dataset]\nname = "pairs"\nversion = {version}\n'
            '[target]\ntype = "fake"\n[[scorers]]\ntype = "exact"\n'
        )
        assert cli(cli_db, "run", str(cfg))[0] == 0
    code, out = cli(cli_db, "gate", "label:one", "label:two")
    assert code == 0
    assert (
        "note: pairs v1 vs v2: compared on the 58 cases both have unchanged; 3 left out "
        "(1 changed, 1 only in the baseline, 1 only in the candidate)"
    ) in out
    code, out = cli(cli_db, "gate", "label:one", "label:two", "--json")
    assert json.loads(out)["dataset_change"]["shared"] == 58


def test_tags_reordered_is_reported() -> None:
    from evalkit.cli import _format_field_change

    assert _format_field_change(FieldChange("tags", "changed", ["a", "b"], ["b", "a"])) == (
        "tags: reordered"
    )
    long = _format_field_change(FieldChange("input", "changed", "x" * 100, "y"))
    assert long == f'input: "{"x" * 56}... -> "y"'
    assert _format_field_change(FieldChange("input.k", "removed", 1)) == (
        "input.k: removed (was 1)"
    )


# API ----------------------------------------------------------------------------

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
async def test_api_patch_log_lineage_and_diff(client: httpx2.AsyncClient) -> None:
    body = "".join(c.canonical() + "\n" for c in cases())
    created = (await client.post("/v1/datasets/qa", content=body, headers=AUTH)).json()
    assert (created["origin"], created["parent_version"], created["changes"]) == (
        "import",
        None,
        None,
    )
    base = created["content_hash"]
    patch = {"base": base, "update": [{"id": "c0", "expected": "new"}], "note": "fix c0"}
    first = await client.patch("/v1/datasets/qa", json=patch, headers=AUTH)
    assert first.status_code == 201
    v2 = first.json()
    assert (v2["version"], v2["origin"], v2["parent_version"], v2["note"]) == (
        2,
        "edit",
        1,
        "fix c0",
    )
    assert v2["changes"] == {"added": 0, "changed": 1, "removed": 0}
    assert v2["existing"] is False
    # A retry of the same edit gets version 2 back; a different stale edit is a conflict.
    retry = await client.patch("/v1/datasets/qa", json=patch, headers=AUTH)
    assert (retry.status_code, retry.json()["version"], retry.json()["existing"]) == (
        200,
        2,
        True,
    )
    stale = await client.patch(
        "/v1/datasets/qa", json={"base": base, "delete": ["c1"]}, headers=AUTH
    )
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "conflict"
    assert "is at version 2" in stale.json()["error"]["message"]
    bad = await client.patch("/v1/datasets/qa", json={"base": base}, headers=AUTH)
    assert bad.status_code == 400
    assert (
        await client.patch("/v1/datasets/-x", json={"base": base, "delete": ["c1"]}, headers=AUTH)
    ).status_code == 400
    missing = await client.patch(
        "/v1/datasets/nope", json={"base": base, "delete": ["c1"]}, headers=AUTH
    )
    assert missing.status_code == 404

    version = (await client.get("/v1/datasets/qa/versions/2", headers=AUTH)).json()
    assert version["parent_version"] == 1
    listing = (await client.get("/v1/datasets", headers=AUTH)).json()["datasets"]
    assert [d["parent_version"] for d in listing] == [None, 1]
    log = (await client.get("/v1/datasets/qa/log", headers=AUTH)).json()["versions"]
    assert [(v["version"], v["origin"], v["note"]) for v in log] == [
        (2, "edit", "fix c0"),
        (1, "import", ""),
    ]
    assert log[0]["created_at"].endswith("+00:00")
    chain = (await client.get("/v1/datasets/qa/versions/2/lineage", headers=AUTH)).json()
    assert [v["version"] for v in chain["versions"]] == [2, 1]
    diff = (await client.get("/v1/datasets/qa/diff?from=1", headers=AUTH)).json()
    assert diff["changed"] == [
        {
            "id": "c0",
            "changes": [{"path": "expected", "kind": "changed", "before": "a0", "after": "new"}],
        }
    ]
    assert diff["unchanged"] == 5
    assert (await client.get("/v1/datasets/qa/diff", headers=AUTH)).status_code == 400
    assert (await client.get("/v1/datasets/qa/diff?from=1&to=9", headers=AUTH)).status_code == 404
    assert (await client.get("/v1/datasets/nope/log", headers=AUTH)).status_code == 404
