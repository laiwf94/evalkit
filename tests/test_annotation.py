"""Annotation queues and Krippendorff's alpha.

The alpha values are checked against the worked examples Krippendorff publishes
("Computing Krippendorff's Alpha-Reliability", 2011): the two-coder binary
table (alpha 0.095) and the four-coder table with missing data (0.743 nominal,
0.815 ordinal, 0.849 interval, 0.797 ratio). The four-coder table is also
labelled through a queue, annotator by annotator, skipping where the table has
no value, and the queue's alpha matches. A brute-force computation from the
definition (every ordered pair within a unit, weighted 1/(m-1)) checks the
matrix computation on random data.
"""

from __future__ import annotations

import io
import itertools
import json
import threading
from collections.abc import Hashable, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from evalkit import annotation, cli
from evalkit.agreement import coincidences, krippendorff_alpha
from evalkit.annotation import (
    QueueSpec,
    adjudicate,
    agreement,
    claim_next,
    create_queue,
    export_labels,
    get_queue,
    list_queues,
    list_tasks,
    queue_summary,
    resolve_label,
    skip,
    submit,
)
from evalkit.datasets import Case, import_dataset
from evalkit.db import make_engine, make_session_factory, upgrade
from evalkit.db.models import AnnotationAssignment, Dataset, Result, Run, utcnow
from evalkit.errors import ConflictError, InvalidInputError, NotFoundError
from evalkit.scorers.calibration import parse_labels

N = None
# Krippendorff (2011), section C: units 1-12, observers A-D, None where missing.
FOUR_CODERS = {
    "A": [1, 2, 3, 3, 2, 1, 4, 1, 2, N, N, N],
    "B": [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, N, 3],
    "C": [N, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, N],
    "D": [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, N],
}
PUBLISHED = {"nominal": 0.743, "ordinal": 0.815, "interval": 0.849, "ratio": 0.797}


def four_coder_units() -> list[list[int]]:
    return [
        [v for v in column if v is not None] for column in zip(*FOUR_CODERS.values(), strict=True)
    ]


# Alpha ---------------------------------------------------------------------------


@pytest.mark.parametrize(("level", "published"), PUBLISHED.items())
def test_alpha_matches_the_published_four_coder_example(level: Any, published: float) -> None:
    r = krippendorff_alpha(four_coder_units(), level)
    assert r.alpha is not None
    assert round(r.alpha, 3) == published
    # Unit 12 has one value: it is labelled but not pairable.
    assert (r.units, r.pairable_units, r.pairable_values) == (12, 11, 40)


def test_alpha_matches_the_published_binary_example() -> None:
    a = [0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
    b = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    r = krippendorff_alpha(list(zip(a, b, strict=True)))
    # The paper's 0.095 is 1 - 19 * 4 / (14 * 6) = 2/21 exactly.
    assert r.alpha == pytest.approx(2 / 21)
    assert r.observed == pytest.approx(8 / 20)


def test_coincidence_matrix_of_the_binary_example() -> None:
    a = [0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
    b = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    o, labelled, pairable = coincidences(list(zip(a, b, strict=True)), [0, 1])
    assert o.tolist() == [[10.0, 4.0], [4.0, 2.0]]
    assert (labelled, pairable) == (10, 10)


def test_alpha_edges() -> None:
    assert krippendorff_alpha([["a", "a"], ["b", "b"]]).alpha == pytest.approx(1.0)
    # No variation, or nothing pairable: undefined, not 1 or 0.
    same = krippendorff_alpha([["a", "a"], ["a", "a", "a"]])
    assert same.alpha is None
    assert same.observed == 0.0
    lone = krippendorff_alpha([["a"], ["b"], []])
    assert (lone.alpha, lone.units, lone.pairable_units) == (None, 2, 0)
    assert krippendorff_alpha([]).alpha is None
    # Systematic disagreement is negative.
    assert (krippendorff_alpha([["a", "b"], ["b", "a"]]).alpha or 0) < 0
    # An unused value changes nothing on the nominal level.
    units = four_coder_units()
    assert krippendorff_alpha(units, values=[1, 2, 3, 4, 5, 6]).alpha == pytest.approx(
        krippendorff_alpha(units).alpha
    )


def test_ordinal_alpha_follows_the_given_order() -> None:
    units = [["low", "mid"], ["mid", "high"], ["low", "low"], ["high", "high"], ["mid", "mid"]]
    scale = krippendorff_alpha(units, "ordinal", ["low", "mid", "high"])
    # Alphabetical order puts "mid" at the end: neighbours become extremes.
    alphabetical = krippendorff_alpha(units, "ordinal")
    assert scale.alpha is not None
    assert alphabetical.alpha is not None
    assert scale.alpha > alphabetical.alpha


def test_alpha_refuses_bad_input() -> None:
    with pytest.raises(ValueError, match="level must be"):
        krippendorff_alpha([[1, 2]], "cardinal")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="interval labels must be numbers"):
        krippendorff_alpha([["a", "b"]], "interval")
    with pytest.raises(ValueError, match="must not be negative"):
        krippendorff_alpha([[-1, 2]], "ratio")
    with pytest.raises(ValueError, match="explicit order"):
        krippendorff_alpha([["a", 1]])
    with pytest.raises(ValueError, match="not one of the values"):
        krippendorff_alpha([["a", "b"]], values=["a"])
    assert krippendorff_alpha([[0, 0], [0, 1]], "ratio").alpha is not None


def brute_force_alpha(
    units: Sequence[Sequence[Hashable]], level: str, values: Sequence[Hashable]
) -> float | None:
    """Alpha straight from the definition, pair by pair."""
    pairable = [u for u in units if len(u) >= 2]
    n = sum(len(u) for u in pairable)
    if n < 2:
        return None
    rank = {v: i for i, v in enumerate(values)}
    totals = dict.fromkeys(values, 0.0)
    for u in pairable:
        for v in u:
            totals[v] += 1

    def delta(c: Any, k: Any) -> float:
        if level == "nominal":
            return 0.0 if c == k else 1.0
        if level == "interval":
            return float(c - k) ** 2
        if level == "ratio":
            return 0.0 if c + k == 0 else ((c - k) / (c + k)) ** 2
        lo, hi = sorted((rank[c], rank[k]))
        between = sum(totals[values[g]] for g in range(lo, hi + 1))
        return (between - (totals[values[lo]] + totals[values[hi]]) / 2) ** 2

    observed = 0.0
    for u in pairable:
        m = len(u)
        for i, j in itertools.permutations(range(m), 2):
            observed += delta(u[i], u[j]) / (m - 1)
    observed /= n
    pooled = [v for u in pairable for v in u]
    expected = sum(
        delta(pooled[i], pooled[j]) for i, j in itertools.permutations(range(len(pooled)), 2)
    ) / (n * (n - 1))
    return None if expected == 0 else 1 - observed / expected


@settings(max_examples=150, deadline=None)
@given(
    units=st.lists(st.lists(st.integers(0, 4), max_size=5), min_size=1, max_size=12),
    level=st.sampled_from(["nominal", "ordinal", "interval", "ratio"]),
)
def test_alpha_equals_the_pairwise_definition(units: list[list[int]], level: Any) -> None:
    values = [0, 1, 2, 3, 4]
    found = krippendorff_alpha(units, level, values).alpha
    expected = brute_force_alpha(units, level, values)
    if expected is None:
        assert found is None
    else:
        assert found == pytest.approx(expected, abs=1e-9)


@settings(max_examples=100, deadline=None)
@given(
    pairs=st.lists(
        st.tuples(st.sampled_from("abc"), st.sampled_from("abc")), min_size=2, max_size=40
    )
)
def test_two_coder_nominal_alpha_is_scotts_pi_corrected(pairs: list[tuple[str, str]]) -> None:
    """With two coders and no missing data, alpha = 1 - (1 - pi)(2N - 1) / 2N."""
    n = len(pairs)
    p_o = sum(a == b for a, b in pairs) / n
    marginals = {c: sum((a == c) + (b == c) for a, b in pairs) / (2 * n) for c in "abc"}
    p_e = sum(p * p for p in marginals.values())
    found = krippendorff_alpha([list(p) for p in pairs]).alpha
    if p_e == 1:
        assert found is None
        return
    pi = (p_o - p_e) / (1 - p_e)
    assert found == pytest.approx(1 - (1 - pi) * (2 * n - 1) / (2 * n))


@settings(max_examples=50, deadline=None)
@given(
    units=st.lists(st.lists(st.integers(1, 5), min_size=2, max_size=4), min_size=2, max_size=15),
    seed=st.randoms(use_true_random=False),
)
def test_alpha_ignores_the_order_of_units_and_labels(units: list[list[int]], seed: Any) -> None:
    shuffled = [seed.sample(u, len(u)) for u in units]
    seed.shuffle(shuffled)
    a = krippendorff_alpha(units, "interval", [1, 2, 3, 4, 5]).alpha
    b = krippendorff_alpha(shuffled, "interval", [1, 2, 3, 4, 5]).alpha
    assert (a is None) == (b is None)
    if a is not None:
        assert b == pytest.approx(a)


# Queues ---------------------------------------------------------------------------


def cases(n: int = 12) -> list[Case]:
    return [
        Case(
            id=f"c{i:02d}",
            input={"question": f"q{i}"},
            expected=f"a{i}",
            tags=("billing",) if i % 3 == 0 else ("account",),
            split="test" if i % 2 else "dev",
        )
        for i in range(1, n + 1)
    ]


def dataset(session: Session, n: int = 12, name: str = "support") -> Dataset:
    ds, _ = import_dataset(session, name, cases(n))
    return ds


def spec(**extra: Any) -> QueueSpec:
    body: dict[str, Any] = {"name": "q", "dataset": "support", "labels": ["good", "bad"]}
    return QueueSpec.model_validate(body | extra)


def test_spec_validation() -> None:
    for bad, message in (
        ({"run": "r1"}, "either dataset or run"),
        ({"dataset": None}, "either dataset or run"),
        ({"labels": ["a"]}, "at least 2"),
        ({"labels": [1, 1.0]}, "unique"),
        ({"labels": ["a", True]}, "string or a finite number"),
        ({"labels": ["a", ""]}, "1-64 characters"),
        ({"labels": ["a", "b"], "level": "interval"}, "must be numbers"),
        ({"labels": [-1, 2], "level": "ratio"}, "must not be negative"),
        ({"claim_timeout": "10"}, "between 1m and 7d"),
        ({"claim_timeout": "soon"}, "not a duration"),
        ({"name": "-x"}, "letters, digits"),
        ({"overlap": 11}, "less than or equal to 10"),
        ({"version": 1, "dataset": None, "run": "r"}, "version comes from the run"),
        ({"surprise": 1}, "Extra inputs"),
    ):
        with pytest.raises(ValueError, match=message):
            spec(**bad)
    assert spec(claim_timeout="2h").claim_seconds == 7200


def test_create_snapshots_the_filtered_cases(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        dataset(s)
        summary, created = create_queue(s, spec(tags=["billing"], split="dev", overlap=2))
        assert created
        # Billing cases are 3, 6, 9, 12; of those, dev (even) are 6 and 12.
        assert summary.tasks == 2
        assert summary.statuses == {"open": 2, "agreed": 0, "disputed": 0, "adjudicated": 0}
        assert (summary.dataset, summary.dataset_version, summary.run) == ("support", 1, None)
        tasks = list_tasks(s, "q")
        assert [t.case_id for t in tasks] == ["c06", "c12"]
        assert tasks[0].item == {"input": {"question": "q6"}, "expected": "a6"}
        # The same settings again: the same queue. Other settings: a conflict.
        again, created = create_queue(s, spec(tags=["billing"], split="dev", overlap=2))
        assert (again.id, created) == (summary.id, False)
        with pytest.raises(ConflictError, match="exists with other settings"):
            create_queue(s, spec(overlap=3))
        # A newer version moves "newest" on, so the same table is a different queue.
        import_dataset(s, "support", cases(13))
        with pytest.raises(ConflictError, match="over support version 1"):
            create_queue(s, spec(tags=["billing"], split="dev", overlap=2))
        with pytest.raises(InvalidInputError, match="no cases to label"):
            create_queue(s, spec(name="empty", tags=["nothing"]))
        limited, _ = create_queue(s, spec(name="first5", limit=5, version=1))
        assert limited.tasks == 5
        assert [q.name for q in list_queues(s)] == ["q", "first5"]
        assert queue_summary(s, limited.id).name == "first5"
        with pytest.raises(NotFoundError):
            queue_summary(s, "nope")
        with pytest.raises(NotFoundError):
            create_queue(s, spec(name="x", dataset="absent"))


def make_run(session: Session, ds: Dataset, outputs: dict[str, str | None]) -> str:
    run = Run(
        id="r" * 32,
        dataset_id=ds.id,
        target_spec={},
        target_hash="t" * 64,
        scorers=[],
        options={},
        seed=1,
        status="completed",
    )
    session.add(run)
    session.flush()
    for case_id, output in outputs.items():
        session.add(
            Result(
                run_id=run.id,
                case_id=case_id,
                status="ok" if output is not None else "error",
                output=output,
            )
        )
    session.commit()
    return run.id


def test_a_run_queue_labels_outputs_and_exports_for_calibration(
    sessions: sessionmaker[Session],
) -> None:
    with sessions() as s:
        ds = dataset(s, 4)
        run_id = make_run(s, ds, {"c01": "yes", "c02": "no", "c03": None, "c04": "maybe"})
        body = {"name": "faith", "run": run_id, "labels": ["faithful", "unfaithful"], "overlap": 2}
        summary, _ = create_queue(s, QueueSpec.model_validate(body))
        # The case whose result failed has no output to label.
        assert (summary.tasks, summary.run, summary.dataset) == (3, run_id, "support")
        with pytest.raises(NotFoundError, match="run"):
            create_queue(s, QueueSpec.model_validate(body | {"name": "x", "run": "missing"}))
        verdicts = {"c01": ("faithful", "faithful"), "c02": ("faithful", "unfaithful")}
        for who in ("ann", "bob"):
            for _ in range(3):
                claim = claim_next(s, "faith", who)
                assert claim is not None
                if claim.case_id in verdicts:
                    submit(s, "faith", who, claim.case_id, verdicts[claim.case_id][who == "bob"])
                else:
                    assert claim.item["output"] == "maybe"
                    skip(s, "faith", who, claim.case_id)
        adjudicate(s, "faith", "lead", "c02", "unfaithful", note="the second claim is unsupported")
        rows = export_labels(s, "faith")
    assert rows == [
        {
            "id": "c01",
            "input": {"question": "q1"},
            "expected": "a1",
            "output": "yes",
            "label": "faithful",
        },
        {
            "id": "c02",
            "input": {"question": "q2"},
            "expected": "a2",
            "output": "no",
            "label": "unfaithful",
        },
    ]
    items = parse_labels("".join(json.dumps(r) + "\n" for r in rows), "faithfulness")
    assert [(i.id, i.label) for i in items] == [("c01", "faithful"), ("c02", "unfaithful")]


def test_claims_respect_overlap_order_and_held_claims(sessions: sessionmaker[Session]) -> None:
    t0 = utcnow()
    with sessions() as s:
        dataset(s, 3)
        create_queue(s, spec(overlap=3))
        got = {who: claim_next(s, "q", who, now=t0) for who in "abcd"}
        assert {w: c.case_id for w, c in got.items() if c} == {
            "a": "c01",
            "b": "c01",
            "c": "c01",
            "d": "c02",
        }
        assert sorted(c.slot for c in got.values() if c and c.case_id == "c01") == [0, 1, 2]
        # Asking again returns the held item and extends the claim.
        later = t0 + timedelta(minutes=10)
        again = claim_next(s, "q", "a", now=later)
        assert again is not None
        assert (again.case_id, again.expires_at) == ("c01", later + timedelta(minutes=30))
        # Labelling without a claim, or another annotator's item, is refused.
        with pytest.raises(ConflictError, match="does not hold"):
            submit(s, "q", "d", "c01", "good", now=later)
        with pytest.raises(NotFoundError, match="no item"):
            submit(s, "q", "a", "c99", "good", now=later)
        with pytest.raises(InvalidInputError, match="is not one of"):
            submit(s, "q", "a", "c01", "fine", now=later)
        with pytest.raises(InvalidInputError, match="annotator must be"):
            claim_next(s, "q", "bad name")
        with pytest.raises(InvalidInputError, match="note is longer"):
            submit(s, "q", "a", "c01", "good", note="x" * 2001)
        done = submit(s, "q", "a", "c01", "good", now=later)
        assert (done.status, done.labels, done.overlap) == ("open", 1, 3)
        # Having labelled c01, a moves on to c02, where d holds one slot of three.
        nxt = claim_next(s, "q", "a", now=later)
        assert nxt is not None
        assert (nxt.case_id, nxt.slot) == ("c02", 1)


def test_lapsed_claims_are_taken_over_or_still_count(sessions: sessionmaker[Session]) -> None:
    t0 = utcnow()
    with sessions() as s:
        dataset(s, 2)
        create_queue(s, spec(overlap=1, claim_timeout="5m"))
        x = claim_next(s, "q", "x", now=t0)
        y = claim_next(s, "q", "y", now=t0)
        assert x is not None
        assert y is not None
        assert (x.case_id, y.case_id) == ("c01", "c02")
        assert claim_next(s, "q", "z", now=t0) is None
        # After x's claim lapses, z takes its slot over; x can no longer label it.
        expired = t0 + timedelta(minutes=6)
        z = claim_next(s, "q", "z", now=expired)
        assert z is not None
        assert (z.case_id, z.slot) == ("c01", 0)
        with pytest.raises(ConflictError, match="does not hold"):
            submit(s, "q", "x", "c01", "good", now=expired)
        # y's claim lapsed too, but nobody took it: asking again renews it, and
        # y's label still counts.
        renewed = claim_next(s, "q", "y", now=expired)
        assert renewed is not None
        assert (renewed.case_id, renewed.expires_at) == ("c02", expired + timedelta(minutes=5))
        late = submit(s, "q", "y", "c02", "bad", now=expired)
        assert (late.status, late.final) == ("agreed", "bad")
        assert claim_next(s, "q", "w", now=expired) is None


def test_statuses_changes_and_adjudication(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        dataset(s, 2)
        create_queue(s, spec(overlap=2, labels=["good", "bad", "unsure"]))
        for who in ("a", "b"):
            for _ in range(2):
                claim = claim_next(s, "q", who)
                assert claim is not None
                submit(s, "q", who, claim.case_id, "good")
        assert queue_summary(s, "q").statuses["agreed"] == 2
        # Changing a label before adjudication re-settles the item.
        changed = submit(s, "q", "b", "c02", "bad", note="missed the date")
        assert (changed.status, changed.final) == ("disputed", None)
        with pytest.raises(ConflictError, match="only disputed"):
            adjudicate(s, "q", "lead", "c01", "bad")
        with pytest.raises(InvalidInputError, match="adjudicator must be"):
            adjudicate(s, "q", "", "c02", "bad")
        final = adjudicate(s, "q", "lead", "c02", "unsure", note="ambiguous question")
        assert (final.status, final.final, final.labels) == ("adjudicated", "unsure", 2)
        # An adjudicated item can be re-adjudicated, but its labels are final.
        assert adjudicate(s, "q", "lead2", "c02", "bad").final == "bad"
        with pytest.raises(ConflictError, match="adjudicated"):
            submit(s, "q", "a", "c02", "bad")
        tasks = {t.case_id: t for t in list_tasks(s, "q")}
        assert tasks["c02"].adjudicator == "lead2"
        assert [(a.annotator, a.label, a.note) for a in tasks["c02"].assignments] == [
            ("a", "good", ""),
            ("b", "bad", "missed the date"),
        ]
        assert [t.case_id for t in list_tasks(s, "q", "agreed")] == ["c01"]
        with pytest.raises(InvalidInputError, match="status must be"):
            list_tasks(s, "q", "done")
        stats = {x.annotator: x for x in agreement(s, "q").annotators}
        assert (stats["a"].compared, stats["a"].matched) == (2, 1)
        assert (stats["b"].matched, stats["b"].agreement) == (2, 1.0)


def test_skips_free_the_slot_and_are_remembered(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        dataset(s, 2)
        create_queue(s, spec(overlap=1))
        first = claim_next(s, "q", "a")
        assert first is not None
        skip(s, "q", "a", first.case_id)
        skip(s, "q", "a", first.case_id)  # twice is fine
        # b gets the skipped item; a is never offered it again.
        b = claim_next(s, "q", "b")
        assert b is not None
        assert b.case_id == "c01"
        a = claim_next(s, "q", "a")
        assert a is not None
        assert a.case_id == "c02"
        submit(s, "q", "a", "c02", "good")
        assert claim_next(s, "q", "a") is None
        with pytest.raises(ConflictError, match="has labelled"):
            skip(s, "q", "a", "c02")
        with pytest.raises(ConflictError, match="does not hold"):
            skip(s, "q", "z", "c01")
        with pytest.raises(ConflictError, match="does not hold"):
            submit(s, "q", "a", "c01", "good")
        shown = list_tasks(s, "q")[0].assignments
        assert [(x.annotator, x.skipped) for x in shown] == [("a", True), ("b", False)]


def test_resolve_label() -> None:
    labels = [1, 2.5, "3", "high"]
    assert resolve_label(labels, 1.0) == 1
    assert resolve_label(labels, "1") == 1
    assert resolve_label(labels, "2.5") == 2.5
    assert resolve_label(labels, "3") == "3"
    assert resolve_label(labels, "high") == "high"
    for bad in (True, "4", 3, None, "nan"):
        with pytest.raises(InvalidInputError):
            resolve_label(labels, bad)


@pytest.mark.parametrize("level", PUBLISHED)
def test_queue_alpha_reproduces_the_published_example(
    sessions: sessionmaker[Session], level: str
) -> None:
    """Four annotators label the 12 units of the published table through a queue,
    skipping where the table has no value."""
    with sessions() as s:
        dataset(s, 12)
        create_queue(s, spec(labels=[1, 2, 3, 4, 5], level=level, overlap=4))
        for who, row in FOUR_CODERS.items():
            for _ in range(12):
                claim = claim_next(s, "q", who)
                assert claim is not None
                value = row[int(claim.case_id[1:]) - 1]
                if value is None:
                    skip(s, "q", who, claim.case_id)
                else:
                    submit(s, "q", who, claim.case_id, str(value))
            assert claim_next(s, "q", who) is None
        found = agreement(s, "q")
    assert found.alpha.alpha is not None
    assert round(found.alpha.alpha, 3) == PUBLISHED[level]
    assert (found.labelled_items, found.labels, found.alpha.pairable_units) == (12, 41, 11)
    assert sum(map(sum, found.coincidences)) == pytest.approx(40)
    # Units 3, 4, 5, 7 and 9 have four equal labels: agreed, and matched by all.
    # Units with fewer than four labels stay open and have no final label.
    assert {(a.annotator, a.compared, a.matched) for a in found.annotators} == {
        (who, 5, 5) for who in "ABCD"
    }
    assert found.alpha.units == 12


def test_agreement_on_an_empty_queue(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        dataset(s, 2)
        create_queue(s, spec())
        found = agreement(s, "q")
    assert (found.alpha.alpha, found.labels, found.annotators) == (None, 0, [])


def test_concurrent_claims_never_exceed_overlap(tmp_path: Path) -> None:
    """Twelve annotators claim and label at once on a file database; each item ends
    with exactly ``overlap`` labels from different annotators."""
    engine = make_engine(f"sqlite:///{tmp_path / 'race.db'}")
    upgrade(engine)
    sessions = make_session_factory(engine)
    with sessions() as s:
        dataset(s, 10)
        create_queue(s, spec(overlap=3))
    barrier = threading.Barrier(12)
    errors: list[BaseException] = []

    def work(i: int) -> None:
        who = f"ann{i}"
        barrier.wait()
        try:
            with sessions() as s:
                while (claim := claim_next(s, "q", who)) is not None:
                    submit(s, "q", who, claim.case_id, "good" if i % 4 else "bad")
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    with sessions() as s:
        rows = list(s.scalars(select(AnnotationAssignment)))
        tasks = list_tasks(s, "q")
    assert len(rows) == 30
    per_task: dict[int, list[str]] = {}
    for r in rows:
        per_task.setdefault(r.task_id, []).append(r.annotator)
    assert all(len(v) == len(set(v)) == 3 for v in per_task.values())
    for task in tasks:
        labels = [a.label for a in task.assignments]
        assert task.status == ("agreed" if len(set(labels)) == 1 else "disputed")
    engine.dispose()


# CLI and API -----------------------------------------------------------------------


def run_cli(db: str, *args: str) -> tuple[int, str]:
    out = io.StringIO()
    return cli.main(["--database-url", db, *args], out=out), out.getvalue()


def labelled_by_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> str:
    """A two-item ordinal queue labelled, skipped and adjudicated through the CLI."""
    db = f"sqlite:///{tmp_path / 'a.db'}"
    data = tmp_path / "cases.jsonl"
    data.write_text("".join(c.model_dump_json() + "\n" for c in cases(2)))
    assert run_cli(db, "dataset", "import", "support", str(data))[0] == 0
    config = tmp_path / "annotate.toml"
    config.write_text(
        '[annotate]\nname = "rubric"\ndataset = "support"\nlabels = [1, 2, 3]\n'
        'level = "ordinal"\noverlap = 2\ninstructions = "Rate the answer"\n'
    )
    code, out = run_cli(db, "annotate", "create", str(config))
    assert (code, out.split("\t")[0]) == (0, "created rubric")
    code, out = run_cli(db, "annotate", "create", str(config))
    assert out.startswith("unchanged, already exists: rubric\tsupport v1\t2 items x 2\tordinal")
    code, out = run_cli(db, "annotate", "next", "rubric", "--annotator", "ann")
    assert code == 0
    assert out.startswith("rubric: item c01 (held until ")
    assert 'input:    {"question": "q1"}' in out
    code, out = run_cli(db, "annotate", "submit", "rubric", "c01", "3", "--annotator", "ann")
    assert out == "c01: 3 (1 of 2 labels; open)\n"
    code, out = run_cli(db, "annotate", "next", "rubric", "--annotator", "ann", "--json")
    assert json.loads(out)["case_id"] == "c02"
    assert run_cli(db, "annotate", "skip", "rubric", "c02", "--annotator", "ann")[0] == 0
    code, out = run_cli(db, "annotate", "next", "rubric", "--annotator", "ann")
    assert (code, out) == (3, "nothing left for ann in rubric\n")
    for case_id, label in (("c01", "1"), ("c02", "2")):
        run_cli(db, "annotate", "next", "rubric", "--annotator", "bob")
        run_cli(db, "annotate", "submit", "rubric", case_id, label, "--annotator", "bob")
    code, out = run_cli(db, "annotate", "submit", "rubric", "c01", "9", "--annotator", "bob")
    assert code == 2
    assert "is not one of: 1, 2, 3" in capsys.readouterr().err
    code, out = run_cli(db, "annotate", "adjudicate", "rubric", "c02", "2", "--adjudicator", "x")
    assert code == 3
    assert "only disputed items" in capsys.readouterr().err
    code, out = run_cli(db, "annotate", "adjudicate", "rubric", "c01", "3", "--adjudicator", "lee")
    assert out == "c01: 3 (2 of 2 labels; adjudicated, final label 3)\n"
    return db


def test_cli_label_adjudicate_and_export(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = labelled_by_cli(tmp_path, capsys)
    code, out = run_cli(db, "annotate", "show", "rubric")
    assert out.splitlines()[1:] == [
        "labels: 1, 2, 3",
        "[adjudicated] c01 -> 3  ann=3, bob=1",
        "[open] c02  ann=(skipped), bob=2",
    ]
    code, out = run_cli(db, "annotate", "show", "rubric", "--status", "open", "--json")
    assert [t["case_id"] for t in json.loads(out)["tasks"]] == ["c02"]
    code, out = run_cli(db, "annotate", "list")
    assert out.startswith("rubric\tsupport v1\t2 items x 2\tordinal\t1 open, 1 adjudicated")
    assert json.loads(run_cli(db, "annotate", "list", "--json")[1])[0]["overlap"] == 2
    code, out = run_cli(db, "annotate", "agreement", "rubric")
    assert out.splitlines()[0] == (
        "rubric: Krippendorff's alpha (ordinal) 0.000 over 3 labels on 2 items "
        "(1 with two or more labels)"
    )
    assert "ann                       1 labels  agrees with the final label 100.0% of 1" in out
    agreed = json.loads(run_cli(db, "annotate", "agreement", "rubric", "--json")[1])
    assert agreed["alpha"]["pairable_values"] == 2
    code, out = run_cli(db, "annotate", "export", "rubric")
    assert [json.loads(line)["label"] for line in out.splitlines()] == [3]
    target = tmp_path / "labels.jsonl"
    code, out = run_cli(db, "annotate", "export", "rubric", "-o", str(target))
    assert code == 0
    assert out == f"wrote 1 labelled items to {target}\n"
    assert json.loads(target.read_text())["id"] == "c01"
    (tmp_path / "bad.toml").write_text("[other]\n")
    assert run_cli(db, "annotate", "create", str(tmp_path / "bad.toml"))[0] == 2
    assert "has no [annotate] table" in capsys.readouterr().err
    (tmp_path / "bad.toml").write_text('[annotate]\nname = "z"\nlabels = [1, 2]\n')
    assert run_cli(db, "annotate", "create", str(tmp_path / "bad.toml"))[0] == 2
    assert "either dataset or run" in capsys.readouterr().err
    assert run_cli(db, "annotate", "list")[0] == 0


def test_cli_lists_no_queues(tmp_path: Path) -> None:
    db = f"sqlite:///{tmp_path / 'b.db'}"
    assert run_cli(db, "annotate", "list") == (0, "no annotation queues\n")
    code, _ = run_cli(db, "annotate", "agreement", "nope")
    assert code == 2


def test_cli_agreement_undefined(tmp_path: Path) -> None:
    db = f"sqlite:///{tmp_path / 'c.db'}"
    data = tmp_path / "cases.jsonl"
    data.write_text("".join(c.model_dump_json() + "\n" for c in cases(1)))
    run_cli(db, "dataset", "import", "support", str(data))
    config = tmp_path / "annotate.toml"
    config.write_text('[annotate]\nname = "q"\ndataset = "support"\nlabels = ["a", "b"]\n')
    run_cli(db, "annotate", "create", str(config))
    _, out = run_cli(db, "annotate", "agreement", "q")
    assert "alpha (nominal) undefined over 0 labels" in out
    assert "alpha needs two labels on one item" in out


@pytest.mark.anyio
async def test_api_queue_lifecycle(sessions: sessionmaker[Session]) -> None:
    from tests.test_api import AUTH, serving

    with sessions() as s:
        dataset(s, 2)
    body = {"name": "q", "dataset": "support", "labels": ["good", "bad"], "overlap": 2}
    async with serving(sessions) as (c, _):
        r = await c.post("/v1/annotation-queues", json=body, headers=AUTH)
        assert (r.status_code, r.json()["tasks"]) == (201, 2)
        r = await c.post("/v1/annotation-queues", json=body, headers=AUTH)
        assert r.status_code == 200
        r = await c.post("/v1/annotation-queues", json=body | {"overlap": 3}, headers=AUTH)
        assert r.status_code == 409
        r = await c.post("/v1/annotation-queues", json=body | {"labels": ["x"]}, headers=AUTH)
        assert r.status_code == 400
        listed = (await c.get("/v1/annotation-queues", headers=AUTH)).json()["queues"]
        assert [q["name"] for q in listed] == ["q"]
        for who, label in (("a", "good"), ("b", "bad")):
            claim = (
                await c.post("/v1/annotation-queues/q/claim", json={"annotator": who}, headers=AUTH)
            ).json()
            assert claim["case_id"] == "c01"
            r = await c.post(
                "/v1/annotation-queues/q/submit",
                json={"annotator": who, "case_id": "c01", "label": label},
                headers=AUTH,
            )
            assert r.status_code == 200
        assert r.json()["status"] == "disputed"
        r = await c.post(
            "/v1/annotation-queues/q/submit",
            json={"annotator": "z", "case_id": "c01", "label": "good"},
            headers=AUTH,
        )
        assert r.status_code == 409
        r = await c.post(
            "/v1/annotation-queues/q/submit",
            json={"annotator": "a", "case_id": "c01", "label": True},
            headers=AUTH,
        )
        assert r.status_code == 400
        r = await c.post(
            "/v1/annotation-queues/q/adjudicate",
            json={"adjudicator": "lead", "case_id": "c01", "label": "bad", "note": "n"},
            headers=AUTH,
        )
        assert (r.status_code, r.json()["final"]) == (200, "bad")
        claim = await c.post("/v1/annotation-queues/q/claim", json={"annotator": "a"}, headers=AUTH)
        assert claim.json()["case_id"] == "c02"
        r = await c.post(
            "/v1/annotation-queues/q/skip", json={"annotator": "a", "case_id": "c02"}, headers=AUTH
        )
        assert r.status_code == 204
        r = await c.post("/v1/annotation-queues/q/claim", json={"annotator": "a"}, headers=AUTH)
        assert r.status_code == 204
        info = (await c.get("/v1/annotation-queues/q", headers=AUTH)).json()
        assert info["statuses"]["adjudicated"] == 1
        assert info["agreement"]["alpha"]["pairable_units"] == 1
        assert {a["annotator"]: a["agreement"] for a in info["agreement"]["annotators"]} == {
            "a": 0.0,
            "b": 1.0,
        }
        tasks = (await c.get("/v1/annotation-queues/q/tasks?status=open", headers=AUTH)).json()
        assert [t["case_id"] for t in tasks["tasks"]] == ["c02"]
        bad = await c.get("/v1/annotation-queues/q/tasks?status=odd", headers=AUTH)
        assert bad.status_code == 400
        assert (await c.get("/v1/annotation-queues/nope", headers=AUTH)).status_code == 404
        r = await c.get("/v1/annotation-queues/q/export", headers=AUTH)
        assert r.headers["content-type"] == "application/x-ndjson"
        assert [json.loads(line)["label"] for line in r.text.splitlines()] == ["bad"]
        assert (await c.get("/v1/annotation-queues")).status_code == 401


def test_a_queue_created_by_another_writer_meanwhile(
    sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two writers create one queue: the loser finds the winner's queue, which is
    the same queue when the settings match and a conflict when they differ."""
    real = annotation._items
    winner: list[QueueSpec] = []

    def racing(session: Session, *args: Any) -> Any:
        if winner:
            with sessions() as other:
                create_queue(other, winner.pop())
        return real(session, *args)

    with sessions() as s:
        dataset(s, 3)
    monkeypatch.setattr(annotation, "_items", racing)
    winner.append(spec(overlap=2))
    with sessions() as s:
        summary, created = create_queue(s, spec(overlap=2))
        assert (created, summary.tasks) == (False, 3)
    winner.append(spec(name="q2", overlap=2))
    with sessions() as s, pytest.raises(ConflictError, match="other settings"):
        create_queue(s, spec(name="q2", overlap=3))
    with sessions() as s:
        assert [q.name for q in list_queues(s)] == ["q", "q2"]


def test_a_claim_that_finds_the_annotator_already_on_the_task(
    sessions: sessionmaker[Session],
) -> None:
    now = utcnow()
    with sessions() as s:
        dataset(s, 1)
        create_queue(s, spec(overlap=2))
        first = claim_next(s, "q", "a", now=now)
        assert first is not None
        q = get_queue(s, "q")
        task_id = s.scalar(select(AnnotationAssignment.task_id))
        assert task_id is not None
        # What a second, concurrent claim by the same annotator would find.
        found = annotation._take_slot(s, q, task_id, "a", now=now, ttl=timedelta(minutes=5))
        assert found is None
