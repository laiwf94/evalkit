"""The labelled RAG sets in examples/rag: they are consistent with their corpus,
the ranking metrics give the values worked out by hand below, the faithfulness
judge is calibrated against the human labels with a kappa worked out by hand,
and failure causes constructed from the sets are attributed as they should be."""

import asyncio
import io
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import httpx2
import pytest
import regex
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from evalkit import cli
from evalkit.datasets import Case, parse_jsonl
from evalkit.db.models import Calibration
from evalkit.errors import InvalidInputError
from evalkit.rag_report import MetricRole, attribute
from evalkit.ratelimit import RetryPolicy
from evalkit.scorers.calibration import (
    FAITHFULNESS_LABELS,
    LabelledItem,
    calibrate,
    cohens_kappa,
    label_mode,
    parse_labels,
)
from evalkit.scorers.faithfulness import (
    EXTRACT_SYSTEM,
    VERIFY_SYSTEM,
    FaithfulnessJudge,
    sources_of,
)
from evalkit.scorers.ranking import RankingMetric
from evalkit.targets.retrieval import MAX_GRADE, Document, Ranking, query_of
from evalkit.targets.retrieval import parse_labels as parse_relevance
from tests.fake_anthropic import FakeAnthropic, json_reply, message

RAG = Path(__file__).parent.parent / "examples" / "rag"
RETRY = RetryPolicy(max_attempts=2, base_delay=0)


def jsonl(name: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (RAG / name).read_text().splitlines() if line]


CORPUS = {d["id"]: d["text"] for d in jsonl("corpus.jsonl")}
CASES = {c.id: c for c in parse_jsonl((RAG / "retrieval.jsonl").read_bytes())}
RANKINGS = {r["id"]: r for r in jsonl("rankings.jsonl")}
LABELLED = parse_labels((RAG / "faithfulness-labels.jsonl").read_bytes(), "faithfulness")


def fields(value: Any) -> dict[str, Any]:
    """A JSON object from a case or labelled item."""
    assert isinstance(value, dict)
    return value


def output(ids: list[str]) -> str:
    return Ranking(documents=[Document(id=d, text=CORPUS[d]) for d in ids]).to_output()


def metric(name: str, case_id: str, retriever: str, **kw: Any) -> float:
    scorer = RankingMetric(name, **kw)
    result = asyncio.run(scorer.score(CASES[case_id], output(RANKINGS[case_id][retriever])))
    assert result.value is not None, result.detail
    return result.value


# ---------------------------------------------------------------- the sets themselves


def test_the_corpus_and_the_retrieval_cases_agree() -> None:
    assert len(CORPUS) == 18
    assert len(CASES) == 12
    assert set(RANKINGS) == set(CASES)
    for case in CASES.values():
        assert query_of(case).query.endswith("?")
        labels = parse_relevance(fields(case.expected)["relevance"])
        assert set(labels) <= set(CORPUS)
        assert all(0 <= grade <= MAX_GRADE for grade in labels.values())
        # Every query has a document that answers it on its own.
        assert 3 in labels.values()
        assert fields(case.expected)["answer"]
    for row in RANKINGS.values():
        for retriever in ("baseline", "regressed"):
            ids = row[retriever]
            assert len(ids) == len(set(ids)) == 5
            assert set(ids) <= set(CORPUS)


def test_the_faithfulness_labels_follow_the_annotated_claims() -> None:
    assert len(LABELLED) == 16
    assert sum(item.label == "faithful" for item in LABELLED) == 9
    for item in LABELLED:
        expected = fields(item.expected)
        case = CASES[expected["case"]]
        assert fields(item.input)["question"] == case.input
        # The answer was written from the baseline's top 3, with the corpus texts.
        sources = sources_of(Case(id=item.id, input=item.input), "contexts")
        assert [s.id for s in sources] == RANKINGS[case.id]["baseline"][:3]
        assert all(s.text == CORPUS[s.id] for s in sources)
        # Unfaithful exactly when the labeller marked a sentence of the answer.
        assert all(claim in sentences(item.output) for claim in expected["unsupported"])
        assert (item.label == "unfaithful") == bool(expected["unsupported"])


def test_the_faithfulness_labels_are_their_own_mode() -> None:
    raw = (RAG / "faithfulness-labels.jsonl").read_text()
    with pytest.raises(InvalidInputError, match="label must be one of 1, 2, 3, 4, 5"):
        parse_labels(raw, "rubric")
    numeric = raw.replace('"label": "unfaithful"', '"label": 0')
    with pytest.raises(
        InvalidInputError, match="line 2: label must be one of unfaithful, faithful"
    ):
        parse_labels(numeric, "faithfulness")
    assert FAITHFULNESS_LABELS == ("unfaithful", "faithful")


# ---------------------------------------------------------------- ranking values by hand
#
# Grades (q01): kb-01 3, kb-03 1, kb-02 0. Relevant means grade >= 1.
# log2(3) = 1.5849625, so 1/log2(3) = 0.6309298 and 2/log2(3) = 1.2618595.


def test_hand_computed_values_for_single_cases() -> None:
    # q01 baseline: kb-01, kb-03, kb-02. DCG@3 = 3/1 + 1/log2(3) = the ideal.
    assert metric("n", "q01", "baseline", metric="ndcg", k=3) == pytest.approx(1.0)
    # q01 regressed: kb-02 (0), kb-03 (1), kb-14 (unlabelled).
    # DCG@3 = 1/log2(3) = 0.6309298; IDCG@3 = 3.6309298; nDCG = 0.1737654.
    assert metric("n", "q01", "regressed", metric="ndcg", k=3) == pytest.approx(
        0.6309298 / 3.6309298, abs=1e-6
    )
    assert metric("r", "q01", "regressed", metric="recall", k=3) == 0.5
    assert metric("m", "q01", "regressed", metric="mrr") == 0.5
    # q10 regressed: kb-01 (2) first, kb-03 (3) at rank 5.
    # DCG@3 = 2; IDCG@3 = 3 + 2/log2(3) = 4.2618595; nDCG = 0.4692787.
    assert metric("n", "q10", "regressed", metric="ndcg", k=3) == pytest.approx(
        2 / 4.2618595, abs=1e-6
    )
    # q04 regressed: kb-05 (1), kb-07 (0), kb-06 (3), kb-13, kb-17 (1).
    # Linear: DCG@3 = 1 + 3/2 = 2.5; IDCG@3 (3, 1, 1) = 3 + 0.6309298 + 0.5.
    assert metric("n", "q04", "regressed", metric="ndcg", k=3) == pytest.approx(
        2.5 / 4.1309298, abs=1e-6
    )
    # Exponential gains 2^g - 1: DCG@3 = 1 + 7/2 = 4.5; IDCG@3 = 7 + 0.6309298 + 0.5.
    assert metric("n", "q04", "regressed", metric="ndcg", k=3, gain="exponential") == pytest.approx(
        4.5 / 8.1309298, abs=1e-6
    )
    # AP@5: relevant at ranks 1, 3 and 5: (1/1 + 2/3 + 3/5) / 3 = 34/45.
    assert metric("a", "q04", "regressed", metric="map", k=5) == pytest.approx(34 / 45)
    assert metric("r", "q04", "regressed", metric="recall", k=3) == pytest.approx(2 / 3)
    # q05 regressed: one relevant document in five.
    assert metric("p", "q05", "regressed", metric="precision", k=5) == 0.2
    # q09 regressed: kb-12 only at rank 5.
    assert metric("r", "q09", "regressed", metric="recall", k=3) == 0.0
    assert metric("h", "q09", "regressed", metric="hit_rate", k=5) == 1.0
    assert metric("m", "q09", "regressed", metric="mrr") == 0.2
    # Grade 2 and above only: kb-03 (3) and kb-01 (2) for q10; kb-01 is first.
    assert metric("m", "q10", "regressed", metric="mrr", min_grade=2) == 1.0
    assert metric("m", "q01", "regressed", metric="mrr", min_grade=2) == 0.25


# Reciprocal ranks of the first relevant document, by hand from rankings.jsonl.
BASELINE_RR = {q: Fraction(1) for q in CASES} | {"q05": Fraction(1, 2), "q11": Fraction(1, 2)}
REGRESSED_RR = {
    "q01": Fraction(1, 2),
    "q02": Fraction(1, 2),
    "q03": Fraction(1),
    "q04": Fraction(1),
    "q05": Fraction(1, 3),
    "q06": Fraction(1, 3),
    "q07": Fraction(1),
    "q08": Fraction(1, 2),
    "q09": Fraction(1, 5),
    "q10": Fraction(1),
    "q11": Fraction(1, 5),
    "q12": Fraction(1),
}
# recall@3: the baseline finds every relevant document in its top 3.
REGRESSED_RECALL_3 = {
    "q01": Fraction(1, 2),
    "q02": Fraction(1, 2),
    "q03": Fraction(1),
    "q04": Fraction(2, 3),
    "q05": Fraction(1),
    "q06": Fraction(1),
    "q07": Fraction(1, 2),
    "q08": Fraction(1, 2),
    "q09": Fraction(0),
    "q10": Fraction(1, 2),
    "q11": Fraction(0),
    "q12": Fraction(1, 2),
}


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_hand_computed_reciprocal_ranks_and_recall(case_id: str) -> None:
    assert metric("m", case_id, "baseline", metric="mrr") == pytest.approx(
        float(BASELINE_RR[case_id])
    )
    assert metric("m", case_id, "regressed", metric="mrr") == pytest.approx(
        float(REGRESSED_RR[case_id])
    )
    assert metric("r", case_id, "baseline", metric="recall", k=3) == 1.0
    assert metric("r", case_id, "regressed", metric="recall", k=3) == pytest.approx(
        float(REGRESSED_RECALL_3[case_id])
    )


def test_hand_computed_means_show_the_regression() -> None:
    def mean(name: str, retriever: str, **kw: Any) -> float:
        return sum(metric(name, q, retriever, **kw) for q in CASES) / len(CASES)

    # Ten firsts and two seconds: 11/12. Regressed: (5 + 3/2 + 2/3 + 2/5) / 12.
    assert sum(BASELINE_RR.values()) / 12 == Fraction(11, 12)
    assert sum(REGRESSED_RR.values()) / 12 == Fraction(227, 360)
    assert mean("m", "baseline", metric="mrr") == pytest.approx(11 / 12)
    assert mean("m", "regressed", metric="mrr") == pytest.approx(227 / 360)
    # Six halves, three ones and a two-thirds: (3 + 3 + 2/3) / 12 = 5/9.
    assert sum(REGRESSED_RECALL_3.values()) / 12 == Fraction(5, 9)
    assert mean("r", "regressed", metric="recall", k=3) == pytest.approx(5 / 9)
    assert mean("r", "baseline", metric="recall", k=3) == 1.0
    for name in ("ndcg", "map"):
        assert mean(name, "regressed", metric=name, k=5) < mean(name, "baseline", metric=name, k=5)


# ---------------------------------------------------------------- faithfulness against the labels


def sentences(text: str) -> list[str]:
    return [s for s in regex.split(r"(?<=\.)\s+", text.strip()) if s]


def system_of(body: dict[str, Any]) -> str:
    return str(body["system"][0]["text"])


def content_of(body: dict[str, Any]) -> str:
    return str(body["messages"][0]["content"])


HUMAN_UNSUPPORTED = {claim for item in LABELLED for claim in fields(item.expected)["unsupported"]}
# Where the scripted judge disagrees with the labeller: it accepts an invented
# permission (f13) and doubts a paraphrase of "over 1 GB" as "large" (f08).
JUDGE_ACCEPTS = {"Admins can also delete the workspace."}
JUDGE_DOUBTS = {"Large exports are emailed as links valid for 7 days."}


def judge_handler(body: dict[str, Any]) -> httpx2.Response:
    """Claims are the answer's sentences; a claim is unsupported when the
    labeller marked it, except where the judge disagrees."""
    content = content_of(body)
    if system_of(body) == EXTRACT_SYSTEM:
        answer = content.split("<answer>\n", 1)[1].split("\n</answer>", 1)[0]
        return json_reply(message(json.dumps({"claims": sentences(answer)})))
    assert system_of(body) == VERIFY_SYSTEM
    block = content.split("<claims>\n", 1)[1].split("\n</claims>", 1)[0]
    verdicts: list[dict[str, Any]] = []
    for number, line in enumerate(block.splitlines(), start=1):
        claim = regex.sub(r"^\[\d+\] ", "", line)
        bad = (claim in HUMAN_UNSUPPORTED or claim in JUDGE_DOUBTS) and claim not in JUDGE_ACCEPTS
        verdicts.append(
            {
                "claim": number,
                "reasoning": "not stated" if bad else "stated",
                "verdict": "unsupported" if bad else "supported",
                "sources": [],
            }
        )
    return json_reply(message(json.dumps({"verdicts": verdicts})))


def faithfulness_judge() -> tuple[FaithfulnessJudge, FakeAnthropic]:
    fake = FakeAnthropic()
    fake.default = judge_handler
    return FaithfulnessJudge("faithfulness", client=fake.client(), retry=RETRY), fake


def test_the_judge_lists_the_unsupported_claim_of_each_unfaithful_answer() -> None:
    judge, _ = faithfulness_judge()
    by_id = {item.id: item for item in LABELLED}
    item = by_id["f02"]
    j = asyncio.run(judge.judge(Case(id=item.id, input=item.input), item.output))
    assert (j.value, j.label) == (0.5, "unfaithful")
    assert '"The reset link expires after 24 hours." (not in the sources' in j.detail
    assert "Open Settings" not in j.detail
    item = by_id["f15"]
    j = asyncio.run(judge.judge(Case(id=item.id, input=item.input), item.output))
    assert (j.value, j.label, j.detail) == (1.0, "faithful", "2/2 claims supported")


def test_calibration_against_the_labels_gives_the_hand_computed_kappa() -> None:
    judge, _ = faithfulness_judge()
    assert label_mode(judge) == "faithfulness"
    report = asyncio.run(calibrate(judge, LABELLED, concurrency=4))
    # Humans: 7 unfaithful, 9 faithful. The judge agrees on 6 + 8, and swaps
    # one each way (f13, f08). Observed 14/16; chance (7*7 + 9*9)/256 = 130/256.
    # kappa = (224 - 130) / (256 - 130) = 47/63 = 0.746.
    assert (report.n, report.judged) == (16, 16)
    assert report.labels == ["unfaithful", "faithful"]
    assert report.confusion == [[6, 1], [1, 8]]
    assert report.agreement == 14 / 16
    assert report.kappa == pytest.approx(47 / 63)
    assert report.kappa >= 0.6
    assert report.weighted_kappa is None
    assert sorted(report.disagreements) == ["f08", "f13"]
    assert report.kappa == pytest.approx(
        cohens_kappa(
            [i.label for i in LABELLED],
            [
                "faithful" if i.id == "f13" else "unfaithful" if i.id == "f08" else i.label
                for i in LABELLED
            ],
            FAITHFULNESS_LABELS,
        )
    )


def test_an_answer_the_judge_cannot_check_is_left_out() -> None:
    fake = FakeAnthropic()
    fake.default = lambda body: json_reply(message("not json"))
    judge = FaithfulnessJudge("faithfulness", client=fake.client(), retry=RETRY)
    with pytest.raises(InvalidInputError, match="could not grade any"):
        asyncio.run(calibrate(judge, LABELLED[:2], concurrency=1))
    # Items without sources are unscorable, not unfaithful.
    bare = [LabelledItem(id="x", input="q", output="o", label="faithful"), *LABELLED[:1]]
    judge, _ = faithfulness_judge()
    report = asyncio.run(calibrate(judge, bare, concurrency=1))
    assert (report.n, report.judged) == (2, 1)


def test_cli_calibrates_the_example_judge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, fake = faithfulness_judge()
    monkeypatch.setattr(cli.App, "claude", lambda self: fake.client())
    db = f"sqlite:///{tmp_path / 'cli.db'}"
    out = io.StringIO()
    args = ["calibrate", "--judge", str(RAG / "faithfulness.toml")]
    code = cli.main(["--database-url", db, *args, str(RAG / "faithfulness-labels.jsonl")], out=out)
    assert code == 0, out.getvalue()
    text = out.getvalue()
    assert "labelled 16  judged 16  agreement 87.5%" in text
    assert "Cohen's kappa 0.746" in text
    assert "quadratic" not in text
    assert "  unfaithful      6      1" in text
    assert "disagreements: f08, f13" in text or "disagreements: f13, f08" in text
    engine = create_engine(db)
    try:
        with Session(engine) as s:
            row = s.scalars(select(Calibration)).one()
            assert row.judge_spec["mode"] == "faithfulness"
            assert row.kappa == pytest.approx(47 / 63)
    finally:
        engine.dispose()
    # A rubric judge cannot be calibrated on faithfulness labels.
    rubric = tmp_path / "rubric.toml"
    rubric.write_text('[judge]\nrubric = "Grounded."\n')
    code = cli.main(
        [
            "--database-url",
            db,
            "calibrate",
            "--judge",
            str(rubric),
            str(RAG / "faithfulness-labels.jsonl"),
        ],
        out=io.StringIO(),
    )
    assert code == 2


# ---------------------------------------------------------------- constructed failure causes

# Recall is evidence (every relevant document in the top 3); the gate is on
# faithfulness (every claim supported, the decision calibrated above) and on
# answer completeness.
ROLES = [
    MetricRole("recall_at_3", "ranking", "retrieval", False, 1.0),
    MetricRole("faithfulness", "faithfulness", "grounding", True, 1.0),
    MetricRole("answer_completeness", "answer_completeness", "answer", True, 0.5),
]


def faithfulness_of(item_id: str) -> float:
    """The labeller's claim-level faithfulness of a labelled answer."""
    item = next(i for i in LABELLED if i.id == item_id)
    bad = fields(item.expected)["unsupported"]
    return 1 - len(bad) / len(sentences(item.output))


@pytest.mark.parametrize(
    ("case_id", "retriever", "faithful", "complete", "expected"),
    [
        # q09 under the regressed retriever: the late-fee page is at rank 5,
        # so the answer, faithful to what it got, says nothing about fees.
        ("q09", "regressed", 1.0, 0.0, ("retrieval_miss", ["recall_at_3"])),
        ("q11", "regressed", 1.0, 0.0, ("retrieval_miss", ["recall_at_3"])),
        # Retrieval missed and the answer invented: retrieval is fixed first.
        ("q01", "regressed", faithfulness_of("f02"), 1.0, ("retrieval_miss", ["recall_at_3"])),
        # q08 retrieved, and f13 adds a permission no source grants.
        ("q08", "baseline", faithfulness_of("f13"), 1.0, ("unsupported_answer", ["faithfulness"])),
        ("q02", "baseline", faithfulness_of("f04"), 0.0, ("unsupported_answer", ["faithfulness"])),
        # q07 retrieved, and a faithful answer leaves out the member limit.
        ("q07", "baseline", 1.0, 0.0, ("ignored_context", ["answer_completeness"])),
        # Retrieved, faithful and complete: not failing.
        ("q01", "baseline", faithfulness_of("f01"), 1.0, None),
        # Half of q12's reference is enough at 0.5.
        ("q12", "baseline", 1.0, 0.5, None),
    ],
)
def test_constructed_failures_get_their_cause(
    case_id: str,
    retriever: str,
    faithful: float,
    complete: float,
    expected: tuple[str, list[str]] | None,
) -> None:
    recall = metric("r", case_id, retriever, metric="recall", k=3)
    scores = {"recall_at_3": recall, "faithfulness": faithful, "answer_completeness": complete}
    found = attribute("ok", scores, ROLES)
    assert (found if found is None else found[:2]) == expected


def test_without_retrieval_evidence_the_cause_is_unattributed() -> None:
    found = attribute("ok", {"faithfulness": 1.0, "answer_completeness": 0.0}, ROLES)
    assert found is not None
    assert found[0] == "unattributed"
