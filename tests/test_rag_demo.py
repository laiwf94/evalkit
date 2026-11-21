"""The RAG demo end to end: examples/rag/server.py (two retrievers, a reader and
a grader, standard library only) behind the configs in examples/rag, run through
the CLI the way examples/rag/demo.sh runs them.

The pipeline on the baseline retriever answers every question; on the regressed
one the answering document drops out of the top 3 for eight questions, every one
of them is attributed to a retrieval miss, and the gate fails the change on the
retrieval metrics and on answer completeness. Then the fakes themselves: the
grader's verdicts, the reader's choice and the server's refusals.
"""

from __future__ import annotations

import http.client
import importlib.util
import io
import json
import os
import threading
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from evalkit import cli
from evalkit.db import make_engine, make_session_factory
from evalkit.db.models import Result, ResultStage
from evalkit.scorers.faithfulness import EXTRACT_SYSTEM, VERIFY_SYSTEM, Source, verify_prompt
from evalkit.scorers.rag import RELEVANCE_SYSTEM, relevance_prompt
from evalkit.targets.pipeline import render_contexts
from evalkit.targets.retrieval import Document

ROOT = Path(__file__).parent.parent
RAG = ROOT / "examples" / "rag"
# The queries whose answering document the regressed retriever pushes out of
# the top 3 (rankings.jsonl): every other query keeps it.
LOST = {"q01", "q02", "q07", "q08", "q09", "q10", "q11", "q12"}


def _load_server() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rag_server", RAG / "server.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fake = _load_server()


def jsonl(name: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (RAG / name).read_text().splitlines() if line]


RANKINGS = {r["id"]: r for r in jsonl("rankings.jsonl")}
ANSWERING = {
    c["id"]: {d for d, grade in c["expected"]["relevance"].items() if grade == 3}
    for c in jsonl("retrieval.jsonl")
}


class Demo:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.server = fake.make_server("127.0.0.1", 0, fake.Service())
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.port = self.server.server_address[1]
        # The checked-in operator files on the port picked, as demo.sh writes them.
        for name, variable in (
            ("endpoints.toml", "EVALKIT_HTTP_TARGETS_FILE"),
            ("providers.toml", "EVALKIT_PROVIDERS_FILE"),
        ):
            text = (RAG / name).read_text()
            assert ":8766" in text
            path = tmp_path / name
            path.write_text(text.replace(":8766", f":{self.port}"))
            monkeypatch.setenv(variable, str(path))
        monkeypatch.setenv("EVALKIT_PRICES_FILE", str(RAG / "prices.toml"))
        url = f"sqlite:///{tmp_path / 'rag.db'}"
        monkeypatch.setenv("EVALKIT_DATABASE_URL", url)
        self.engine = make_engine(url)
        self.sessions: sessionmaker[Session] = make_session_factory(self.engine)

    def evalkit(self, *args: str, expect: int = 0) -> str:
        out = io.StringIO()
        code = cli.main(list(args), out=out)
        assert code == expect, f"exit {code}\n{out.getvalue()}"
        return out.getvalue()

    def json(self, *args: str, expect: int = 0) -> Any:
        return json.loads(self.evalkit(*args, "--json", expect=expect))

    def scores(self, run_id: str) -> dict[str, dict[str, float | None]]:
        with self.sessions() as s:
            return {
                r.case_id: {k: v["value"] for k, v in (r.scores or {}).items()}
                for r in s.scalars(select(Result).where(Result.run_id == run_id))
            }

    def stages(self, run_id: str) -> dict[str, list[ResultStage]]:
        found: dict[str, list[ResultStage]] = {}
        with self.sessions() as s:
            rows = s.scalars(
                select(ResultStage)
                .where(ResultStage.run_id == run_id)
                .order_by(ResultStage.case_id, ResultStage.position)
            )
            for row in rows:
                s.expunge(row)
                found.setdefault(row.case_id, []).append(row)
        return found

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.engine.dispose()


@pytest.fixture
def demo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Demo]:
    for key in list(os.environ):
        if key.upper().endswith("_PROXY") or key.startswith(("EVALKIT_", "ANTHROPIC_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.chdir(ROOT)
    d = Demo(tmp_path, monkeypatch)
    yield d
    d.close()


def run_id(output: str) -> str:
    first = output.split()
    assert first[0] == "run", output
    return first[1]


# ---------------------------------------------------------------- end to end


def test_the_demo_gates_a_retriever_regression_and_blames_retrieval(demo: Demo) -> None:
    base = run_id(demo.evalkit("run", "examples/rag/pipeline.toml"))
    nxt = run_id(demo.evalkit("run", "examples/rag/pipeline-next.toml"))

    # Both stages of every case are stored: the ranking of 5, then the answer,
    # which is the text of an answering document the reader was given.
    corpus = {d["id"]: d["text"] for d in jsonl("corpus.jsonl")}
    for rid, retriever in ((base, "baseline"), (nxt, "regressed")):
        stages = demo.stages(rid)
        assert set(stages) == set(RANKINGS)
        for case_id, (retrieval, generation) in stages.items():
            assert (retrieval.stage, generation.stage) == ("retrieval", "generation")
            ranked = [d["id"] for d in json.loads(retrieval.output)["documents"]]
            assert ranked == RANKINGS[case_id][retriever]
            if case_id in LOST and retriever == "regressed":
                assert not ANSWERING[case_id] & set(ranked[:3])
                assert generation.output not in {corpus[d] for d in ANSWERING[case_id]}
            else:
                assert generation.output in {corpus[d] for d in ANSWERING[case_id]}
                assert generation.output in {corpus[d] for d in ranked[:3]}

    # The baseline answers everything; recall@3 is 1 by construction of the set.
    for case_id, scores in demo.scores(base).items():
        assert scores["recall_at_3"] == 1.0, case_id
        assert scores["answer_completeness"] == 1.0, case_id
        assert scores["faithfulness"] == 1.0, case_id
        assert scores["context_recall"] == 1.0, case_id
    report = demo.json("report", "rag", "label:rag")
    assert report["run_id"] == base
    assert report["failing"] == 0

    # The regressed retriever: mean recall@3 5/9 (examples/rag/README.md), and
    # the answers to the eight lost queries miss every reference point.
    regressed = demo.scores(nxt)
    recall = sum(Fraction(s["recall_at_3"] or 0).limit_denominator(9) for s in regressed.values())
    assert recall / len(regressed) == Fraction(5, 9)
    for case_id, scores in regressed.items():
        expected = 0.0 if case_id in LOST else 1.0
        assert scores["answer_completeness"] == expected, case_id
        assert scores["context_recall"] == expected, case_id
        # The reader only ever copies a source, or says it cannot answer (no claims).
        assert scores["faithfulness"] in (1.0, None), case_id

    report = demo.json("report", "rag", "label:rag-next")
    assert report["failing"] == len(LOST)
    assert report["causes"]["retrieval_miss"] == len(LOST)
    assert {c["case_id"] for c in report["shown"]} == LOST
    assert {c["cause"] for c in report["shown"]} == {"retrieval_miss"}
    text = demo.evalkit("report", "rag", "label:rag-next")
    assert "retrieval miss           8   100.0%" in text

    gate = demo.json("gate", "label:rag", "label:rag-next", expect=1)
    assert set(gate["regressions"]) == {"recall_at_3", "ndcg_at_5", "answer_completeness"}


def test_a_rerun_of_the_pipeline_is_served_from_the_caches(demo: Demo) -> None:
    demo.evalkit("run", "examples/rag/pipeline.toml")
    before = dict(json.loads(_get(demo.port, "/stats")))
    again = run_id(demo.evalkit("run", "examples/rag/pipeline.toml", "--label", "rag-rerun"))
    # No retrieval, generation or grading request reached the fakes the second time.
    assert json.loads(_get(demo.port, "/stats")) == before
    for retrieval, generation in demo.stages(again).values():
        assert retrieval.cached
        assert generation.cached
    demo.evalkit("gate", "label:rag", "label:rag-rerun")


def test_the_demo_configs_differ_only_in_label_and_retriever() -> None:
    base = (RAG / "pipeline.toml").read_text().split("\nlabel = ", 1)[1].splitlines()
    nxt = (RAG / "pipeline-next.toml").read_text().split("\nlabel = ", 1)[1].splitlines()
    changed = [(a, b) for a, b in zip(base, nxt, strict=True) if a != b]
    assert changed == [
        ('"rag"', '"rag-next"'),
        ('endpoint = "kb-search"', 'endpoint = "kb-search-next"'),
    ]


# ---------------------------------------------------------------- the fakes


def test_the_fake_recognises_evalkits_judge_prompts() -> None:
    assert EXTRACT_SYSTEM.startswith(fake.EXTRACT)
    assert VERIFY_SYSTEM.startswith(fake.VERIFY)
    assert RELEVANCE_SYSTEM.startswith(fake.RELEVANCE)


def test_claims_are_sentences_that_state_something() -> None:
    answer = "Yes. Annual plans are refunded within 14 days. I hope this helps! Ok."
    assert fake.extract_claims(answer) == ["Annual plans are refunded within 14 days."]
    assert fake.extract_claims(fake.NO_ANSWER) == []


def test_the_grader_supports_contradicts_or_leaves_unsupported() -> None:
    sources = [
        ("kb-05", "Annual plans can be refunded in full within 14 days of purchase."),
        ("kb-13", "The Free plan allows 3 members."),
    ]
    assert fake.verdict("Annual plans are refunded in full within 14 days.", sources) == (
        "supported",
        ["kb-05"],
    )
    # Every other word is there, the number is not.
    assert fake.verdict("Annual plans are refunded in full within 30 days.", sources) == (
        "contradicted",
        ["kb-05"],
    )
    assert fake.verdict("Monthly plans include phone support.", sources) == ("unsupported", [])
    prompt = verify_prompt(
        [Source(i, t) for i, t in sources],
        ["The Free plan allows 3 members.", "The Free plan allows 5 members."],
    )
    verdicts = fake.verify(prompt)["verdicts"]
    assert [(v["claim"], v["verdict"], v["sources"]) for v in verdicts] == [
        (1, "supported", ["kb-13"]),
        (2, "contradicted", ["kb-13"]),
    ]
    with pytest.raises(fake.BadRequestError, match="no claims"):
        fake.verify(verify_prompt([Source(i, t) for i, t in sources], []))


def test_the_grader_judges_relevance_by_shared_words() -> None:
    prompt = relevance_prompt(
        "How do I download an invoice PDF?",
        [
            Source("kb-04", "Select an invoice and choose Download to get a PDF."),
            Source("kb-13", "The Free plan allows 3 members."),
        ],
    )
    assert [s["relevant"] for s in fake.relevance(prompt)["sources"]] == [True, False]
    with pytest.raises(fake.BadRequestError, match="no sources"):
        fake.relevance("<question>\nanything\n</question>")


def test_the_reader_copies_the_document_closest_to_the_question() -> None:
    docs = [
        Document(id="a", text="Billing details are changed by owners."),
        Document(id="b", text="Invoices are under Billing > Invoices. Choose Download."),
        Document(id="c", text="Download any invoice from Billing."),
    ]

    def prompt(documents: list[Document]) -> str:
        contexts = render_contexts(documents)
        question = "<question>\nWhere do I download an invoice?\n</question>"
        return f"{question}\n\n<documents>\n{contexts}\n</documents>"

    # b and c share "download" and "invoice"; b comes first.
    assert fake.read(prompt(docs)) == docs[1].text
    assert fake.read(prompt([docs[0]])) == fake.NO_ANSWER
    assert fake.read(prompt([])) == fake.NO_ANSWER
    with pytest.raises(fake.BadRequestError, match="<question> and <documents>"):
        fake.read("Where do I download an invoice?")


def test_the_retrievers_replay_the_recorded_rankings_or_rank_by_words() -> None:
    retrievers = fake.Retrievers()
    q09 = next(c for c in jsonl("retrieval.jsonl") if c["id"] == "q09")
    for name in fake.RETRIEVERS:
        found = retrievers.search(name, {"query": q09["input"], "k": 5})["results"]
        assert [r["id"] for r in found] == RANKINGS["q09"][name]
        assert [r["score"] for r in found] == [1.0, 0.5, 0.3333, 0.25, 0.2]
    other = retrievers.search("baseline", {"query": "Is SMS two-factor supported?", "k": 2})
    assert other["results"][0]["id"] == "kb-02"
    assert len(other["results"]) <= 2
    assert retrievers.search("baseline", {"query": "zzz qqq"}) == {"results": []}
    bad: list[Any] = [{"k": 3}, {"query": "x", "k": 0}, {"query": "x", "k": True}, []]
    for body in bad:
        with pytest.raises(fake.BadRequestError):
            retrievers.search("baseline", body)


def _request(port: int, method: str, path: str, body: bytes | None = None) -> tuple[int, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        headers = {"Content-Type": "application/json"} if body is not None else {}
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def _get(port: int, path: str) -> str:
    status, body = _request(port, "GET", path)
    assert status == 200
    return body.decode()


@pytest.fixture
def served() -> Iterator[int]:
    server = fake.make_server("127.0.0.1", 0, fake.Service())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def test_the_server_speaks_chat_completions_and_refuses_what_it_cannot_answer(
    served: int,
) -> None:
    assert _get(served, "/healthz") == "ok"
    contexts = render_contexts([Document(id="kb-13", text="The Free plan allows 3 members.")])
    body: dict[str, Any] = {
        "model": "reader-1",
        "messages": [
            {"role": "system", "content": "Answer from the documents."},
            {
                "role": "user",
                "content": f"<question>\nHow many members on the Free plan?\n</question>\n"
                f"<documents>\n{contexts}\n</documents>",
            },
        ],
    }
    status, raw = _request(served, "POST", "/v1/chat/completions", json.dumps(body).encode())
    assert status == 200
    answer = json.loads(raw)
    assert answer["choices"][0]["message"]["content"] == "The Free plan allows 3 members."
    assert answer["model"] == "reader-1"
    assert answer["usage"]["prompt_tokens"] > 0

    refused = [
        ("/v1/chat/completions", {"model": "x", "messages": []}, 400),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}, 400),
        (
            "/v1/chat/completions",
            {"model": "x", "messages": [{"role": "user", "content": [{"type": "text"}]}]},
            400,
        ),
        (
            "/v1/chat/completions",
            {
                "model": "x",
                "messages": [{"role": "user", "content": "<question>\nq\n</question>"}],
                "response_format": {"type": "json_schema"},
            },
            400,
        ),
        ("/search/other", {"query": "x"}, 404),
        ("/search/baseline", {"query": 3}, 400),
    ]
    for path, value, expected in refused:
        status, raw = _request(served, "POST", path, json.dumps(value).encode())
        assert status == expected, (path, value, raw)
        assert json.loads(raw)["error"]["message"]
    status, _ = _request(served, "POST", "/search/baseline", b"{not json")
    assert status == 400
    status, _ = _request(served, "GET", "/nowhere")
    assert status == 404
    stats = json.loads(_get(served, "/stats"))
    assert stats["/v1/chat/completions"] == 5
    assert stats["/search/baseline"] == 1


def test_the_server_closes_a_connection_whose_body_it_will_not_read(served: int) -> None:
    conn = http.client.HTTPConnection("127.0.0.1", served, timeout=10)
    try:
        conn.putrequest("POST", "/search/baseline")
        conn.putheader("Content-Length", str(fake.MAX_BODY + 1))
        conn.endheaders()
        response = conn.getresponse()
        assert response.status == 400
        assert response.getheader("Connection") == "close"
    finally:
        conn.close()


def test_main_writes_the_port_file_and_serves(tmp_path: Path) -> None:
    port_file = tmp_path / "port"
    done = threading.Event()

    def serve() -> None:
        fake.main(["--port", "0", "--port-file", str(port_file)])
        done.set()

    original = fake.ThreadingHTTPServer.serve_forever

    def once(self: Any, poll_interval: float = 0.5) -> None:
        raise KeyboardInterrupt

    fake.ThreadingHTTPServer.serve_forever = once
    try:
        thread = threading.Thread(target=serve)
        thread.start()
        thread.join(10)
    finally:
        fake.ThreadingHTTPServer.serve_forever = original
    assert done.is_set()
    assert int(port_file.read_text()) > 0
