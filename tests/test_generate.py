"""Synthetic generation: the spec, seed documents, requests, a scripted generator,
near-duplicate removal against the dataset and within the drafts, failures,
caching, and the review step (approve, reject) through the library, the CLI and
the API."""

import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import pytest
import regex
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from evalkit import cli
from evalkit import generate as gen
from evalkit.cache import VerdictStore
from evalkit.dataset_edits import DatasetEdit, apply_edit, dataset_log
from evalkit.datasets import Case, get_dataset, import_dataset, load_cases
from evalkit.db.models import CaseDraft, Generation
from evalkit.errors import ConflictError, InvalidInputError, NotFoundError
from evalkit.generate import (
    GENERATE_SYSTEM,
    GenerateSpec,
    Generator,
    SeedDocument,
    approve,
    build_case,
    generate,
    generation_info,
    get_generation,
    list_drafts,
    list_generations,
    load_documents,
    pick_examples,
    plan_requests,
    reject,
    reply_schema,
)
from evalkit.ratelimit import RetryPolicy
from tests.fake_anthropic import FakeAnthropic, error_reply, json_reply, message

RETRY = RetryPolicy(max_attempts=1, base_delay=0)
DOCS = [
    SeedDocument("refunds.md", "Refunds are paid within 14 days to the original card."),
    SeedDocument("shipping.md", "Orders ship in 2 working days. Express costs 9 euros."),
    SeedDocument("account.md", "Reset a password from the sign-in page with the link."),
]
EXISTING = [
    Case(id="old-1", input={"question": "How long do refunds take to reach my card?"}),
    Case(id="old-2", input={"question": "Can I change the delivery address?"}),
]
# What the scripted model writes for each document.
SCRIPT: dict[str, list[dict[str, Any]]] = {
    "refunds.md": [
        # A near-duplicate of old-1, already in the dataset.
        {"input": "How long do refunds take to reach my card ?", "expected": "14 days", "tags": []},
        {"input": "Where is my refund paid?", "expected": "To the original card.", "tags": []},
        {"input": "   ", "expected": "", "tags": []},
    ],
    "shipping.md": [
        {"input": "How fast do orders ship?", "expected": "2 working days", "tags": ["shipping"]},
        {"input": "What does express cost?", "expected": "9 euros", "tags": ["made-up"]},
        # The same input as the first: an exact repeat within the generation.
        {"input": "How fast do orders ship?", "expected": "In 2 days", "tags": []},
    ],
    "account.md": [
        {"input": "How do I reset my password?", "expected": "", "tags": ["account"]},
        # A near-duplicate of the case above, from the same reply.
        {"input": "How do I reset my password??", "expected": "", "tags": []},
    ],
}
SPEC = GenerateSpec(
    dataset="support",
    instructions="Write questions customers ask about the document.",
    documents=("kb/*.md",),
    per_document=3,
    input_key="question",
    allowed_tags=("shipping", "account"),
    split="synthetic",
)


def document_of(body: dict[str, Any]) -> str | None:
    found = regex.search(r'<document id="([^"]+)">', body["messages"][0]["content"])
    return found[1] if found else None


def scripted(
    script: dict[str, list[dict[str, Any]]] | None = None,
) -> Callable[[dict[str, Any]], httpx2.Response]:
    script = SCRIPT if script is None else script

    def handler(body: dict[str, Any]) -> httpx2.Response:
        doc = document_of(body)
        assert doc is not None
        # Documents loaded from files are named by their path (kb/refunds.md).
        return json_reply(message(json.dumps({"cases": script[doc.rsplit("/", 1)[-1]]})))

    return handler


def generator(fake: FakeAnthropic, spec: GenerateSpec = SPEC, **kw: Any) -> Generator:
    return Generator(spec, fake.client(), retry=RETRY, **kw)


def seed(sessions: sessionmaker[Session], cases: list[Case] = EXISTING) -> None:
    with sessions() as s:
        import_dataset(s, "support", cases)


# ---------------------------------------------------------------- the spec


def test_spec_needs_documents_or_a_count() -> None:
    base = {"dataset": "d", "instructions": "x"}
    with pytest.raises(ValidationError, match="either documents or count"):
        GenerateSpec.model_validate(base)
    with pytest.raises(ValidationError, match="either documents or count"):
        GenerateSpec.model_validate({**base, "documents": "a.md", "count": 3})
    with pytest.raises(ValidationError, match="examples or examples_from_dataset"):
        GenerateSpec.model_validate(
            {**base, "count": 3, "examples": "e.jsonl", "examples_from_dataset": True}
        )
    with pytest.raises(ValidationError, match="tags must be unique"):
        GenerateSpec.model_validate({**base, "count": 3, "allowed_tags": ["a", "a"]})
    with pytest.raises(ValidationError, match="dataset"):
        GenerateSpec.model_validate({**base, "count": 3, "dataset": "../x"})
    spec = GenerateSpec.model_validate({**base, "documents": "a.md", "dedupe": {"threshold": 0.6}})
    assert spec.documents == ("a.md",)
    assert spec.dedupe.deduper().threshold == 0.6
    assert spec.tags == ("synthetic",)


def test_documents_from_files_and_jsonl(tmp_path: Path) -> None:
    (tmp_path / "kb").mkdir()
    (tmp_path / "kb" / "b.md").write_text("second")
    (tmp_path / "kb" / "a.md").write_text("first")
    (tmp_path / "more.jsonl").write_text(
        '{"id": "x1", "text": "from jsonl"}\n\n{"id": 7, "text": "numbered"}\n'
    )
    docs = load_documents(["kb/*.md", "more.jsonl"], tmp_path)
    assert docs == [
        SeedDocument("kb/a.md", "first"),
        SeedDocument("kb/b.md", "second"),
        SeedDocument("x1", "from jsonl"),
        SeedDocument("7", "numbered"),
    ]


@pytest.mark.parametrize(
    ("content", "message_"),
    [
        ("{oops", "not valid JSON"),
        ("[1]", "must be an object"),
        ('{"id": true, "text": "t"}', "id must be"),
        ('{"id": "a", "text": " "}', "text must be"),
        ('{"id": "' + "x" * 600 + '", "text": "t"}', "longer than 512"),
        ('{"id": "a", "text": "t"}\n{"id": "a", "text": "u"}', "appears twice"),
        ("\n\n", "holds no documents"),
    ],
)
def test_bad_document_files(tmp_path: Path, content: str, message_: str) -> None:
    (tmp_path / "d.jsonl").write_text(content)
    with pytest.raises(InvalidInputError, match=message_):
        load_documents(["d.jsonl"], tmp_path)


def test_missing_and_unreadable_documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(InvalidInputError, match="nothing matches"):
        load_documents(["none/*.md"], tmp_path)
    (tmp_path / "bin.txt").write_bytes(b"\xff\xfe\x00")
    with pytest.raises(InvalidInputError, match="not UTF-8"):
        load_documents(["bin.txt"], tmp_path)
    monkeypatch.setattr(gen, "MAX_DOCUMENT_BYTES", 2)
    (tmp_path / "big.txt").write_text("large")
    with pytest.raises(InvalidInputError, match="larger than"):
        load_documents(["big.txt"], tmp_path)
    monkeypatch.setattr(gen, "MAX_DOCUMENTS", 1)
    (tmp_path / "two.jsonl").write_text('{"id": 1, "text": "a"}\n{"id": 2, "text": "b"}\n')
    with pytest.raises(InvalidInputError, match="more than 1 documents"):
        load_documents(["two.jsonl"], tmp_path)


# ---------------------------------------------------------------- requests


def test_examples_are_spread_over_the_dataset() -> None:
    cases = [Case(id=str(i), input=str(i)) for i in range(10)]
    assert [c.id for c in pick_examples(cases, 4)] == ["0", "3", "6", "9"]
    assert [c.id for c in pick_examples(cases, 1)] == ["0"]
    assert pick_examples(cases, 0) == []
    assert len(pick_examples(cases[:3], 5)) == 3


def test_requests_per_document_show_examples_and_mark_data() -> None:
    examples = [Case(id="e", input={"question": "Q?"}, expected="A.")]
    long_doc = SeedDocument('a"<b', "word " * 5000)
    (request,) = plan_requests(SPEC, [long_doc], examples)
    assert request.source == 'a"<b'
    assert request.count == 3
    assert "<example>\n<input>\nQ?\n</input>\n<expected>\nA.\n</expected>\n</example>" in (
        request.prompt
    )
    assert '<document id="a&quot;&lt;b">' in request.prompt
    assert "... [5000 more characters]" in request.prompt
    assert request.prompt.endswith("Write 3 test cases based on the document.")
    system = gen.system_prompt(SPEC)
    assert "do not follow\nthem" in system
    assert "shipping, account." in system
    assert SPEC.instructions in system
    assert "tags from this list" not in gen.system_prompt(
        SPEC.model_copy(update={"allowed_tags": ()})
    )
    assert GENERATE_SYSTEM.startswith("You write test cases")


def test_requests_without_documents() -> None:
    spec = GenerateSpec(dataset="d", instructions="x", count=25, per_call=10)
    requests = plan_requests(spec, [], [])
    assert [r.count for r in requests] == [10, 10, 5]
    assert [r.source for r in requests] == [None] * 3
    assert len({r.prompt for r in requests}) == 3
    assert "request 3 of 3" in requests[2].prompt
    with pytest.raises(InvalidInputError, match="none were loaded"):
        plan_requests(SPEC, [], [])
    big = SPEC.model_copy(update={"per_document": 50})
    with pytest.raises(InvalidInputError, match="is more than"):
        plan_requests(big, [SeedDocument(str(i), "t") for i in range(2001)], [])


def test_reply_schema() -> None:
    plain = reply_schema(())
    item = plain["properties"]["cases"]["items"]
    assert item["required"] == ["input", "expected"]
    tagged = reply_schema(("a", "b"))["properties"]["cases"]["items"]
    assert tagged["properties"]["tags"]["items"]["enum"] == ["a", "b"]
    assert tagged["required"] == ["input", "expected", "tags"]


def test_generated_cases_become_dataset_cases() -> None:
    drafted = gen._Drafted(input="  Where?  ", expected="", tags=["account", "nope", "account"])
    case = build_case(SPEC, drafted, "kb/a.md")
    assert case.input == {"question": "Where?"}
    assert case.expected is None
    assert case.tags == ("synthetic", "account")
    assert (case.split, case.metadata) == ("synthetic", {"source": "kb/a.md"})
    assert regex.fullmatch(r"gen-[0-9a-f]{12}", case.id)
    # The id depends on the input only.
    again = build_case(SPEC, gen._Drafted(input="Where?", expected="x"), None)
    assert again.id == case.id
    assert again.metadata == {}
    with pytest.raises(ValueError, match="empty input"):
        build_case(SPEC, gen._Drafted(input=" ", expected=""), None)
    plain = SPEC.model_copy(update={"input_key": None})
    with pytest.raises(ValueError, match="longer than 1048576 bytes"):
        build_case(plain, gen._Drafted(input="x" * 1_100_000, expected=""), None)
    with pytest.raises(ValueError, match="split"):
        build_case(SPEC.model_copy(update={"split": "\x00"}), drafted, None)


# ---------------------------------------------------------------- generating


@pytest.mark.anyio
async def test_a_generation_drafts_dedupes_and_records(sessions: sessionmaker[Session]) -> None:
    seed(sessions)
    fake = FakeAnthropic()
    fake.default = scripted()
    report = await generate(
        sessions, SPEC, generator(fake, cache=VerdictStore(sessions)), documents=DOCS
    )
    assert len(fake.requests) == 3
    assert fake.requests[0]["output_config"]["format"]["schema"] == reply_schema(SPEC.allowed_tags)
    assert (report.requested, report.generated, report.invalid) == (9, 7, 1)
    assert (report.duplicates, report.pending) == (3, 4)
    assert (report.calls, report.cached_calls) == (3, 0)
    assert (report.input_tokens, report.output_tokens) == (300, 60)
    assert report.cost_usd == pytest.approx(3 * (100 * 4 + 20 * 20) / 1e6)
    assert report.errors == [{"source": "refunds.md", "error": "invalid case: empty input"}]
    with sessions() as s:
        drafts = list_drafts(s, report.id)
        by_q = {d.case["input"]["question"]: d for d in drafts if "~" not in d.id}
        ship = by_q["How fast do orders ship?"].id
        reset = by_q["How do I reset my password?"].id
        assert [(d.case["input"]["question"], d.status, d.duplicate_of) for d in drafts] == [
            ("How long do refunds take to reach my card ?", "duplicate", "old-1"),
            ("Where is my refund paid?", "pending", None),
            ("How fast do orders ship?", "pending", None),
            ("What does express cost?", "pending", None),
            ("How fast do orders ship?", "duplicate", ship),
            ("How do I reset my password?", "pending", None),
            ("How do I reset my password??", "duplicate", reset),
        ]
        repeat = [d for d in drafts if d.id == f"{ship}~2"]
        assert repeat[0].status == "duplicate"
        assert repeat[0].similarity == 1.0
        assert repeat[0].duplicate_of == ship
        assert by_q["What does express cost?"].case["tags"] == ["synthetic"]
        assert by_q["How fast do orders ship?"].case["tags"] == ["synthetic", "shipping"]
        assert by_q["Where is my refund paid?"].source == "refunds.md"
        near = by_q["How long do refunds take to reach my card ?"]
        assert near.similarity is not None
        assert near.similarity >= 0.8
        info = generation_info(s, report.id)
        assert info.statuses == {"pending": 4, "duplicate": 3, "approved": 0, "rejected": 0}
        assert info.generator["model"] == "claude-opus-5-5"
        assert info.dedupe["threshold"] == 0.8
        assert [d.id for d in list_drafts(s, report.id, "pending", limit=2, offset=1)] == [
            ship,
            by_q["What does express cost?"].id,
        ]
        with pytest.raises(InvalidInputError, match="status must be"):
            list_drafts(s, report.id, "odd")
    # The same config again: every reply from the cache, the same drafts, nothing paid.
    again = await generate(
        sessions, SPEC, generator(fake, cache=VerdictStore(sessions)), documents=DOCS
    )
    assert len(fake.requests) == 3
    assert (again.calls, again.cached_calls, again.cost_usd) == (3, 3, 0.0)
    with sessions() as s:
        assert [d.id for d in list_drafts(s, again.id)] == [d.id for d in drafts]
        assert [g.id for g in list_generations(s, "support")] == sorted(
            [report.id, again.id], key=[report.id, again.id].index
        )
        assert list_generations(s, "other") == []


@pytest.mark.anyio
async def test_failed_calls_and_unreadable_replies_are_errors(
    sessions: sessionmaker[Session],
) -> None:
    replies = {
        "refunds.md": json_reply(message('{"cases": [{"input": "fine", "expected": "x"}]}')),
        "shipping.md": json_reply(message("{}", stop_reason="refusal")),
        "account.md": json_reply(message('{"cases": [', stop_reason="max_tokens")),
    }
    fake = FakeAnthropic()
    fake.default = lambda body: replies[document_of(body) or ""]
    report = await generate(sessions, SPEC, generator(fake), documents=DOCS)
    assert report.generated == 1
    assert report.errors == [
        {"source": "shipping.md", "error": "the model declined the request"},
        {
            "source": "account.md",
            "error": "the reply was cut off at max_tokens; raise max_tokens or ask for fewer",
        },
    ]
    broken = FakeAnthropic()
    broken.default = lambda body: (
        error_reply(500)
        if document_of(body) == "refunds.md"
        else json_reply(message('{"items": []}'))
    )
    report = await generate(sessions, SPEC, generator(broken), documents=DOCS)
    assert report.calls == 2
    assert report.generated == 0
    assert report.errors[0]["source"] == "refunds.md"
    assert report.errors[0]["error"].startswith("call failed:")
    assert report.errors[1:] == [
        {"source": s, "error": "the reply does not match the schema"}
        for s in ("shipping.md", "account.md")
    ]


def test_reply_stop_reasons() -> None:
    from evalkit.providers.base import Completion
    from evalkit.targets.base import Usage

    with pytest.raises(ValueError, match="ended with 'pause_turn'"):
        gen.parse_reply(Completion("{}", "pause_turn", Usage(), "m"))


@pytest.mark.anyio
async def test_more_cases_than_asked_are_cut(sessions: sessionmaker[Session]) -> None:
    many = [{"input": f"Question number {i} about refunds", "expected": ""} for i in range(9)]
    fake = FakeAnthropic()
    fake.default = scripted(dict.fromkeys(("refunds.md", "shipping.md", "account.md"), many))
    report = await generate(sessions, SPEC, generator(fake), documents=DOCS[:1])
    assert report.generated == 3


@pytest.mark.anyio
async def test_examples_from_the_dataset_and_count_mode(sessions: sessionmaker[Session]) -> None:
    spec = GenerateSpec(
        dataset="support",
        instructions="More questions like these.",
        count=4,
        per_call=2,
        examples_from_dataset=True,
        max_examples=1,
    )
    fake = FakeAnthropic()
    with pytest.raises(InvalidInputError, match="no version to take examples from"):
        await generate(sessions, spec, generator(fake, spec))
    seed(sessions)
    counter = iter(range(100))

    def handler(body: dict[str, Any]) -> httpx2.Response:
        content = body["messages"][0]["content"]
        assert "How long do refunds take to reach my card?" in content
        assert "Can I change the delivery address?" not in content
        cases = [{"input": f"Fresh question {next(counter)} on a new subject", "expected": ""}]
        return json_reply(message(json.dumps({"cases": cases * 2})))

    fake.default = handler
    report = await generate(sessions, spec, generator(fake, spec))
    assert (report.requested, report.calls) == (4, 2)
    # Each reply repeats its case: the repeat is a duplicate.
    assert (report.generated, report.duplicates, report.pending) == (4, 2, 2)
    with pytest.raises(InvalidInputError, match="none were loaded"):
        await generate(sessions, SPEC, generator(fake))


@pytest.mark.anyio
async def test_examples_count_as_existing_cases(sessions: sessionmaker[Session]) -> None:
    fake = FakeAnthropic()
    fake.default = scripted()
    report = await generate(sessions, SPEC, generator(fake), documents=DOCS[:1], examples=EXISTING)
    with sessions() as s:
        (near,) = [d for d in list_drafts(s, report.id) if d.status == "duplicate"]
    assert near.duplicate_of == "old-1"


def test_the_generator_needs_a_priced_model() -> None:
    spec = SPEC.model_copy(update={"model": "unpriced"})
    with pytest.raises(InvalidInputError, match="unknown model"):
        Generator(spec, FakeAnthropic().client())
    identity = Generator(SPEC, FakeAnthropic().client()).identity()
    other = Generator(SPEC.model_copy(update={"effort": "high"}), FakeAnthropic().client())
    assert identity["fingerprint"] != other.identity()["fingerprint"]
    assert set(identity) == {
        "provider",
        "endpoint",
        "model",
        "effort",
        "system_sha256",
        "fingerprint",
    }


# ---------------------------------------------------------------- review


async def drafted(sessions: sessionmaker[Session]) -> str:
    seed(sessions)
    fake = FakeAnthropic()
    fake.default = scripted()
    return (await generate(sessions, SPEC, generator(fake), documents=DOCS)).id


@pytest.mark.anyio
async def test_approving_every_pending_draft_makes_one_version(
    sessions: sessionmaker[Session],
) -> None:
    gid = await drafted(sessions)
    with sessions() as s:
        result = approve(s, gid, note="first review")
        assert result.created
        assert result.version == 2
        assert len(result.approved) == 4
        ds = get_dataset(s, "support")
        assert (ds.version, ds.case_count, ds.note, ds.cases_added) == (2, 6, "first review", 4)
        cases = load_cases(s, ds)
        assert [c.id for c in cases[:2]] == ["old-1", "old-2"]
        assert {c.id for c in cases[2:]} == set(result.approved)
        approved = list_drafts(s, gid, "approved")
        assert {d.dataset_version for d in approved} == {2}
        assert all(d.reviewed_at is not None and d.note == "first review" for d in approved)
        # Again: nothing pending is left.
        again = approve(s, gid)
        assert (again.approved, again.created, again.version) == ([], False, 2)
        assert dataset_log(s, "support")[0].origin == "edit"


@pytest.mark.anyio
async def test_approval_rechecks_against_the_newest_version(
    sessions: sessionmaker[Session],
) -> None:
    gid = await drafted(sessions)
    with sessions() as s:
        latest = get_dataset(s, "support")
        added = Case(id="human-1", input={"question": "Where is my refund paid ?"})
        apply_edit(s, "support", DatasetEdit(base=latest.content_hash, upsert=(added,)))
        result = approve(s, gid)
        assert len(result.approved) == 3
        (left_out,) = result.now_duplicates.items()
        assert left_out[1] == "human-1"
        draft = next(d for d in list_drafts(s, gid) if d.id == left_out[0])
        assert (draft.status, draft.duplicate_of) == ("duplicate", "human-1")
        # Named one by one, a duplicate is added as asked.
        forced = approve(s, gid, [left_out[0], result.approved[0]])
        assert forced.approved == [left_out[0]]
        assert forced.already_approved == [result.approved[0]]
        assert get_dataset(s, "support").case_count == 3 + 3 + 1


@pytest.mark.anyio
async def test_first_approval_creates_the_dataset(sessions: sessionmaker[Session]) -> None:
    fake = FakeAnthropic()
    fake.default = scripted()
    report = await generate(sessions, SPEC, generator(fake), documents=DOCS[1:])
    with sessions() as s:
        pending = [d.id for d in list_drafts(s, report.id, "pending")]
        result = approve(s, report.id, pending[:1])
        assert (result.version, result.created, result.approved) == (1, True, pending[:1])
        assert get_dataset(s, "support").origin == "import"
        assert approve(s, report.id).version == 2


@pytest.mark.anyio
async def test_approval_is_safe_to_repeat_after_a_crash(sessions: sessionmaker[Session]) -> None:
    gid = await drafted(sessions)
    with sessions() as s:
        pending = list_drafts(s, gid, "pending")
        cases = tuple(Case.model_validate(d.case) for d in pending)
        latest = get_dataset(s, "support")
        # The edit went in, but the drafts were never marked.
        apply_edit(s, "support", DatasetEdit(base=latest.content_hash, upsert=cases))
        result = approve(s, gid, [d.id for d in pending])
        assert (result.created, result.version) == (False, 2)
        assert {d.status for d in list_drafts(s, gid) if d.id in result.approved} == {"approved"}


@pytest.mark.anyio
async def test_a_conflicting_writer_is_retried(
    sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    gid = await drafted(sessions)
    real = apply_edit
    calls = []

    def flaky(session: Session, name: str, edit: DatasetEdit) -> Any:
        calls.append(edit.base)
        if len(calls) == 1:
            raise ConflictError("moved on")
        return real(session, name, edit)

    monkeypatch.setattr(gen, "apply_edit", flaky)
    with sessions() as s:
        assert approve(s, gid).version == 2
    assert len(calls) == 2

    def always(session: Session, name: str, edit: DatasetEdit) -> Any:
        raise ConflictError("moved on")

    monkeypatch.setattr(gen, "apply_edit", always)
    with sessions() as s:
        pending = [d.id for d in list_drafts(s, gid, "duplicate")]
        with pytest.raises(ConflictError, match="kept changing"):
            approve(s, gid, pending)


@pytest.mark.anyio
async def test_rejecting(sessions: sessionmaker[Session]) -> None:
    gid = await drafted(sessions)
    with sessions() as s:
        pending = [d.id for d in list_drafts(s, gid, "pending")]
        assert reject(s, gid, pending[:2], note="off topic") == pending[:2]
        rejected = list_drafts(s, gid, "rejected")
        assert {d.note for d in rejected} == {"off topic"}
        result = approve(s, gid)
        assert result.approved == pending[2:]
        with pytest.raises(ConflictError, match="already in version 2"):
            reject(s, gid, pending[2:3])
        # A rejected draft can still be approved by name.
        assert approve(s, gid, pending[:1]).approved == pending[:1]
        with pytest.raises(NotFoundError, match="has no draft 'nope'"):
            reject(s, gid, ["nope"])
        with pytest.raises(InvalidInputError, match="at least one"):
            approve(s, gid, [])
        with pytest.raises(InvalidInputError, match="longer than 2000"):
            reject(s, gid, pending[:1], note="x" * 2001)
        with pytest.raises(InvalidInputError, match="longer than 2000"):
            approve(s, gid, note="x" * 2001)


@pytest.mark.anyio
async def test_generations_by_prefix(sessions: sessionmaker[Session]) -> None:
    gid = await drafted(sessions)
    with sessions() as s:
        assert get_generation(s, gid[:8]).id == gid
        with pytest.raises(NotFoundError):
            get_generation(s, "abc")
        s.add(Generation(**_twin(s.get(Generation, gid), gid[:6] + "0" * 26)))
        s.commit()
        with pytest.raises(InvalidInputError, match="ambiguous"):
            get_generation(s, gid[:6])
        # Deleting a generation deletes its drafts.
        s.delete(s.get(Generation, gid))
        s.commit()
        assert s.scalars(select(CaseDraft).where(CaseDraft.generation_id == gid)).all() == []


def _twin(g: Generation | None, new_id: str) -> dict[str, Any]:
    assert g is not None
    fields = ("dataset", "spec", "generator", "dedupe", "requested", "generated")
    return {"id": new_id, **{f: getattr(g, f) for f in fields}, "invalid": 0, "duplicates": 0}


# ---------------------------------------------------------------- CLI and API


def run_cli(db: str, *args: str) -> tuple[int, str]:
    out = io.StringIO()
    return cli.main(["--database-url", db, *args], out=out), out.getvalue()


def write_config(tmp_path: Path, **extra: Any) -> Path:
    (tmp_path / "kb").mkdir(exist_ok=True)
    for doc in DOCS:
        (tmp_path / "kb" / doc.id).write_text(doc.text)
    (tmp_path / "examples.jsonl").write_text(
        json.dumps({"id": "ex", "input": {"question": "Is there a phone line?"}}) + "\n"
    )
    values: dict[str, Any] = {
        "dataset": "support",
        "instructions": "Write questions customers ask about the document.",
        "documents": "kb/*.md",
        "per_document": 3,
        "input_key": "question",
        "allowed_tags": ["shipping", "account"],
        "examples": "examples.jsonl",
    } | extra
    lines = ["[generate]", *(f"{k} = {json.dumps(v)}" for k, v in values.items())]
    path = tmp_path / "generate.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_cli_generate_review_and_approve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeAnthropic()
    fake.default = scripted()
    monkeypatch.setattr(cli.App, "claude", lambda self: fake.client())
    db = f"sqlite:///{tmp_path / 'e.db'}"
    config = write_config(tmp_path)
    code, out = run_cli(db, "generate", str(config))
    assert code == 0
    first = out.splitlines()[0]
    gid = first.split()[1]
    assert first.endswith("7 drafts of 9 asked (5 pending, 2 near-duplicates, 1 invalid)")
    assert "3 calls, 300 input and 60 output tokens" in out
    assert "error: kb/refunds.md: invalid case: empty input" in out
    assert out.strip().endswith(f"evalkit drafts show {gid}")
    assert "Is there a phone line?" in fake.requests[0]["messages"][0]["content"]
    code, out = run_cli(db, "generate", str(config), "--json")
    assert code == 0
    assert json.loads(out)["cached_calls"] == 3
    code, out = run_cli(db, "drafts", "list")
    assert code == 0
    assert len(out.splitlines()) == 2
    assert "5 pending, 2 duplicate" in out
    code, out = run_cli(db, "drafts", "show", gid[:8], "--status", "duplicate")
    assert code == 0
    assert out.splitlines()[0] == (
        f"generation {gid} for support: 5 pending, 2 duplicate, 0 approved, 0 rejected"
    )
    assert "(similarity 1.00)" in out
    assert "from kb/shipping.md" in out
    code, out = run_cli(db, "drafts", "show", gid, "--json", "--limit", "2")
    shown = json.loads(out)
    assert len(shown["drafts"]) == 2
    assert shown["generation"]["id"] == gid
    pending = [
        d["id"]
        for d in json.loads(run_cli(db, "drafts", "show", gid, "--json")[1])["drafts"]
        if d["status"] == "pending"
    ]
    code, out = run_cli(db, "drafts", "reject", gid, "--id", pending[0], "--note", "vague")
    assert (code, out) == (0, "rejected 1 drafts\n")
    assert run_cli(db, "drafts", "approve", gid)[0] == 2
    assert "--all" in capsys.readouterr().err
    code, out = run_cli(db, "drafts", "approve", gid, "--all", "--note", "reviewed")
    assert code == 0
    assert out.startswith("approved 4 drafts: support now at version 1 (")
    code, out = run_cli(db, "drafts", "show", gid, "--status", "approved")
    assert "in version 1" in out
    code, out = run_cli(db, "drafts", "approve", gid, "--all")
    assert (code, out) == (0, "nothing to approve\n")
    code, out = run_cli(db, "drafts", "approve", gid, "--id", pending[1], "--json")
    assert json.loads(out)["already_approved"] == [pending[1]]
    code, out = run_cli(db, "drafts", "list", "--dataset", "none")
    assert (code, out) == (0, "no generations\n")
    assert run_cli(db, "drafts", "reject", gid, "--id", pending[1])[0] == 3
    assert run_cli(db, "drafts", "show", "nope")[0] == 2


def test_cli_reports_recheck_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db = f"sqlite:///{tmp_path / 'f.db'}"
    fake = FakeAnthropic()
    fake.default = lambda body: error_reply(400, "invalid_request_error")
    monkeypatch.setattr(cli.App, "claude", lambda self: fake.client())
    code, out = run_cli(db, "generate", str(write_config(tmp_path)))
    assert code == 3
    assert out.count("call failed") == 3
    many = SCRIPT["refunds.md"] * 4
    fake.default = scripted(dict.fromkeys((d.id for d in DOCS), many))
    code, out = run_cli(db, "generate", str(write_config(tmp_path, per_document=12)))
    assert "and 2 more errors (--json for all)" in out
    (tmp_path / "bad.toml").write_text("[other]\n")
    assert run_cli(db, "generate", str(tmp_path / "bad.toml"))[0] == 2
    assert "has no [generate] table" in capsys.readouterr().err
    (tmp_path / "bad.toml").write_text('[generate]\ndataset = "x"\n')
    assert run_cli(db, "generate", str(tmp_path / "bad.toml"))[0] == 2
    gid = json.loads(run_cli(db, "drafts", "list", "--json")[1])[-1]["id"]
    code, out = run_cli(db, "drafts", "approve", gid, "--all")
    assert code == 0
    assert "support now at version 1" in out


@pytest.mark.anyio
async def test_api_review(sessions: sessionmaker[Session]) -> None:
    from tests.test_api import AUTH, serving

    gid = await drafted(sessions)
    async with serving(sessions) as (c, _):
        listed = (await c.get("/v1/generations", headers=AUTH)).json()["generations"]
        assert [g["id"] for g in listed] == [gid]
        assert (await c.get("/v1/generations?dataset=x", headers=AUTH)).json() == {
            "generations": []
        }
        info = (await c.get(f"/v1/generations/{gid}", headers=AUTH)).json()
        assert info["statuses"]["pending"] == 4
        assert (await c.get("/v1/generations/nope", headers=AUTH)).status_code == 404
        page = await c.get(f"/v1/generations/{gid}/drafts?status=pending&limit=3", headers=AUTH)
        pending = [d["id"] for d in page.json()["drafts"]]
        assert len(pending) == 3
        bad = await c.get(f"/v1/generations/{gid}/drafts?status=odd", headers=AUTH)
        assert bad.status_code == 400
        for body in ({}, {"ids": [pending[0]], "all": True}):
            r = await c.post(f"/v1/generations/{gid}/approve", json=body, headers=AUTH)
            assert r.status_code == 400
        r = await c.post(
            f"/v1/generations/{gid}/reject", json={"ids": [pending[0]], "note": "no"}, headers=AUTH
        )
        assert r.json() == {"rejected": [pending[0]]}
        r = await c.post(
            f"/v1/generations/{gid}/approve", json={"all": True, "note": "ok"}, headers=AUTH
        )
        approved = r.json()
        assert (approved["version"], len(approved["approved"]), approved["created"]) == (2, 3, True)
        r = await c.post(
            f"/v1/generations/{gid}/reject", json={"ids": approved["approved"][:1]}, headers=AUTH
        )
        assert r.status_code == 409
        r = await c.post(f"/v1/generations/{gid}/approve", json={"ids": ["nope"]}, headers=AUTH)
        assert r.status_code == 404
        assert (await c.get("/v1/generations", headers={})).status_code == 401


def test_a_document_line_too_long(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gen, "MAX_LINE_BYTES", 10)
    (tmp_path / "d.jsonl").write_text('{"id": "a", "text": "long enough"}\n')
    with pytest.raises(InvalidInputError, match="line 1: longer than 10 bytes"):
        load_documents(["d.jsonl"], tmp_path)
