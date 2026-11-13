import asyncio
import io
import json
import re
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from evalkit import cli
from evalkit.errors import InvalidInputError
from evalkit.tools import (
    FixtureRecord,
    Fixtures,
    SandboxSpec,
    ToolCall,
    ToolSandbox,
    load_sandbox,
    parse_calls,
    record,
    sandbox_identity,
)

FLIGHTS: dict[str, Any] = {
    "name": "search_flights",
    "description": "Find flights between two airports on a date.",
    "input_schema": {
        "type": "object",
        "properties": {
            "origin": {"type": "string", "pattern": "^[A-Z]{3}$"},
            "destination": {"type": "string", "pattern": "^[A-Z]{3}$"},
            "date": {"type": "string"},
            "passengers": {"type": "integer", "minimum": 1},
        },
        "required": ["origin", "destination", "date"],
        "additionalProperties": False,
    },
    "output_schema": {
        "type": "object",
        "properties": {"flights": {"type": "array"}},
        "required": ["flights"],
    },
}
BOOK: dict[str, Any] = {
    "name": "book",
    "input_schema": {
        "type": "object",
        "properties": {"flight": {"type": "string"}},
        "required": ["flight"],
    },
}
LHR_JFK = {"origin": "LHR", "destination": "JFK", "date": "2026-12-01"}


class Sleeps:
    def __init__(self) -> None:
        self.seconds: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.seconds.append(seconds)


def spec(*tools: dict[str, Any], **options: Any) -> SandboxSpec:
    return SandboxSpec.model_validate({"tools": list(tools or (FLIGHTS, BOOK)), **options})


def calls(sandbox: ToolSandbox, *made: tuple[str, Any]) -> list[ToolCall]:
    async def main() -> list[ToolCall]:
        return [await sandbox.call(name, args) for name, args in made]

    return asyncio.run(main())


def one(sandbox: ToolSandbox, name: str, args: Any) -> ToolCall:
    return calls(sandbox, (name, args))[0]


# --- the spec ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tools", "options", "message"),
    [
        ([{**FLIGHTS, "input_schema": {"type": "nonsense"}}], {}, "not a valid JSON Schema"),
        ([{**FLIGHTS, "input_schema": {"type": "string"}}], {}, 'type "object"'),
        ([{**FLIGHTS, "output_schema": {"minLength": -1}}], {}, "output_schema is not a valid"),
        ([FLIGHTS, FLIGHTS], {}, "unique: search_flights"),
        ([{**BOOK, "name": "1bad"}], {}, "String should match"),
        ([{**BOOK, "fail_calls": [0]}], {}, "counted from 1"),
        ([{**BOOK, "responses": [{}]}], {}, "result or an error"),
        ([{**BOOK, "responses": [{"result": 1, "error": "x"}]}], {}, "result or an error"),
        ([{**BOOK, "responses": [{"result": "x" * (1024 * 1024)}]}], {}, "larger than"),
        (
            [{**FLIGHTS, "responses": [{"result": {"nope": 1}}]}],
            {},
            "responses[0].result breaks output_schema: (root): 'flights' is a required",
        ),
        ([], {}, "at least 1"),
        ([BOOK], {"max_calls": 0}, "greater than or equal to 1"),
    ],
)
def test_invalid_specs_are_refused(
    tools: list[dict[str, Any]], options: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=re.escape(message)):
        SandboxSpec.model_validate({"tools": tools, **options})


def test_definitions_are_provider_neutral() -> None:
    s = spec()
    assert s.definitions()[0] == {
        "name": "search_flights",
        "description": FLIGHTS["description"],
        "input_schema": FLIGHTS["input_schema"],
    }
    assert ToolSandbox(s).definitions() == s.definitions()
    assert s.tools[1].input_schema["required"] == ["flight"]
    assert spec(BOOK, {"name": "noop"}).tools[1].input_schema == {"type": "object"}


# --- calls that are never answered ----------------------------------------------------


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"origin": "LHR", "date": "x"}, "(root): 'destination' is a required property"),
        ({**LHR_JFK, "origin": "london"}, "origin: 'london' does not match"),
        ({**LHR_JFK, "passengers": 0}, "passengers: 0 is less than the minimum of 1"),
        ({**LHR_JFK, "extra": True}, "Additional properties are not allowed"),
        (["LHR"], "arguments must be a JSON object"),
        ({**LHR_JFK, "date": "x" * 70_000}, "larger than 65536 bytes"),
        ({**LHR_JFK, "date": float("nan")}, "arguments must be JSON"),
        ({**LHR_JFK, "date": object()}, "arguments must be JSON"),
    ],
)
def test_invalid_arguments_are_answered_with_the_reason(args: Any, message: str) -> None:
    call = one(ToolSandbox(spec()), "search_flights", args)
    assert call.status == "invalid_arguments"
    assert call.is_error
    assert call.error is not None
    assert call.error.startswith("invalid arguments for search_flights: ")
    assert message in call.error
    assert call.content() == {"error": call.error}


def test_unresolvable_references_are_reported_not_raised() -> None:
    tool = {"name": "t", "input_schema": {"type": "object", "properties": {"a": {"$ref": "#/x"}}}}
    call = one(ToolSandbox(spec(tool)), "t", {"a": 1})
    assert call.status == "invalid_arguments"
    assert "cannot be resolved" in (call.error or "")


def test_unknown_tools_and_missing_responses() -> None:
    sandbox = ToolSandbox(spec())
    unknown, missing = calls(sandbox, ("cancel", {}), ("book", {"flight": "BA1"}))
    assert unknown.status == "unknown_tool"
    assert unknown.error == "there is no tool 'cancel'; tools: book, search_flights"
    assert missing.status == "no_response"
    assert missing.source is None
    assert [c.index for c in sandbox.calls] == [1, 2]


def test_session_and_tool_call_limits() -> None:
    limited = {**BOOK, "max_calls": 2, "responses": [{"result": "ok"}]}
    sandbox = ToolSandbox(spec(limited, FLIGHTS, max_calls=4))
    made = calls(
        sandbox,
        ("book", {"flight": "A"}),
        ("book", {}),  # invalid, but still one of the tool's two calls
        ("book", {"flight": "B"}),
        ("nope", {}),
        ("book", {"flight": "C"}),
    )
    assert [c.status for c in made] == [
        "ok",
        "invalid_arguments",
        "limit_reached",
        "unknown_tool",
        "limit_reached",
    ]
    assert made[2].error == "book may be called at most 2 times"
    assert made[4].error == "the session's 4 calls are used"
    assert len(sandbox.calls) == 5


# --- scripted responses --------------------------------------------------------------


def test_scripted_rules_match_subsets_in_order_with_uses() -> None:
    tool = {
        **FLIGHTS,
        "output_schema": None,
        "responses": [
            {"when": {"origin": "LHR", "passengers": 2}, "result": {"flights": ["two"]}},
            {"when": {"origin": "LHR"}, "times": 1, "result": {"flights": ["BA1"]}},
            {"when": {"origin": "LHR"}, "error": "sold out"},
            {"result": {"flights": []}},
        ],
    }
    sandbox = ToolSandbox(spec(tool))
    made = calls(
        sandbox,
        ("search_flights", LHR_JFK),
        ("search_flights", LHR_JFK),
        ("search_flights", {**LHR_JFK, "passengers": 2}),
        ("search_flights", {**LHR_JFK, "origin": "CDG"}),
    )
    assert [(c.status, c.content(), c.source) for c in made] == [
        ("ok", {"flights": ["BA1"]}, "script"),
        ("error", {"error": "sold out"}, "script"),
        ("ok", {"flights": ["two"]}, "script"),
        ("ok", {"flights": []}, "script"),
    ]


@pytest.mark.parametrize(
    ("when", "args", "matched"),
    [
        ({"a": {"b": 1}}, {"a": {"b": 1, "c": 2}}, True),
        ({"a": {"b": 1}}, {"a": {"c": 2}}, False),
        ({"a": {"b": 1}}, {"a": 1}, False),
        ({"a": 1}, {"a": True}, False),
        ({"a": True}, {"a": 1}, False),
        ({"a": True}, {"a": True}, True),
        ({"a": 1}, {"a": 1.0}, True),
        ({"a": [1, {"b": 2}]}, {"a": [1, {"b": 2, "c": 3}]}, True),
        ({"a": [1, 2]}, {"a": [1]}, False),
        ({"a": [1]}, {"a": "x"}, False),
        ({"a": None}, {"a": None}, True),
        ({"a": None}, {}, False),
    ],
)
def test_when_is_a_typed_subset(when: dict[str, Any], args: dict[str, Any], matched: bool) -> None:
    tool = {"name": "t", "responses": [{"when": when, "result": "hit"}]}
    call = one(ToolSandbox(spec(tool)), "t", args)
    assert (call.status == "ok") is matched


# --- fixtures -------------------------------------------------------------------------


def fixture_lines(*records: dict[str, Any]) -> str:
    return "".join(json.dumps(r) + "\n" for r in records)


def test_fixtures_replay_in_order_by_canonical_arguments() -> None:
    s = spec()
    fixtures = Fixtures.parse(
        fixture_lines(
            {"tool": "search_flights", "arguments": LHR_JFK, "result": {"flights": ["BA1"]}},
            {"tool": "search_flights", "arguments": LHR_JFK, "result": {"flights": []}},
            {"tool": "book", "arguments": {"flight": "BA1"}, "error": "card declined"},
        ),
        s,
    )
    sandbox = ToolSandbox(s, fixtures)
    reordered = dict(reversed(list(LHR_JFK.items())))
    made = calls(
        sandbox,
        ("search_flights", reordered),
        ("search_flights", LHR_JFK),
        ("search_flights", LHR_JFK),
        ("book", {"flight": "BA1"}),
        ("book", {"flight": "BA2"}),
    )
    assert [(c.status, c.content(), c.source) for c in made] == [
        ("ok", {"flights": ["BA1"]}, "fixture"),
        ("ok", {"flights": []}, "fixture"),
        ("ok", {"flights": []}, "fixture"),  # the last recorded answer repeats
        ("error", {"error": "card declined"}, "fixture"),
        ("no_response", {"error": "book has no response for these arguments"}, None),
    ]
    assert made[0].to_dict()["arguments"] == reordered


def test_scripts_win_over_fixtures() -> None:
    tool = {**BOOK, "responses": [{"when": {"flight": "BA1"}, "result": "scripted"}]}
    s = spec(tool)
    fixtures = Fixtures([FixtureRecord("book", {"flight": "BA1"}, "recorded")])
    assert one(ToolSandbox(s, fixtures), "book", {"flight": "BA1"}).result == "scripted"


def test_fixtures_round_trip_and_hash() -> None:
    s = spec()
    text = fixture_lines(
        {"tool": "book", "arguments": {"flight": "A"}, "result": [1, 2], "latency_ms": 12.5},
        {"tool": "book", "arguments": {"flight": "B"}, "error": "no"},
        {"tool": "book", "arguments": {"flight": "C"}, "result": None},
    )
    fixtures = Fixtures.parse(text.encode() + b"\n  \n", s)
    assert len(fixtures) == 3
    assert ("book", '{"flight":"A"}') in fixtures
    again = Fixtures.parse(fixtures.to_jsonl(), s)
    assert again.to_jsonl() == fixtures.to_jsonl()
    assert again.content_hash == fixtures.content_hash
    assert again.lookup("book", {"flight": "C"}) == FixtureRecord("book", {"flight": "C"})
    assert fixtures.records[0].latency_ms == 12.5


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("{", "line 1 is not JSON"),
        ("[1]", "must be a JSON object"),
        ('{"tool": "book", "arguments": {"flight": "A"}, "result": NaN}', "not JSON"),
        ('{"tool": "book", "arguments": {}, "result": 1, "x": 1}', "unknown key(s) x"),
        ('{"tool": "cancel", "arguments": {}, "result": 1}', "no tool 'cancel'"),
        ('{"tool": 3, "arguments": {}, "result": 1}', "no tool 3"),
        ('{"tool": "book", "arguments": {}, "result": 1}', "'flight' is a required"),
        ('{"tool": "book", "arguments": 1, "result": 1}', "must be a JSON object"),
        ('{"tool": "book", "arguments": {"flight": "A"}}', "give a result or an error"),
        ('{"tool": "book", "arguments": {"flight": "A"}, "error": 5}', "error must be a string"),
        ('{"tool": "book", "arguments": {"flight": "A"}, "error": ""}', "error must be a string"),
        (
            '{"tool": "book", "arguments": {"flight": "A"}, "result": 1, "latency_ms": -1}',
            "latency_ms must be >= 0",
        ),
        (
            '{"tool": "book", "arguments": {"flight": "A"}, "result": 1, "latency_ms": true}',
            "latency_ms must be >= 0",
        ),
        (
            json.dumps({"tool": "search_flights", "arguments": LHR_JFK, "result": {"f": 1}}),
            "result breaks output_schema",
        ),
        (
            json.dumps({"tool": "book", "arguments": {"flight": "A"}, "result": "x" * 1_100_000}),
            "is larger than",
        ),
    ],
)
def test_bad_fixtures_are_refused_with_the_line(line: str, message: str) -> None:
    with pytest.raises(InvalidInputError, match="line 1") as raised:
        Fixtures.parse(line, spec())
    assert message in raised.value.message


def test_fixture_files_are_bounded_and_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(InvalidInputError, match="not UTF-8"):
        Fixtures.parse(b"\xff\n", spec())
    monkeypatch.setattr("evalkit.tools.sandbox.MAX_FILE_BYTES", 10)
    with pytest.raises(InvalidInputError, match="at most 10 bytes"):
        Fixtures.parse(b"x" * 11, spec())
    monkeypatch.setattr("evalkit.tools.sandbox.MAX_RECORDS", 1)
    line = '{"tool": "book", "arguments": {"flight": "A"}, "result": 1}\n'
    with pytest.raises(InvalidInputError, match="at most 1 records"):
        Fixtures.parse(line * 2, spec())
    with pytest.raises(InvalidInputError, match="at most 1 calls"):
        parse_calls('{"tool": "book", "arguments": {"flight": "A"}}\n' * 2, spec())


# --- latency and failures --------------------------------------------------------------


def test_latency_jitter_and_overrides_are_injected_deterministically() -> None:
    tool = {
        **BOOK,
        "latency_ms": 100,
        "jitter_ms": 50,
        "responses": [{"when": {"flight": "fast"}, "latency_ms": 0, "result": 1}],
    }
    fixtures = Fixtures([FixtureRecord("book", {"flight": "rec"}, 1, latency_ms=700)])
    args = [("book", {"flight": f}) for f in ("a", "fast", "rec", "a")]

    def sleeps(session: str, **options: Any) -> list[float]:
        slept = Sleeps()
        calls(ToolSandbox(spec(tool, **options), fixtures, session=session, sleep=slept), *args)
        return slept.seconds

    first = sleeps("case-1")
    assert len(first) == 3  # the "fast" rule sleeps not at all
    assert all(0.1 <= s <= 0.15 for s in first)
    assert first == sleeps("case-1")
    assert first != sleeps("case-2")
    replayed = sleeps("case-1", replay_latency=True)
    assert replayed[1] == pytest.approx(0.7)


def test_measured_latency_uses_the_clock() -> None:
    ticks = iter([1.0, 1.25])
    sandbox = ToolSandbox(spec(), clock=lambda: next(ticks))
    assert one(sandbox, "book", {"flight": "A"}).latency_ms == pytest.approx(250)


def test_fail_calls_fail_the_nth_executed_call() -> None:
    tool = {**BOOK, "fail_calls": [2], "failure_message": "503", "responses": [{"result": 1}]}
    sandbox = ToolSandbox(spec(tool))
    made = calls(
        sandbox,
        ("book", {"flight": "A"}),
        ("book", {}),  # invalid: not executed, so not the second call
        ("book", {"flight": "B"}),
        ("book", {"flight": "C"}),
    )
    assert [c.status for c in made] == ["ok", "invalid_arguments", "injected_failure", "ok"]
    assert made[2].content() == {"error": "503"}


def test_failure_rate_is_seeded_per_session_and_near_the_rate() -> None:
    tool = {**BOOK, "failure_rate": 0.2, "responses": [{"result": 1}]}
    args = [("book", {"flight": str(i)}) for i in range(2000)]

    def failures(session: str, seed: int = 0) -> list[bool]:
        sandbox = ToolSandbox(spec(tool, max_calls=2000, seed=seed), session=session)
        return [c.status == "injected_failure" for c in calls(sandbox, *args)]

    first = failures("case-1")
    assert 0.17 < sum(first) / len(first) < 0.23
    assert first == failures("case-1")
    assert first != failures("case-2")
    assert first != failures("case-1", seed=1)
    always = ToolSandbox(spec({**BOOK, "failure_rate": 1}))
    assert one(always, "book", {"flight": "A"}).status == "injected_failure"


def test_trajectory_is_kept_in_call_order() -> None:
    tool = {**BOOK, "responses": [{"result": {"ref": "X1"}}]}
    sandbox = ToolSandbox(spec(tool), clock=lambda: 0.0)
    calls(sandbox, ("book", {"flight": "A"}), ("lookup", {}))
    assert [c.to_dict() for c in sandbox.calls] == [
        {
            "index": 1,
            "tool": "book",
            "arguments": {"flight": "A"},
            "status": "ok",
            "result": {"ref": "X1"},
            "error": None,
            "source": "script",
            "latency_ms": 0.0,
        },
        {
            "index": 2,
            "tool": "lookup",
            "arguments": {},
            "status": "unknown_tool",
            "result": None,
            "error": "there is no tool 'lookup'; tools: book",
            "source": None,
            "latency_ms": 0.0,
        },
    ]


# --- identity and files ------------------------------------------------------------------


def test_identity_covers_answers_not_recording_details() -> None:
    base = spec(BOOK)
    fixtures = Fixtures([FixtureRecord("book", {"flight": "A"}, 1)])
    ident = sandbox_identity(base, fixtures)
    assert ident == sandbox_identity(
        spec({**BOOK, "endpoint": "booking"}, fixtures="f.jsonl"), fixtures
    )
    assert ident != sandbox_identity(base, None)
    assert ident != sandbox_identity(base, Fixtures([FixtureRecord("book", {"flight": "A"}, 2)]))
    assert ident != sandbox_identity(spec({**BOOK, "latency_ms": 1}), fixtures)
    assert ident != sandbox_identity(spec(BOOK, seed=1), fixtures)


SANDBOX_TOML = """\
[sandbox]
max_calls = 10
fixtures = "fixtures/booking.jsonl"

[[sandbox.tools]]
name = "book"
description = "Book a flight."
endpoint = "booking"
output_schema = { type = "object", required = ["ref"] }

[sandbox.tools.input_schema]
type = "object"
properties = { flight = { type = "string" } }
required = ["flight"]

[[sandbox.tools.responses]]
when = { flight = "XX0" }
error = "no such flight"
"""


def write_sandbox(tmp_path: Path, fixtures: str | None = "") -> Path:
    path = tmp_path / "sandbox.toml"
    path.write_text(SANDBOX_TOML)
    if fixtures is not None:
        (tmp_path / "fixtures").mkdir()
        (tmp_path / "fixtures" / "booking.jsonl").write_text(fixtures)
    return path


def test_load_sandbox_resolves_fixtures_beside_the_file(tmp_path: Path) -> None:
    line = '{"tool": "book", "arguments": {"flight": "BA1"}, "result": {"ref": "R1"}}\n'
    loaded, fixtures = load_sandbox(write_sandbox(tmp_path, line))
    assert loaded.max_calls == 10
    assert loaded.tools[0].endpoint == "booking"
    assert fixtures is not None
    assert len(fixtures) == 1
    sandbox = ToolSandbox(loaded, fixtures)
    first, second = calls(sandbox, ("book", {"flight": "BA1"}), ("book", {"flight": "XX0"}))
    assert first.result == {"ref": "R1"}
    assert second.error == "no such flight"
    assert load_sandbox(tmp_path / "sandbox.toml", fixtures=False)[1] is None


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, "cannot read"),
        (b"\xff", "not valid TOML"),
        (b"x = 1\n", "no [sandbox] table"),
        (b"[sandbox]\ntools = []\n", "sandbox.tools: List should have at least 1"),
        (b'[sandbox]\nfixtures = "none.jsonl"\n[[sandbox.tools]]\nname = "t"\n', "none.jsonl"),
    ],
)
def test_load_sandbox_errors(tmp_path: Path, content: bytes | None, message: str) -> None:
    path = tmp_path / "sandbox.toml"
    if content is not None:
        path.write_bytes(content)
    with pytest.raises(InvalidInputError) as raised:
        load_sandbox(path)
    assert message in raised.value.message


def test_load_sandbox_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_sandbox(tmp_path)
    monkeypatch.setattr("evalkit.tools.sandbox.MAX_FILE_BYTES", 10)
    with pytest.raises(InvalidInputError, match="larger than 10 bytes"):
        load_sandbox(path)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ('{"tool": "book"}', "arguments must be a JSON object"),
        ('{"tool": "book", "arguments": {"flight": 1}}', "1 is not of type 'string'"),
        ('{"tool": "x", "arguments": {}}', "no tool 'x'"),
        ('{"tool": "book", "arguments": {}, "result": 1}', "unknown key(s) result"),
    ],
)
def test_parse_calls_checks_each_call(line: str, message: str) -> None:
    with pytest.raises(InvalidInputError) as raised:
        parse_calls(line, spec())
    assert message in raised.value.message


# --- recording -------------------------------------------------------------------------


def test_record_keeps_answers_skips_recorded_calls_and_reports_failures() -> None:
    s = spec()
    sent: list[tuple[str, Any]] = []

    async def booking(name: str, args: Any) -> Any:
        sent.append((name, args))
        flight = args["flight"]
        if flight == "boom":
            raise InvalidInputError("endpoint 'booking' returned 500")
        if flight == "weird":
            return {"result": 1, "error": "both"}
        if flight == "empty-error":
            return {"error": ""}
        if flight == "full":
            return {"error": "sold out"}
        return {"result": {"ref": flight.upper()}}

    async def flights(name: str, args: Any) -> Any:
        return {"result": {"wrong": True}}

    made = [
        ("book", {"flight": "ba1"}),
        ("book", {"flight": "ba1"}),  # the same call twice is made once
        ("book", {"flight": "full"}),
        ("book", {"flight": "boom"}),
        ("book", {"flight": "weird"}),
        ("book", {"flight": "empty-error"}),
        ("search_flights", LHR_JFK),
    ]
    ticks = iter(float(i) for i in range(100))
    fixtures, report = asyncio.run(
        record(s, made, {"book": booking, "search_flights": flights}, clock=lambda: next(ticks))
    )
    assert report.recorded == 2
    assert report.skipped == 1
    assert len(report.failed) == 4
    assert "returned 500" in report.failed[0]
    assert 'must answer {"result": ...}' in report.failed[1]
    assert "error must be a string" in report.failed[2]
    assert "breaks output_schema" in report.failed[3]
    assert [r.to_json() for r in fixtures.records] == [
        '{"arguments":{"flight":"ba1"},"latency_ms":1000.0,"result":{"ref":"BA1"},"tool":"book"}',
        '{"arguments":{"flight":"full"},"error":"sold out","latency_ms":1000.0,"tool":"book"}',
    ]

    sent.clear()
    again, report = asyncio.run(
        record(s, [*made[:3], ("book", {"flight": "new"})], {"book": booking}, fixtures)
    )
    assert sent == [("book", {"flight": "new"})]
    assert (report.recorded, report.skipped, report.failed) == (1, 3, [])
    assert len(again) == 3

    sent.clear()
    refreshed, report = asyncio.run(
        record(s, [("book", {"flight": "ba1"})], {"book": booking}, again, refresh=True)
    )
    assert sent == [("book", {"flight": "ba1"})]
    assert [r.arguments["flight"] for r in refreshed.records] == ["full", "new", "ba1"]

    _, report = asyncio.run(record(s, [("book", {"flight": "x"})], {}))
    assert report.failed == ["book: no endpoint to record from"]


class ToolServer(BaseHTTPRequestHandler):
    received: list[Any] = []  # noqa: RUF012

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        ToolServer.received.append(body)
        flight = body["arguments"]["flight"]
        answer = {"error": "sold out"} if flight == "full" else {"result": {"ref": flight}}
        if flight == "broken":
            answer = {"result": {"no": "ref"}}
        data = json.dumps(answer).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def tool_server() -> Iterator[str]:
    ToolServer.received = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), ToolServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/tools"
    finally:
        server.shutdown()
        server.server_close()


def run_cli(*args: str) -> tuple[int, str]:
    out = io.StringIO()
    return cli.main(list(args), out=out), out.getvalue()


def test_cli_record_and_check(
    tmp_path: Path, tool_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    endpoints = tmp_path / "endpoints.toml"
    endpoints.write_text(f'[endpoints.booking]\nurl = "{tool_server}"\nallow_private = true\n')
    monkeypatch.setenv("EVALKIT_HTTP_TARGETS_FILE", str(endpoints))
    monkeypatch.setenv("EVALKIT_DATABASE_URL", f"sqlite:///{tmp_path / 'unused.db'}")
    sandbox = write_sandbox(tmp_path, fixtures=None)
    calls_file = tmp_path / "calls.jsonl"
    calls_file.write_text(
        '{"tool": "book", "arguments": {"flight": "BA1"}}\n'
        '{"tool": "book", "arguments": {"flight": "full"}}\n'
        '{"tool": "book", "arguments": {"flight": "broken"}}\n'
    )
    (tmp_path / "fixtures").mkdir()
    code, out = run_cli("tools", "record", str(sandbox), str(calls_file))
    assert code == cli.EXIT_UNUSABLE
    assert out.startswith("recorded 2, already recorded 0, failed 1\n  failed: book")
    assert ToolServer.received[0] == {"tool": "book", "arguments": {"flight": "BA1"}}
    assert not (tmp_path / "unused.db").exists()

    calls_file.write_text('{"tool": "book", "arguments": {"flight": "BA1"}}\n')
    code, out = run_cli("tools", "record", str(sandbox), str(calls_file), "--json")
    assert code == cli.EXIT_OK
    assert json.loads(out) == {"recorded": 0, "skipped": 1, "failed": []}
    assert len(ToolServer.received) == 3

    code, out = run_cli("tools", "check", str(sandbox))
    assert code == cli.EXIT_OK
    assert out.startswith("1 tools (book), 2 fixtures, identity ")
    code, out = run_cli("tools", "check", str(sandbox), "--json")
    spec_, fixtures = load_sandbox(sandbox)
    assert json.loads(out)["identity"] == sandbox_identity(spec_, fixtures)

    other = tmp_path / "other.jsonl"
    code, _ = run_cli(
        "tools", "record", str(sandbox), str(calls_file), "-o", str(other), "--refresh"
    )
    assert code == cli.EXIT_OK
    assert other.read_text().count("\n") == 1


def test_cli_record_needs_the_endpoint_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("EVALKIT_HTTP_TARGETS_FILE", raising=False)
    sandbox = write_sandbox(tmp_path)
    calls_file = tmp_path / "calls.jsonl"
    calls_file.write_text('{"tool": "book", "arguments": {"flight": "BA1"}}\n')
    code, _ = run_cli("tools", "record", str(sandbox), str(calls_file))
    assert code == cli.EXIT_INPUT
    assert "not configured in EVALKIT_HTTP_TARGETS_FILE" in capsys.readouterr().err
    code, _ = run_cli("tools", "check", str(tmp_path / "missing.toml"))
    assert code == cli.EXIT_INPUT
