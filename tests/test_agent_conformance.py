"""``evalkit agent conformance`` and the reference agent (examples/agent_server.py):
the reference agent passes every check over real HTTP, and agents that break
a part of the session protocol fail or warn on the check for it."""

import asyncio
import importlib.util
import io
import itertools
import json
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx2
import pytest

from evalkit import cli
from evalkit.conformance import (
    CHECKS,
    TOOL_FAILURE,
    UNKNOWN_PROTOCOL,
    AgentConformance,
    ConformanceReport,
)
from evalkit.targets import TargetError
from evalkit.targets.http import HttpEndpoint, HttpTarget
from evalkit.targets.http_agent import PROTOCOL
from evalkit.targets.specs import HttpTargetSpec
from evalkit.tools.sandbox import load_sandbox

EXAMPLES = Path(__file__).parent.parent / "examples"
SANDBOX_FILE = EXAMPLES / "agents" / "booking.toml"
SANDBOX, FIXTURES = load_sandbox(SANDBOX_FILE)
TASKS = ["Book a table on 2026-11-20", "A table on 2026-11-21 at 20:30", "Book me something"]
REMOTE = HttpEndpoint(url="https://agent.example.com/v1/step", timeout_seconds=10)


def _load_reference() -> ModuleType:
    spec = importlib.util.spec_from_file_location("agent_server", EXAMPLES / "agent_server.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = _load_reference()


async def public(host: str, port: int) -> list[str]:
    return ["93.184.216.34"]


Handler = Callable[[dict[str, Any]], Any]


def reference(body: dict[str, Any]) -> Any:
    """The reference agent's answer, or 400 as its server would give it."""
    try:
        return ref.respond(body)
    except ref.BadRequestError as exc:
        return httpx2.Response(400, json={"error": str(exc)})


def transport(handler: Handler) -> httpx2.MockTransport:
    def answer(request: httpx2.Request) -> httpx2.Response:
        out = handler(json.loads(request.content))
        return out if isinstance(out, httpx2.Response) else httpx2.Response(200, json=out)

    return httpx2.MockTransport(answer)


def check(
    handler: Handler,
    tasks: list[str] | None = None,
    *,
    max_steps: int = 20,
    endpoint: HttpEndpoint = REMOTE,
    clock: Callable[[], float] | None = None,
    mock: httpx2.AsyncBaseTransport | None = None,
) -> ConformanceReport:
    extra: dict[str, Any] = {} if clock is None else {"clock": clock}
    checker = AgentConformance(
        "booker",
        endpoint,
        SANDBOX,
        tasks or TASKS,
        fixtures=FIXTURES,
        max_steps=max_steps,
        resolver=public,
        transport=mock or transport(handler),
        nonce="test",
        **extra,
    )
    return asyncio.run(checker.run())


def statuses(report: ConformanceReport) -> dict[str, str]:
    return {c.name: c.status for c in report.checks}


def detail(report: ConformanceReport, name: str) -> str:
    return next(c.detail for c in report.checks if c.name == name)


def request(messages: list[dict[str, Any]], step: int = 0, steps_left: int = 20) -> Any:
    return {
        "protocol": PROTOCOL,
        "session_id": "s-1",
        "step": step,
        "request_id": f"r-{step}",
        "messages": messages,
        "tools": SANDBOX.definitions(),
        "budget": {"steps_remaining": steps_left, "tokens_remaining": None},
    }


def user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def called(name: str, arguments: dict[str, Any], call_id: str = "c") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": call_id, "name": name, "arguments": arguments}],
    }


def answered(name: str, call_id: str = "c", **outcome: Any) -> dict[str, Any]:
    return {"role": "tool", "results": [{"id": call_id, "name": name, **outcome}]}


# --- the reference agent over real HTTP ------------------------------------------


@pytest.fixture
def agent_url() -> Iterator[tuple[str, Any]]:
    agent = ref.Agent()
    server = ref.make_server("127.0.0.1", 0, agent)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", agent
    finally:
        server.shutdown()
        server.server_close()


def test_the_reference_agent_passes_every_check_over_http(agent_url: tuple[str, Any]) -> None:
    url, agent = agent_url
    endpoint = HttpEndpoint(url=f"{url}/v1/step", allow_private=True, timeout_seconds=10)
    checker = AgentConformance("booker", endpoint, SANDBOX, TASKS, fixtures=FIXTURES)
    report = asyncio.run(checker.run())
    assert statuses(report) == dict.fromkeys(CHECKS, "pass"), report.checks
    assert report.passed(strict=True)
    assert detail(report, "protocol_version") == f"answered 400 to {UNKNOWN_PROTOCOL}"
    # Three sessions: find and book (3 replies), find and book at 20:30 (3), ask for a date (1).
    assert detail(report, "schemas") == "3 sessions, 7 replies and 4 tool calls, all valid"
    # Every repeated request_id was answered from the agent's store, not worked out again.
    assert agent.replayed == 3 + 2
    assert report.requests == agent.answered + agent.replayed + 1  # and the refused version
    assert 0 < report.latency_ms_p50 <= report.latency_ms_max


def test_the_reference_server_refuses_what_is_not_a_protocol_request(
    agent_url: tuple[str, Any],
) -> None:
    url, _ = agent_url
    with httpx2.Client(base_url=url, trust_env=False) as client:
        assert client.get("/healthz").text == "ok"
        assert client.get("/stats").json() == {"answered": 0, "replayed": 0}
        assert client.get("/nope").status_code == 404
        assert client.post("/v1/step", content=b"{not json").status_code == 400
        wrong = client.post("/v1/step", json={**request([user("x")]), "protocol": "other/1"})
        assert wrong.status_code == 400
        assert "speaks evalkit-agent/1" in wrong.json()["error"]
        big = client.post("/v1/step", content=b" " * (ref.MAX_BODY + 1))
        assert big.status_code == 413
        ok = client.post("/v1/step", json=request([user("on 2026-11-20")]))
        assert ok.json()["message"]["tool_calls"][0]["name"] == "find_slots"


def test_the_cli_checks_an_endpoint_and_exits_1_on_a_failing_check(
    agent_url: tuple[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url, _ = agent_url
    endpoints = tmp_path / "endpoints.toml"
    endpoints.write_text(
        f'[endpoints.booker]\nurl = "{url}/v1/step"\nallow_private = true\n'
        f'[endpoints.missing]\nurl = "{url}/v1/step"\n'  # private, so refused
    )
    monkeypatch.setenv("EVALKIT_HTTP_TARGETS_FILE", str(endpoints))
    monkeypatch.setenv("EVALKIT_DATABASE_URL", f"sqlite:///{tmp_path / 'unused.db'}")

    def run(*args: str) -> tuple[int, str]:
        out = io.StringIO()
        return cli.main(list(args), out=out), out.getvalue()

    code, text = run("agent", "conformance", "booker", "--sandbox", str(SANDBOX_FILE))
    assert code == 0, text
    assert text.startswith("agent booker on evalkit-agent/1: 6 pass\n")
    assert "4 sessions" in text  # the sandbox file's [conformance] tasks
    code, text = run(
        "agent", "conformance", "booker", "--sandbox", str(SANDBOX_FILE),
        "--task", "Book on 2026-11-20", "--json",
    )  # fmt: skip
    report = json.loads(text)
    assert code == 0
    assert report["passed"] is True
    assert [c["name"] for c in report["checks"]] == list(CHECKS)
    assert "1 sessions" in report["checks"][1]["detail"]

    code, text = run("agent", "conformance", "missing", "--sandbox", str(SANDBOX_FILE))
    assert code == cli.EXIT_REGRESSION
    assert "fail  protocol_version" in text
    assert "non-public address" in text

    bare = tmp_path / "bare.toml"
    bare.write_text(SANDBOX_FILE.read_text().split("[conformance]")[0])
    code, _ = run("agent", "conformance", "booker", "--sandbox", str(bare))
    assert code == cli.EXIT_INPUT
    code, _ = run("agent", "conformance", "nowhere", "--sandbox", str(SANDBOX_FILE))
    assert code == cli.EXIT_INPUT
    code, _ = run(
        "agent", "conformance", "booker", "--sandbox", str(SANDBOX_FILE), "--max-steps", "0"
    )
    assert code == cli.EXIT_INPUT


def test_the_cli_strict_mode_fails_on_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = ref.Agent()
    original = ref.respond

    def lenient(body: Any, *, skip_lookup: bool = False) -> Any:
        return original({**body, "protocol": PROTOCOL}, skip_lookup=skip_lookup)

    monkeypatch.setattr(ref, "respond", lenient)  # accepts any version: a warning
    server = ref.make_server("127.0.0.1", 0, agent)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoints = tmp_path / "endpoints.toml"
        port = server.server_address[1]
        endpoints.write_text(
            f'[endpoints.booker]\nurl = "http://127.0.0.1:{port}/"\nallow_private = true\n'
        )
        monkeypatch.setenv("EVALKIT_HTTP_TARGETS_FILE", str(endpoints))
        args = ["agent", "conformance", "booker", "--sandbox", str(SANDBOX_FILE)]
        out = io.StringIO()
        assert cli.main(args, out=out) == 0
        assert "warn  protocol_version" in out.getvalue()
        assert cli.main([*args, "--strict"], out=io.StringIO()) == cli.EXIT_REGRESSION
    finally:
        server.shutdown()
        server.server_close()


# --- the reference agent's policy ----------------------------------------------


def test_the_reference_agent_asks_for_a_date_then_looks_up_and_books() -> None:
    asked = ref.respond(request([user("Book me something")]))
    assert asked["message"] == {
        "content": "Which date would you like a table on? Please give it as YYYY-MM-DD.",
        "tool_calls": [],
    }
    assert asked["protocol"] == PROTOCOL
    assert asked["usage"]["input_tokens"] > 0
    messages = [user("Book me something"), {"role": "assistant", "content": "Which date?"}]
    messages.append(user("2026-11-20 at 20:30 please"))
    first = ref.respond(request(messages, step=1))["message"]["tool_calls"]
    assert first == [{"id": "call-1-find_slots", "name": "find_slots",
                      "arguments": {"date": "2026-11-20"}}]  # fmt: skip
    messages += [
        called("find_slots", {"date": "2026-11-20"}),
        answered("find_slots", result=["19:00", "20:30"]),
    ]
    book = ref.respond(request(messages, step=2))["message"]["tool_calls"][0]
    assert book["arguments"] == {"date": "2026-11-20", "slot": "20:30"}  # the asked-for time
    messages += [
        called("book", book["arguments"], "b"),
        answered("book", "b", result={"confirmation": "BK-2030"}),
    ]
    done = ref.respond(request(messages, step=3))
    assert done["message"]["content"] == (
        "Booked a table on 2026-11-20 at 20:30, confirmation BK-2030."
    )
    assert done["stop_reason"] == "booked"


def test_the_reference_agent_takes_the_first_slot_or_says_none_is_free() -> None:
    base = [user("on 2026-11-20 at 23:00"), called("find_slots", {"date": "2026-11-20"})]
    taken = ref.respond(request([*base, answered("find_slots", result=["19:00", 7])]))
    assert taken["message"]["tool_calls"][0]["arguments"]["slot"] == "19:00"
    none = ref.respond(request([*base, answered("find_slots", result=[])]))
    assert none["message"]["content"] == "There are no free tables on 2026-11-20."
    junk = ref.respond(request([*base, answered("find_slots", result={"x": 1})]))
    assert junk["message"]["content"] == "There are no free tables on 2026-11-20."


def test_the_reference_agent_retries_a_failed_tool_once_then_apologises() -> None:
    once = [
        user("on 2026-11-20"),
        called("find_slots", {"date": "2026-11-20"}),
        answered("find_slots", error="down"),
    ]
    assert ref.respond(request(once))["message"]["tool_calls"][0]["name"] == "find_slots"
    twice = [
        *once,
        called("find_slots", {"date": "2026-11-20"}, "d"),
        answered("find_slots", "d", error="down"),
    ]
    gave_up = ref.respond(request(twice))
    assert gave_up["stop_reason"] == "tool_failed"
    assert gave_up["message"]["tool_calls"] == []
    booking = [
        user("on 2026-11-20"),
        called("book", {"date": "2026-11-20", "slot": "19:00"}),
        answered("book", error="taken"),
        called("book", {"date": "2026-11-20", "slot": "19:00"}, "d"),
        answered("book", "d", error="taken"),
    ]
    failed = ref.respond(request(booking), skip_lookup=True)
    assert failed["stop_reason"] == "tool_failed"
    assert "did not go through" in failed["message"]["content"]


def test_skip_lookup_books_blind_and_a_spent_budget_ends_in_text() -> None:
    blind = ref.respond(request([user("on 2026-11-20")]), skip_lookup=True)
    assert blind["message"]["tool_calls"][0]["name"] == "book"
    assert blind["message"]["tool_calls"][0]["arguments"]["slot"] == "19:00"
    wanted = ref.respond(request([user("on 2026-11-20 at 21:15")]), skip_lookup=True)
    assert wanted["message"]["tool_calls"][0]["arguments"]["slot"] == "21:15"
    last = ref.respond(request([user("on 2026-11-20")], steps_left=1))
    assert last["stop_reason"] == "out_of_steps"
    assert last["message"]["tool_calls"] == []
    no_tools = {**request([user("on 2026-11-20")]), "tools": []}
    assert "no find_slots tool" in ref.respond(no_tools)["message"]["content"]


def test_only_this_turns_tool_results_count() -> None:
    earlier = [
        user("on 2026-11-20"),
        called("find_slots", {"date": "2026-11-20"}),
        answered("find_slots", result=["19:00"]),
        called("book", {"date": "2026-11-20", "slot": "19:00"}, "b"),
        answered("book", "b", result={"confirmation": "BK-1900"}),
        {"role": "assistant", "content": "Booked", "tool_calls": []},
        user("And another on 2026-12-31"),
    ]
    again = ref.respond(request(earlier))["message"]["tool_calls"][0]
    assert again["arguments"] == {"date": "2026-12-31"}


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ([], "JSON object"),
        ({**request([user("x")]), "protocol": None}, "not None"),
        ({**request([user("x")]), "messages": []}, "non-empty"),
        ({**request([user("x")]), "messages": [{"role": "robot"}]}, "a role"),
        ({**request([{"role": "tool", "results": []}])}, "start with the user"),
        ({**request([user("x"), {"role": "assistant"}])}, "not end with the assistant"),
        ({**request([user("x")]), "step": True}, "step must be"),
        ({**request([user("x")]), "step": -1}, "step must be"),
        ({**request([user("x")]), "tools": None}, "tools must be"),
    ],
)
def test_the_reference_agent_refuses_requests_it_cannot_read(body: Any, message: str) -> None:
    with pytest.raises(ref.BadRequestError, match=message):
        ref.respond(body)


def test_the_reference_agent_ignores_a_malformed_budget() -> None:
    body = {**request([user("on 2026-11-20")]), "budget": {"steps_remaining": "1"}}
    assert ref.respond(body)["message"]["tool_calls"]


def test_the_reference_agent_stores_answers_by_request_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ref, "MAX_STORED", 2)
    agent = ref.Agent(delay_ms=1)
    bodies = [json.dumps(request([user(f"on 2026-11-2{i}")], step=i)).encode() for i in range(3)]
    first = agent.handle(bodies[0], None)
    assert agent.handle(bodies[0], None) == first
    assert (agent.answered, agent.replayed) == (1, 1)
    agent.handle(bodies[1], None)
    agent.handle(bodies[2], None)  # pushes out the least recently used: step 0
    assert list(agent.stored) == ["r-1", "r-2"]
    agent.handle(bodies[0], None)
    assert agent.answered == 4
    # Without a request_id in the body the Idempotency-Key header is the key.
    keyless = json.dumps({k: v for k, v in request([user("x")]).items() if k != "request_id"})
    agent.handle(keyless.encode(), "k-1")
    assert "k-1" in agent.stored
    assert agent.handle(b"[1, 2", None)[0] == 400
    assert agent.handle(b'{"protocol": "x"}', None)[0] == 400


def test_the_reference_server_main_writes_its_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    port_file = tmp_path / "port"
    done: list[int] = []

    def serve() -> None:
        done.append(ref.main(["--port", "0", "--port-file", str(port_file)]))

    thread = threading.Thread(target=serve, daemon=True)
    original = ref.make_server
    servers: list[Any] = []

    def capture(*args: Any) -> Any:
        servers.append(original(*args))
        return servers[-1]

    monkeypatch.setattr(ref, "make_server", capture)
    try:
        thread.start()
        for _ in range(200):
            if port_file.exists():
                break
            threading.Event().wait(0.01)
        port = int(port_file.read_text())
        with httpx2.Client(trust_env=False) as client:
            assert client.get(f"http://127.0.0.1:{port}/healthz").text == "ok"
    finally:
        if servers:
            servers[0].shutdown()
        thread.join(5)
    assert done == [0]


# --- agents that break the protocol -----------------------------------------------


def test_an_agent_that_accepts_any_version_warns() -> None:
    report = check(lambda b: ref.respond({**b, "protocol": PROTOCOL}))
    assert statuses(report)["protocol_version"] == "warn"
    assert report.passed()
    assert not report.passed(strict=True)


@pytest.mark.parametrize(("status", "verdict"), [(404, "warn"), (422, "warn"), (500, "fail")])
def test_refusing_an_unknown_version_with_another_status(status: int, verdict: str) -> None:
    def agent(body: dict[str, Any]) -> Any:
        if body["protocol"] != PROTOCOL:
            return httpx2.Response(status, json={})
        return reference(body)

    report = check(agent)
    assert statuses(report)["protocol_version"] == verdict
    assert str(status) in detail(report, "protocol_version")


def test_calling_a_tool_that_was_not_offered_fails_schemas() -> None:
    def agent(body: dict[str, Any]) -> Any:
        out = reference(body)
        if (
            body["step"] == 0
            and not isinstance(out, httpx2.Response)
            and out["message"]["tool_calls"]
        ):
            out["message"]["tool_calls"][0]["name"] = "cancel_everything"
        return out

    report = check(agent)
    assert statuses(report)["schemas"] == "fail"
    assert detail(report, "schemas") == (
        "task 1: step 0: called 'cancel_everything', which was not offered (and 1 more)"
    )
    assert not report.passed()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"date": 20261120}, "break its input_schema at date: 20261120 is not of type 'string'"),
        ({"date": "2026-11-20", "extra": 1}, "break its input_schema at arguments"),
        ("2026-11-20", "are not a JSON object"),
    ],
)
def test_arguments_that_break_the_schema_fail(arguments: Any, message: str) -> None:
    def agent(body: dict[str, Any]) -> Any:
        out = reference(body)
        if not isinstance(out, httpx2.Response) and out["message"]["tool_calls"]:
            out["message"]["tool_calls"][0]["arguments"] = arguments
        return out

    report = check(agent, ["on 2026-11-20"])
    assert statuses(report)["schemas"] == "fail"
    assert message in detail(report, "schemas")


def test_an_agent_that_answers_off_protocol_fails_and_the_rest_is_skipped() -> None:
    report = check(lambda b: {"text": "hello"})
    assert statuses(report) == {
        "protocol_version": "warn",
        "schemas": "fail",
        "idempotent_retries": "skip",
        "ordering": "skip",
        "tool_errors": "skip",
        "timeouts": "pass",
    }
    assert "broke the session protocol: message: Field required" in detail(report, "schemas")


def test_an_agent_that_never_stops_calling_tools_fails() -> None:
    def agent(body: dict[str, Any]) -> Any:
        call = {"id": f"c{body['step']}", "name": "find_slots", "arguments": {"date": "2026-11-20"}}
        return {"message": {"tool_calls": [call]}}

    report = check(agent, ["x"], max_steps=4)
    assert detail(report, "schemas") == "task 1 did not finish within 4 replies"


def test_an_agent_that_answers_differently_each_time_warns() -> None:
    counter = itertools.count()

    def agent(body: dict[str, Any]) -> Any:
        if body["protocol"] != PROTOCOL:
            return httpx2.Response(400)
        return {"message": {"content": f"reply {next(counter)}"}}

    report = check(agent)
    assert statuses(report) == {
        "protocol_version": "pass",
        "schemas": "pass",
        "idempotent_retries": "warn",
        "ordering": "warn",
        "tool_errors": "pass",
        "timeouts": "pass",
    }
    assert "different answer at step 0" in detail(report, "idempotent_retries")
    assert detail(report, "ordering").startswith(
        "step 0 of a resumed session (and 3 more) answered differently"
    )
    assert report.passed()


def test_a_differing_concurrent_retry_warns() -> None:
    seen: dict[str, int] = {}

    def agent(body: dict[str, Any]) -> Any:
        seen[body["request_id"]] = seen.get(body["request_id"], 0) + 1
        out = reference(body)
        if seen[body["request_id"]] >= 3 and not isinstance(out, httpx2.Response):
            out["message"]["content"] = "changed my mind"
        return out

    report = check(agent, ["Book me something"])
    assert statuses(report)["idempotent_retries"] == "warn"


def test_an_agent_that_keys_state_by_arrival_fails() -> None:
    """It expects each session's steps in order and once, and refuses anything else."""
    expected: dict[str, int] = {}

    def agent(body: dict[str, Any]) -> Any:
        if body["protocol"] != PROTOCOL:
            return httpx2.Response(400)
        session = body["session_id"]
        if body["step"] != expected.get(session, 0):
            return httpx2.Response(409, json={"error": "out of order"})
        expected[session] = body["step"] + 1
        return reference(body)

    report = check(agent)
    assert statuses(report)["idempotent_retries"] == "fail"
    assert "returned 409" in detail(report, "idempotent_retries")
    assert statuses(report)["ordering"] == "fail"
    assert detail(report, "ordering").startswith("a session resumed at a later step")


def test_a_concurrent_session_that_fails_fails_ordering() -> None:
    namer = AgentConformance("booker", REMOTE, SANDBOX, TASKS[:1], nonce="test")
    concurrent = {namer.session_id(f"concurrent-{i}") for i in range(2)}
    asyncio.run(namer.http.aclose())

    def agent(body: dict[str, Any]) -> Any:
        if body["session_id"] in concurrent:
            return httpx2.Response(503)
        return reference(body)

    report = check(agent, TASKS[:1])
    assert statuses(report)["ordering"] == "fail"
    assert detail(report, "ordering").startswith("concurrent session 1: step 0:")
    assert "returned 503" in detail(report, "ordering")


def test_an_agent_that_retries_failed_tools_forever_fails() -> None:
    def agent(body: dict[str, Any]) -> Any:
        last = body["messages"][-1]
        if last["role"] == "tool" and "error" in last["results"][0]:
            call = {"id": f"c{body['step']}", "name": "find_slots",
                    "arguments": {"date": "2026-11-20"}}  # fmt: skip
            return {"message": {"tool_calls": [call]}}
        return reference(body)

    report = check(agent, max_steps=6)
    assert statuses(report)["schemas"] == "pass"
    assert detail(report, "tool_errors") == (
        "with every tool failing, the session did not finish within 6 replies"
    )


def test_tool_failures_reach_the_agent_as_errors() -> None:
    seen: list[Any] = []

    def agent(body: dict[str, Any]) -> Any:
        for m in body["messages"]:
            if m["role"] == "tool":
                seen.extend(m["results"])
        return reference(body)

    check(agent, ["on 2026-11-20"])
    assert {"id": "call-0-find_slots", "name": "find_slots", "error": TOOL_FAILURE} in seen


def test_a_tool_error_session_that_breaks_the_protocol_fails() -> None:
    def agent(body: dict[str, Any]) -> Any:
        last = body["messages"][-1]
        if last["role"] == "tool" and last["results"][0].get("error") == TOOL_FAILURE:
            return {"message": "oops"}
        return reference(body)

    report = check(agent, ["on 2026-11-20"])
    assert detail(report, "tool_errors").startswith("with every tool failing: step 1:")


def test_a_timeout_fails_timeouts_and_the_check_it_happened_in() -> None:
    def answer(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        if body["protocol"] == PROTOCOL and body["step"] == 1:
            raise httpx2.ReadTimeout("slow", request=request)
        out = reference(body)
        return out if isinstance(out, httpx2.Response) else httpx2.Response(200, json=out)

    report = check(reference, ["on 2026-11-20"], mock=httpx2.MockTransport(answer))
    assert statuses(report)["schemas"] == "fail"
    assert "timed out" in detail(report, "schemas")
    assert statuses(report)["timeouts"] == "fail"
    assert detail(report, "timeouts").endswith("ran out of 10 s")
    assert detail(report, "timeouts").startswith("step 1 of s-")  # each probe that reached it


def test_several_timeouts_are_counted() -> None:
    def answer(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectTimeout("slow", request=request)

    report = check(reference, mock=httpx2.MockTransport(answer))
    assert "(and 3 more) ran out of 10 s" in detail(report, "timeouts")
    assert statuses(report)["protocol_version"] == "fail"


def test_a_slow_agent_warns_near_its_timeout() -> None:
    ticks = itertools.count()

    def clock() -> float:
        return next(ticks) * 4.5  # every request takes 4.5 s of a 5 s timeout

    endpoint = HttpEndpoint(url=REMOTE.url, timeout_seconds=5)
    report = check(reference, ["Book me something"], endpoint=endpoint, clock=clock)
    assert statuses(report)["timeouts"] == "warn"
    assert detail(report, "timeouts") == (
        "the slowest reply took 4500 ms, close to the 5 s timeout"
    )
    assert report.latency_ms_max == 4500.0
    assert report.to_dict()["latency_ms"] == {"p50": 4500.0, "max": 4500.0}


def test_the_checker_takes_one_to_twenty_tasks_and_a_sane_step_budget() -> None:
    for tasks, steps in (([], 20), (["x"] * 21, 20), (["x"], 0), (["x"], 501)):
        with pytest.raises(ValueError, match=r"task|max_steps"):
            AgentConformance("booker", REMOTE, SANDBOX, tasks, max_steps=steps)


def test_a_session_id_is_new_for_every_invocation() -> None:
    a = AgentConformance("booker", REMOTE, SANDBOX, ["x"])
    b = AgentConformance("booker", REMOTE, SANDBOX, ["x"])
    assert a.session_id("task-0") != b.session_id("task-0")
    assert a.session_id("task-0").startswith("s-")
    assert len(a.session_id("task-0")) == 34
    asyncio.run(a.http.aclose())
    asyncio.run(b.http.aclose())


def test_requests_are_the_ones_a_run_sends() -> None:
    seen: list[httpx2.Request] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        out = reference(json.loads(request.content))
        return out if isinstance(out, httpx2.Response) else httpx2.Response(200, json=out)

    check(reference, ["on 2026-11-20"], mock=httpx2.MockTransport(answer))
    body = json.loads(seen[1].content)
    assert set(body) == {
        "protocol", "session_id", "step", "request_id", "messages", "tools", "budget"
    }  # fmt: skip
    assert seen[1].headers["Idempotency-Key"] == body["request_id"]
    assert seen[1].headers["X-Evalkit-Protocol"] == PROTOCOL
    assert seen[1].headers["X-Evalkit-Session"] == body["session_id"]
    assert body["budget"] == {"steps_remaining": 20, "tokens_remaining": None}
    assert json.loads(seen[0].content)["protocol"] == UNKNOWN_PROTOCOL


# --- the HTTP client reports statuses and timeouts ----------------------------------


def test_http_errors_carry_the_status_and_body_timeouts_are_timeouts() -> None:
    class SlowBody(httpx2.AsyncByteStream):
        async def __aiter__(self) -> Any:
            raise httpx2.ReadTimeout("slow body")
            yield b""  # pragma: no cover

    def answer(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/slow":
            return httpx2.Response(200, stream=SlowBody())
        if request.url.path == "/moved":
            return httpx2.Response(302, headers={"Location": "/x"})
        return httpx2.Response(429, headers={"Retry-After": "3"})

    async def post(path: str) -> TargetError:
        target = HttpTarget(
            HttpTargetSpec(type="http", endpoint="t"),
            HttpEndpoint(url=f"https://agent.example.com{path}"),
            resolver=public,
            transport=httpx2.MockTransport(answer),
        )
        try:
            with pytest.raises(TargetError) as caught:
                await target.post({})
            return caught.value
        finally:
            await target.aclose()

    limited = asyncio.run(post("/limited"))
    assert (limited.status, limited.retryable, limited.retry_after) == (429, True, 3.0)
    assert not limited.timed_out
    moved = asyncio.run(post("/moved"))
    assert (moved.status, moved.retryable) == (302, False)
    slow = asyncio.run(post("/slow"))
    assert slow.timed_out
    assert slow.retryable
    assert slow.status is None
