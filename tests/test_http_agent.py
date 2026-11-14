"""The HTTP agent target and the session protocol (docs/agent-protocol.md):
requests and headers on the wire, tool calls answered by the sandbox, the
per-session step and token budgets, protocol violations, idempotent retries,
the step cache, sessions with a simulated user, and whole runs that are
stored, rerun for free and resumed without asking for a step twice."""

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx2
import pytest
from pydantic import SecretStr, TypeAdapter, ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from evalkit.config import Settings
from evalkit.datasets import Case, import_dataset
from evalkit.db.models import CachedResponse, Result, ResultStep, ResultTurn, Run
from evalkit.errors import InvalidInputError
from evalkit.providers.registry import ProviderConfig
from evalkit.ratelimit import Clock, RealClock
from evalkit.redaction import RedactionConfig, Redactor
from evalkit.runner import Runner
from evalkit.runspec import RunConfig
from evalkit.simuser import LLMSimulator, ScriptedSimulator
from evalkit.targets import HttpAgentTargetSpec, TargetError, TargetSpec, Usage
from evalkit.targets.factory import build_target, identity, simulator_for
from evalkit.targets.http import HttpEndpoint
from evalkit.targets.http_agent import (
    END_TURN,
    MAX_STEPS,
    PROTOCOL,
    TOKEN_BUDGET,
    HttpAgentTarget,
    parse_reply,
    request_id,
)
from evalkit.targets.specs import ScriptedSimulatorSpec
from tests.test_agent import FIXTURES, SANDBOX
from tests.test_conversation import LOCAL, Registry
from tests.test_factory import registry

SPEC: TypeAdapter[Any] = TypeAdapter(TargetSpec)
ENDPOINT = HttpEndpoint(
    url="https://agent.example.com/v1/step",
    headers={"X-Team": "evals"},
    secret_headers={"Authorization": SecretStr("Bearer s3cret")},
)
PUBLIC = "93.184.216.34"
CASE = Case(id="c1", input={"date": "2026-11-20"})
USAGE = {"input_tokens": 100, "output_tokens": 20, "cost_usd": 0.001}


async def public(host: str, port: int) -> list[str]:
    return [PUBLIC]


async def private(host: str, port: int) -> list[str]:
    return ["10.0.0.5"]


def reply(content: str = "", calls: list[dict[str, Any]] | None = None, **extra: Any) -> Any:
    return {"message": {"content": content, "tool_calls": calls or []}, "usage": USAGE, **extra}


def booking_plan(body: dict[str, Any]) -> Any:
    """The reference behaviour: look up slots, book the first, confirm. In a
    conversation it waits for the user to ask for a table first."""
    messages = body["messages"]
    results = [r for m in messages if m["role"] == "tool" for r in m["results"]]
    turn = next(m for m in reversed(messages) if m["role"] == "user")["content"]
    if "table" not in turn and not results:
        return reply("What can I do for you?")
    looked = [r for r in results if r["name"] == "find_slots"]
    booked = [r for r in results if r["name"] == "book"]
    if not looked:
        return reply("", [{"id": "c1", "name": "find_slots", "arguments": {"date": "2026-11-20"}}])
    if not booked:
        slot = looked[-1]["result"][0]
        return reply("Booking", [{"id": "c2", "name": "book", "arguments": {"slot": slot}}])
    confirmation = booked[-1]["result"]["confirmation"]
    return reply(f"Booked, confirmation {confirmation}", stop_reason="done")


@dataclass
class Seen:
    body: dict[str, Any]
    headers: httpx2.Headers
    url: str


class FakeAgent:
    """An agent on the protocol, in process. ``answer`` maps a request body to a
    response (a document, or an ``httpx2.Response``); every request is kept."""

    def __init__(self, answer: Callable[[dict[str, Any]], Any] = booking_plan) -> None:
        self.answer = answer
        self.seen: list[Seen] = []
        self.fail: dict[int, httpx2.Response] = {}

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.seen.append(Seen(body, request.headers, str(request.url)))
        failure = self.fail.pop(len(self.seen) - 1, None)
        if failure is not None:
            return failure
        out = self.answer(body)
        return out if isinstance(out, httpx2.Response) else httpx2.Response(200, json=out)

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [s.body for s in self.seen]


def spec(**fields: Any) -> HttpAgentTargetSpec:
    base: dict[str, Any] = {
        "type": "http_agent",
        "endpoint": "booker",
        "template": "Book a table on {{ date }}",
        "sandbox": SANDBOX,
        "fixtures": FIXTURES,
    }
    return HttpAgentTargetSpec.model_validate({**base, **fields})


def target(
    agent: FakeAgent, s: HttpAgentTargetSpec | None = None, resolver: Any = public, **kw: Any
) -> HttpAgentTarget:
    s = s or spec()
    simulator = (
        ScriptedSimulator(s.simulator) if isinstance(s.simulator, ScriptedSimulatorSpec) else None
    )
    return HttpAgentTarget(
        s,
        kw.pop("endpoint", ENDPOINT),
        simulator,
        resolver=resolver,
        transport=httpx2.MockTransport(agent),
        **kw,
    )


def generate(t: HttpAgentTarget, case: Case = CASE) -> Any:
    async def main() -> Any:
        try:
            return await t.generate(case)
        finally:
            await t.aclose()

    return asyncio.run(main())


class MemoryStore:
    def __init__(self) -> None:
        self.entries: dict[str, tuple[str, str | None]] = {}

    async def get(self, key: str) -> Any:
        found = self.entries.get(key)
        if found is None:
            return None

        @dataclass
        class Hit:
            output: str
            stop_reason: str | None

        return Hit(*found)

    async def put(
        self, key: str, owner_hash: str, text: str, stop_reason: str | None, usage: Usage
    ) -> None:
        self.entries[key] = (text, stop_reason)


# The wire


def test_a_task_session_on_the_wire() -> None:
    agent = FakeAgent()
    t = target(agent)
    generation = generate(t)
    assert generation.text == "Booked, confirmation BK-19"
    assert generation.stop_reason == "done"
    first, second, third = agent.seen
    # Requests go to the pinned address with the original host, and carry the
    # endpoint's headers, its secret headers and the protocol's own.
    assert first.url == f"https://{PUBLIC}/v1/step"
    assert first.headers["host"] == "agent.example.com"
    assert first.headers["authorization"] == "Bearer s3cret"
    assert first.headers["x-team"] == "evals"
    assert first.headers["x-evalkit-protocol"] == PROTOCOL
    session = first.body["session_id"]
    assert first.headers["x-evalkit-session"] == session
    assert session.startswith("s-")
    assert len(session) == 34
    for i, s in enumerate(agent.seen):
        assert s.body["protocol"] == PROTOCOL
        assert s.body["session_id"] == session
        assert s.body["step"] == i
        assert s.headers["idempotency-key"] == s.body["request_id"]
        assert s.body["budget"] == {"steps_remaining": 20 - i, "tokens_remaining": None}
        assert [t["name"] for t in s.body["tools"]] == ["find_slots", "book"]
        assert s.body["tools"][0]["input_schema"]["required"] == ["date"]
        without = {k: v for k, v in s.body.items() if k != "request_id"}
        assert s.body["request_id"] == request_id(without)
    assert first.body["messages"] == [{"role": "user", "content": "Book a table on 2026-11-20"}]
    assert third.body["messages"][1:] == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "name": "find_slots", "arguments": {"date": "2026-11-20"}}],
        },
        {
            "role": "tool",
            "results": [{"id": "c1", "name": "find_slots", "result": ["19:00", "20:30"]}],
        },
        {
            "role": "assistant",
            "content": "Booking",
            "tool_calls": [{"id": "c2", "name": "book", "arguments": {"slot": "19:00"}}],
        },
        {
            "role": "tool",
            "results": [{"id": "c2", "name": "book", "result": {"confirmation": "BK-19"}}],
        },
    ]
    assert len(second.body["messages"]) == 3
    kinds = [(s.kind, s.tool, s.source) for s in generation.steps]
    assert kinds == [
        ("model", None, None),
        ("tool", "find_slots", "fixture"),
        ("model", None, None),
        ("tool", "book", "script"),
        ("model", None, None),
    ]
    assert [s.index for s in generation.steps] == list(range(5))
    assert generation.usage == Usage(300, 60, 0, 0, pytest.approx(0.003))  # type: ignore[arg-type]
    assert generation.turns == ()
    assert t.fresh_steps == 3


def test_sessions_are_stable_per_case_and_differ_between_cases() -> None:
    agent = FakeAgent()
    generate(target(agent))
    generate(target(agent))
    generate(target(agent), Case(id="c2", input={"date": "2026-11-20"}))
    ids = [s.body["session_id"] for s in agent.seen]
    assert ids[0] == ids[3] != ids[6]
    # The same step of the same session has the same idempotency key.
    assert [s.body["request_id"] for s in agent.seen[:3]] == [
        s.body["request_id"] for s in agent.seen[3:6]
    ]


def test_tool_errors_go_back_to_the_agent() -> None:
    def answer(body: dict[str, Any]) -> Any:
        if len(body["messages"]) == 1:
            return reply(
                "",
                [
                    {"id": "x", "name": "launch", "arguments": {}},
                    {"id": "y", "name": "find_slots", "arguments": {"date": 3}},
                ],
            )
        return reply("Sorry")

    agent = FakeAgent(answer)
    generation = generate(target(agent))
    results = agent.bodies[1]["messages"][-1]["results"]
    assert [sorted(r) for r in results] == [["error", "id", "name"], ["error", "id", "name"]]
    assert [s.status for s in generation.steps if s.kind == "tool"] == [
        "unknown_tool",
        "invalid_arguments",
    ]
    assert generation.stop_reason == END_TURN


# Budgets


def test_the_step_budget_ends_a_session_that_never_stops() -> None:
    def answer(body: dict[str, Any]) -> Any:
        return reply(
            "again", [{"id": "c", "name": "find_slots", "arguments": {"date": "2026-11-20"}}]
        )

    agent = FakeAgent(answer)
    generation = generate(target(agent, spec(max_steps=3)))
    assert len(agent.seen) == 3
    assert [b["budget"]["steps_remaining"] for b in agent.bodies] == [3, 2, 1]
    assert generation.stop_reason == MAX_STEPS
    assert generation.text == "again"
    # The last reply's tool calls are still answered and stored.
    assert [s.kind for s in generation.steps] == ["model", "tool"] * 3


def test_the_token_budget_counts_what_the_agent_reports() -> None:
    def answer(body: dict[str, Any]) -> Any:
        return reply(
            "more", [{"id": "c", "name": "find_slots", "arguments": {"date": "2026-11-20"}}]
        )

    agent = FakeAgent(answer)
    generation = generate(target(agent, spec(max_tokens=300)))
    # 120 tokens a reply: 300, 180, 60 left, then nothing.
    assert [b["budget"]["tokens_remaining"] for b in agent.bodies] == [300, 180, 60]
    assert generation.stop_reason == TOKEN_BUDGET


def test_budgets_are_validated() -> None:
    with pytest.raises(ValidationError):
        spec(max_steps=0)
    with pytest.raises(ValidationError):
        spec(max_tokens=0)


# Protocol violations and transport failures


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ([1, 2], "answer"),
        ({}, "message"),
        ({"message": {"content": 3}}, "message.content"),
        ({"message": {"tool_calls": [{"name": "book"}]}}, "message.tool_calls.0.id"),
        ({"message": {"tool_calls": [{"id": "", "name": "book"}]}}, "message.tool_calls.0.id"),
        (
            {"message": {"tool_calls": [{"id": "a", "name": "b"}, {"id": "a", "name": "b"}]}},
            "unique",
        ),
        ({"message": {}, "usage": {"input_tokens": -1}}, "usage.input_tokens"),
        ({"message": {}, "usage": {"input_tokens": "5"}}, "usage.input_tokens"),
        ({"message": {}, "usage": {"cost_usd": float("inf")}}, "usage.cost_usd"),
        ({"message": {}, "protocol": "evalkit-agent/2"}, "evalkit-agent/2"),
        (
            {"message": {"tool_calls": [{"id": str(i), "name": "b"} for i in range(65)]}},
            "tool_calls",
        ),
    ],
)
def test_protocol_violations_fail_the_case_without_a_retry(document: Any, message: str) -> None:
    with pytest.raises(TargetError, match="broke the session protocol") as info:
        parse_reply(document, "booker")
    assert message in info.value.message
    assert not info.value.retryable


def test_unknown_fields_are_ignored_and_defaults_apply() -> None:
    parsed = parse_reply({"message": {"content": "hi", "extra": 1}, "trace": "x"}, "booker")
    assert (parsed.text, parsed.tool_uses, parsed.tokens, parsed.stop_reason) == ("hi", (), 0, None)
    parsed = parse_reply({"protocol": PROTOCOL, "message": {}, "usage": {"cost_usd": 1}}, "b")
    assert parsed.usage.cost_usd == 1.0


def test_a_malformed_answer_from_the_endpoint_fails_the_case() -> None:
    agent = FakeAgent(lambda body: httpx2.Response(200, text="not json"))
    with pytest.raises(TargetError, match="did not return JSON") as info:
        generate(target(agent))
    assert not info.value.retryable
    agent = FakeAgent(lambda body: {"message": {"content": None}})
    with pytest.raises(TargetError, match="broke the session protocol"):
        generate(target(agent))


@pytest.mark.parametrize(
    ("status", "retryable"), [(503, True), (429, True), (408, True), (400, False), (401, False)]
)
def test_http_failures_follow_the_retry_policy(status: int, retryable: bool) -> None:
    agent = FakeAgent()
    agent.fail[0] = httpx2.Response(status, headers={"retry-after": "3"})
    with pytest.raises(TargetError) as info:
        generate(target(agent))
    assert info.value.retryable is retryable
    assert info.value.retry_after == 3.0


def test_a_retry_after_cools_down_the_endpoint() -> None:
    class Limiter:
        def __init__(self) -> None:
            self.clock: Clock = RealClock()
            self.acquired = 0
            self.cooled: list[float] = []

        async def acquire(self) -> None:
            self.acquired += 1

        async def cool_down(self, seconds: float) -> None:
            self.cooled.append(seconds)

    agent = FakeAgent()
    agent.fail[1] = httpx2.Response(429, headers={"retry-after": "500"})
    t = target(agent)
    limiter = Limiter()
    t.attach(store=None, redactor=Redactor(), limiter=limiter, max_retry_after=60)
    with pytest.raises(TargetError):
        generate(t)
    assert (limiter.acquired, limiter.cooled) == (2, [60])


def test_private_addresses_are_refused_unless_the_endpoint_allows_them() -> None:
    agent = FakeAgent()
    with pytest.raises(TargetError, match="non-public address"):
        generate(target(agent, resolver=private))
    assert agent.seen == []
    allowed = HttpEndpoint(url="http://agent.internal:8080/step", allow_private=True)
    generation = generate(target(agent, resolver=private, endpoint=allowed))
    assert generation.text == "Booked, confirmation BK-19"
    assert agent.seen[0].url == "http://10.0.0.5:8080/step"


# Idempotent retries and the step cache


def test_a_retried_session_resends_the_same_request_ids() -> None:
    agent = FakeAgent()
    agent.fail[2] = httpx2.Response(503)
    t = target(agent)
    with pytest.raises(TargetError):
        generate(t)
    generate(target(agent))
    ids = [s.body["request_id"] for s in agent.seen]
    # Without a cache every step is asked again, under the key it had the first time.
    assert ids[3:6] == ids[0:3]


def test_a_retried_session_asks_only_for_the_steps_it_lacks() -> None:
    agent = FakeAgent()
    agent.fail[2] = httpx2.Response(503)
    store = MemoryStore()
    first = target(agent)
    first.attach(store=store, redactor=Redactor())
    with pytest.raises(TargetError):
        generate(first)
    second = target(agent)
    second.attach(store=store, redactor=Redactor())
    generation = generate(second)
    assert second.fresh_steps == 1
    # The failed third step goes out again with the key it was first sent with.
    assert agent.seen[3].body["request_id"] == agent.seen[2].body["request_id"]
    assert [s.cached for s in generation.steps if s.kind == "model"] == [True, True, False]
    assert generation.usage.cost_usd == pytest.approx(0.001)
    rerun = target(agent)
    rerun.attach(store=store, redactor=Redactor())
    again = generate(rerun)
    assert rerun.fresh_steps == 0
    assert again.text == generation.text
    assert again.usage == Usage()


def test_a_replayed_session_spends_its_token_budget_the_same_way() -> None:
    def answer(body: dict[str, Any]) -> Any:
        return reply(
            "more", [{"id": "c", "name": "find_slots", "arguments": {"date": "2026-11-20"}}]
        )

    agent = FakeAgent(answer)
    agent.fail[2] = httpx2.Response(503)
    store = MemoryStore()
    first = target(agent, spec(max_tokens=300))
    first.attach(store=store, redactor=Redactor())
    with pytest.raises(TargetError):
        generate(first)
    second = target(agent, spec(max_tokens=300))
    second.attach(store=store, redactor=Redactor())
    generation = generate(second)
    assert generation.stop_reason == TOKEN_BUDGET
    assert agent.seen[-1].body["budget"]["tokens_remaining"] == 60
    assert second.fresh_steps == 1


def test_a_cache_entry_this_version_did_not_write_is_asked_again() -> None:
    agent = FakeAgent()
    store = MemoryStore()
    first = target(agent)
    first.attach(store=store, redactor=Redactor())
    generate(first)
    for key in store.entries:
        store.entries[key] = ('{"text": 1}', None)
    second = target(agent)
    second.attach(store=store, redactor=Redactor())
    generate(second)
    assert second.fresh_steps == 3


def test_reply_text_is_redacted_before_the_session_goes_on() -> None:
    def answer(body: dict[str, Any]) -> Any:
        if len(body["messages"]) == 1:
            return reply(
                "Mail ops@example.com",
                [{"id": "c1", "name": "find_slots", "arguments": {"date": "x@y.com"}}],
            )
        return reply("done")

    agent = FakeAgent(answer)
    t = target(agent)
    t.attach(store=MemoryStore(), redactor=Redactor(RedactionConfig(builtins=["email"])))
    generation = generate(t)
    sent = agent.bodies[1]["messages"][1]
    assert "ops@example.com" not in sent["content"]
    # Tool arguments are not redacted: the sandbox matches on them.
    assert sent["tool_calls"][0]["arguments"] == {"date": "x@y.com"}
    assert "ops@example.com" not in generation.steps[0].text


# Sessions with a simulated user


def conversation(**fields: Any) -> Case:
    base = {
        "goal": "book a table",
        "user_turns": ["Hello", "I need a table on 2026-11-20"],
        "success": ["confirmation BK-19"],
        "max_turns": 5,
    }
    return Case(id="d1", input={**base, **fields})


def dialogue_spec(**fields: Any) -> HttpAgentTargetSpec:
    return spec(**{"template": "", "simulator": {"type": "scripted"}, **fields})


def test_a_simulated_user_talks_to_the_agent_until_the_goal_is_reached() -> None:
    agent = FakeAgent()
    generation = generate(target(agent, dialogue_spec()), conversation())
    assert generation.stop_reason == "goal_reached"
    assert generation.text == "Booked, confirmation BK-19"
    roles = [(t.index, t.role, t.stop_reason) for t in generation.turns]
    assert roles == [
        (0, "user", "continue"),
        (1, "assistant", END_TURN),
        (2, "user", "continue"),
        (3, "assistant", "done"),
        (4, "user", "goal_reached"),
    ]
    assert generation.turns[1].content == "What can I do for you?"
    # One session across the dialogue, steps numbered across turns.
    assert len({b["session_id"] for b in agent.bodies}) == 1
    assert [b["step"] for b in agent.bodies] == [0, 1, 2, 3]
    last = agent.bodies[-1]["messages"]
    assert [m["role"] for m in last] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    # Turn and step costs both add up to the session's.
    assert sum(t.usage.cost_usd for t in generation.turns) == pytest.approx(
        generation.usage.cost_usd
    )
    assert sum(s.usage.cost_usd for s in generation.steps) == pytest.approx(0.004)


def test_a_dialogue_ends_on_give_up_turn_limit_no_reply_or_a_budget() -> None:
    def chatty(body: dict[str, Any]) -> Any:
        return reply("Tell me more")

    generation = generate(target(FakeAgent(chatty), dialogue_spec()), conversation())
    assert generation.stop_reason == "gave_up"
    generation = generate(target(FakeAgent(chatty), dialogue_spec()), conversation(max_turns=1))
    assert generation.stop_reason == "max_turns"
    generation = generate(target(FakeAgent(lambda b: reply("")), dialogue_spec()), conversation())
    assert generation.stop_reason == "no_reply"
    generation = generate(target(FakeAgent(), dialogue_spec(max_steps=2)), conversation())
    assert generation.stop_reason == MAX_STEPS
    assert generation.turns[-1].role == "assistant"
    assert generation.turns[-1].stop_reason == MAX_STEPS
    # A budget spent exactly at the end of a turn adds no empty agent turn.
    generation = generate(target(FakeAgent(), dialogue_spec(max_steps=1)), conversation())
    assert generation.stop_reason == MAX_STEPS
    assert [t.role for t in generation.turns] == ["user", "assistant", "user"]
    assert generation.text == "What can I do for you?"


def test_a_dialogue_can_open_with_messages_from_the_case() -> None:
    agent = FakeAgent()
    case = conversation(
        messages=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
        user_turns=["A table on 2026-11-20 please"],
    )
    generation = generate(target(agent, dialogue_spec()), case)
    assert generation.stop_reason == "goal_reached"
    assert agent.bodies[0]["messages"][:2] == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!", "tool_calls": []},
    ]
    assert generation.turns[0].index == 2


def test_a_dialogue_case_must_be_a_conversation() -> None:
    with pytest.raises(InvalidInputError, match="not a conversation"):
        generate(target(FakeAgent(), dialogue_spec()), CASE)


# Specs, identity and the factory


def test_specs_are_validated() -> None:
    assert isinstance(SPEC.validate_python(spec().model_dump()), HttpAgentTargetSpec)
    with pytest.raises(ValidationError, match="template for the task"):
        spec(template="")
    with pytest.raises(ValidationError, match="give no template"):
        spec(simulator={"type": "scripted"})
    with pytest.raises(ValidationError, match="not a valid"):
        spec(template="{{ date")
    with pytest.raises(ValidationError, match="names a file"):
        spec(sandbox={**SANDBOX, "fixtures": "answers.jsonl"})
    with pytest.raises(ValidationError):
        spec(endpoint="Not A Name")
    with pytest.raises(ValueError, match="simulator exactly"):
        HttpAgentTarget(dialogue_spec(), ENDPOINT)


def test_identity_covers_the_endpoint_the_sandbox_and_the_simulator() -> None:
    endpoints = {"booker": ENDPOINT}
    base = identity(spec(), endpoints)
    moved = {"booker": HttpEndpoint(url="https://elsewhere.example.com/step")}
    assert identity(spec(), moved) != base
    assert identity(spec(fixtures=[]), endpoints) != base
    assert identity(spec(max_tokens=10), endpoints) != base
    llm = dialogue_spec(simulator={"type": "llm", "provider": "local", "model": "m"})
    a = identity(
        llm, endpoints, {"local": ProviderConfig(kind="openai_compat", base_url="http://a/v1")}
    )
    b = identity(
        llm, endpoints, {"local": ProviderConfig(kind="openai_compat", base_url="http://b/v1")}
    )
    assert a != b
    with pytest.raises(InvalidInputError, match="unknown HTTP endpoint"):
        identity(spec(), {})


def test_the_factory_builds_http_agents() -> None:
    providers = registry()
    endpoints = {"booker": ENDPOINT}
    built = build_target(spec(), endpoints=endpoints, providers=providers)
    assert isinstance(built, HttpAgentTarget)
    assert built.provider == "http:booker"
    assert built.simulator is None
    built = build_target(dialogue_spec(), endpoints=endpoints, providers=providers)
    assert isinstance(built, HttpAgentTarget)
    assert isinstance(built.simulator, ScriptedSimulator)
    llm = dialogue_spec(simulator={"type": "llm", "provider": "local", "model": "gpt-test"})
    built = build_target(llm, endpoints=endpoints, providers=providers)
    assert isinstance(built, HttpAgentTarget)
    assert isinstance(built.simulator, LLMSimulator)
    assert built.simulator.provider == "local"


# Runs


class Factory:
    def __init__(self, answer: Callable[[dict[str, Any]], Any] = booking_plan) -> None:
        self.agent = FakeAgent(answer)
        self.targets: list[HttpAgentTarget] = []
        self.delay = 0.0

    def __call__(self, s: Any) -> HttpAgentTarget:
        agent = self.agent
        delay = self.delay

        async def handler(request: httpx2.Request) -> httpx2.Response:
            if delay:
                await asyncio.sleep(delay)
            return agent(request)

        simulator = ScriptedSimulator(s.simulator) if s.simulator is not None else None
        t = HttpAgentTarget(
            s, ENDPOINT, simulator, resolver=public, transport=httpx2.MockTransport(handler)
        )
        self.targets.append(t)
        return t


def run_config(**target: Any) -> RunConfig:
    options = target.pop("options", {})
    return RunConfig.model_validate(
        {
            "label": "http-agent",
            "dataset": {"name": target.pop("dataset", "trips")},
            "target": {
                "type": "http_agent",
                "endpoint": "booker",
                "template": "Book a table on {{ date }}",
                "sandbox": SANDBOX,
                "fixtures": FIXTURES,
                **target,
            },
            "scorers": [{"type": "regex", "pattern": "BK-19", "name": "booked"}],
            "options": {"concurrency": 4, **options},
        }
    )


SETTINGS = Settings.model_validate(
    {**Settings.from_env({}).model_dump(), "http_endpoints": {"booker": ENDPOINT}}
)


@pytest.fixture
def dataset(sessions: sessionmaker[Session]) -> str:
    with sessions() as s:
        import_dataset(
            s, "trips", [Case(id=f"t{i:03d}", input={"date": f"day {i}"}) for i in range(6)]
        )
        import_dataset(
            s,
            "chats",
            [
                Case(
                    id=f"d{i}",
                    input={
                        "goal": "book a table",
                        "user_turns": ["Hi", "A table on 2026-11-20"],
                        "success": ["BK-19"],
                    },
                )
                for i in range(3)
            ],
        )
    return "trips"


def results(sessions: sessionmaker[Session], run_id: str) -> dict[str, Result]:
    with sessions() as s:
        return {r.case_id: r for r in s.scalars(select(Result).where(Result.run_id == run_id))}


def rows(sessions: sessionmaker[Session], model: Any, run_id: str) -> int:
    with sessions() as s:
        return int(
            s.scalar(select(func.count()).select_from(model).where(model.run_id == run_id)) or 0
        )


def test_a_run_stores_each_session_and_reruns_for_free(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    factory = Factory()
    runner = Runner(sessions, SETTINGS, target_factory=factory)
    run = runner.create_run(run_config())
    report = asyncio.run(runner.execute(run.id))
    assert (report.status, report.errors) == ("completed", 0)
    found = results(sessions, run.id)
    assert {r.scores["booked"]["value"] for r in found.values()} == {1.0}
    assert {r.stop_reason for r in found.values()} == {"done"}
    assert sum(r.cost_usd for r in found.values()) == pytest.approx(6 * 0.003)
    assert rows(sessions, ResultStep, run.id) == 6 * 5
    assert len(factory.agent.seen) == 18
    rerun = runner.create_run(run_config())
    asyncio.run(runner.execute(rerun.id))
    assert len(factory.agent.seen) == 18
    assert all(r.cached and r.cost_usd == 0 for r in results(sessions, rerun.id).values())
    nocache = runner.create_run(run_config(options={"use_cache": False}))
    asyncio.run(runner.execute(nocache.id))
    assert len(factory.agent.seen) == 36


def test_a_run_with_a_simulated_user_stores_turns_and_steps(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    factory = Factory()
    runner = Runner(sessions, SETTINGS, target_factory=factory)
    run = runner.create_run(
        run_config(dataset="chats", template="", simulator={"type": "scripted"})
    )
    report = asyncio.run(runner.execute(run.id))
    assert (report.status, report.errors) == ("completed", 0)
    assert {r.stop_reason for r in results(sessions, run.id).values()} == {"goal_reached"}
    assert rows(sessions, ResultTurn, run.id) == 3 * 5
    assert rows(sessions, ResultStep, run.id) == 3 * 6


def test_a_run_refuses_cases_that_are_not_conversations(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    runner = Runner(sessions, SETTINGS, target_factory=Factory())
    with pytest.raises(InvalidInputError, match="not a conversation"):
        runner.create_run(run_config(template="", simulator={"type": "scripted"}))


def test_a_run_killed_mid_session_resumes_without_asking_twice(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    factory = Factory()
    factory.delay = 0.01
    runner = Runner(sessions, SETTINGS, target_factory=factory)
    run = runner.create_run(run_config(options={"concurrency": 1}))

    def cached() -> int:
        if not factory.targets:
            return 0
        owner = factory.targets[-1].agent_hash
        with sessions() as s:
            return int(
                s.scalar(
                    select(func.count())
                    .select_from(CachedResponse)
                    .where(CachedResponse.target_hash == owner)
                )
                or 0
            )

    async def interrupt() -> None:
        task = asyncio.create_task(runner.execute(run.id))
        while cached() < 2:
            await asyncio.sleep(0.002)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(interrupt())
    assert "t000" not in results(sessions, run.id)
    with sessions() as s:
        assert s.get_one(Run, run.id).status == "interrupted"
    report = asyncio.run(Runner(sessions, SETTINGS, target_factory=factory).execute(run.id))
    assert report.status == "completed"
    asked = [s.body["request_id"] for s in factory.agent.seen]
    assert len(asked) == len(set(asked)) == 18
    assert results(sessions, run.id)["t000"].cost_usd == pytest.approx(0.001)


def test_a_run_through_the_real_factory_resolves_the_endpoint(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    runner = Runner(sessions, SETTINGS)
    run = runner.create_run(run_config())
    assert run.target_hash == identity(run_config().target, {"booker": ENDPOINT})
    unknown = Settings.from_env({})
    with pytest.raises(InvalidInputError, match="unknown HTTP endpoint"):
        Runner(sessions, unknown).create_run(run_config())


def test_an_llm_user_talks_to_the_agent_through_the_registry(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    with sessions() as s:
        import_dataset(s, "live", [Case(id=f"l{i}", input={"goal": "Book"}) for i in range(2)])
    fakes = Registry()
    providers = fakes.registry({"local": LOCAL})
    agent = FakeAgent()

    def build(s: Any) -> HttpAgentTarget:
        return HttpAgentTarget(
            s,
            ENDPOINT,
            simulator_for(s.simulator, providers),
            simulator_endpoint=LOCAL.endpoint,
            resolver=public,
            transport=httpx2.MockTransport(agent),
        )

    runner = Runner(sessions, SETTINGS, providers=providers, target_factory=build)
    simulator = {"type": "llm", "provider": "local", "model": "gpt-test"}
    cfg = run_config(dataset="live", template="", simulator=simulator)
    run = runner.create_run(cfg)
    with sessions() as s:
        # Only the simulated user is priced: the agent reports its own cost.
        assert set(s.get_one(Run, run.id).prices or {}) == {"local"}
    report = asyncio.run(runner.execute(run.id))
    assert (report.status, report.errors) == ("completed", 0)
    found = results(sessions, run.id)
    assert {r.stop_reason for r in found.values()} == {"goal_reached"}
    assert {r.output for r in found.values()} == {"Booked, confirmation BK-19"}
    with sessions() as s:
        turn_total = s.scalar(
            select(func.sum(ResultTurn.cost_usd)).where(ResultTurn.run_id == run.id)
        )
    assert turn_total == pytest.approx(sum(r.cost_usd for r in found.values()))
    unpriced = run_config(dataset="live", template="", simulator={**simulator, "model": "nope"})
    with pytest.raises(InvalidInputError):
        runner.create_run(unpriced)
