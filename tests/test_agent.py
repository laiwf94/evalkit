"""Agents with tools: the provider-neutral transcript, one tool loop on Claude,
OpenAI-compatible servers and Gemini, the agent target with its step cache,
``result_steps`` rows and the trajectory endpoint, and resuming a run killed
mid-trajectory without paying for a step twice."""

import asyncio
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx2
import pytest
from pydantic import SecretStr, TypeAdapter, ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from evalkit.cli import CliError, inline_sandbox
from evalkit.config import Settings
from evalkit.datasets import Case, import_dataset
from evalkit.db.models import CachedResponse, Result, ResultStep, Run
from evalkit.errors import InvalidInputError, PreconditionError
from evalkit.pricing import ModelPrice, PriceBook, PriceTable
from evalkit.providers.anthropic import ClaudeClient
from evalkit.providers.base import ErrorKind, ProviderError
from evalkit.providers.gemini import GeminiClient
from evalkit.providers.openai_compat import OpenAICompatClient
from evalkit.providers.registry import ProviderConfig, ProviderRegistry
from evalkit.providers.tools import (
    AgentMessage,
    ToolCompletion,
    ToolDefinition,
    ToolOutcome,
    ToolUse,
    check_transcript,
    parse_arguments,
)
from evalkit.redaction import RedactionConfig
from evalkit.runner import Runner
from evalkit.runspec import RunConfig
from evalkit.targets import AgentTargetSpec, Step, TargetSpec, Usage
from evalkit.targets.agent import AgentTarget
from evalkit.targets.factory import identity
from evalkit.tools.loop import MAX_STEPS, ToolLoop, owner_hash, step_key
from evalkit.tools.sandbox import SandboxSpec, ToolSandbox
from tests.fake_anthropic import FakeAnthropic, json_reply, message
from tests.fake_gemini import FakeGemini, function_calls, generated
from tests.fake_openai import FakeOpenAI, tool_call_response
from tests.test_api import AUTH, serving

SPEC: TypeAdapter[Any] = TypeAdapter(TargetSpec)
SETTINGS = Settings.from_env({})
CLAUDE = "claude-opus-5-5"
PRICES = {
    "gpt-test": ModelPrice(input=2.0, output=8.0),
    "gemini-test": ModelPrice(input=1.0, output=4.0),
    CLAUDE: ModelPrice(input=5.0, output=25.0),
}
TABLE = PriceTable(PRICES)
LOCAL = ProviderConfig(kind="openai_compat", base_url="http://llm.test/v1")

SANDBOX: dict[str, Any] = {
    "seed": 7,
    "tools": [
        {
            "name": "find_slots",
            "description": "Free tables on a date",
            "input_schema": {
                "type": "object",
                "properties": {"date": {"type": "string"}},
                "required": ["date"],
                "additionalProperties": False,
            },
        },
        {
            "name": "book",
            "description": "Book a slot",
            "input_schema": {
                "type": "object",
                "properties": {"slot": {"type": "string"}},
                "required": ["slot"],
            },
            "responses": [{"when": {"slot": "19:00"}, "result": {"confirmation": "BK-19"}}],
        },
    ],
}
FIXTURES = [
    {"tool": "find_slots", "arguments": {"date": "2026-11-20"}, "result": ["19:00", "20:30"]},
]


def agent_spec(**fields: Any) -> AgentTargetSpec:
    raw = {
        "type": "agent",
        "model": CLAUDE,
        "template": "Book a table on {{ date }}",
        "sandbox": SANDBOX,
        "fixtures": FIXTURES,
        **fields,
    }
    spec = SPEC.validate_python(raw)
    assert isinstance(spec, AgentTargetSpec)
    return spec


def plan(results: list[Any]) -> tuple[str, Any]:
    """The reference agent's policy, from the tool results it has seen so far:
    look up slots, book the first one, then confirm."""
    if not results:
        return "call", ("find_slots", {"date": "2026-11-20"})
    if len(results) == 1:
        slots = results[0].get("result") or ["none"]
        return "call", ("book", {"slot": slots[0]})
    confirmation = (results[1].get("result") or {}).get("confirmation", "none")
    return "say", f"Booked, confirmation {confirmation}"


# Each provider's fake, answering with the policy above.


def claude_results(body: dict[str, Any]) -> list[Any]:
    found = []
    for m in body["messages"]:
        if isinstance(m["content"], list):
            for block in m["content"]:
                if block.get("type") == "tool_result":
                    found.append(json.loads(block["content"]))
    return found


def claude_agent(body: dict[str, Any]) -> httpx2.Response:
    action, what = plan(claude_results(body))
    if action == "say":
        return json_reply(message(what))
    name, args = what
    n = len(claude_results(body))
    blocks = [{"type": "tool_use", "id": f"toolu_{n}", "name": name, "input": args}]
    reply = message("Let me check.", stop_reason="tool_use", extra_blocks=blocks)
    # Text first, then the tool call, as Claude sends them.
    reply["content"] = [reply["content"][-1], *blocks]
    return json_reply(reply)


def openai_results(body: dict[str, Any]) -> list[Any]:
    return [json.loads(m["content"]) for m in body["messages"] if m["role"] == "tool"]


def openai_agent(body: dict[str, Any]) -> str | httpx2.Response:
    results = openai_results(body)
    action, what = plan(results)
    if action == "say":
        return str(what)
    name, args = what
    return tool_call_response([(f"call_{len(results)}", name, json.dumps(args))])


def gemini_results(body: dict[str, Any]) -> list[Any]:
    return [
        p["functionResponse"]["response"]
        for c in body["contents"]
        for p in c["parts"]
        if "functionResponse" in p
    ]


def gemini_agent(model: str, body: dict[str, Any]) -> str | httpx2.Response:
    action, what = plan(gemini_results(body))
    if action == "say":
        return str(what)
    return function_calls([what])


def claude_client(fake: FakeAnthropic) -> ClaudeClient:
    return fake.client(TABLE)


def make_provider(kind: str) -> tuple[Any, Any, str]:
    if kind == "claude":
        claude = FakeAnthropic()
        claude.default = claude_agent
        return claude, claude_client(claude), CLAUDE
    if kind == "openai":
        server = FakeOpenAI("http://llm.test/v1", api_key=None, answer=openai_agent)
        local = OpenAICompatClient(
            TABLE, None, name="local", base_url="http://llm.test/v1", transport=server.transport()
        )
        return server, local, "gpt-test"
    gemini = FakeGemini(answer=gemini_agent)
    client = GeminiClient(TABLE, SecretStr("g-key"), transport=gemini.transport())
    return gemini, client, "gemini-test"


def generate(target: AgentTarget, case: Case) -> Any:
    return asyncio.run(target.generate(case))


# The transcript


def test_transcripts_are_checked() -> None:
    use = ToolUse("a", "book", {"slot": "1"})
    asked = AgentMessage("assistant", "", (use,))
    answer = AgentMessage("tool", outcomes=(ToolOutcome("a", "book", {"ok": True}),))
    user = AgentMessage("user", "hi")
    check_transcript([user, asked, answer])
    check_transcript([user, AgentMessage("assistant", "hello"), user])
    bad: list[tuple[list[AgentMessage], str]] = [
        ([], "at least"),
        ([asked, answer], "starts with the user"),
        ([user, user], "must speak"),
        ([user, asked, asked], "two assistant"),
        ([user, asked, user], "answered next"),
        ([user, AgentMessage("assistant", "x"), answer], "without tool uses"),
        ([user, asked, AgentMessage("tool", outcomes=())], "do not answer"),
        ([AgentMessage("user", "x", outcomes=answer.outcomes)], "only a tool message"),
        ([AgentMessage("user", "x", (use,))], "only the assistant"),
        ([user, AgentMessage("assistant", "x")], "must not end"),
    ]
    for messages, match in bad:
        with pytest.raises(InvalidInputError, match=match):
            check_transcript(messages)


def test_tool_values_round_trip() -> None:
    use = ToolUse("a", "book", {"slot": "1"}, signature="s")
    assert ToolUse.from_json(use.to_json()) == use
    assert ToolUse.from_json({"id": "b", "name": "n", "arguments": 3}).signature is None
    broken_uses: list[Any] = [[], {"id": 1, "name": "n"}, {"id": "a", "name": "n", "signature": 2}]
    for broken in broken_uses:
        with pytest.raises(TypeError):
            ToolUse.from_json(broken)
    assert ToolOutcome("a", "n", "down", is_error=True).payload() == {"error": "down"}
    assert json.loads(ToolOutcome("a", "n", [1]).payload_text()) == {"result": [1]}
    assert parse_arguments('{"a": 1}') == {"a": 1}
    assert parse_arguments("{broken") == "{broken"
    assert parse_arguments({"a": 1}) == {"a": 1}
    definition = ToolDefinition.of({"name": "x", "input_schema": {"type": "object"}})
    assert definition.description == ""


# One loop on every provider


def check_claude_wire(fake: Any, steps: Sequence[Step]) -> None:
    sent = fake.requests
    assert len(sent) == 3
    assert sent[0]["tools"][0] == {
        "name": "find_slots",
        "description": "Free tables on a date",
        "input_schema": SANDBOX["tools"][0]["input_schema"],
    }
    assert sent[0]["system"][0]["text"] == "You book tables."
    assistant, results = sent[1]["messages"][1:]
    assert assistant["content"] == [
        {"type": "text", "text": "Let me check."},
        {
            "type": "tool_use",
            "id": "toolu_0",
            "name": "find_slots",
            "input": {"date": "2026-11-20"},
        },
    ]
    assert results["role"] == "user"
    assert results["content"][0]["tool_use_id"] == "toolu_0"
    assert results["content"][0]["is_error"] is False
    assert steps[0].stop_reason == "tool_use"


def check_openai_wire(fake: Any, steps: Sequence[Step]) -> None:
    assert fake.rejected == []
    sent = fake.bodies
    assert len(sent) == 3
    assert sent[0]["messages"][0] == {"role": "system", "content": "You book tables."}
    assert sent[0]["tools"][1]["function"]["name"] == "book"
    call = sent[1]["messages"][2]
    assert call["tool_calls"][0]["function"] == {
        "name": "find_slots",
        "arguments": json.dumps({"date": "2026-11-20"}),
    }
    assert call["content"] is None
    assert sent[1]["messages"][3]["role"] == "tool"
    assert steps[0].stop_reason == "tool_calls"


def check_gemini_wire(fake: Any, steps: Sequence[Step]) -> None:
    assert fake.rejected == []
    sent = fake.bodies
    assert len(sent) == 3
    declarations = sent[0]["tools"][0]["functionDeclarations"]
    assert declarations[0]["parametersJsonSchema"] == SANDBOX["tools"][0]["input_schema"]
    model_turn = sent[1]["contents"][1]
    # The thought signature goes back on the call it came with.
    assert model_turn["parts"][0]["thoughtSignature"] == "sig-1"
    assert model_turn["parts"][0]["functionCall"]["id"] == "call_1_0"
    response = sent[1]["contents"][2]["parts"][0]["functionResponse"]
    assert response["name"] == "find_slots"
    assert response["response"] == {"result": ["19:00", "20:30"]}


WIRE_CHECKS: dict[str, Callable[[Any, Sequence[Step]], None]] = {
    "claude": check_claude_wire,
    "openai": check_openai_wire,
    "gemini": check_gemini_wire,
}


@pytest.mark.parametrize("kind", ["claude", "openai", "gemini"])
def test_the_same_loop_runs_on_every_provider(kind: str) -> None:
    fake, client, model = make_provider(kind)
    provider = "anthropic" if kind == "claude" else kind
    spec = agent_spec(model=model, provider=provider, system="You book tables.")
    target = AgentTarget(spec, client)
    generation = generate(target, Case(id="c1", input={"date": "2026-11-20"}))
    assert generation.text == "Booked, confirmation BK-19"
    steps = generation.steps
    assert [s.kind for s in steps] == ["model", "tool", "model", "tool", "model"]
    assert [s.index for s in steps] == [0, 1, 2, 3, 4]
    lookup, booking = steps[1], steps[3]
    assert (lookup.tool, lookup.arguments, lookup.status) == (
        "find_slots",
        {"date": "2026-11-20"},
        "ok",
    )
    assert (lookup.result, lookup.source) == (["19:00", "20:30"], "fixture")
    assert (booking.tool, booking.result, booking.source) == (
        "book",
        {"confirmation": "BK-19"},
        "script",
    )
    assert lookup.call_id
    assert booking.call_id
    assert lookup.call_id != booking.call_id
    assert all(s.usage == Usage() for s in steps if s.kind == "tool")
    model_steps = [s for s in steps if s.kind == "model"]
    assert all(s.usage.cost_usd > 0 for s in model_steps)
    assert generation.usage == sum((s.usage for s in steps), Usage())
    WIRE_CHECKS[kind](fake, steps)


def test_claude_thinking_blocks_are_sent_back_first() -> None:
    thinking = {"type": "thinking", "thinking": "Look it up.", "signature": "sig"}
    redacted = {"type": "redacted_thinking", "data": "opaque"}
    call = {"type": "tool_use", "id": "t1", "name": "find_slots", "input": {"date": "x"}}
    first = message("", stop_reason="tool_use")
    first["content"] = [thinking, redacted, call]
    claude = FakeAnthropic(json_reply(first), json_reply(message("done")))
    target = AgentTarget(agent_spec(), claude_client(claude))
    generation = generate(target, Case(id="c", input={"date": "x"}))
    assert generation.text == "done"
    sent = claude.requests[1]["messages"][1]["content"]
    # No empty text block: the API refuses those.
    assert sent == [thinking, redacted, call]
    assert generation.steps[1].status == "no_response"
    result = claude.requests[1]["messages"][2]["content"][0]
    assert result["is_error"] is True
    assert json.loads(result["content"]) == {
        "error": "find_slots has no response for these arguments"
    }


def test_a_truncated_or_refused_reply_runs_no_tool() -> None:
    call = {"type": "tool_use", "id": "t1", "name": "book", "input": {"slot": "19:00"}}
    for stop in ("max_tokens", "refusal"):
        reply = message("partial", stop_reason=stop, extra_blocks=[call])
        claude = FakeAnthropic(json_reply(reply))
        target = AgentTarget(agent_spec(), claude_client(claude))
        generation = generate(target, Case(id="c", input={"date": "x"}))
        assert [s.kind for s in generation.steps] == ["model"]
        assert generation.stop_reason == stop
        assert len(claude.requests) == 1


def test_arguments_that_are_not_json_are_answered_as_invalid() -> None:
    def answer(body: dict[str, Any]) -> str | httpx2.Response:
        tools = [m for m in body["messages"] if m["role"] == "tool"]
        if tools:
            return "sorry: " + str(json.loads(tools[0]["content"])["error"])
        return tool_call_response([("c1", "book", "{slot: 19}"), ("c2", "cancel", "{}")])

    server = FakeOpenAI("http://llm.test/v1", api_key=None, answer=answer)
    client = OpenAICompatClient(
        TABLE, None, name="local", base_url="http://llm.test/v1", transport=server.transport()
    )
    target = AgentTarget(agent_spec(model="gpt-test", provider="local"), client)
    generation = generate(target, Case(id="c", input={"date": "x"}))
    assert server.rejected == []
    bad, unknown = generation.steps[1], generation.steps[2]
    assert (bad.status, bad.arguments) == ("invalid_arguments", "{slot: 19}")
    assert unknown.status == "unknown_tool"
    assert generation.text.startswith("sorry: invalid arguments for book")
    # The raw text goes back as it was sent.
    resent = server.bodies[1]["messages"][1]["tool_calls"][0]["function"]["arguments"]
    assert resent == "{slot: 19}"


def test_openai_calls_without_ids_or_of_other_types() -> None:
    payload: dict[str, Any] = {
        "model": "gpt-test",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"type": "function", "function": {"name": "book", "arguments": None}},
                        {"id": "x", "type": "retrieval", "retrieval": {}},
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    def answer(body: dict[str, Any]) -> str | httpx2.Response:
        if body["messages"][-1]["role"] == "tool":
            return "ok"
        return httpx2.Response(200, json=payload)

    server = FakeOpenAI("http://llm.test/v1", api_key=None, answer=answer)
    client = OpenAICompatClient(
        TABLE, None, name="local", base_url="http://llm.test/v1", transport=server.transport()
    )
    target = AgentTarget(agent_spec(model="gpt-test", provider="local"), client)
    generation = generate(target, Case(id="c", input={"date": "x"}))
    calls = [s for s in generation.steps if s.kind == "tool"]
    assert [(c.call_id, c.arguments, c.status) for c in calls] == [
        ("call_0", {}, "invalid_arguments")
    ]
    payload["choices"][0]["message"]["tool_calls"] = [
        {"id": "a", "type": "function", "function": {"name": "", "arguments": "{}"}}
    ]
    with pytest.raises(ProviderError) as caught:
        generate(
            AgentTarget(agent_spec(model="gpt-test", provider="local"), client),
            Case(id="d", input={"date": "x"}),
        )
    assert caught.value.kind is ErrorKind.MALFORMED_RESPONSE


def test_gemini_ids_and_malformed_calls() -> None:
    def answer(model: str, body: dict[str, Any]) -> str | httpx2.Response:
        if gemini_results(body):
            return "done"
        return function_calls([("book", {"slot": "19:00"})], ids=True, signature=None)

    gemini = FakeGemini(answer=answer)
    client = GeminiClient(TABLE, SecretStr("g-key"), transport=gemini.transport())
    target = AgentTarget(agent_spec(model="gemini-test", provider="gemini"), client)
    generation = generate(target, Case(id="c", input={"date": "x"}))
    assert gemini.rejected == []
    assert generation.steps[1].call_id == "g0"
    part = gemini.bodies[1]["contents"][1]["parts"][0]
    assert "thoughtSignature" not in part
    assert gemini.bodies[1]["contents"][2]["parts"][0]["functionResponse"]["id"] == "g0"

    def nameless(model: str, body: dict[str, Any]) -> httpx2.Response:
        response = function_calls([("book", {})])
        data = json.loads(response.content)
        data["candidates"][0]["content"]["parts"][0]["functionCall"]["name"] = ""
        return httpx2.Response(200, json=data)

    broken = FakeGemini(answer=nameless)
    client = GeminiClient(TABLE, SecretStr("g-key"), transport=broken.transport())
    with pytest.raises(ProviderError) as caught:
        generate(
            AgentTarget(agent_spec(model="gemini-test", provider="gemini"), client),
            Case(id="c", input={"date": "x"}),
        )
    assert caught.value.kind is ErrorKind.MALFORMED_RESPONSE


def test_a_safety_block_or_bad_transcript_never_reaches_the_loop_as_text() -> None:
    gemini = FakeGemini(answer=lambda m, b: generated("", finish="SAFETY"))
    client = GeminiClient(TABLE, SecretStr("g-key"), transport=gemini.transport())
    with pytest.raises(ProviderError) as caught:
        asyncio.run(
            client.complete_tools(
                model="gemini-test", messages=[AgentMessage("user", "hi")], tools=[]
            )
        )
    assert caught.value.kind is ErrorKind.SAFETY_BLOCK
    assert "tools" not in gemini.bodies[0]
    openai = OpenAICompatClient(TABLE, None, transport=FakeOpenAI().transport())
    clients: list[tuple[Any, str]] = [
        (claude_client(FakeAnthropic()), CLAUDE),
        (openai, "gpt-test"),
        (client, "gemini-test"),
    ]
    for c, model in clients:
        with pytest.raises(InvalidInputError, match="at least"):
            asyncio.run(c.complete_tools(model=model, messages=[], tools=[]))


def test_max_steps_ends_an_agent_that_never_stops() -> None:
    def forever(body: dict[str, Any]) -> httpx2.Response:
        n = len(claude_results(body))
        blocks = [
            {
                "type": "tool_use",
                "id": f"t{n}",
                "name": "find_slots",
                "input": {"date": "2026-11-20"},
            }
        ]
        reply = message("again", stop_reason="tool_use", extra_blocks=blocks)
        return json_reply(reply)

    claude = FakeAnthropic()
    claude.default = forever
    target = AgentTarget(agent_spec(max_steps=3), claude_client(claude))
    generation = generate(target, Case(id="c", input={"date": "x"}))
    assert generation.stop_reason == MAX_STEPS
    assert [s.kind for s in generation.steps].count("model") == 3
    assert len(claude.requests) == 3
    # Recorded answers replay in order and the last one repeats.
    assert {s.source for s in generation.steps if s.kind == "tool"} == {"fixture"}


# The spec and its identity


def test_agent_specs_are_validated() -> None:
    spec = agent_spec()
    assert len(spec.recorded()) == 1
    with pytest.raises(ValidationError, match="names a file"):
        agent_spec(sandbox={**SANDBOX, "fixtures": "answers.jsonl"})
    with pytest.raises(ValidationError, match="fixture line 1"):
        agent_spec(fixtures=[{"tool": "find_slots", "arguments": {"date": 3}, "result": []}])
    with pytest.raises(ValidationError, match="placeholder"):
        agent_spec(template="Book {{")
    with pytest.raises(ValidationError):
        agent_spec(max_steps=0)
    assert set(spec.model_part()) == {
        "provider",
        "model",
        "system",
        "max_tokens",
        "effort",
        "fallbacks",
    }


def test_agent_identity_covers_the_sandbox_and_where_the_model_is() -> None:
    base = identity(agent_spec(), {})
    other_answer = {**SANDBOX, "seed": 8}
    assert identity(agent_spec(sandbox=other_answer), {}) != base
    assert identity(agent_spec(fixtures=[]), {}) != base
    local = agent_spec(provider="local", model="gpt-test")
    moved = ProviderConfig(kind="openai_compat", base_url="http://other.test/v1")
    assert identity(local, {}, {"local": LOCAL}) != identity(local, {}, {"local": moved})
    with pytest.raises(InvalidInputError, match="unknown provider"):
        identity(local, {}, {})


def test_step_keys_ignore_how_the_sandbox_answers_but_not_what_the_model_saw() -> None:
    tools = [ToolDefinition.of(d) for d in SandboxSpec.model_validate(SANDBOX).definitions()]
    owner = owner_hash({"m": 1}, tools)
    assert owner != owner_hash({"m": 2}, tools)
    assert owner != owner_hash({"m": 1}, tools[:1])
    user = [AgentMessage("user", "hi")]
    assert step_key(owner, user, "f") != step_key(owner, [AgentMessage("user", "ho")], "f")
    assert step_key(owner, user, "f") != step_key(owner, user, "g")


def test_a_provider_without_tool_use_is_refused() -> None:
    class Plain:
        name = "plain"
        prices = TABLE

    with pytest.raises(InvalidInputError, match="cannot call tools"):
        AgentTarget(agent_spec(), Plain())  # type: ignore[arg-type]


def test_inline_sandbox_reads_the_file_and_its_fixtures(tmp_path: Path) -> None:
    lines = "".join(json.dumps(f) + "\n" for f in FIXTURES)
    (tmp_path / "answers.jsonl").write_text(lines)
    (tmp_path / "booking.toml").write_text(
        """
[sandbox]
seed = 7
fixtures = "answers.jsonl"

[[sandbox.tools]]
name = "find_slots"
input_schema = { type = "object", properties = { date = { type = "string" } } }
"""
    )
    target: dict[str, Any] = {"type": "agent", "template": "{{ date }}", "sandbox": "booking.toml"}
    inline_sandbox(target, tmp_path)
    assert "fixtures" not in target["sandbox"]
    assert target["fixtures"] == [{**FIXTURES[0], "latency_ms": 0.0}]
    spec = SPEC.validate_python(target)
    assert isinstance(spec, AgentTargetSpec)
    assert len(spec.recorded()) == 1
    with pytest.raises(CliError, match="fixtures in the sandbox file"):
        inline_sandbox({"sandbox": "booking.toml", "fixtures": []}, tmp_path)
    with pytest.raises(InvalidInputError, match="cannot read"):
        inline_sandbox({"sandbox": "missing.toml"}, tmp_path)


# Runs: steps stored, cached, retried and resumed


class Agent:
    """A tool-using model in process, following ``plan``. It records every reply
    it was paid for, so a test can see one asked for twice."""

    name = "anthropic"

    def __init__(self, delay: float = 0.0) -> None:
        self.prices = SETTINGS.price_book().table("anthropic")
        self.delay = delay
        self.paid: list[str] = []
        self.fail_once: set[int] = set()

    async def complete_tools(
        self,
        *,
        model: str,
        messages: Sequence[AgentMessage],
        tools: Sequence[ToolDefinition],
        system: str = "",
        max_tokens: int = 4096,
        effort: str = "medium",
        fallbacks: bool = True,
    ) -> ToolCompletion:
        check_transcript(messages)
        if self.delay:
            await asyncio.sleep(self.delay)
        number = len(self.paid)
        if number in self.fail_once:
            self.fail_once.discard(number)
            raise ProviderError(
                "overloaded", kind=ErrorKind.OVERLOADED, provider="anthropic", retry_after=0
            )
        self.paid.append(json.dumps([m.to_json() for m in messages], sort_keys=True))
        results = [o.payload() for m in messages for o in m.outcomes]
        action, what = plan(results)
        usage = Usage(100, 20, 0, 0, 0.001)
        if action == "say":
            text = f"{what} for {messages[0].text}"
            return ToolCompletion(text, (), "end_turn", usage, model)
        name, args = what
        use = ToolUse(f"t{len(results)}", name, args)
        return ToolCompletion("", (use,), "tool_use", usage, model)


class Factory:
    def __init__(self, delay: float = 0.0) -> None:
        self.agent = Agent(delay)
        self.targets: list[AgentTarget] = []

    def __call__(self, spec: Any) -> AgentTarget:
        target = AgentTarget(spec, self.agent)  # type: ignore[arg-type]
        self.targets.append(target)
        return target


def config(dataset: str = "trips", **target: Any) -> RunConfig:
    options = target.pop("options", {})
    return RunConfig.model_validate(
        {
            "label": "agent",
            "dataset": {"name": dataset},
            "target": {
                "type": "agent",
                "model": CLAUDE,
                "template": "Book a table on {{ date }}",
                "sandbox": SANDBOX,
                "fixtures": FIXTURES,
                **target,
            },
            "scorers": [{"type": "regex", "pattern": "BK-19", "name": "booked"}],
            "options": {"concurrency": 4, **options},
        }
    )


@pytest.fixture
def dataset(sessions: sessionmaker[Session]) -> str:
    with sessions() as s:
        import_dataset(
            s, "trips", [Case(id=f"t{i:03d}", input={"date": f"day {i}"}) for i in range(6)]
        )
    return "trips"


def results(sessions: sessionmaker[Session], run_id: str) -> dict[str, Result]:
    with sessions() as s:
        return {r.case_id: r for r in s.scalars(select(Result).where(Result.run_id == run_id))}


def steps(sessions: sessionmaker[Session], run_id: str) -> dict[str, list[ResultStep]]:
    found: dict[str, list[ResultStep]] = {}
    with sessions() as s:
        rows = s.scalars(
            select(ResultStep)
            .where(ResultStep.run_id == run_id)
            .order_by(ResultStep.case_id, ResultStep.step)
        )
        for row in rows:
            found.setdefault(row.case_id, []).append(row)
    return found


def test_a_run_stores_each_trajectory_and_reruns_for_free(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    factory = Factory()
    runner = Runner(sessions, SETTINGS, target_factory=factory)
    run = runner.create_run(config())
    report = asyncio.run(runner.execute(run.id))
    assert (report.status, report.errors) == ("completed", 0)
    found = results(sessions, run.id)
    assert len(found) == 6
    stored = steps(sessions, run.id)
    for case_id, result in found.items():
        assert result.scores["booked"]["value"] == 1.0
        day = int(case_id[1:])
        assert result.output == f"Booked, confirmation BK-19 for Book a table on day {day}"
        rows = stored[case_id]
        assert [r.kind for r in rows] == ["model", "tool", "model", "tool", "model"]
        assert rows[1].arguments == {"date": "2026-11-20"}
        assert rows[1].result == ["19:00", "20:30"]
        assert rows[3].result == {"confirmation": "BK-19"}
        assert (rows[3].status, rows[3].source) == ("ok", "script")
        assert sum(r.cost_usd for r in rows) == pytest.approx(result.cost_usd)
        assert result.cost_usd == pytest.approx(0.003)
        assert not result.cached
    assert len(factory.agent.paid) == 18

    again = runner.create_run(config())
    asyncio.run(runner.execute(again.id))
    assert len(factory.agent.paid) == 18
    replayed = results(sessions, again.id)
    assert all(r.cached and r.cost_usd == 0 for r in replayed.values())
    assert all(
        r.cached for rows in steps(sessions, again.id).values() for r in rows if r.kind == "model"
    )
    assert [r.output for r in replayed.values()] == [r.output for r in found.values()]

    fresh = runner.create_run(config(options={"use_cache": False}))
    asyncio.run(runner.execute(fresh.id))
    assert len(factory.agent.paid) == 36


def test_a_changed_tool_answer_keeps_the_steps_before_it(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    factory = Factory()
    runner = Runner(sessions, SETTINGS, target_factory=factory)
    asyncio.run(runner.execute(runner.create_run(config()).id))
    paid = len(factory.agent.paid)
    tools = [
        dict(SANDBOX["tools"][0]),
        {**SANDBOX["tools"][1], "responses": [{"result": {"confirmation": "BK-20"}}]},
    ]
    changed = runner.create_run(config(sandbox={**SANDBOX, "tools": tools}))
    asyncio.run(runner.execute(changed.id))
    # The first two replies saw the same transcript; only the last one is new.
    assert len(factory.agent.paid) == paid + 6
    rows = steps(sessions, changed.id)["t000"]
    assert [r.cached for r in rows if r.kind == "model"] == [True, True, False]
    assert all(r.scores["booked"]["value"] == 0.0 for r in results(sessions, changed.id).values())


def test_a_retried_case_pays_only_for_the_steps_it_lacks(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    factory = Factory()
    factory.agent.fail_once = {1}
    runner = Runner(sessions, SETTINGS, target_factory=factory)
    run = runner.create_run(config(options={"concurrency": 1, "retry": {"base_delay": 0.0}}))
    asyncio.run(runner.execute(run.id))
    found = results(sessions, run.id)
    first = found["t000"]
    assert first.attempts == 2
    assert first.status == "ok"
    assert len(factory.agent.paid) == len(set(factory.agent.paid)) == 18
    rows = [r for r in steps(sessions, run.id)["t000"] if r.kind == "model"]
    assert [r.cached for r in rows] == [True, False, False]


def test_a_run_killed_mid_trajectory_resumes_without_repaying(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    factory = Factory(delay=0.01)
    runner = Runner(sessions, SETTINGS, target_factory=factory)
    run = runner.create_run(config(options={"concurrency": 1}))

    def cached_steps() -> int:
        if not factory.targets:
            return 0
        owner = factory.targets[-1].loop.owner_hash
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
        # The first case needs three replies: stop it after two.
        while cached_steps() < 2:
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
    paid = factory.agent.paid
    assert len(paid) == len(set(paid)) == 18
    resumed = [r for r in steps(sessions, run.id)["t000"] if r.kind == "model"]
    assert [r.cached for r in resumed] == [True, True, False]
    assert results(sessions, run.id)["t000"].cost_usd == pytest.approx(0.001)


def test_reply_text_is_redacted_before_the_loop_goes_on(sessions: sessionmaker[Session]) -> None:
    class Leaky(Agent):
        async def complete_tools(self, **kw: Any) -> ToolCompletion:
            reply = await super().complete_tools(**kw)
            return ToolCompletion(
                reply.text + " mail me at jo@example.com",
                reply.tool_uses,
                reply.stop_reason,
                reply.usage,
                reply.model,
            )

    settings = Settings(redaction=RedactionConfig())
    with sessions() as s:
        import_dataset(s, "one", [Case(id="x", input={"date": "d"})])
    leaky = Leaky()
    runner = Runner(sessions, settings, target_factory=lambda spec: AgentTarget(spec, leaky))  # type: ignore[arg-type]
    run = runner.create_run(config("one"))
    asyncio.run(runner.execute(run.id))
    rows = steps(sessions, run.id)["x"]
    assert all("jo@example.com" not in r.text for r in rows)
    assert all("jo@example.com" not in p for p in leaky.paid)
    assert "jo@example.com" not in (results(sessions, run.id)["x"].output or "")


class Registry:
    """The real factory path: Claude (the fake API) and an OpenAI-compatible server."""

    def __init__(self) -> None:
        self.claude = FakeAnthropic()
        self.claude.default = claude_agent
        self.openai = FakeOpenAI("http://llm.test/v1", api_key=None, answer=openai_agent)

    def registry(self, configs: dict[str, ProviderConfig]) -> ProviderRegistry:
        return ProviderRegistry(
            configs,
            PriceBook(PRICES),
            claude=claude_client(self.claude),
            transport=httpx2.MockTransport(self.openai.handle),
        )


@pytest.mark.parametrize("provider", ["anthropic", "local"])
def test_agent_runs_through_the_registry(sessions: sessionmaker[Session], provider: str) -> None:
    with sessions() as s:
        import_dataset(
            s, "live", [Case(id=f"l{i}", input={"date": "2026-11-20"}) for i in range(3)]
        )
    fakes = Registry()
    runner = Runner(sessions, Settings(), providers=fakes.registry({"local": LOCAL}))
    model = CLAUDE if provider == "anthropic" else "gpt-test"
    cfg = config("live", provider=provider, model=model, options={"concurrency": 1})
    run = runner.create_run(cfg)
    with sessions() as s:
        assert set(s.get_one(Run, run.id).prices or {}) == {provider}
    report = asyncio.run(runner.execute(run.id))
    assert report.errors == 0
    assert fakes.openai.rejected == []
    found = results(sessions, run.id)
    assert {r.output for r in found.values()} == {"Booked, confirmation BK-19"}
    # Identical cases: each step was paid for once and replayed for the rest.
    paid = fakes.claude.requests if provider == "anthropic" else fakes.openai.bodies
    assert len(paid) == 3
    if provider == "local":
        moved = {"local": ProviderConfig(kind="openai_compat", base_url="http://other.test/v1")}
        stale = Runner(sessions, Settings(), providers=fakes.registry(moved))
        with pytest.raises(PreconditionError, match="configuration changed"):
            asyncio.run(stale.execute(runner.create_run(cfg).id))
    unpriced = config("live", provider=provider, model="nope")
    with pytest.raises(InvalidInputError):
        runner.create_run(unpriced)


def test_long_tool_names_are_clipped_for_storage(sessions: sessionmaker[Session]) -> None:
    long_name = "x" * 300

    class Wild(Agent):
        async def complete_tools(self, **kw: Any) -> ToolCompletion:
            if any(m.outcomes for m in kw["messages"]):
                return ToolCompletion("gave up", (), "end_turn", Usage(), kw["model"])
            use = ToolUse("i" * 300, long_name, {})
            return ToolCompletion("", (use,), "tool_use", Usage(), kw["model"])

    with sessions() as s:
        import_dataset(s, "wild", [Case(id="w", input={"date": "d"})])
    wild = Wild()
    runner = Runner(sessions, SETTINGS, target_factory=lambda spec: AgentTarget(spec, wild))  # type: ignore[arg-type]
    run = runner.create_run(config("wild"))
    asyncio.run(runner.execute(run.id))
    row = steps(sessions, run.id)["w"][1]
    assert (row.status, len(row.tool or ""), len(row.call_id or "")) == ("unknown_tool", 64, 128)


# The trajectory endpoint


@pytest.mark.anyio
async def test_the_trajectory_endpoint_serves_every_step(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        import_dataset(s, "api", [Case(id="a/1", input={"date": "d"})])
    factory = Factory()
    runner = Runner(sessions, SETTINGS, target_factory=factory)
    run = runner.create_run(config("api"))
    await runner.execute(run.id)
    async with serving(sessions, EVALKIT_MAX_ACTIVE_RUNS="0") as (c, _):
        response = await c.get(f"/v1/runs/{run.id}/results/a/1/trajectory", headers=AUTH)
        assert response.status_code == 200
        body = response.json()
        assert body["case_id"] == "a/1"
        assert body["run_id"] == run.id
        assert body["turns"] == []
        assert [s["kind"] for s in body["steps"]] == ["model", "tool", "model", "tool", "model"]
        first, lookup = body["steps"][:2]
        assert first["stop_reason"] == "tool_use"
        assert first["cost_usd"] == pytest.approx(0.001)
        assert "tool" not in first
        assert lookup["tool"] == "find_slots"
        assert lookup["arguments"] == {"date": "2026-11-20"}
        assert lookup["result"] == ["19:00", "20:30"]
        assert (lookup["status"], lookup["source"], lookup["error"]) == ("ok", "fixture", None)
        assert "text" not in lookup
        missing = await c.get(f"/v1/runs/{run.id}/results/nope/trajectory", headers=AUTH)
        assert missing.status_code == 404
        unknown = await c.get("/v1/runs/nope/results/a/1/trajectory", headers=AUTH)
        assert unknown.status_code == 404
        unauthorised = await c.get(f"/v1/runs/{run.id}/results/a/1/trajectory")
        assert unauthorised.status_code == 401


def test_the_loop_runs_without_a_store_or_limiter() -> None:
    agent = Agent()
    spec = agent_spec()
    tools = [ToolDefinition.of(d) for d in spec.sandbox.definitions()]
    loop = ToolLoop(agent, model=CLAUDE, tools=tools, owner_hash="o")
    sandbox = ToolSandbox(spec.sandbox, spec.recorded(), session="s")
    result = asyncio.run(loop.run("task", sandbox))
    assert result.text == "Booked, confirmation BK-19 for task"
    assert loop.fresh_steps == 3
    assert len(sandbox.calls) == 2
