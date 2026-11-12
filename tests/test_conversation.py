"""Conversation cases: the case format, multi-turn requests on every provider,
simulated users, the conversation target with its turn cache, per-turn result
rows, and resuming a run killed mid-dialogue without paying for a turn twice."""

import asyncio
import json
from collections import Counter
from collections.abc import Sequence
from typing import Any

import httpx2
import pytest
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from evalkit.config import Settings
from evalkit.datasets import (
    Case,
    Conversation,
    Message,
    conversation_of,
    import_dataset,
    parse_jsonl,
)
from evalkit.db.models import CachedResponse, Result, ResultTurn, Run
from evalkit.errors import InvalidInputError, PreconditionError
from evalkit.pricing import ModelPrice, PriceBook, PriceTable, TokenCounts
from evalkit.providers.anthropic import ClaudeClient
from evalkit.providers.base import (
    Completion,
    Embeddings,
    ErrorKind,
    Prompt,
    Provider,
    ProviderError,
    as_messages,
)
from evalkit.providers.gemini import GeminiClient
from evalkit.providers.openai_compat import OpenAICompatClient
from evalkit.providers.registry import ProviderConfig, ProviderRegistry
from evalkit.redaction import RedactionConfig
from evalkit.runner import Runner
from evalkit.runspec import RunConfig
from evalkit.simuser import (
    INSTRUCTIONS,
    OUTPUT_SCHEMA,
    LLMSimulator,
    ScriptedSimulator,
    UserTurn,
    build_simulator,
    render_prompt,
)
from evalkit.targets import ConversationTargetSpec, TargetError, TargetSpec, Usage
from evalkit.targets.conversation import MAX_TURNS, NO_REPLY, ConversationTarget, turn_key
from evalkit.targets.factory import identity
from evalkit.targets.specs import LLMSimulatorSpec, ScriptedSimulatorSpec
from tests.fake_anthropic import FakeAnthropic, json_reply, message
from tests.fake_gemini import FakeGemini
from tests.fake_openai import FakeOpenAI

SPEC: TypeAdapter[Any] = TypeAdapter(TargetSpec)
MODEL = "claude-sonnet-5-5"
LOCAL = ProviderConfig(kind="openai_compat", base_url="http://llm.test/v1")
PRICES = {"gpt-test": ModelPrice(input=2.0, output=8.0)}


def booking(i: int, kind: str) -> dict[str, Any]:
    """A booking dialogue that reaches its goal, gives up, or runs out of turns."""
    goal = f"Book a table for {i % 6 + 1} at 7pm"
    if kind == "reach":
        turns = [f"Hi, this is guest {i}", "A table please", "Make it 7pm and book it"]
        return {"goal": goal, "user_turns": turns, "success": ["confirmation"], "max_turns": 5}
    if kind == "give_up":
        return {"goal": goal, "user_turns": [f"Hello from {i}"], "success": ["confirmation"]}
    if kind == "limit":
        turns = [f"Turn {n} from {i}" for n in range(5)]
        return {"goal": goal, "user_turns": turns, "success": ["confirmation"], "max_turns": 2}
    # Opens with the user's message already in the case: the assistant speaks first.
    return {
        "goal": goal,
        "messages": [{"role": "user", "content": f"Please book for guest {i}"}],
        "success": ["confirmation"],
    }


KINDS = ("reach", "give_up", "limit", "opening")


def cases(n: int = 8) -> list[Case]:
    return [Case(id=f"b{i:03d}", input=booking(i, KINDS[i % 4])) for i in range(n)]


class Assistant:
    """An assistant model in process: books when asked to, otherwise echoes.
    It records every completed request, so a test can see a turn paid twice."""

    name = "anthropic"

    def __init__(self, prices: PriceTable, delay: float = 0.0) -> None:
        self.prices = prices
        self.delay = delay
        self.completed: list[str] = []
        self.fail_once: set[int] = set()
        self.reply: str | None = None

    async def complete(
        self,
        *,
        model: str,
        prompt: Prompt,
        system: str = "",
        max_tokens: int = 4096,
        effort: str = "medium",
        output_schema: dict[str, Any] | None = None,
        fallbacks: bool = True,
    ) -> Completion:
        messages = as_messages(prompt)
        if self.delay:
            await asyncio.sleep(self.delay)
        number = len(self.completed)
        if number in self.fail_once:
            self.fail_once.discard(number)
            raise ProviderError(
                "overloaded", kind=ErrorKind.OVERLOADED, provider="anthropic", retry_after=0
            )
        last = messages[-1]["content"]
        if self.reply is not None:
            text = self.reply
        elif "book" in last.lower():
            text = f"Booked. Your confirmation for {last[-12:]} is BK-{len(messages)}"
        else:
            text = f"Noted: {last}"
        self.completed.append(json.dumps(messages))
        return Completion(text, "end_turn", Usage(10, 5, cost_usd=0.001), model)

    async def count_tokens(self, *, model: str, prompt: str, system: str = "") -> int:
        raise NotImplementedError

    async def embed(self, *, model: str, texts: Sequence[str]) -> Embeddings:
        raise NotImplementedError

    def price(self, model: str, tokens: TokenCounts, *, batch: bool = False) -> float:
        return 0.0

    def classify_error(self, exc: BaseException) -> ProviderError:
        raise NotImplementedError

    async def aclose(self) -> None:
        return


SETTINGS = Settings.from_env({})
TABLE = SETTINGS.price_book().table("anthropic")


class Factory:
    """Builds conversation targets on one in-process assistant."""

    def __init__(self, delay: float = 0.0) -> None:
        self.assistant = Assistant(TABLE, delay)
        self.targets: list[ConversationTarget] = []

    def __call__(self, spec: Any) -> ConversationTarget:
        target = ConversationTarget(spec, self.assistant, build_simulator(spec.simulator))
        self.targets.append(target)
        return target


def config(dataset: str = "bookings", **target: Any) -> RunConfig:
    options = target.pop("options", {})
    return RunConfig.model_validate(
        {
            "label": "talk",
            "dataset": {"name": dataset},
            "target": {"type": "conversation", "model": MODEL, **target},
            "scorers": [{"type": "regex", "pattern": "confirmation", "name": "booked"}],
            "options": {"concurrency": 4, **options},
        }
    )


@pytest.fixture
def dataset(sessions: sessionmaker[Session]) -> str:
    with sessions() as s:
        import_dataset(s, "bookings", cases())
    return "bookings"


def results(sessions: sessionmaker[Session], run_id: str) -> dict[str, Result]:
    with sessions() as s:
        rows = s.scalars(select(Result).where(Result.run_id == run_id))
        return {r.case_id: r for r in rows}


def turns(sessions: sessionmaker[Session], run_id: str) -> dict[str, list[ResultTurn]]:
    found: dict[str, list[ResultTurn]] = {}
    with sessions() as s:
        rows = s.scalars(
            select(ResultTurn)
            .where(ResultTurn.run_id == run_id)
            .order_by(ResultTurn.case_id, ResultTurn.turn)
        )
        for row in rows:
            found.setdefault(row.case_id, []).append(row)
    return found


# The case format


def test_conversation_input_is_validated() -> None:
    conv = Conversation.model_validate(booking(0, "opening"))
    assert conv.messages == (Message(role="user", content="Please book for guest 0"),)
    assert conv.max_turns == 10
    assert conv.persona == ""
    with pytest.raises(ValidationError, match="goal"):
        Conversation.model_validate({"messages": []})
    with pytest.raises(ValidationError, match="must be from the user"):
        Conversation.model_validate(
            {"goal": "g", "messages": [{"role": "assistant", "content": "hi"}]}
        )
    with pytest.raises(ValidationError, match="must be from the assistant"):
        Conversation.model_validate(
            {
                "goal": "g",
                "messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
            }
        )
    for bad in ({"goal": "g", "max_turns": 0}, {"goal": "g", "max_turns": 51}):
        with pytest.raises(ValidationError):
            Conversation.model_validate(bad)
    with pytest.raises(ValidationError):
        Conversation.model_validate({"goal": "g", "extra": 1})


def test_conversation_of_names_the_case() -> None:
    assert conversation_of(cases(1)[0]).goal.startswith("Book a table")
    with pytest.raises(InvalidInputError, match=r"case 'x' is not a conversation: goal"):
        conversation_of(Case(id="x", input={"persona": "p"}))
    with pytest.raises(InvalidInputError, match="case 'y' is not a conversation"):
        conversation_of(Case(id="y", input="just a question"))


def test_conversation_cases_import_like_any_other() -> None:
    lines = "\n".join(json.dumps({"id": c.id, "input": c.input}) for c in cases(4))
    parsed = parse_jsonl(lines)
    assert [conversation_of(c).max_turns for c in parsed] == [5, 10, 2, 10]


# Conversations on every provider


def test_as_messages() -> None:
    assert as_messages("hi") == [{"role": "user", "content": "hi"}]
    talk = [Message(role="user", content="a"), Message(role="assistant", content="b")]
    with pytest.raises(InvalidInputError, match="must end with a user message"):
        as_messages(talk)
    with pytest.raises(InvalidInputError, match="must end with a user message"):
        as_messages([])
    with pytest.raises(InvalidInputError, match="roles alternate"):
        as_messages([Message(role="assistant", content="b")])
    talk.append(Message(role="user", content="c"))
    assert [m["role"] for m in as_messages(talk)] == ["user", "assistant", "user"]


TALK = [
    Message(role="user", content="one"),
    Message(role="assistant", content="two"),
    Message(role="user", content="three"),
]


def test_claude_sends_the_whole_conversation() -> None:
    fake = FakeAnthropic(json_reply(message("four")))
    client = fake.client(PriceTable({MODEL: ModelPrice(input=1, output=1)}))
    done = asyncio.run(client.complete(model=MODEL, prompt=TALK, system="be brief"))
    assert done.text == "four"
    assert fake.requests[0]["messages"] == [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]


def test_openai_compatible_sends_the_whole_conversation() -> None:
    fake = FakeOpenAI("http://llm.test/v1", api_key=None, answer=lambda body: "four")
    client = OpenAICompatClient(
        PriceTable(PRICES),
        None,
        name="local",
        base_url="http://llm.test/v1",
        transport=httpx2.MockTransport(fake.handle),
    )
    done = asyncio.run(client.complete(model="gpt-test", prompt=TALK, system="be brief"))
    assert done.text == "four"
    assert fake.rejected == []
    assert fake.bodies[0]["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]


def test_gemini_sends_the_whole_conversation_as_user_and_model() -> None:
    fake = FakeGemini("http://gemini.test/v1beta", api_key=None, answer=lambda model, body: "four")
    client = GeminiClient(
        PriceTable({"gemini-test": ModelPrice(input=1, output=1)}),
        None,
        name="gem",
        base_url="http://gemini.test/v1beta",
        transport=httpx2.MockTransport(fake.handle),
    )
    done = asyncio.run(client.complete(model="gemini-test", prompt=TALK))
    assert done.text == "four"
    assert fake.rejected == []
    assert fake.bodies[0]["contents"] == [
        {"role": "user", "parts": [{"text": "one"}]},
        {"role": "model", "parts": [{"text": "two"}]},
        {"role": "user", "parts": [{"text": "three"}]},
    ]


# Simulated users


def conv(**fields: Any) -> Conversation:
    return Conversation.model_validate({"goal": "Get a refund", **fields})


def test_scripted_simulator_follows_its_script() -> None:
    sim = ScriptedSimulator(ScriptedSimulatorSpec(type="scripted"))
    c = conv(user_turns=["hi", "refund please"], success=["Refund ISSUED"])

    def say(*messages: tuple[str, str]) -> UserTurn:
        talk = [Message(role=r, content=t) for r, t in messages]
        return asyncio.run(sim.next_turn(c, talk))

    assert say() == UserTurn("continue", "hi")
    assert say(("user", "hi"), ("assistant", "hello")) == UserTurn("continue", "refund please")
    done = say(("user", "hi"), ("assistant", "your refund issued."))
    assert done.status == "goal_reached"
    assert say(
        ("user", "hi"), ("assistant", "?"), ("user", "refund please"), ("assistant", "no")
    ) == UserTurn("gave_up")
    # Nothing scripted: the user opens with the goal.
    assert asyncio.run(sim.next_turn(conv(), [])) == UserTurn("continue", "Get a refund")
    exact = ScriptedSimulator(ScriptedSimulatorSpec(type="scripted", ignore_case=False))
    reply = [Message(role="user", content="x"), Message(role="assistant", content="refund issued")]
    assert asyncio.run(exact.next_turn(c, reply)) == UserTurn("continue", "refund please")


def test_scripted_simulator_counts_only_its_own_turns() -> None:
    sim = ScriptedSimulator(ScriptedSimulatorSpec(type="scripted"))
    c = conv(messages=[{"role": "user", "content": "opening"}], user_turns=["next"])
    talk = [*c.messages, Message(role="assistant", content="ok")]
    assert asyncio.run(sim.next_turn(c, talk)) == UserTurn("continue", "next")


def test_user_turns_parse_and_serialise() -> None:
    turn = UserTurn("continue", "hello", Usage(1, 2))
    assert UserTurn.from_json(turn.to_json()) == UserTurn("continue", "hello")
    assert UserTurn.from_json('{"status": "gave_up"}') == UserTurn("gave_up")
    for bad, why in (
        ("nope", "Invalid JSON"),
        ("[1]", "answer"),
        ('{"status": "maybe", "message": ""}', "status"),
        ('{"status": "continue", "message": 3}', "message"),
    ):
        with pytest.raises(ValueError, match=why):
            UserTurn.from_json(bad)
    with pytest.raises(ValueError, match="needs a message"):
        UserTurn("continue", "  ")
    with pytest.raises(ValueError, match="needs its provider"):
        build_simulator(LLMSimulatorSpec(type="llm"))


def test_render_prompt() -> None:
    c = conv(persona="An impatient customer")
    opening = render_prompt(c, [])
    assert "<persona>\nAn impatient customer\n</persona>" in opening
    assert "<goal>\nGet a refund\n</goal>" in opening
    assert "has not started" in opening
    later = render_prompt(conv(), TALK[:2])
    assert "<persona>" not in later
    assert "<conversation>\nUSER: one\n\nASSISTANT: two\n</conversation>" in later


def test_llm_simulator_answers_in_json_under_a_schema() -> None:
    answers = iter(['{"status": "continue", "message": "Where is my refund?"}', "not json"])
    fake = FakeOpenAI("http://llm.test/v1", api_key=None, answer=lambda body: next(answers))
    client = OpenAICompatClient(
        PriceTable(PRICES),
        None,
        name="local",
        base_url="http://llm.test/v1",
        transport=httpx2.MockTransport(fake.handle),
    )
    spec = LLMSimulatorSpec(type="llm", provider="local", model="gpt-test", instructions="Be rude.")
    sim = build_simulator(spec, client)
    assert isinstance(sim, LLMSimulator)
    assert sim.provider == "local"
    turn = asyncio.run(sim.next_turn(conv(), []))
    assert turn.status == "continue"
    assert turn.message == "Where is my refund?"
    assert turn.usage.input_tokens == 100
    body = fake.bodies[0]
    assert body["messages"][0] == {"role": "system", "content": f"{INSTRUCTIONS}\n\nBe rude."}
    assert body["response_format"]["json_schema"]["schema"] == OUTPUT_SCHEMA
    with pytest.raises(TargetError, match="simulated user's answer is unusable") as caught:
        asyncio.run(sim.next_turn(conv(), []))
    assert not caught.value.retryable
    assert fake.rejected == []
    plain = LLMSimulator(LLMSimulatorSpec(type="llm", provider="local", model="gpt-test"), client)
    assert plain.system == INSTRUCTIONS
    with pytest.raises(InvalidInputError):
        LLMSimulator(LLMSimulatorSpec(type="llm", provider="local", model="unpriced"), client)


# The target spec


def test_conversation_spec_and_identity() -> None:
    spec = SPEC.validate_python({"type": "conversation", "model": MODEL})
    assert isinstance(spec, ConversationTargetSpec)
    assert spec.simulator == ScriptedSimulatorSpec(type="scripted")
    llm = SPEC.validate_python(
        {"type": "conversation", "model": MODEL, "simulator": {"type": "llm", "provider": "local"}}
    )
    configs = {"local": LOCAL}
    assert identity(spec, {}, configs) != identity(llm, {}, configs)
    moved = {"local": ProviderConfig(kind="openai_compat", base_url="http://other.test/v1")}
    assert identity(llm, {}, configs) != identity(llm, {}, moved)
    with pytest.raises(InvalidInputError, match="unknown provider 'local'"):
        identity(llm, {}, {})
    with pytest.raises(ValidationError):
        SPEC.validate_python({"type": "conversation", "simulator": {"type": "psychic"}})
    # Who plays the user is not part of what the assistant answered.
    assert spec.assistant_part() == llm.assistant_part()
    assert "simulator" not in spec.assistant_part()


def test_turn_keys_cover_who_speaks_and_what_they_saw() -> None:
    a = turn_key("assistant", "h", [{"role": "user", "content": "x"}], "fp")
    assert a == turn_key("assistant", "h", [{"role": "user", "content": "x"}], "fp")
    assert a != turn_key("user", "h", [{"role": "user", "content": "x"}], "fp")
    assert a != turn_key("assistant", "h2", [{"role": "user", "content": "x"}], "fp")
    assert a != turn_key("assistant", "h", [{"role": "user", "content": "y"}], "fp")
    assert a != turn_key("assistant", "h", [{"role": "user", "content": "x"}], "fp2")


# Runs


def test_a_run_ends_each_dialogue_and_stores_every_turn(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    factory = Factory()
    runner = Runner(sessions, SETTINGS, target_factory=factory)
    run = runner.create_run(config())
    report = asyncio.run(runner.execute(run.id))
    assert report.status == "completed"
    assert report.executed == 8
    assert report.errors == 0
    found, rows = results(sessions, run.id), turns(sessions, run.id)
    ends = {cid: r.stop_reason for cid, r in found.items()}
    assert ends == {
        "b000": "goal_reached",
        "b001": "gave_up",
        "b002": MAX_TURNS,
        "b003": "goal_reached",
        "b004": "goal_reached",
        "b005": "gave_up",
        "b006": MAX_TURNS,
        "b007": "goal_reached",
    }
    reached = rows["b000"]
    assert [(t.turn, t.role) for t in reached] == [
        (0, "user"),
        (1, "assistant"),
        (2, "user"),
        (3, "assistant"),
        (4, "user"),
        (5, "assistant"),
        (6, "user"),
    ]
    assert reached[0].content == "Hi, this is guest 0"
    assert [t.stop_reason for t in reached if t.role == "user"] == [
        "continue",
        "continue",
        "continue",
        "goal_reached",
    ]
    assert found["b000"].output == reached[5].content
    assert "confirmation" in (found["b000"].output or "")
    assert found["b000"].scores["booked"]["value"] == 1.0
    assert found["b001"].scores["booked"]["value"] == 0.0
    # The turn limit counts assistant replies; the user's last verdict is kept.
    limited = rows["b002"]
    assert [t.role for t in limited] == ["user", "assistant", "user", "assistant", "user"]
    assert limited[-1].stop_reason == "continue"
    # A case that opens with the user's message: the assistant speaks first.
    assert [(t.turn, t.role) for t in rows["b003"]] == [(1, "assistant"), (2, "user")]
    # The result's cost is the sum of its turns', and nothing came from a cache.
    for cid, result in found.items():
        assert not result.cached
        assert result.cost_usd == pytest.approx(sum(t.cost_usd for t in rows[cid]))
        assert result.input_tokens == sum(t.input_tokens for t in rows[cid])
    assert factory.assistant.completed == list(dict.fromkeys(factory.assistant.completed))
    assert len(factory.assistant.completed) == sum(
        1 for ts in rows.values() for t in ts if t.role == "assistant"
    )


def test_a_rerun_replays_every_turn_for_free(sessions: sessionmaker[Session], dataset: str) -> None:
    factory = Factory()
    runner = Runner(sessions, SETTINGS, target_factory=factory)
    first = runner.create_run(config())
    asyncio.run(runner.execute(first.id))
    paid = len(factory.assistant.completed)
    second = runner.create_run(config())
    asyncio.run(runner.execute(second.id))
    assert len(factory.assistant.completed) == paid
    before, after = results(sessions, first.id), results(sessions, second.id)
    assert {c: r.output for c, r in after.items()} == {c: r.output for c, r in before.items()}
    assert {c: r.stop_reason for c, r in after.items()} == {
        c: r.stop_reason for c, r in before.items()
    }
    assert all(r.cached and r.cost_usd == 0 for r in after.values())
    replayed = turns(sessions, second.id)
    assert all(t.cached and t.cost_usd == 0 for ts in replayed.values() for t in ts)
    assert {c: [t.content for t in ts] for c, ts in replayed.items()} == {
        c: [t.content for t in ts] for c, ts in turns(sessions, first.id).items()
    }
    # Without the cache every turn is paid for again.
    third = runner.create_run(config(options={"use_cache": False}))
    asyncio.run(runner.execute(third.id))
    assert len(factory.assistant.completed) == 2 * paid
    assert not any(r.cached for r in results(sessions, third.id).values())


def test_another_simulator_keeps_the_replies_it_already_has(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    factory = Factory()
    runner = Runner(sessions, SETTINGS, target_factory=factory)
    asyncio.run(runner.execute(runner.create_run(config()).id))
    paid = len(factory.assistant.completed)
    exact = runner.create_run(config(simulator={"type": "scripted", "ignore_case": False}))
    asyncio.run(runner.execute(exact.id))
    # The same dialogues: the assistant is asked nothing new, the user's turns are new.
    assert len(factory.assistant.completed) == paid
    rows = turns(sessions, exact.id)
    assert all(t.cached for ts in rows.values() for t in ts if t.role == "assistant")
    assert not any(t.cached for ts in rows.values() for t in ts if t.role == "user")


def test_a_retried_dialogue_pays_only_for_the_turns_it_lacks(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    factory = Factory()
    factory.assistant.fail_once = {2, 5}
    runner = Runner(sessions, SETTINGS, target_factory=factory)
    run = runner.create_run(config(options={"concurrency": 1, "retry": {"base_delay": 0}}))
    report = asyncio.run(runner.execute(run.id))
    assert report.errors == 0
    found = results(sessions, run.id)
    assert sorted(r.attempts for r in found.values()) == [1] * 6 + [2] * 2
    completed = factory.assistant.completed
    assert len(completed) == len(set(completed))
    retried = [r for r in found.values() if r.attempts == 2]
    for result in retried:
        rows = turns(sessions, run.id)[result.case_id]
        assistant = [t for t in rows if t.role == "assistant"]
        # The turns before the failure were replayed from the cache.
        assert assistant[0].cached
        assert not assistant[-1].cached
        assert not result.cached
        assert result.cost_usd == pytest.approx(sum(t.cost_usd for t in rows))


def test_a_run_killed_mid_dialogue_resumes_without_repeating_paid_turns(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    factory = Factory(delay=0.01)
    runner = Runner(sessions, SETTINGS, target_factory=factory)
    run = runner.create_run(config(options={"concurrency": 1}))

    def cached_replies() -> int:
        target = factory.targets[-1]
        with sessions() as s:
            return int(
                s.scalar(
                    select(func.count())
                    .select_from(CachedResponse)
                    .where(CachedResponse.target_hash == target.assistant_hash)
                )
                or 0
            )

    async def interrupt() -> None:
        task = asyncio.create_task(runner.execute(run.id))
        # The first dialogue needs three replies: stop it after two.
        while not factory.targets or cached_replies() < 2:
            await asyncio.sleep(0.002)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(interrupt())
    assert "b000" not in results(sessions, run.id)
    paid_before = len(factory.assistant.completed)
    assert paid_before >= 2
    with sessions() as s:
        assert s.get_one(Run, run.id).status == "interrupted"

    report = asyncio.run(Runner(sessions, SETTINGS, target_factory=factory).execute(run.id))
    assert report.status == "completed"
    completed = factory.assistant.completed
    # No reply the assistant gave was asked for again.
    assert len(completed) == len(set(completed))
    found = results(sessions, run.id)
    assert len(found) == 8
    resumed = turns(sessions, run.id)["b000"]
    replies = [t for t in resumed if t.role == "assistant"]
    assert [t.cached for t in replies][:2] == [True, True]
    assert not replies[2].cached
    assert found["b000"].stop_reason == "goal_reached"
    assert found["b000"].cost_usd == pytest.approx(0.001)


def test_turns_are_redacted_before_the_dialogue_goes_on(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        import_dataset(
            s,
            "pii",
            [Case(id="p", input={"goal": "g", "user_turns": ["mail me at ann@example.com"]})],
        )
    factory = Factory()
    settings = Settings.from_env({})
    runner = Runner(sessions, settings, target_factory=factory)
    run = runner.create_run(config("pii"))
    asyncio.run(runner.execute(run.id))
    rows = turns(sessions, run.id)["p"]
    assert rows[0].content == "mail me at [EMAIL]"
    assert rows[1].content == "Noted: mail me at [EMAIL]"
    assert "ann@example.com" not in factory.assistant.completed[0]
    with sessions() as s:
        stored = list(s.scalars(select(CachedResponse.output)))
    assert all("ann@example.com" not in text for text in stored)
    off = Settings.from_env({}).model_copy(update={"redaction": RedactionConfig(enabled=False)})
    plain = Runner(sessions, off, target_factory=factory)
    asyncio.run(plain.execute(plain.create_run(config("pii")).id))
    assert "ann@example.com" in factory.assistant.completed[-1]


def test_an_empty_reply_ends_the_dialogue(sessions: sessionmaker[Session], dataset: str) -> None:
    factory = Factory()
    factory.assistant.reply = ""
    runner = Runner(sessions, SETTINGS, target_factory=factory)
    run = runner.create_run(config())
    asyncio.run(runner.execute(run.id))
    found = results(sessions, run.id)
    assert {r.stop_reason for r in found.values()} == {NO_REPLY}
    assert {r.output for r in found.values()} == {""}
    assert Counter(t.role for ts in turns(sessions, run.id).values() for t in ts) == {
        "user": 6,
        "assistant": 8,
    }


def test_a_conversation_run_needs_conversation_cases(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        import_dataset(s, "plain", [Case(id="q1", input={"question": "2+2?"})])
    runner = Runner(sessions, SETTINGS, target_factory=Factory())
    with pytest.raises(InvalidInputError, match="case 'q1' is not a conversation"):
        runner.create_run(config("plain"))
    with pytest.raises(InvalidInputError, match="batch mode"):
        runner.create_run(config(options={"mode": "batch"}))


class Registry:
    """The real factory path: Claude as the assistant (the fake API), and a
    simulated user played by a model on an OpenAI-compatible server."""

    def __init__(self) -> None:
        self.claude = FakeAnthropic()
        self.claude.default = self._claude
        self.openai = FakeOpenAI("http://llm.test/v1", api_key=None, answer=self._user)

    @staticmethod
    def _claude(body: dict[str, Any]) -> httpx2.Response:
        last = body["messages"][-1]["content"]
        text = "Done, confirmation BK-1" if "book" in last else "What time?"
        return json_reply(message(text, model=body["model"]))

    @staticmethod
    def _user(body: dict[str, Any]) -> str:
        prompt = body["messages"][-1]["content"]
        if "confirmation" in prompt:
            return json.dumps({"status": "goal_reached", "message": "Thanks"})
        if "What time?" in prompt:
            return json.dumps({"status": "continue", "message": "7pm, book it"})
        return json.dumps({"status": "continue", "message": "A table for two"})

    def registry(self, configs: dict[str, ProviderConfig]) -> ProviderRegistry:
        claude: ClaudeClient = self.claude.client(PriceTable(PRICES))
        return ProviderRegistry(
            configs,
            PriceBook(PRICES),
            claude=claude,
            transport=httpx2.MockTransport(self.openai.handle),
        )


def test_an_llm_user_talks_to_claude_through_the_registry(
    sessions: sessionmaker[Session],
) -> None:
    with sessions() as s:
        import_dataset(s, "live", [Case(id=f"l{i}", input={"goal": "Book"}) for i in range(3)])
    fakes = Registry()
    runner = Runner(sessions, Settings(), providers=fakes.registry({"local": LOCAL}))
    cfg = config(
        "live",
        simulator={"type": "llm", "provider": "local", "model": "gpt-test"},
        options={"concurrency": 1},
    )
    run = runner.create_run(cfg)
    with sessions() as s:
        assert set(s.get_one(Run, run.id).prices or {}) == {"anthropic", "local"}
    report = asyncio.run(runner.execute(run.id))
    assert report.errors == 0
    assert fakes.openai.rejected == []
    found = results(sessions, run.id)
    assert {r.stop_reason for r in found.values()} == {"goal_reached"}
    # Identical dialogues: each turn was paid for once and replayed for the rest.
    assert len(fakes.claude.requests) == 2
    assert len(fakes.openai.bodies) == 3
    assert fakes.claude.requests[1]["messages"] == [
        {"role": "user", "content": "A table for two"},
        {"role": "assistant", "content": "What time?"},
        {"role": "user", "content": "7pm, book it"},
    ]
    total = sum(r.cost_usd for r in found.values())
    assert total > 0
    with sessions() as s:
        turn_total = s.scalar(
            select(func.sum(ResultTurn.cost_usd)).where(ResultTurn.run_id == run.id)
        )
    assert turn_total == pytest.approx(total)
    # The simulated user's provider moved: the run's identity no longer holds.
    moved = {"local": ProviderConfig(kind="openai_compat", base_url="http://other.test/v1")}
    stale = Runner(sessions, Settings(), providers=fakes.registry(moved))
    again = runner.create_run(cfg)
    with pytest.raises(PreconditionError, match="configuration changed"):
        asyncio.run(stale.execute(again.id))
    unpriced = config("live", simulator={"type": "llm", "provider": "local", "model": "nope"})
    with pytest.raises(InvalidInputError):
        runner.create_run(unpriced)


def test_a_provider_retry_after_holds_back_the_turns_provider(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    factory = Factory()
    factory.assistant.fail_once = {0}
    runner = Runner(sessions, SETTINGS, target_factory=factory)
    run = runner.create_run(config(options={"concurrency": 1, "retry": {"base_delay": 0}}))
    held: list[float] = []
    asyncio.run(runner.execute(run.id))
    target = factory.targets[-1]
    limiter = target.assistant_limiter
    assert limiter is not None
    assert target.simulator_limiter is None  # the scripted user makes no calls
    original = limiter.cool_down

    async def record(seconds: float) -> None:
        held.append(seconds)
        await original(seconds)

    limiter.cool_down = record  # type: ignore[method-assign]
    factory.assistant.fail_once = {len(factory.assistant.completed)}
    target.store = None
    with pytest.raises(ProviderError):
        asyncio.run(target.generate(cases(1)[0]))
    assert held == [0]
    assert isinstance(target.assistant, Provider)
