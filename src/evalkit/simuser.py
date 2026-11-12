"""Simulated users: who talks to the assistant in a conversation case.

A simulator reads the case's ``Conversation`` (persona, goal, the dialogue so
far) and returns the user's next turn with a verdict: ``continue`` with a
message for the assistant, ``goal_reached`` or ``gave_up``. The conversation
target (``evalkit.targets.conversation``) asks it after every assistant reply,
and caches what it says like any other paid turn.

* ``ScriptedSimulator`` makes no calls: it says the case's ``user_turns`` in
  order, reaches the goal when an assistant reply contains one of the case's
  ``success`` phrases, and gives up when it has nothing left to say. Runs on it
  are deterministic, so they are what tests and demos use.
* ``LLMSimulator`` role-plays the persona on any provider and answers in JSON
  under an output schema.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from evalkit.datasets import Conversation, Message
from evalkit.providers.base import Provider
from evalkit.targets.base import TargetError, Usage
from evalkit.targets.specs import LLMSimulatorSpec, ScriptedSimulatorSpec

Status = Literal["continue", "goal_reached", "gave_up"]
STATUSES: tuple[Status, ...] = ("continue", "goal_reached", "gave_up")


@dataclass(frozen=True)
class UserTurn:
    status: Status
    message: str = ""
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        if self.status == "continue" and not self.message.strip():
            raise ValueError("a user turn that continues the conversation needs a message")

    def to_json(self) -> str:
        return json.dumps({"status": self.status, "message": self.message}, sort_keys=True)

    @classmethod
    def from_json(cls, text: str, usage: Usage | None = None) -> UserTurn:
        """Parse a simulator's JSON answer. Raises ValueError when it is not one."""
        try:
            answer = _Answer.model_validate_json(text)
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(p) for p in first["loc"]) or "answer"
            raise ValueError(f"{where}: {first['msg']}") from None
        return cls(answer.status, answer.message, usage or Usage())


class _Answer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: Status
    message: str = ""


class Simulator(ABC):
    #: The provider the simulator calls, or None when it makes no calls.
    provider: str | None = None

    @abstractmethod
    async def next_turn(
        self, conversation: Conversation, transcript: Sequence[Message]
    ) -> UserTurn:
        """The user's next turn, given the case and the dialogue so far, which is
        empty or ends with an assistant reply."""


def generated_user_turns(conversation: Conversation, transcript: Sequence[Message]) -> int:
    """How many user messages in ``transcript`` the simulator wrote (not the case)."""
    opening = sum(1 for m in conversation.messages if m.role == "user")
    return sum(1 for m in transcript if m.role == "user") - opening


class ScriptedSimulator(Simulator):
    def __init__(self, spec: ScriptedSimulatorSpec) -> None:
        self.spec = spec

    def _reached(self, conversation: Conversation, reply: str) -> bool:
        if self.spec.ignore_case:
            reply = reply.casefold()
            return any(p.casefold() in reply for p in conversation.success)
        return any(p in reply for p in conversation.success)

    async def next_turn(
        self, conversation: Conversation, transcript: Sequence[Message]
    ) -> UserTurn:
        replied = bool(transcript) and transcript[-1].role == "assistant"
        if replied and self._reached(conversation, transcript[-1].content):
            return UserTurn("goal_reached")
        said = generated_user_turns(conversation, transcript)
        if said < len(conversation.user_turns):
            return UserTurn("continue", conversation.user_turns[said])
        if not transcript:
            # Nothing scripted and nothing said yet: open with the goal itself.
            return UserTurn("continue", conversation.goal)
        return UserTurn("gave_up")


INSTRUCTIONS = """\
You are role-playing a user talking to an AI assistant, to test the assistant.
Stay in character as the persona, pursue the goal, and write only what that \
user would type: short, natural messages, one at a time. Do not help the \
assistant or reveal that you are a simulation. Give information the persona \
has when the assistant asks for it.

After each assistant reply, decide:
- "goal_reached" when the assistant has fully achieved the goal;
- "gave_up" when the assistant is clearly unable or unwilling to achieve it, \
or the conversation is going in circles;
- otherwise "continue", with your next message.

Answer with JSON only: {"status": "continue" | "goal_reached" | "gave_up", \
"message": "..."}. The message may be empty unless the status is "continue"."""

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": list(STATUSES)},
        "message": {"type": "string"},
    },
    "required": ["status", "message"],
    "additionalProperties": False,
}


def render_prompt(conversation: Conversation, transcript: Sequence[Message]) -> str:
    parts = []
    if conversation.persona:
        parts.append(f"<persona>\n{conversation.persona}\n</persona>")
    parts.append(f"<goal>\n{conversation.goal}\n</goal>")
    if transcript:
        lines = "\n\n".join(f"{m.role.upper()}: {m.content}" for m in transcript)
        parts.append(f"<conversation>\n{lines}\n</conversation>")
        parts.append("Decide on the assistant's last reply, and write your next message.")
    else:
        parts.append("The conversation has not started. Write your opening message.")
    return "\n\n".join(parts)


class LLMSimulator(Simulator):
    def __init__(self, spec: LLMSimulatorSpec, client: Provider) -> None:
        client.prices.require(spec.model)
        self.spec = spec
        self.client = client
        self.provider = client.name

    @property
    def system(self) -> str:
        if self.spec.instructions:
            return f"{INSTRUCTIONS}\n\n{self.spec.instructions}"
        return INSTRUCTIONS

    async def next_turn(
        self, conversation: Conversation, transcript: Sequence[Message]
    ) -> UserTurn:
        completion = await self.client.complete(
            model=self.spec.model,
            prompt=render_prompt(conversation, transcript),
            system=self.system,
            max_tokens=self.spec.max_tokens,
            effort=self.spec.effort,
            output_schema=OUTPUT_SCHEMA,
        )
        try:
            return UserTurn.from_json(completion.text, completion.usage)
        except ValueError as exc:
            # A paid answer that cannot be used; asking again costs as much and
            # usually fails the same way, so the case fails.
            raise TargetError(
                f"the simulated user's answer is unusable: {exc}", retryable=False
            ) from None


def build_simulator(
    spec: LLMSimulatorSpec | ScriptedSimulatorSpec, client: Provider | None = None
) -> Simulator:
    if isinstance(spec, ScriptedSimulatorSpec):
        return ScriptedSimulator(spec)
    if client is None:
        raise ValueError("an LLM simulator needs its provider's client")
    return LLMSimulator(spec, client)
