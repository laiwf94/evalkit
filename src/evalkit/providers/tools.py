"""Tool use in one shape for every provider.

An agent loop (``evalkit.tools.loop``) holds the dialogue with a model as
``AgentMessage`` values: the user's task, the assistant's replies with the
tools they asked for (``ToolUse``), and the answers those tools gave
(``ToolOutcome``). A provider that supports tool use implements
``ToolProvider.complete_tools``, which turns that transcript into its own wire
format (Claude ``tool_use`` and ``tool_result`` blocks, chat completions
``tool_calls`` and ``tool`` messages, Gemini ``functionCall`` and
``functionResponse`` parts) and the answer back into a ``ToolCompletion``.

The loop never sees a provider's own format, so it is written once. Two
provider details travel through it unchanged, because the provider must get
them back on the next request: Claude's thinking blocks (``reasoning``) and
Gemini's thought signatures on function calls (``ToolUse.signature``).

Tool results go back to every provider as ``{"result": ...}`` or
``{"error": message}``: Gemini needs an object there, and one form everywhere
keeps the transcript, and with it the step cache, provider-neutral.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from evalkit.errors import InvalidInputError
from evalkit.providers.base import TRUNCATED_STOP_REASONS
from evalkit.targets.base import Usage

Role = Literal["user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolDefinition:
    """What the model is told about one tool: its arguments are a JSON object."""

    name: str
    description: str
    input_schema: dict[str, Any]

    @classmethod
    def of(cls, definition: dict[str, Any]) -> ToolDefinition:
        """From a sandbox definition (``ToolSpec.definition``)."""
        return cls(
            str(definition["name"]),
            str(definition.get("description", "")),
            dict(definition["input_schema"]),
        )


@dataclass(frozen=True)
class ToolUse:
    """One tool call the model asked for. ``arguments`` is the parsed JSON, or the
    raw text when the model sent arguments that are not JSON (the sandbox then
    answers ``invalid_arguments``). ``signature`` is Gemini's thought signature."""

    id: str
    name: str
    arguments: Any
    signature: str | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "name": self.name, "arguments": self.arguments}
        if self.signature is not None:
            out["signature"] = self.signature
        return out

    @classmethod
    def from_json(cls, value: Any) -> ToolUse:
        if not isinstance(value, dict):
            raise TypeError("a tool use is an object")
        signature = value.get("signature")
        return cls(
            id=_text(value.get("id"), "id"),
            name=_text(value.get("name"), "name"),
            arguments=value.get("arguments"),
            signature=None if signature is None else _text(signature, "signature"),
        )


@dataclass(frozen=True)
class ToolOutcome:
    """What one tool call answered, sent back to the model."""

    id: str
    name: str
    content: Any
    is_error: bool = False

    def payload(self) -> dict[str, Any]:
        """The object the model receives."""
        return {"error": self.content} if self.is_error else {"result": self.content}

    def payload_text(self) -> str:
        return json.dumps(self.payload(), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class AgentMessage:
    """One message of an agent transcript.

    A ``user`` message has ``text``; an ``assistant`` message has ``text`` and the
    ``tool_uses`` it asked for (and any ``reasoning`` blocks to replay); a
    ``tool`` message answers every tool use of the assistant message before it,
    in order."""

    role: Role
    text: str = ""
    tool_uses: tuple[ToolUse, ...] = ()
    outcomes: tuple[ToolOutcome, ...] = ()
    reasoning: tuple[dict[str, Any], ...] = ()

    def to_json(self) -> dict[str, Any]:
        """Canonical form, for cache keys: every field that reaches the model."""
        return {
            "role": self.role,
            "text": self.text,
            "tool_uses": [u.to_json() for u in self.tool_uses],
            "outcomes": [{"id": o.id, "name": o.name, **o.payload()} for o in self.outcomes],
            "reasoning": list(self.reasoning),
        }


@dataclass(frozen=True)
class ToolCompletion:
    """One model reply in a tool loop: text, the tools it asks for, and usage."""

    text: str
    tool_uses: tuple[ToolUse, ...]
    stop_reason: str | None
    usage: Usage
    model: str
    reasoning: tuple[dict[str, Any], ...] = field(default=())
    refusal_category: str | None = None

    @property
    def truncated(self) -> bool:
        """Cut off at the output limit; a tool call in it may be incomplete."""
        return self.stop_reason in TRUNCATED_STOP_REASONS


@runtime_checkable
class ToolProvider(Protocol):
    """A ``Provider`` that can also call tools."""

    name: str

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
        """The model's next reply to ``messages``, which must end with the user or
        with tool results. Raises ``ProviderError``; a malformed transcript
        raises ``InvalidInputError`` before anything is sent."""
        ...


def check_transcript(messages: Sequence[AgentMessage]) -> None:
    """Raise InvalidInputError unless ``messages`` start with the user, alternate
    between the model and the user or tool results, answer each assistant
    message's tool uses exactly and in order, and end waiting for the model."""
    if not messages:
        raise InvalidInputError("an agent transcript needs at least the user's message")
    if messages[0].role != "user":
        raise InvalidInputError("an agent transcript starts with the user")
    previous: AgentMessage | None = None
    for i, message in enumerate(messages):
        if previous is not None:
            if message.role == "assistant" and previous.role == "assistant":
                raise InvalidInputError(f"message {i}: two assistant messages in a row")
            if message.role != "assistant" and previous.role != "assistant":
                raise InvalidInputError(
                    f"message {i}: the model must speak between {message.role}s"
                )
            if previous.role == "assistant" and previous.tool_uses and message.role != "tool":
                raise InvalidInputError(f"message {i}: tool uses must be answered next")
            if previous.role == "assistant" and not previous.tool_uses and message.role == "tool":
                raise InvalidInputError(f"message {i}: tool results without tool uses")
        if message.role == "tool":
            asked = [u.id for u in previous.tool_uses] if previous is not None else []
            answered = [o.id for o in message.outcomes]
            if asked != answered:
                raise InvalidInputError(
                    f"message {i}: tool results {answered} do not answer the tool uses {asked}"
                )
        elif message.outcomes:
            raise InvalidInputError(f"message {i}: only a tool message carries tool results")
        if message.role != "assistant" and (message.tool_uses or message.reasoning):
            raise InvalidInputError(f"message {i}: only the assistant asks for tools")
        previous = message
    if messages[-1].role == "assistant":
        raise InvalidInputError("an agent transcript sent to a model must not end with it")


def parse_arguments(raw: Any) -> Any:
    """Arguments sent as JSON text (chat completions): the parsed value, or the
    text itself when it is not JSON."""
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a string")
    return value
