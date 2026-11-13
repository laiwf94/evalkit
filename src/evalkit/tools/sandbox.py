"""A tool sandbox: the tools an agent may call, answered without touching real systems.

An agent under test calls tools by name with JSON arguments. In an evaluation
those calls must not book real flights or send real email, and the answers must
be the same on every run, so a ``ToolSandbox`` answers them instead. Each tool
has a JSON Schema for its arguments (and optionally for its result), checked
with ``jsonschema`` (draft 2020-12) when the sandbox is loaded, and every call
is checked against it before anything answers. A call is answered by, in
order:

1. the tool's scripted ``responses``: the first rule whose ``when`` is a subset
   of the arguments and that has uses left (``times``);
2. recorded fixtures: what the real tool answered to exactly these arguments
   (canonical JSON), replayed in recorded order with the last one repeating.
   ``evalkit tools record`` makes the fixture file by sending a list of calls
   to each tool's HTTP endpoint, through the SSRF-checked client HTTP targets use;
3. otherwise the call gets a ``no_response`` error.

The agent always gets an answer, never an exception: an unknown tool, invalid
arguments, a call over the sandbox's or the tool's ``max_calls``, an injected
failure and a missing response are all error results the agent can react to,
and every call, answered or not, is kept in order in ``calls`` as the session's
trajectory. Latency (``latency_ms`` plus up to ``jitter_ms``) and failures
(``failure_rate``, or the tool's nth calls in ``fail_calls``) are injected
deterministically from the seed, the session id (the case) and the call number,
so a rerun of a case sees the same failures whatever else runs beside it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import tomllib
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from referencing.exceptions import Unresolvable

from evalkit.datasets import canonical_json, sha256_hex
from evalkit.errors import InvalidInputError

TOOL_NAME = r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$"
ENDPOINT_NAME = r"^[a-z0-9][a-z0-9_-]{0,63}$"
MAX_TOOLS = 64
MAX_ARGUMENT_BYTES = 64 * 1024
MAX_RESULT_BYTES = 1024 * 1024
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_RECORDS = 100_000
MAX_MESSAGE = 300

CallStatus = Literal[
    "ok",
    "error",
    "invalid_arguments",
    "unknown_tool",
    "limit_reached",
    "injected_failure",
    "no_response",
]
Source = Literal["script", "fixture"]
Sleep = Callable[[float], Awaitable[None]]
Backend = Callable[[str, Any], Awaitable[Any]]


def _short(text: str) -> str:
    return text if len(text) <= MAX_MESSAGE else text[: MAX_MESSAGE - 3] + "..."


def _validator(schema: dict[str, Any], what: str) -> Draft202012Validator:
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ValueError(f"{what} is not a valid JSON Schema: {_short(exc.message)}") from None
    return Draft202012Validator(schema)


def _violation(validator: Draft202012Validator, value: Any) -> str | None:
    """The first way ``value`` breaks the schema, as ``path: message``, or None."""
    try:
        errors = sorted(validator.iter_errors(value), key=lambda e: list(e.absolute_path))
    except Unresolvable as exc:
        return f"the schema has a reference that cannot be resolved: {exc}"
    if not errors:
        return None
    first = errors[0]
    where = "/".join(str(p) for p in first.absolute_path) or "(root)"
    return _short(f"{where}: {first.message}")


def _uniform(*parts: object) -> float:
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _matches(pattern: Any, value: Any) -> bool:
    """``pattern`` is a subset of ``value``: every key of a pattern object is in the
    value and matches recursively; anything else must be equal, type included."""
    if isinstance(pattern, dict):
        return isinstance(value, dict) and all(
            k in value and _matches(v, value[k]) for k, v in pattern.items()
        )
    if isinstance(pattern, bool) or isinstance(value, bool):
        return type(pattern) is type(value) and pattern == value
    if isinstance(pattern, list):
        return (
            isinstance(value, list)
            and len(pattern) == len(value)
            and all(_matches(p, v) for p, v in zip(pattern, value, strict=True))
        )
    return bool(pattern == value)


def _json_size(value: Any) -> int:
    return len(canonical_json(value).encode())


class _Spec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ScriptedResponse(_Spec):
    """One rule of a tool's script: answer ``result`` (any JSON) or ``error``."""

    when: dict[str, Any] | None = None
    result: Any = None
    error: str | None = Field(default=None, min_length=1, max_length=4000)
    times: int | None = Field(default=None, ge=1, le=10_000)
    latency_ms: float | None = Field(default=None, ge=0, le=60_000)

    @model_validator(mode="after")
    def _one_answer(self) -> ScriptedResponse:
        has_result = "result" in self.model_fields_set
        if has_result == (self.error is not None):
            raise ValueError("give a response either a result or an error")
        if has_result and _json_size(self.result) > MAX_RESULT_BYTES:
            raise ValueError(f"result is larger than {MAX_RESULT_BYTES} bytes")
        return self


class ToolSpec(_Spec):
    name: str = Field(pattern=TOOL_NAME)
    description: str = Field(default="", max_length=4096)
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})
    output_schema: dict[str, Any] | None = None
    responses: list[ScriptedResponse] = Field(default_factory=list, max_length=1000)
    max_calls: int | None = Field(default=None, ge=1, le=10_000)
    latency_ms: float = Field(default=0.0, ge=0, le=60_000)
    jitter_ms: float = Field(default=0.0, ge=0, le=60_000)
    failure_rate: float = Field(default=0.0, ge=0, le=1)
    fail_calls: list[int] = Field(default_factory=list, max_length=1000)
    failure_message: str = Field(
        default="the tool is temporarily unavailable", min_length=1, max_length=4000
    )
    # Only for ``evalkit tools record``: an endpoint from EVALKIT_HTTP_TARGETS_FILE.
    endpoint: str | None = Field(default=None, pattern=ENDPOINT_NAME)

    @model_validator(mode="after")
    def _valid_schemas(self) -> ToolSpec:
        _validator(self.input_schema, "input_schema")
        if self.input_schema.get("type") != "object":
            raise ValueError('input_schema must have type "object": arguments are an object')
        if any(n < 1 for n in self.fail_calls):
            raise ValueError("fail_calls are call numbers, counted from 1")
        if self.output_schema is not None:
            output = _validator(self.output_schema, "output_schema")
            for i, rule in enumerate(self.responses):
                if rule.error is None:
                    problem = _violation(output, rule.result)
                    if problem is not None:
                        raise ValueError(f"responses[{i}].result breaks output_schema: {problem}")
        return self

    def definition(self) -> dict[str, Any]:
        """What the agent is told about the tool, in provider-neutral form."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class SandboxSpec(_Spec):
    tools: list[ToolSpec] = Field(min_length=1, max_length=MAX_TOOLS)
    # Calls in one session (one case), answered or not.
    max_calls: int = Field(default=50, ge=1, le=10_000)
    # A JSONL fixture file, relative to the sandbox file.
    fixtures: str | None = Field(default=None, min_length=1, max_length=1024)
    # Replay each fixture's recorded latency instead of the tool's latency_ms.
    replay_latency: bool = False
    seed: int = 0

    @model_validator(mode="after")
    def _unique_names(self) -> SandboxSpec:
        counts = Counter(t.name for t in self.tools)
        repeated = sorted(n for n, c in counts.items() if c > 1)
        if repeated:
            raise ValueError(f"tool names must be unique: {', '.join(repeated)}")
        return self

    def tool(self, name: str) -> ToolSpec | None:
        return next((t for t in self.tools if t.name == name), None)

    def definitions(self) -> list[dict[str, Any]]:
        return [t.definition() for t in self.tools]


@dataclass(frozen=True)
class FixtureRecord:
    """One recorded answer: ``result`` when the tool answered, else ``error``."""

    tool: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None
    latency_ms: float = 0.0

    @property
    def key(self) -> tuple[str, str]:
        return self.tool, canonical_json(self.arguments)

    def to_json(self) -> str:
        line: dict[str, Any] = {"tool": self.tool, "arguments": self.arguments}
        if self.error is None:
            line["result"] = self.result
        else:
            line["error"] = self.error
        line["latency_ms"] = round(self.latency_ms, 3)
        return canonical_json(line)


class Fixtures:
    """Recorded answers, looked up by tool and canonical arguments."""

    def __init__(self, records: Iterable[FixtureRecord] = ()) -> None:
        self.records: list[FixtureRecord] = []
        self._by_key: dict[tuple[str, str], list[FixtureRecord]] = {}
        for record in records:
            self.add(record)

    def add(self, record: FixtureRecord) -> None:
        self.records.append(record)
        self._by_key.setdefault(record.key, []).append(record)

    def __len__(self) -> int:
        return len(self.records)

    def __contains__(self, key: object) -> bool:
        return key in self._by_key

    def lookup(self, tool: str, arguments: Any, occurrence: int = 0) -> FixtureRecord | None:
        """The answer to the ``occurrence``-th (from 0) call with these arguments:
        recorded answers in order, then the last one again."""
        found = self._by_key.get((tool, canonical_json(arguments)))
        if not found:
            return None
        return found[min(occurrence, len(found) - 1)]

    def to_jsonl(self) -> str:
        return "".join(r.to_json() + "\n" for r in self.records)

    @property
    def content_hash(self) -> str:
        return sha256_hex(self.to_jsonl())

    @classmethod
    def parse(cls, data: bytes | str, spec: SandboxSpec) -> Fixtures:
        """Read a fixture file, refusing records the sandbox could never answer
        with: unknown tools, arguments or results that break the tool's schemas."""
        fixtures = cls()
        for number, line in _jsonl(data, "fixture"):
            raw = _object(line, number, "fixture", {"tool", "arguments", "result", "error"})
            tool = _known_tool(spec, raw.get("tool"), number)
            arguments = raw.get("arguments")
            problem = _arguments_problem(tool, arguments)
            if problem is not None or not isinstance(arguments, dict):
                raise InvalidInputError(f"fixture line {number}: arguments {problem}")
            error = raw.get("error")
            if ("result" in raw) == (error is not None):
                raise InvalidInputError(f"fixture line {number}: give a result or an error")
            if error is not None and (not isinstance(error, str) or not error):
                raise InvalidInputError(f"fixture line {number}: error must be a string")
            if error is None:
                problem = _result_problem(tool, raw["result"])
                if problem is not None:
                    raise InvalidInputError(f"fixture line {number}: result {problem}")
            latency = raw.get("latency_ms", 0.0)
            if isinstance(latency, bool) or not isinstance(latency, int | float) or latency < 0:
                raise InvalidInputError(f"fixture line {number}: latency_ms must be >= 0")
            fixtures.add(
                FixtureRecord(tool.name, arguments, raw.get("result"), error, float(latency))
            )
            if len(fixtures) > MAX_RECORDS:
                raise InvalidInputError(f"a fixture file holds at most {MAX_RECORDS} records")
        return fixtures


def _jsonl(data: bytes | str, what: str) -> Iterable[tuple[int, str]]:
    if isinstance(data, bytes):
        if len(data) > MAX_FILE_BYTES:
            raise InvalidInputError(f"a {what} file is at most {MAX_FILE_BYTES} bytes")
        try:
            data = data.decode("utf-8")
        except UnicodeDecodeError:
            raise InvalidInputError(f"the {what} file is not UTF-8") from None
    for number, line in enumerate(data.splitlines(), start=1):
        if line.strip():
            yield number, line


def _object(line: str, number: int, what: str, keys: set[str]) -> dict[str, Any]:
    try:
        raw = json.loads(line, parse_constant=_no_constant)
    except (ValueError, RecursionError):
        raise InvalidInputError(f"{what} line {number} is not JSON") from None
    if not isinstance(raw, dict):
        raise InvalidInputError(f"{what} line {number} must be a JSON object")
    unknown = sorted(set(raw) - keys - {"latency_ms"})
    if unknown:
        raise InvalidInputError(f"{what} line {number}: unknown key(s) {', '.join(unknown)}")
    return raw


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def _known_tool(spec: SandboxSpec, name: Any, number: int) -> ToolSpec:
    tool = spec.tool(name) if isinstance(name, str) else None
    if tool is None:
        raise InvalidInputError(f"line {number}: the sandbox has no tool {name!r}")
    return tool


def _arguments_problem(tool: ToolSpec, arguments: Any) -> str | None:
    if not isinstance(arguments, dict):
        return "must be a JSON object"
    try:
        size = _json_size(arguments)
    except ValueError:
        return "must be JSON (no NaN or infinity)"
    if size > MAX_ARGUMENT_BYTES:
        return f"are larger than {MAX_ARGUMENT_BYTES} bytes"
    return _violation(Draft202012Validator(tool.input_schema), arguments)


def _result_problem(tool: ToolSpec, result: Any) -> str | None:
    if _json_size(result) > MAX_RESULT_BYTES:
        return f"is larger than {MAX_RESULT_BYTES} bytes"
    if tool.output_schema is None:
        return None
    problem = _violation(Draft202012Validator(tool.output_schema), result)
    return None if problem is None else f"breaks output_schema: {problem}"


@dataclass(frozen=True)
class ToolCall:
    """One call in a session's trajectory, with what the agent was answered."""

    index: int
    tool: str
    arguments: Any
    status: CallStatus
    result: Any = None
    error: str | None = None
    source: Source | None = None
    latency_ms: float = 0.0

    @property
    def is_error(self) -> bool:
        return self.status != "ok"

    def content(self) -> Any:
        """What goes back to the agent: the result, or ``{"error": message}``."""
        return {"error": self.error} if self.is_error else self.result

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "tool": self.tool,
            "arguments": self.arguments,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "source": self.source,
            "latency_ms": round(self.latency_ms, 3),
        }


@dataclass
class _ToolState:
    attempts: int = 0  # calls that named this tool
    executed: int = 0  # of those, calls with valid arguments under the limits
    rule_uses: Counter[int] = field(default_factory=Counter)


class ToolSandbox:
    """The tools of one session (one case). Not shared between sessions."""

    def __init__(
        self,
        spec: SandboxSpec,
        fixtures: Fixtures | None = None,
        *,
        session: str = "",
        sleep: Sleep = asyncio.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.spec = spec
        self.fixtures = fixtures or Fixtures()
        self.session = session
        self.calls: list[ToolCall] = []
        self._sleep = sleep
        self._clock = clock
        self._tools = {t.name: t for t in spec.tools}
        self._inputs = {t.name: Draft202012Validator(t.input_schema) for t in spec.tools}
        self._state = {t.name: _ToolState() for t in spec.tools}
        self._replays: Counter[tuple[str, str]] = Counter()

    def definitions(self) -> list[dict[str, Any]]:
        return self.spec.definitions()

    async def call(self, name: str, arguments: Any) -> ToolCall:
        start = self._clock()
        index = len(self.calls) + 1
        status, result, error, source = await self._answer(index, name, arguments)
        call = ToolCall(
            index=index,
            tool=name,
            arguments=arguments,
            status=status,
            result=result,
            error=error,
            source=source,
            latency_ms=max(0.0, (self._clock() - start) * 1000),
        )
        self.calls.append(call)
        return call

    async def _answer(
        self, index: int, name: str, arguments: Any
    ) -> tuple[CallStatus, Any, str | None, Source | None]:
        if index > self.spec.max_calls:
            message = f"the session's {self.spec.max_calls} calls are used"
            return "limit_reached", None, message, None
        tool = self._tools.get(name)
        if tool is None:
            known = ", ".join(sorted(self._tools))
            return "unknown_tool", None, f"there is no tool {name!r}; tools: {known}", None
        state = self._state[name]
        state.attempts += 1
        if tool.max_calls is not None and state.attempts > tool.max_calls:
            message = f"{name} may be called at most {tool.max_calls} times"
            return "limit_reached", None, message, None
        problem = self._arguments_problem(tool, arguments)
        if problem is not None:
            return "invalid_arguments", None, f"invalid arguments for {name}: {problem}", None
        state.executed += 1
        n = state.executed
        rule = self._rule(tool, state, arguments)
        replayed = None if rule is not None else self._replay(name, arguments)
        await self._delay(tool, n, rule, replayed)
        if n in tool.fail_calls or (
            tool.failure_rate > 0
            and _uniform(self.spec.seed, self.session, name, n) < tool.failure_rate
        ):
            return "injected_failure", None, tool.failure_message, None
        if rule is not None:
            if rule.error is not None:
                return "error", None, rule.error, "script"
            return "ok", rule.result, None, "script"
        if replayed is not None:
            if replayed.error is not None:
                return "error", None, replayed.error, "fixture"
            return "ok", replayed.result, None, "fixture"
        return "no_response", None, f"{name} has no response for these arguments", None

    def _arguments_problem(self, tool: ToolSpec, arguments: Any) -> str | None:
        if not isinstance(arguments, dict):
            return "arguments must be a JSON object"
        try:
            size = _json_size(arguments)
        except (ValueError, TypeError):
            return "arguments must be JSON"
        if size > MAX_ARGUMENT_BYTES:
            return f"arguments are larger than {MAX_ARGUMENT_BYTES} bytes"
        return _violation(self._inputs[tool.name], arguments)

    @staticmethod
    def _rule(tool: ToolSpec, state: _ToolState, arguments: Any) -> ScriptedResponse | None:
        for i, rule in enumerate(tool.responses):
            if rule.times is not None and state.rule_uses[i] >= rule.times:
                continue
            if rule.when is None or _matches(rule.when, arguments):
                state.rule_uses[i] += 1
                return rule
        return None

    def _replay(self, name: str, arguments: Any) -> FixtureRecord | None:
        key = (name, canonical_json(arguments))
        found = self.fixtures.lookup(name, arguments, self._replays[key])
        if found is not None:
            self._replays[key] += 1
        return found

    async def _delay(
        self,
        tool: ToolSpec,
        n: int,
        rule: ScriptedResponse | None,
        replayed: FixtureRecord | None,
    ) -> None:
        if rule is not None and rule.latency_ms is not None:
            ms = rule.latency_ms
        elif replayed is not None and self.spec.replay_latency:
            ms = replayed.latency_ms
        else:
            jitter = _uniform("latency", self.spec.seed, self.session, tool.name, n)
            ms = tool.latency_ms + tool.jitter_ms * jitter
        if ms > 0:
            await self._sleep(ms / 1000)


def sandbox_identity(spec: SandboxSpec, fixtures: Fixtures | None) -> str:
    """A hash of everything that can change what the agent is answered: the tools
    (not the recording endpoints or the fixture file's path) and the fixtures."""
    tools = [t.model_dump(mode="json", exclude={"endpoint"}) for t in spec.tools]
    dumped = spec.model_dump(mode="json", exclude={"tools", "fixtures"})
    fixture_hash = None if fixtures is None else fixtures.content_hash
    return sha256_hex(canonical_json({**dumped, "tools": tools, "fixtures": fixture_hash}))


def load_sandbox(path: Path, *, fixtures: bool = True) -> tuple[SandboxSpec, Fixtures | None]:
    """Read a TOML file with a ``[sandbox]`` table, and its fixture file if it names
    one (a missing fixture file is an error unless ``fixtures`` is false)."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise InvalidInputError(f"cannot read {path}: {exc.strerror}") from None
    if len(data) > MAX_FILE_BYTES:
        raise InvalidInputError(f"{path} is larger than {MAX_FILE_BYTES} bytes")
    try:
        raw = tomllib.loads(data.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise InvalidInputError(f"{path} is not valid TOML: {exc}") from None
    table = raw.get("sandbox")
    if not isinstance(table, dict):
        raise InvalidInputError(f"{path} has no [sandbox] table")
    try:
        spec = SandboxSpec.model_validate(table)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        raise InvalidInputError(f"{path}: sandbox.{where}: {first['msg']}") from None
    if spec.fixtures is None or not fixtures:
        return spec, None
    fixture_path = fixtures_path(path, spec)
    try:
        content = fixture_path.read_bytes()
    except OSError as exc:
        raise InvalidInputError(f"cannot read {fixture_path}: {exc.strerror}") from None
    return spec, Fixtures.parse(content, spec)


def fixtures_path(sandbox_file: Path, spec: SandboxSpec) -> Path:
    if spec.fixtures is None:
        raise InvalidInputError("the sandbox names no fixture file")
    return sandbox_file.parent / spec.fixtures


def parse_calls(data: bytes | str, spec: SandboxSpec) -> list[tuple[str, dict[str, Any]]]:
    """A calls file for ``evalkit tools record``: JSONL of ``{"tool", "arguments"}``,
    each checked against the tool's input schema before anything is called."""
    calls = []
    for number, line in _jsonl(data, "calls"):
        raw = _object(line, number, "calls", {"tool", "arguments"})
        tool = _known_tool(spec, raw.get("tool"), number)
        problem = _arguments_problem(tool, raw.get("arguments"))
        if problem is not None:
            raise InvalidInputError(f"calls line {number}: arguments {problem}")
        calls.append((tool.name, raw["arguments"]))
        if len(calls) > MAX_RECORDS:
            raise InvalidInputError(f"a calls file holds at most {MAX_RECORDS} calls")
    return calls


@dataclass
class RecordReport:
    recorded: int = 0
    skipped: int = 0
    failed: list[str] = field(default_factory=list)


async def record(
    spec: SandboxSpec,
    calls: Sequence[tuple[str, dict[str, Any]]],
    backends: Mapping[str, Backend],
    existing: Fixtures | None = None,
    *,
    refresh: bool = False,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[Fixtures, RecordReport]:
    """Call the real tools and keep what they answer as fixtures.

    ``backends`` maps a tool name to a function that sends one call and returns
    the endpoint's JSON document, ``{"result": ...}`` or ``{"error": "..."}``.
    Calls whose tool and arguments are already in ``existing`` are skipped (so
    an interrupted recording resumes where it stopped) unless ``refresh``; with
    ``refresh`` those records are dropped and the calls made again. A call that
    fails, or whose answer breaks the tool's output schema, is reported and not
    recorded."""
    report = RecordReport()
    wanted = {(tool, canonical_json(args)) for tool, args in calls}
    kept = [] if existing is None else existing.records
    if refresh:
        kept = [r for r in kept if r.key not in wanted]
    fixtures = Fixtures(kept)
    done = {r.key for r in kept}
    for tool_name, arguments in calls:
        key = (tool_name, canonical_json(arguments))
        if key in done:
            report.skipped += 1
            continue
        tool = spec.tool(tool_name)
        backend = backends.get(tool_name)
        if tool is None or backend is None:
            report.failed.append(f"{tool_name}: no endpoint to record from")
            continue
        start = clock()
        try:
            document = await backend(tool_name, arguments)
        except Exception as exc:
            message = getattr(exc, "message", None) or f"{type(exc).__name__}: {exc}"
            report.failed.append(_short(f"{tool_name} {key[1]}: {message}"))
            continue
        latency = max(0.0, (clock() - start) * 1000)
        problem = _answer_problem(tool, document)
        if problem is not None:
            report.failed.append(_short(f"{tool_name} {key[1]}: {problem}"))
            continue
        error = document.get("error")
        fixtures.add(FixtureRecord(tool_name, arguments, document.get("result"), error, latency))
        done.add(key)
        report.recorded += 1
    return fixtures, report


def _answer_problem(tool: ToolSpec, document: Any) -> str | None:
    if not isinstance(document, dict) or ("result" in document) == ("error" in document):
        return 'the endpoint must answer {"result": ...} or {"error": "..."}'
    error = document.get("error")
    if "error" in document:
        return None if isinstance(error, str) and error else "error must be a string"
    problem = _result_problem(tool, document["result"])
    return None if problem is None else f"result {problem}"
