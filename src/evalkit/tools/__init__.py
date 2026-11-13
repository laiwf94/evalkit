"""Tools an agent under test may call, answered by a sandbox instead of real systems."""

from evalkit.tools.sandbox import (
    CallStatus,
    FixtureRecord,
    Fixtures,
    RecordReport,
    SandboxSpec,
    ScriptedResponse,
    ToolCall,
    ToolSandbox,
    ToolSpec,
    load_sandbox,
    parse_calls,
    record,
    sandbox_identity,
)

__all__ = [
    "CallStatus",
    "FixtureRecord",
    "Fixtures",
    "RecordReport",
    "SandboxSpec",
    "ScriptedResponse",
    "ToolCall",
    "ToolSandbox",
    "ToolSpec",
    "load_sandbox",
    "parse_calls",
    "record",
    "sandbox_identity",
]
