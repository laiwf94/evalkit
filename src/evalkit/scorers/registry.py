"""Scorer specs from run configs, and how they become scorer objects.

Plugins are found two ways: an installed package that declares an
``evalkit.scorers`` entry point, or an explicit ``module:attribute`` path.
Import paths execute arbitrary code, so only the CLI (which runs with the
operator's own config) allows them; the API accepts entry-point names only.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from importlib.metadata import entry_points
from typing import Annotated, Any, Literal

import regex
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from evalkit.errors import InvalidInputError
from evalkit.pricing import PriceTable
from evalkit.providers.anthropic import ClaudeClient
from evalkit.providers.registry import ANTHROPIC, PROVIDER_NAME, ProviderRegistry
from evalkit.scorers.base import Scorer
from evalkit.scorers.builtin import ExactMatch, JsonSchemaMatch, NumericMatch, RegexMatch, TokenF1
from evalkit.scorers.judge import LLMJudge

ENTRY_POINT_GROUP = "evalkit.scorers"
Name = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")]


class _Spec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    type: str
    name: Name | None = None
    gate: bool = True

    def resolved_name(self) -> str:
        return self.name or self.type


class ExactSpec(_Spec):
    type: Literal["exact"]
    case_sensitive: bool = True
    normalize_whitespace: bool = False
    expected_key: str | None = None


class RegexSpec(_Spec):
    type: Literal["regex"]
    pattern: str | None = Field(default=None, min_length=1, max_length=1000)
    ignore_case: bool = False
    full_match: bool = False


class JsonSchemaSpec(_Spec):
    type: Literal["json_schema"]
    json_schema: dict[str, JsonValue] | None = Field(default=None, alias="schema")


class NumericSpec(_Spec):
    type: Literal["numeric"]
    abs_tol: float = Field(default=0.0, ge=0)
    rel_tol: float = Field(default=1e-9, ge=0, le=1)
    extract: Literal["first", "last", "whole"] = "last"
    expected_key: str | None = None


class TokenF1Spec(_Spec):
    type: Literal["token_f1"]
    expected_key: str | None = None


class JudgeSpec(_Spec):
    type: Literal["judge"]
    # A name from the provider registry: anthropic, openai, gemini or a configured server.
    provider: str = Field(default=ANTHROPIC, pattern=PROVIDER_NAME)
    mode: Literal["rubric", "pairwise"] = "rubric"
    rubric: str = Field(min_length=1, max_length=20_000)
    model: str = Field(default="claude-opus-5-5", min_length=1, max_length=64)
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_tokens: int = Field(default=4096, ge=256, le=16_000)


class PluginSpec(_Spec):
    type: Literal["plugin"]
    entry: str = Field(min_length=1, max_length=200)
    options: dict[str, JsonValue] = Field(default_factory=dict)


ScorerSpec = Annotated[
    ExactSpec | RegexSpec | JsonSchemaSpec | NumericSpec | TokenF1Spec | JudgeSpec | PluginSpec,
    Field(discriminator="type"),
]


def build_scorers(
    specs: Sequence[ScorerSpec],
    *,
    allow_import_paths: bool,
    claude: ClaudeClient | None = None,
    providers: ProviderRegistry | None = None,
) -> list[Scorer]:
    """Scorer objects for ``specs``. Judges get their client from ``providers``;
    without one, only Claude judges can be built, on ``claude``."""
    if providers is None:
        providers = ProviderRegistry({}, PriceTable(), claude=claude)
    if not specs:
        raise InvalidInputError("at least one scorer is required")
    names = [s.resolved_name() for s in specs]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise InvalidInputError(f"scorer names must be unique; repeated: {', '.join(duplicates)}")
    return [
        _build(spec, allow_import_paths=allow_import_paths, providers=providers) for spec in specs
    ]


def _build(spec: ScorerSpec, *, allow_import_paths: bool, providers: ProviderRegistry) -> Scorer:
    name = spec.resolved_name()
    try:
        match spec:
            case ExactSpec():
                return ExactMatch(
                    name,
                    case_sensitive=spec.case_sensitive,
                    normalize_whitespace=spec.normalize_whitespace,
                    expected_key=spec.expected_key,
                )
            case RegexSpec():
                return RegexMatch(
                    name,
                    pattern=spec.pattern,
                    ignore_case=spec.ignore_case,
                    full_match=spec.full_match,
                )
            case JsonSchemaSpec():
                return JsonSchemaMatch(name, schema=spec.json_schema)
            case NumericSpec():
                return NumericMatch(
                    name,
                    abs_tol=spec.abs_tol,
                    rel_tol=spec.rel_tol,
                    extract=spec.extract,
                    expected_key=spec.expected_key,
                )
            case TokenF1Spec():
                return TokenF1(name, expected_key=spec.expected_key)
            case JudgeSpec():
                return judge_from_spec(spec, providers, name)
            case PluginSpec():
                return _load_plugin(spec, name, allow_import_paths=allow_import_paths)
    except SchemaError as exc:
        raise InvalidInputError(f"scorer {name}: invalid JSON schema: {exc.message}") from None
    except regex.error as exc:
        raise InvalidInputError(f"scorer {name}: invalid pattern: {exc}") from None
    raise InvalidInputError(f"unknown scorer type for {name}")  # pragma: no cover


def judge_from_spec(spec: JudgeSpec, providers: ProviderRegistry, name: str = "") -> LLMJudge:
    name = name or spec.resolved_name()
    try:
        client = providers.get(spec.provider)
        endpoint = providers.endpoint(spec.provider)
    except InvalidInputError as exc:
        raise InvalidInputError(f"scorer {name}: {exc.message}") from None
    return LLMJudge(
        name,
        client=client,
        endpoint=endpoint,
        rubric=spec.rubric,
        mode=spec.mode,
        model=spec.model,
        effort=spec.effort,
        max_tokens=spec.max_tokens,
    )


def _load_plugin(spec: PluginSpec, name: str, *, allow_import_paths: bool) -> Scorer:
    factory: Any
    if ":" in spec.entry:
        if not allow_import_paths:
            raise InvalidInputError(
                f"scorer {name}: import paths are not accepted here; "
                f"install the plugin with an '{ENTRY_POINT_GROUP}' entry point"
            )
        module_name, _, attr = spec.entry.partition(":")
        try:
            factory = getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError) as exc:
            raise InvalidInputError(f"scorer {name}: cannot load {spec.entry}: {exc}") from None
    else:
        found = [ep for ep in entry_points(group=ENTRY_POINT_GROUP) if ep.name == spec.entry]
        if not found:
            raise InvalidInputError(f"scorer {name}: no installed plugin named {spec.entry!r}")
        factory = found[0].load()
    if not (isinstance(factory, type) and issubclass(factory, Scorer)):
        raise InvalidInputError(f"scorer {name}: {spec.entry} is not a Scorer subclass")
    scorer: Scorer = factory(name, dict(spec.options))
    return scorer
