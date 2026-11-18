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
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from evalkit.errors import InvalidInputError
from evalkit.pricing import PriceBook
from evalkit.providers.anthropic import ClaudeClient
from evalkit.providers.registry import ANTHROPIC, PROVIDER_NAME, ProviderRegistry
from evalkit.scorers.base import Scorer
from evalkit.scorers.builtin import ExactMatch, JsonSchemaMatch, NumericMatch, RegexMatch, TokenF1
from evalkit.scorers.faithfulness import (
    DEFAULT_BATCH,
    MAX_BATCH,
    CitationAccuracyJudge,
    FaithfulnessJudge,
)
from evalkit.scorers.judge import LLMJudge
from evalkit.scorers.rag import (
    AnswerCompletenessJudge,
    ContextRecallJudge,
    ContextRelevanceJudge,
    NoiseSensitivityJudge,
)
from evalkit.scorers.ranking import METRICS as RANKING_METRICS
from evalkit.scorers.ranking import Metric as RankingMetricName
from evalkit.scorers.ranking import RankingMetric, default_name
from evalkit.scorers.trajectory import (
    MAX_EXPECTED_CALLS,
    ForbiddenTools,
    StepEfficiency,
    ToolCallMatch,
    TrajectoryJudge,
)
from evalkit.targets.retrieval import MAX_GRADE

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


class ToolCallsSpec(_Spec):
    type: Literal["tool_calls"]
    # The expected calls for every case; without them, each case's own
    # (``expected[expected_key]``, or ``expected`` itself when the key is null).
    calls: list[JsonValue] | None = Field(default=None, max_length=MAX_EXPECTED_CALLS)
    expected_key: str | None = "tool_calls"
    order: Literal["ordered", "unordered"] = "ordered"
    arguments: Literal["exact", "subset", "schema", "ignore"] = "subset"
    allow_extra: bool = True
    only_ok: bool = False


class StepEfficiencySpec(_Spec):
    type: Literal["step_efficiency"]
    budget: int | None = Field(default=None, ge=0, le=10_000)
    count: Literal["tool_calls", "model_steps"] = "tool_calls"
    budget_key: str | None = "optimal_steps"
    calls_key: str | None = "tool_calls"


class ForbiddenToolsSpec(_Spec):
    type: Literal["forbidden_tools"]
    tools: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(
        default_factory=list, max_length=1000
    )
    expected_key: str | None = "forbidden_tools"


class TrajectoryJudgeSpec(_Spec):
    type: Literal["trajectory_judge"]
    provider: str = Field(default=ANTHROPIC, pattern=PROVIDER_NAME)
    rubric: str = Field(min_length=1, max_length=20_000)
    model: str = Field(default="claude-opus-5-5", min_length=1, max_length=64)
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_tokens: int = Field(default=4096, ge=256, le=16_000)


class RankingSpec(_Spec):
    """A ranking metric over a retrieval target's output (``evalkit.scorers.ranking``).
    Its name defaults to ``recall_at_10``, ``ndcg_at_5``, ``mrr`` and so on."""

    type: Literal["ranking"]
    metric: RankingMetricName
    # The cut-off: 10 by default, except for MRR, which reads the whole ranking.
    k: int | None = Field(default=None, ge=1, le=1000)
    expected_key: str | None = "relevance"
    min_grade: int = Field(default=1, ge=1, le=MAX_GRADE)
    gain: Literal["linear", "exponential"] = "linear"

    @model_validator(mode="before")
    @classmethod
    def _named(cls, data: Any) -> Any:
        """Fill in the cut-off and the name, so the run stores the metric under the
        name it is reported and gated by."""
        if not isinstance(data, dict) or data.get("metric") not in RANKING_METRICS:
            return data
        data = dict(data)
        if data.get("k") is None and data["metric"] != "mrr":
            data["k"] = 10
        if data.get("name") is None and isinstance(data.get("k"), int | None):
            data["name"] = default_name(data["metric"], data.get("k"))
        return data


class FaithfulnessSpec(_Spec):
    """Claim-level faithfulness of the answer to the case's sources
    (``evalkit.scorers.faithfulness``)."""

    type: Literal["faithfulness"]
    provider: str = Field(default=ANTHROPIC, pattern=PROVIDER_NAME)
    model: str = Field(default="claude-opus-5-5", min_length=1, max_length=64)
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_tokens: int = Field(default=4096, ge=256, le=16_000)
    # Where the sources are: this key of the case's input, or else of its expected value.
    contexts_key: str = Field(default="contexts", min_length=1, max_length=200)
    # Claims checked per judge call.
    batch_size: int = Field(default=DEFAULT_BATCH, ge=1, le=MAX_BATCH)


class CitationAccuracySpec(_Spec):
    """The share of the answer's citations whose source supports the sentence."""

    type: Literal["citation_accuracy"]
    provider: str = Field(default=ANTHROPIC, pattern=PROVIDER_NAME)
    model: str = Field(default="claude-opus-5-5", min_length=1, max_length=64)
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_tokens: int = Field(default=4096, ge=256, le=16_000)
    contexts_key: str = Field(default="contexts", min_length=1, max_length=200)
    batch_size: int = Field(default=DEFAULT_BATCH, ge=1, le=MAX_BATCH)
    # An answer that cites nothing scores 0 instead of being unscorable.
    require_citations: bool = False


class _RagSpec(_Spec):
    """The options every RAG judge (``evalkit.scorers.rag``) takes."""

    provider: str = Field(default=ANTHROPIC, pattern=PROVIDER_NAME)
    model: str = Field(default="claude-opus-5-5", min_length=1, max_length=64)
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_tokens: int = Field(default=4096, ge=256, le=16_000)
    contexts_key: str = Field(default="contexts", min_length=1, max_length=200)
    batch_size: int = Field(default=DEFAULT_BATCH, ge=1, le=MAX_BATCH)


class ContextRelevanceSpec(_RagSpec):
    """Relevant sources / sources."""

    type: Literal["context_relevance"]
    # "output": the sources are the target's output (a retrieval target's ranking).
    contexts_from: Literal["case", "output"] = "case"


class ContextRecallSpec(_RagSpec):
    """Reference claims the sources support / reference claims."""

    type: Literal["context_recall"]
    contexts_from: Literal["case", "output"] = "case"
    reference_key: str | None = Field(default="answer", min_length=1, max_length=200)


class AnswerCompletenessSpec(_RagSpec):
    """Reference claims the answer states / reference claims."""

    type: Literal["answer_completeness"]
    reference_key: str | None = Field(default="answer", min_length=1, max_length=200)


class NoiseSensitivitySpec(_RagSpec):
    """1 - claims that are wrong and taken from a source / claims."""

    type: Literal["noise_sensitivity"]
    reference_key: str | None = Field(default="answer", min_length=1, max_length=200)


class PluginSpec(_Spec):
    type: Literal["plugin"]
    entry: str = Field(min_length=1, max_length=200)
    options: dict[str, JsonValue] = Field(default_factory=dict)


ScorerSpec = Annotated[
    ExactSpec
    | RegexSpec
    | JsonSchemaSpec
    | NumericSpec
    | TokenF1Spec
    | JudgeSpec
    | ToolCallsSpec
    | StepEfficiencySpec
    | ForbiddenToolsSpec
    | TrajectoryJudgeSpec
    | RankingSpec
    | FaithfulnessSpec
    | CitationAccuracySpec
    | ContextRelevanceSpec
    | ContextRecallSpec
    | AnswerCompletenessSpec
    | NoiseSensitivitySpec
    | PluginSpec,
    Field(discriminator="type"),
]
# What ``evalkit calibrate`` accepts in its [judge] table.
AnyJudgeSpec = Annotated[JudgeSpec | TrajectoryJudgeSpec, Field(discriminator="type")]


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
        providers = ProviderRegistry({}, PriceBook(), claude=claude)
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
            case JudgeSpec() | TrajectoryJudgeSpec() | FaithfulnessSpec() | CitationAccuracySpec():
                return judge_from_spec(spec, providers, name)
            case (
                ContextRelevanceSpec()
                | ContextRecallSpec()
                | AnswerCompletenessSpec()
                | NoiseSensitivitySpec()
            ):
                return rag_judge_from_spec(spec, providers, name)
            case ToolCallsSpec():
                return ToolCallMatch(
                    name,
                    calls=spec.calls,
                    expected_key=spec.expected_key,
                    order=spec.order,
                    arguments=spec.arguments,
                    allow_extra=spec.allow_extra,
                    only_ok=spec.only_ok,
                )
            case StepEfficiencySpec():
                return StepEfficiency(
                    name,
                    budget=spec.budget,
                    count=spec.count,
                    budget_key=spec.budget_key,
                    calls_key=spec.calls_key,
                )
            case ForbiddenToolsSpec():
                return ForbiddenTools(name, tools=spec.tools, expected_key=spec.expected_key)
            case RankingSpec():
                return RankingMetric(
                    name,
                    metric=spec.metric,
                    k=spec.k,
                    expected_key=spec.expected_key,
                    min_grade=spec.min_grade,
                    gain=spec.gain,
                )
            case PluginSpec():
                return _load_plugin(spec, name, allow_import_paths=allow_import_paths)
    except SchemaError as exc:
        raise InvalidInputError(f"scorer {name}: invalid JSON schema: {exc.message}") from None
    except regex.error as exc:
        raise InvalidInputError(f"scorer {name}: invalid pattern: {exc}") from None
    except ValueError as exc:
        raise InvalidInputError(f"scorer {name}: {exc}") from None
    raise InvalidInputError(f"unknown scorer type for {name}")  # pragma: no cover


AnySourcesSpec = FaithfulnessSpec | CitationAccuracySpec


def judge_from_spec(
    spec: JudgeSpec | TrajectoryJudgeSpec | AnySourcesSpec,
    providers: ProviderRegistry,
    name: str = "",
) -> LLMJudge:
    name = name or spec.resolved_name()
    try:
        client = providers.get(spec.provider)
        endpoint = providers.endpoint(spec.provider)
    except InvalidInputError as exc:
        raise InvalidInputError(f"scorer {name}: {exc.message}") from None
    if isinstance(spec, FaithfulnessSpec):
        return FaithfulnessJudge(
            name,
            client=client,
            endpoint=endpoint,
            model=spec.model,
            effort=spec.effort,
            max_tokens=spec.max_tokens,
            contexts_key=spec.contexts_key,
            batch_size=spec.batch_size,
        )
    if isinstance(spec, CitationAccuracySpec):
        return CitationAccuracyJudge(
            name,
            client=client,
            endpoint=endpoint,
            model=spec.model,
            effort=spec.effort,
            max_tokens=spec.max_tokens,
            contexts_key=spec.contexts_key,
            batch_size=spec.batch_size,
            require_citations=spec.require_citations,
        )
    if isinstance(spec, TrajectoryJudgeSpec):
        return TrajectoryJudge(
            name,
            client=client,
            endpoint=endpoint,
            rubric=spec.rubric,
            model=spec.model,
            effort=spec.effort,
            max_tokens=spec.max_tokens,
        )
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


AnyRagSpec = (
    ContextRelevanceSpec | ContextRecallSpec | AnswerCompletenessSpec | NoiseSensitivitySpec
)


def rag_judge_from_spec(spec: AnyRagSpec, providers: ProviderRegistry, name: str = "") -> LLMJudge:
    name = name or spec.resolved_name()
    try:
        client = providers.get(spec.provider)
        endpoint = providers.endpoint(spec.provider)
    except InvalidInputError as exc:
        raise InvalidInputError(f"scorer {name}: {exc.message}") from None
    common: dict[str, Any] = {
        "client": client,
        "endpoint": endpoint,
        "model": spec.model,
        "effort": spec.effort,
        "max_tokens": spec.max_tokens,
        "contexts_key": spec.contexts_key,
        "batch_size": spec.batch_size,
    }
    match spec:
        case ContextRelevanceSpec():
            return ContextRelevanceJudge(name, contexts_from=spec.contexts_from, **common)
        case ContextRecallSpec():
            return ContextRecallJudge(
                name,
                contexts_from=spec.contexts_from,
                reference_key=spec.reference_key,
                **common,
            )
        case AnswerCompletenessSpec():
            return AnswerCompletenessJudge(name, reference_key=spec.reference_key, **common)
    return NoiseSensitivityJudge(name, reference_key=spec.reference_key, **common)


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
