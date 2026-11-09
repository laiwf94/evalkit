"""Judges on any provider, judge identities stored with runs, and the gate that
refuses to compare runs whose judges differ."""

import asyncio
import functools
import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import httpx2
import pytest
from pydantic import SecretStr, ValidationError
from sqlalchemy import inspect, update
from sqlalchemy.orm import Session, sessionmaker

from evalkit import cli
from evalkit.api import create_app
from evalkit.config import ConfigError, Settings
from evalkit.datasets import Case, canonical_json, import_dataset, sha256_hex
from evalkit.db import make_engine, upgrade
from evalkit.db.models import Run
from evalkit.errors import InvalidInputError, PreconditionError
from evalkit.estimate import estimate
from evalkit.pricing import ModelPrice, PriceBook, PriceTable, estimate_tokens
from evalkit.providers.base import Completion
from evalkit.providers.gemini import GeminiClient
from evalkit.providers.openai_compat import OpenAICompatClient
from evalkit.providers.registry import ProviderConfig, ProviderRegistry
from evalkit.ratelimit import RetryPolicy
from evalkit.report import compare_runs, run_judges, summarize
from evalkit.runner import Runner
from evalkit.runspec import RunConfig
from evalkit.scorers import build_scorers
from evalkit.scorers.judge import RUBRIC_SCHEMA, LLMJudge, identity_changes, judge_identity
from evalkit.targets.base import Usage
from tests.fake_anthropic import FakeAnthropic, json_reply, message
from tests.fake_gemini import FakeGemini, generated
from tests.fake_openai import FakeOpenAI, chat_response

PRICES = {
    "gpt-test": ModelPrice(input=2.0, output=8.0),
    "gemini-test": ModelPrice(input=1.0, output=4.0),
}
RUBRIC = "The answer is correct."
CASE = Case(id="c1", input={"q": "capital of France?"}, expected="Paris")
LOCAL = ProviderConfig(kind="openai_compat", base_url="http://llm.test/v1")
GEM = ProviderConfig(kind="gemini", base_url="http://gemini.test/v1beta")


class Server:
    """The OpenAI-compatible and Gemini fakes behind one transport, answering
    every judge with the same verdict. Each fake checks the exact shape of the
    requests it gets; ``no_rejected_requests`` fails a test that sent a bad one."""

    instances: ClassVar[list["Server"]] = []

    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []
        self.score = 5
        self.finish = {"llm.test": "stop", "gemini.test": "STOP"}
        self.openai = FakeOpenAI("http://llm.test/v1", api_key=None, answer=self._chat)
        self.gemini = FakeGemini("http://gemini.test/v1beta", api_key=None, answer=self._generate)
        Server.instances.append(self)

    def _verdict(self) -> str:
        return json.dumps({"reasoning": "fine", "score": self.score})

    def _chat(self, body: dict[str, Any]) -> httpx2.Response:
        return chat_response(self._verdict(), model="gpt-test", finish=self.finish["llm.test"])

    def _generate(self, model: str, body: dict[str, Any]) -> httpx2.Response:
        return generated(self._verdict(), model=model, finish=self.finish["gemini.test"])

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        fake = self.gemini if request.url.host == "gemini.test" else self.openai
        return fake.handle(request)

    def bodies(self, host: str) -> list[Any]:
        return [json.loads(r.content) for r in self.requests if r.url.host == host]


@pytest.fixture(autouse=True)
def no_rejected_requests() -> Iterator[None]:
    Server.instances.clear()
    yield
    for server in Server.instances:
        assert server.openai.rejected == [], server.openai.rejected
        assert server.gemini.rejected == [], server.gemini.rejected


def registry(server: Server, **configs: ProviderConfig) -> ProviderRegistry:
    return ProviderRegistry(
        configs or {"local": LOCAL, "gem": GEM},
        PriceBook(PRICES),
        claude=FakeAnthropic(json_reply(message('{"reasoning": "ok", "score": 3}'))).client(),
        transport=httpx2.MockTransport(server.handle),
    )


def judge_scorer(provider: str, model: str, **extra: Any) -> dict[str, Any]:
    return {
        "type": "judge",
        "name": "quality",
        "provider": provider,
        "model": model,
        "rubric": RUBRIC,
        **extra,
    }


def run_config(*scorers: dict[str, Any], label: str = "r") -> RunConfig:
    return RunConfig.model_validate(
        {
            "label": label,
            "dataset": {"name": "qa"},
            "target": {"type": "fake", "accuracy": 0.8, "seed": 1},
            "scorers": [{"type": "exact"}, *scorers],
            "options": {"concurrency": 4},
        }
    )


@pytest.fixture
def dataset(sessions: sessionmaker[Session]) -> None:
    cases = [Case(id=f"c{i:02d}", input={"q": f"q{i}"}, expected=f"a{i}") for i in range(12)]
    with sessions() as s:
        import_dataset(s, "qa", cases)


def settings() -> Settings:
    return Settings.from_env({}).model_copy(update={"prices": PRICES})


def execute(sessions: sessionmaker[Session], providers: ProviderRegistry, config: RunConfig) -> str:
    runner = Runner(sessions, settings(), providers=providers)
    run_id = runner.create_run(config).id
    asyncio.run(runner.execute(run_id))
    return run_id


# Judges on OpenAI-compatible servers and Gemini


def http_judge(server: Server, provider: str, model: str, **kw: Any) -> LLMJudge:
    reg = registry(server)
    return LLMJudge(
        "quality",
        client=reg.get(provider),
        endpoint=reg.endpoint(provider),
        rubric=RUBRIC,
        model=model,
        retry=RetryPolicy(max_attempts=1, base_delay=0),
        **kw,
    )


def test_openai_compatible_judge_sends_the_rubric_and_a_strict_schema() -> None:
    server = Server()
    server.score = 4
    judge = http_judge(server, "local", "gpt-test")
    result = asyncio.run(judge.score(CASE, "Paris."))
    assert result.value == 0.75
    assert result.detail == "4/5: fine"
    assert result.cost_usd == pytest.approx((100 * 2.0 + 10 * 8.0) / 1e6)
    (prompt,) = judge.prompts(CASE, "Paris.")
    assert "<rubric>\nThe answer is correct.\n</rubric>" in judge.system
    assert prompt.endswith("<response>\nParis.\n</response>")
    assert server.bodies("llm.test") == [
        {
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": judge.system},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": judge.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": RUBRIC_SCHEMA, "strict": True},
            },
        }
    ]


def test_gemini_judge_in_pairwise_mode() -> None:
    server = Server()
    tie = json.dumps({"reasoning": "same", "winner": "tie"})
    server.gemini.answer = lambda model, body: generated(
        tie, model=model, prompt_tokens=50, output_tokens=5
    )
    judge = http_judge(server, "gem", "gemini-test", mode="pairwise")
    result = asyncio.run(judge.score(CASE, "Paris"))
    assert result.value == 0.5
    bodies = server.bodies("gemini.test")
    assert len(bodies) == 2
    assert bodies[0]["generationConfig"]["responseJsonSchema"]["required"] == [
        "reasoning",
        "winner",
    ]
    assert server.requests[0].url.path == "/v1beta/models/gemini-test:generateContent"


@pytest.mark.parametrize(
    ("provider", "host", "finish"),
    [("local", "llm.test", "length"), ("gem", "gemini.test", "MAX_TOKENS")],
)
def test_a_truncated_verdict_is_reported_on_every_provider(
    provider: str, host: str, finish: str
) -> None:
    server = Server()
    server.finish[host] = finish
    model = "gpt-test" if provider == "local" else "gemini-test"
    result = asyncio.run(http_judge(server, provider, model).score(CASE, "Paris"))
    assert result.value is None
    assert result.detail == "judge ran out of output tokens; raise max_tokens"


def test_completion_stop_reasons_are_read_the_same_way_for_every_provider() -> None:
    def c(reason: str | None) -> Completion:
        return Completion(text="", stop_reason=reason, usage=Usage(), model="m")

    assert all(c(r).finished for r in ("end_turn", "stop", "STOP"))
    assert all(c(r).truncated for r in ("max_tokens", "length", "MAX_TOKENS"))
    assert not c("refusal").finished
    assert not c(None).truncated


def test_judge_rate_limits_are_kept_per_provider() -> None:
    server = Server()
    assert http_judge(server, "local", "gpt-test").limiter is not None
    reg = registry(server)
    from evalkit.ratelimit import Limiters

    limiters = Limiters()
    local = LLMJudge("a", client=reg.get("local"), rubric="r", model="gpt-test", limiters=limiters)
    gem = LLMJudge("b", client=reg.get("gem"), rubric="r", model="gemini-test", limiters=limiters)
    assert local.limiter is limiters.get("local:judge")
    assert gem.limiter is limiters.get("gem:judge")
    assert local.limiter is not gem.limiter


# Identity and fingerprints


def test_claude_judge_fingerprints_are_unchanged() -> None:
    """Existing cached verdicts and calibrations stay valid for Claude judges."""
    judge = LLMJudge("q", client=FakeAnthropic().client(), rubric=RUBRIC, effort="low")
    old = sha256_hex(
        canonical_json(
            {"mode": "rubric", "model": "claude-opus-5-5", "effort": "low", "rubric": RUBRIC}
        )
    )
    assert judge.fingerprint() == old
    assert judge.identity() == {
        "provider": "anthropic",
        "endpoint": None,
        "model": "claude-opus-5-5",
        "mode": "rubric",
        "effort": "low",
        "rubric_sha256": sha256_hex(RUBRIC),
        "fingerprint": old,
    }
    assert judge.spec()["provider"] == "anthropic"


def test_other_providers_never_share_a_fingerprint() -> None:
    server = Server()
    same_model = PriceTable({"m": ModelPrice(input=1, output=1)})
    a = OpenAICompatClient(same_model, name="a", transport=httpx2.MockTransport(server.handle))
    b = GeminiClient(same_model, name="b", transport=httpx2.MockTransport(server.handle))
    prints = {
        LLMJudge("q", client=a, rubric="r", model="m", endpoint="x").fingerprint(),
        LLMJudge("q", client=a, rubric="r", model="m", endpoint="y").fingerprint(),
        LLMJudge("q", client=b, rubric="r", model="m", endpoint="x").fingerprint(),
    }
    assert len(prints) == 3


def test_identity_changes_name_each_field() -> None:
    base = judge_identity(
        provider="local", endpoint="e1", model="m1", mode="rubric", effort="low", rubric="r1"
    )
    assert identity_changes(base, dict(base)) == []
    assert identity_changes(None, None) == []
    assert identity_changes(base, None) == ["a judge in one run only"]
    assert identity_changes(None, base) == ["a judge in one run only"]
    cand = judge_identity(
        provider="gem", endpoint="e2", model="m2", mode="pairwise", effort="high", rubric="r2"
    )
    assert identity_changes(base, cand) == [
        "provider local -> gem",
        "endpoint e1 -> e2",
        "model m1 -> m2",
        "mode rubric -> pairwise",
        "effort low -> high",
        "rubric changed",
    ]


# The registry and its configuration


def test_registry_builds_each_provider_once_and_closes_what_it_built() -> None:
    server = Server()
    claude = FakeAnthropic().client()
    reg = ProviderRegistry(
        {"local": LOCAL, "gem": GEM},
        PriceBook(),
        claude=claude,
        transport=httpx2.MockTransport(server.handle),
    )
    assert reg.names() == ["anthropic", "gem", "local"]
    assert reg.get("anthropic") is claude
    assert reg.endpoint("anthropic") is None
    local = reg.get("local")
    assert isinstance(local, OpenAICompatClient)
    assert local.name == "local"
    assert reg.get("local") is local
    assert isinstance(reg.get("gem"), GeminiClient)
    assert reg.endpoint("local") == "openai_compat:http://llm.test/v1"
    assert reg.endpoint("gem") == "gemini:http://gemini.test/v1beta"
    asyncio.run(reg.aclose())
    assert reg.get("local") is not local  # rebuilt after close


def test_registry_errors() -> None:
    reg = ProviderRegistry(Settings.from_env({}).provider_configs(), PriceBook())
    with pytest.raises(InvalidInputError, match="ANTHROPIC_API_KEY"):
        reg.get("anthropic")
    with pytest.raises(InvalidInputError, match="unknown provider 'nope'; configured: anthropic"):
        reg.get("nope")
    with pytest.raises(InvalidInputError, match="unknown provider"):
        reg.endpoint("nope")
    with pytest.raises(InvalidInputError, match="set OPENAI_API_KEY"):
        reg.get("openai")
    with pytest.raises(InvalidInputError, match="set GEMINI_API_KEY"):
        reg.get("gemini")
    with pytest.raises(InvalidInputError, match="reserved"):
        ProviderRegistry({"anthropic": LOCAL}, PriceBook())


def test_registry_makes_the_claude_client_only_when_asked() -> None:
    made: list[int] = []
    fake = FakeAnthropic()

    def factory() -> Any:
        made.append(1)
        return fake.client()

    reg = ProviderRegistry({"local": LOCAL}, PriceBook(), claude_factory=factory)
    reg.get("local")
    assert made == []
    claude = reg.get("anthropic")
    assert reg.get("anthropic") is claude
    assert made == [1]


@pytest.mark.parametrize(
    "url", ["ftp://x/v1", "http://", "http://u:p@host/v1", "http://host/v1?k=1", "http://h/#f"]
)
def test_provider_urls_are_plain_http(url: str) -> None:
    with pytest.raises(ValidationError):
        ProviderConfig(kind="openai_compat", base_url=url)


def test_settings_read_provider_keys_and_the_providers_file(tmp_path: Path) -> None:
    providers = tmp_path / "providers.toml"
    providers.write_text(
        "[providers.local]\n"
        'kind = "openai_compat"\n'
        'base_url = "http://vllm.internal:8000/v1/"\n'
        'output_limit_field = "max_tokens"\n'
        "[providers.proxy]\n"
        'kind = "gemini"\n'
        'base_url = "https://proxy.example/v1beta"\n'
        'api_key_env = "PROXY_KEY"\n'
        "thinking_level = true\n"
    )
    key_file = tmp_path / "proxy.key"
    key_file.write_text("proxy-secret\n")
    s = Settings.from_env(
        {
            "OPENAI_API_KEY": "sk-openai",
            "GEMINI_API_KEY": "g-gemini",
            "EVALKIT_PROVIDERS_FILE": str(providers),
            "PROXY_KEY_FILE": str(key_file),
        }
    )
    configs = s.provider_configs()
    assert sorted(configs) == ["gemini", "local", "openai", "proxy"]
    assert configs["openai"].api_key == SecretStr("sk-openai")
    assert configs["gemini"].api_key == SecretStr("g-gemini")
    local = configs["local"]
    assert (local.base_url, local.output_limit_field, local.api_key) == (
        "http://vllm.internal:8000/v1",
        "max_tokens",
        None,
    )
    assert configs["proxy"].api_key == SecretStr("proxy-secret")
    assert configs["proxy"].key_required
    assert configs["proxy"].thinking_level
    assert {"sk-openai", "g-gemini", "proxy-secret"} <= set(s.secret_values())


@pytest.mark.parametrize(
    ("toml", "error"),
    [
        ('[providers.anthropic]\nkind = "gemini"\nbase_url = "http://x/v1"\n', "anthropic"),
        ('[providers."Bad Name"]\nkind = "gemini"\nbase_url = "http://x/v1"\n', "must match"),
        ('[providers]\nlocal = "http://x"\n', "must be a table"),
        (
            '[providers.l]\nkind = "gemini"\nbase_url = "http://x/v1"\napi_key = "k"\n',
            "api_key_env",
        ),
        (
            '[providers.l]\nkind = "gemini"\nbase_url = "http://x/v1"\napi_key_env = "NOPE"\n',
            "needs NOPE or NOPE_FILE",
        ),
        ('[providers.l]\nkind = "grpc"\nbase_url = "http://x/v1"\n', "providers.l.kind"),
    ],
)
def test_bad_provider_files_are_refused(tmp_path: Path, toml: str, error: str) -> None:
    path = tmp_path / "p.toml"
    path.write_text(toml)
    with pytest.raises(ConfigError, match=error):
        Settings.from_env({"EVALKIT_PROVIDERS_FILE": str(path)})


def test_a_judge_on_an_unknown_provider_is_refused_when_the_run_is_created(
    sessions: sessionmaker[Session], dataset: None
) -> None:
    runner = Runner(sessions, settings(), providers=registry(Server()))
    with pytest.raises(InvalidInputError, match="scorer quality: unknown provider 'other'"):
        runner.create_run(run_config(judge_scorer("other", "gpt-test")))
    with pytest.raises(ValidationError):
        run_config(judge_scorer("Bad Name", "gpt-test"))


# Runs record their judges


def test_a_run_records_each_judge_and_reuses_cached_verdicts(
    sessions: sessionmaker[Session], dataset: None
) -> None:
    server = Server()
    providers = registry(server)
    config = run_config(judge_scorer("local", "gpt-test"))
    run_id = execute(sessions, providers, config)
    calls = len(server.requests)
    assert calls == 12
    with sessions() as s:
        run = s.get_one(Run, run_id)
        assert run.status == "completed"
        assert run.judges == {
            "quality": judge_identity(
                provider="local",
                endpoint="openai_compat:http://llm.test/v1",
                model="gpt-test",
                mode="rubric",
                effort="medium",
                rubric=RUBRIC,
            )
        }
        assert run.scorers[1]["provider"] == "local"
        summary = summarize(s, run_id)
    assert summary.judges["quality"]["model"] == "gpt-test"
    assert {m.name: m.mean for m in summary.metrics}["quality"] == 1.0
    assert summary.judge_cost_usd == pytest.approx(12 * (100 * 2.0 + 10 * 8.0) / 1e6)
    # The same judge on the same outputs: every verdict comes from the cache.
    execute(sessions, providers, config)
    assert len(server.requests) == calls


def test_a_process_with_another_judge_config_will_not_execute_the_run(
    sessions: sessionmaker[Session], dataset: None
) -> None:
    server = Server()
    runner = Runner(sessions, settings(), providers=registry(server))
    run_id = runner.create_run(run_config(judge_scorer("local", "gpt-test"))).id
    moved = ProviderConfig(kind="openai_compat", base_url="http://elsewhere.test/v1")
    other = Runner(sessions, settings(), providers=registry(server, local=moved))
    with pytest.raises(InvalidInputError, match="not the judge the run recorded"):
        asyncio.run(other.execute(run_id))
    with sessions() as s:
        run = s.get_one(Run, run_id)
        assert run.status == "failed"
        assert "endpoint openai_compat:http://llm.test/v1 -> " in run.error
    assert server.requests == []


def test_runner_closes_the_registry_it_built(sessions: sessionmaker[Session]) -> None:
    runner = Runner(sessions, settings())
    asyncio.run(runner.aclose())
    given = registry(Server())
    local = given.get("local")
    asyncio.run(Runner(sessions, settings(), providers=given).aclose())
    assert given.get("local") is local  # the caller's registry stays open


# The gate


@pytest.fixture
def two_judges(sessions: sessionmaker[Session], dataset: None) -> tuple[str, str]:
    server = Server()
    providers = registry(server)
    base = execute(sessions, providers, run_config(judge_scorer("local", "gpt-test")))
    cand = execute(sessions, providers, run_config(judge_scorer("gem", "gemini-test")))
    return base, cand


def test_compare_refuses_a_changed_judge(
    sessions: sessionmaker[Session], two_judges: tuple[str, str]
) -> None:
    base, cand = two_judges
    with sessions() as s, pytest.raises(PreconditionError) as err:
        compare_runs(s, base, cand)
    message = err.value.message
    assert "the judge changed between the runs (quality: provider local -> gem" in message
    assert "model gpt-test -> gemini-test" in message
    assert "--allow-judge-change" in message
    # A comparison of metrics without a judge is unaffected.
    with sessions() as s:
        assert compare_runs(s, base, cand, metrics=["exact"]).judge_changes == []


def test_compare_allows_a_changed_judge_when_asked(
    sessions: sessionmaker[Session], two_judges: tuple[str, str]
) -> None:
    base, cand = two_judges
    with sessions() as s:
        report = compare_runs(s, base, cand, allow_judge_change=True)
    assert [c.metric for c in report.comparisons] == ["exact", "quality"]
    (change,) = report.judge_changes
    assert change.metric == "quality"
    assert change.changes[0] == "provider local -> gem"
    assert report.passed


def test_a_judge_added_to_a_metric_counts_as_a_change(
    sessions: sessionmaker[Session], dataset: None
) -> None:
    providers = registry(Server())
    base = execute(sessions, providers, run_config({"type": "exact", "name": "quality"}))
    cand = execute(sessions, providers, run_config(judge_scorer("local", "gpt-test")))
    with sessions() as s, pytest.raises(PreconditionError, match="a judge in one run only"):
        compare_runs(s, base, cand)


def test_runs_from_before_judge_identities_compare_with_new_claude_runs(
    sessions: sessionmaker[Session], dataset: None
) -> None:
    providers = registry(Server())
    claude_judge = {"type": "judge", "name": "quality", "rubric": RUBRIC, "effort": "low"}
    old = execute(sessions, providers, run_config(claude_judge))
    new = execute(sessions, providers, run_config(claude_judge))
    with sessions() as s:
        recorded = s.get_one(Run, old).judges
        # An old run: no recorded judges, and a stored spec without a provider.
        spec = dict(s.get_one(Run, old).scorers[1])
        del spec["provider"]
        s.execute(
            update(Run).where(Run.id == old).values(judges=None, scorers=[{"type": "exact"}, spec])
        )
        s.commit()
        assert run_judges(s.get_one(Run, old)) == recorded
        assert compare_runs(s, old, new).judge_changes == []
        assert summarize(s, old).judges == recorded


# Estimates


def test_estimate_counts_each_judge_on_its_own_provider(
    sessions: sessionmaker[Session], dataset: None
) -> None:
    fake = FakeAnthropic()
    fake.default = lambda body: json_reply({"input_tokens": 40})
    claude = fake.client()
    server = Server()
    providers = ProviderRegistry(
        {"local": LOCAL},
        PriceBook(PRICES),
        claude=claude,
        transport=httpx2.MockTransport(server.handle),
    )
    config = run_config(
        judge_scorer("local", "gpt-test"),
        {"type": "judge", "name": "claude_judge", "rubric": RUBRIC},
    )
    result = asyncio.run(estimate(sessions, settings(), config, claude, providers=providers))
    assert result.judge_calls == 24
    # The Claude judge is counted; the local judge is estimated from text length.
    local = next(
        s
        for s in build_scorers(config.scorers, allow_import_paths=True, providers=providers)
        if s.name == "quality"
    )
    assert isinstance(local, LLMJudge)
    estimated = sum(
        estimate_tokens(p, local.system)
        for i in range(12)
        for p in local.prompts(Case(id=f"c{i:02d}", input={"q": f"q{i}"}, expected=f"a{i}"), "")
    )
    assert result.judge_input_tokens == 12 * 40 + estimated
    assert result.judge_input_estimated
    assert not result.target_input_estimated
    assert result.price_versions == {
        "anthropic": claude.prices.version,
        "local": PriceBook(PRICES).table("local").version,
    }
    assert result.price_versions["anthropic"] == "anthropic@2026-11-09"
    assert result.price_versions["local"].startswith("none@none+local.")
    assert result.notes == [
        "the fake target's own cost is not estimated",
        "judge quality: local cannot count tokens; its input tokens are estimated "
        "from the text's length",
    ]
    assert result.judge_max_output_cost_usd == pytest.approx(
        12 * 4096 * 8.0 / 1e6 + 12 * 4096 * 20.0 / 1e6
    )
    assert server.requests == []


# CLI and API


def write_run(tmp_path: Path, name: str, judge: str) -> Path:
    lines = [
        json.dumps({"id": f"q{i}", "input": {"q": f"q{i}"}, "expected": f"a{i}"}) for i in range(30)
    ]
    (tmp_path / "qa.jsonl").write_text("\n".join(lines) + "\n")
    path = tmp_path / f"{name}.toml"
    path.write_text(
        f'label = "{name}"\n[dataset]\nname = "qa"\npath = "qa.jsonl"\n'
        '[target]\ntype = "fake"\naccuracy = 0.7\nseed = 1\n'
        '[[scorers]]\ntype = "exact"\n'
        f'[[scorers]]\ntype = "judge"\nname = "quality"\nrubric = "be right"\n{judge}'
    )
    return path


def test_cli_gate_refuses_a_changed_judge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    server = Server()
    monkeypatch.setattr(
        cli,
        "ProviderRegistry",
        functools.partial(ProviderRegistry, transport=httpx2.MockTransport(server.handle)),
    )
    providers = tmp_path / "providers.toml"
    providers.write_text(
        '[providers.local]\nkind = "openai_compat"\nbase_url = "http://llm.test/v1"\n'
    )
    prices = tmp_path / "prices.toml"
    prices.write_text("[prices.gpt-test]\ninput = 2.0\noutput = 8.0\n")
    monkeypatch.setenv("EVALKIT_PROVIDERS_FILE", str(providers))
    monkeypatch.setenv("EVALKIT_PRICES_FILE", str(prices))
    db = f"sqlite:///{tmp_path / 'cli.db'}"

    def run_cli(*args: str) -> tuple[int, str]:
        out = io.StringIO()
        return cli.main(["--database-url", db, *args], out=out), out.getvalue()

    base = write_run(tmp_path, "base", 'provider = "local"\nmodel = "gpt-test"\n')
    code, text = run_cli("run", str(base))
    assert code == 0, text
    assert "judge quality: gpt-test on local (openai_compat:http://llm.test/v1), rubric" in text
    cand = write_run(tmp_path, "cand", 'provider = "local"\nmodel = "gpt-test"\neffort = "high"\n')
    assert run_cli("run", str(cand))[0] == 0
    assert run_cli("gate", "label:base", "label:cand") == (3, "")
    code, text = run_cli("gate", "label:base", "label:cand", "--allow-judge-change")
    assert code == 0
    assert "warning: judge changed for quality: effort medium -> high" in text
    code, raw = run_cli("compare", "label:base", "label:cand", "--allow-judge-change", "--json")
    assert json.loads(raw)["judge_changes"] == [
        {"metric": "quality", "changes": ["effort medium -> high"]}
    ]
    # Resuming a run whose judges use no Claude model needs no Claude key.
    engine = make_engine(db)
    with engine.connect() as conn:
        run_id = conn.exec_driver_sql("SELECT id FROM runs WHERE label = 'cand'").scalar()
    engine.dispose()
    assert run_cli("run", "--resume", str(run_id))[0] == 0


@pytest.mark.anyio
async def test_api_compare_takes_allow_judge_change(
    sessions: sessionmaker[Session], two_judges: tuple[str, str]
) -> None:
    base, cand = two_judges
    token = "test-token-0123456789"
    app = create_app(
        Settings.from_env({"EVALKIT_API_TOKENS": token, "EVALKIT_MAX_ACTIVE_RUNS": "0"}),
        sessions=sessions,
        claude=FakeAnthropic().client(),
    )
    auth = {"Authorization": f"Bearer {token}"}
    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://test") as c,
    ):
        body = {"baseline": base, "candidate": cand}
        refused = await c.post("/v1/compare", json=body, headers=auth)
        assert refused.status_code == 412
        assert "judge changed" in refused.json()["error"]["message"]
        allowed = await c.post(
            "/v1/compare", json={**body, "allow_judge_change": True}, headers=auth
        )
        assert allowed.status_code == 200
        assert allowed.json()["judge_changes"][0]["metric"] == "quality"
        assert allowed.json()["passed"] is True


# Migration


def test_migration_0008_adds_and_drops_the_judges_column(tmp_path: Path) -> None:
    from alembic import command

    from evalkit.db.session import alembic_config

    engine = make_engine(f"sqlite:///{tmp_path / 'm.db'}")
    upgrade(engine)
    assert "judges" in {c["name"] for c in inspect(engine).get_columns("runs")}
    command.downgrade(alembic_config(engine), "0007")
    assert "judges" not in {c["name"] for c in inspect(engine).get_columns("runs")}
    engine.dispose()
