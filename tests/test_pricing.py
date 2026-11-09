"""Versioned per-provider price tables, prices frozen with each run, and token estimates."""

import asyncio
import json
import math
from pathlib import Path
from typing import Any

import httpx2
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from sqlalchemy.orm import Session, sessionmaker

from evalkit import cli
from evalkit.config import ConfigError, Settings
from evalkit.datasets import Case, import_dataset
from evalkit.db.models import Run
from evalkit.errors import InvalidInputError, PreconditionError
from evalkit.estimate import estimate
from evalkit.pricing import (
    ANTHROPIC_PRICES,
    OPENAI_PRICES,
    PRICE_LISTS,
    ModelPrice,
    PriceBook,
    PriceTable,
    TokenCounts,
    estimate_tokens,
    price_changes,
    price_record,
)
from evalkit.providers.registry import ProviderConfig, ProviderRegistry
from evalkit.report import summarize
from evalkit.runner import Runner
from evalkit.runspec import RunConfig
from tests.fake_anthropic import FakeAnthropic, json_reply, message
from tests.fake_gemini import FakeGemini, count_text, text_of
from tests.fake_openai import FakeOpenAI

LOCAL = ProviderConfig(kind="openai_compat", base_url="http://llm.test/v1")
GEM = ProviderConfig(kind="gemini", base_url="http://gemini.test/v1beta")
GPT = {"gpt-test": ModelPrice(input=2.0, output=8.0)}

# The built-in lists


@pytest.mark.parametrize("price_list", list(PRICE_LISTS.values()), ids=list(PRICE_LISTS))
def test_built_in_lists_are_versioned_and_consistent(price_list: Any) -> None:
    assert price_list.version == "2026-11-09"
    assert price_list.source.startswith("https://")
    assert price_list.prices
    for model, price in price_list.prices.items():
        assert price.input >= 0, model
        assert price.output >= 0, model
        if price.cache_read is not None:
            assert price.cache_read < price.input, model
        if price.long_context_threshold is not None:
            assert price.long_input is not None, model
            assert price.long_input > price.input, model
            assert price.long_output is not None, model
            assert price.long_output >= price.output, model
    with pytest.raises(TypeError):
        price_list.prices["x"] = ModelPrice(input=1, output=1)  # read-only


def test_each_provider_starts_from_its_own_list() -> None:
    book = PriceBook()
    assert book.table("anthropic").version == "anthropic@2026-11-09"
    assert book.table("openai").version == "openai@2026-11-09"
    assert book.table("gemini").version == "gemini@2026-11-09"
    assert "gpt-5" in book.table("openai")
    assert "gpt-5" not in book.table("anthropic")
    assert "gemini-2.5-pro" in book.table("gemini")
    assert "gemini-2.5-pro" not in book.table("openai")
    assert "claude-opus-5-5" not in book.table("gemini")
    # A self-hosted server prices nothing until the operator says so.
    local = book.table("local")
    assert local.version == "none@none"
    assert local.models() == []
    with pytest.raises(InvalidInputError, match=r"unknown model 'm' in price table none@none"):
        local.require("m")
    # ... unless it names a list, e.g. a proxy in front of OpenAI.
    assert book.table("proxy", "openai").models() == sorted(OPENAI_PRICES.prices)
    assert book.table("proxy", "openai").require("gpt-4o").input == 2.5
    assert book.table("local") is book.table("local")
    with pytest.raises(InvalidInputError, match="unknown price list 'azure'"):
        book.table("proxy", "azure")


def test_the_claude_default_table_is_unchanged() -> None:
    assert PriceTable().models() == sorted(ANTHROPIC_PRICES.prices)
    assert PriceTable().version == "anthropic@2026-11-09"


def test_openai_costs_cached_tokens_at_the_cached_rate() -> None:
    table = PriceBook().table("openai")
    tokens = TokenCounts(input_tokens=600_000, output_tokens=100_000, cache_read_tokens=400_000)
    # gpt-5: 1.25 in, 0.125 cached, 10 out.
    assert table.cost("gpt-5", tokens) == pytest.approx(0.6 * 1.25 + 0.4 * 0.125 + 0.1 * 10.0)
    # gpt-4o caches at half price, not a tenth.
    assert table.cost("gpt-4o", TokenCounts(cache_read_tokens=1_000_000)) == pytest.approx(1.25)


def test_gemini_long_prompts_cost_the_long_context_rates() -> None:
    table = PriceBook().table("gemini")
    short = TokenCounts(input_tokens=150_000, cache_read_tokens=50_000, output_tokens=10_000)
    assert table.cost("gemini-2.5-pro", short) == pytest.approx(
        0.15 * 1.25 + 0.05 * 0.125 + 0.01 * 10.0
    )
    long = TokenCounts(input_tokens=150_000, cache_read_tokens=50_001, output_tokens=10_000)
    assert table.cost("gemini-2.5-pro", long) == pytest.approx(
        0.15 * 2.5 + 0.050001 * 0.25 + 0.01 * 15.0
    )


# Overrides and versions


def test_overrides_apply_to_every_provider_or_to_one() -> None:
    book = PriceBook(
        {
            "shared-model": ModelPrice(input=1.0, output=2.0),
            "local": {"llama-test": ModelPrice(input=0.1, output=0.2)},
            "openai": {"shared-model": ModelPrice(input=3.0, output=4.0)},
        }
    )
    assert book.table("anthropic").require("shared-model").input == 1.0
    assert book.table("local").require("shared-model").input == 1.0
    assert book.table("local").require("llama-test").input == 0.1
    assert "llama-test" not in book.table("gemini")
    # The provider's own table wins over the shared one.
    assert book.table("openai").require("shared-model").input == 3.0


def test_a_version_names_the_list_and_a_hash_of_what_the_overrides_change() -> None:
    plain = PriceTable(base=OPENAI_PRICES)
    assert plain.version == "openai@2026-11-09"
    # Restating a built-in price changes nothing, so the version stays.
    same = PriceTable({"gpt-5": OPENAI_PRICES.prices["gpt-5"]}, base=OPENAI_PRICES)
    assert same.version == plain.version
    cheaper = PriceTable({"gpt-5": ModelPrice(input=1.0, output=10.0)}, base=OPENAI_PRICES)
    assert cheaper.version.startswith("openai@2026-11-09+local.")
    assert len(cheaper.version.split("+local.")[1]) == 8
    again = PriceTable({"gpt-5": ModelPrice(input=1.0, output=10.0)}, base=OPENAI_PRICES)
    assert again.version == cheaper.version
    other = PriceTable({"gpt-5": ModelPrice(input=1.1, output=10.0)}, base=OPENAI_PRICES)
    assert other.version != cheaper.version


def test_prices_file_with_shared_and_per_provider_tables(tmp_path: Path) -> None:
    f = tmp_path / "prices.toml"
    f.write_text(
        "[prices.my-model]\ninput = 1.5\noutput = 3\n\n"
        '[prices.openai."gpt-5"]\ninput = 1.0\noutput = 9\ncache_read = 0.1\n\n'
        '[prices.local."llama-test"]\ninput = 0\noutput = 0\n'
    )
    settings = Settings.from_env({"EVALKIT_PRICES_FILE": str(f)})
    book = settings.price_book()
    assert book.table("anthropic").require("my-model").output == 3
    assert book.table("openai").require("gpt-5").output == 9
    assert book.table("openai").version.startswith("openai@2026-11-09+local.")
    assert book.table("gemini").require("my-model").input == 1.5
    assert book.table("local").require("llama-test").input == 0
    f.write_text('[prices.openai."gpt-5"]\ninput = -1\noutput = 3\n')
    with pytest.raises(ConfigError, match="invalid configuration"):
        Settings.from_env({"EVALKIT_PRICES_FILE": str(f)})


def test_a_server_names_its_price_list() -> None:
    proxy = ProviderConfig(
        kind="openai_compat", base_url="http://proxy.test/v1", price_list="openai"
    )
    registry = ProviderRegistry(
        {"proxy": proxy, "local": LOCAL}, PriceBook(), claude=FakeAnthropic().client()
    )
    assert registry.price_table("proxy").require("gpt-5").input == 1.25
    assert registry.price_table("local").models() == []
    assert registry.price_table("anthropic").version == "anthropic@2026-11-09"
    builtin = ProviderRegistry(Settings.from_env({}).provider_configs(), PriceBook())
    assert builtin.price_table("openai").version == "openai@2026-11-09"
    assert builtin.price_table("gemini").version == "gemini@2026-11-09"
    assert builtin.price_table("anthropic").version == "anthropic@2026-11-09"
    with pytest.raises(ValidationError, match="unknown price list 'azure'"):
        ProviderConfig(kind="openai_compat", base_url="http://x.test", price_list="azure")


# What a run records


def test_price_record_and_changes() -> None:
    gpt = PriceTable(GPT, base=OPENAI_PRICES)
    record = price_record(
        [("openai", gpt, "gpt-test"), ("openai", gpt, "gpt-5"), ("anthropic", PriceTable(), "x")]
    )
    assert record == {
        "openai": {
            "version": gpt.version,
            "models": {
                "gpt-test": {"input": 2.0, "output": 8.0},
                "gpt-5": {"input": 1.25, "output": 10.0, "cache_read": 0.125},
            },
        },
        "anthropic": {"version": "anthropic@2026-11-09", "models": {}},
    }
    assert price_changes(record, record) == []
    # Another version that prices the run's models the same is not a change.
    restated = PriceTable({**GPT, "gpt-4o": ModelPrice(input=9, output=9)}, base=OPENAI_PRICES)
    assert restated.version != gpt.version
    assert price_changes(record, price_record([("openai", restated, "gpt-test")])) == []
    dearer = PriceTable({"gpt-test": ModelPrice(input=3.0, output=8.0)}, base=OPENAI_PRICES)
    assert price_changes(record, price_record([("openai", dearer, "gpt-test")])) == [
        f"openai/gpt-test: input 2, output 8 in {gpt.version}, "
        f"input 3, output 8 in {dearer.version}"
    ]
    # A provider or a model the run did not record is not compared.
    assert price_changes({}, price_record([("openai", dearer, "gpt-test")])) == []


@pytest.fixture
def dataset(sessions: sessionmaker[Session]) -> None:
    cases = [Case(id=f"c{i}", input={"q": f"q{i}"}, expected="ok") for i in range(6)]
    with sessions() as s:
        import_dataset(s, "qa", cases)


def gpt_config(**target: Any) -> RunConfig:
    return RunConfig.model_validate(
        {
            "label": "priced",
            "dataset": {"name": "qa"},
            "target": {
                "type": "prompt",
                "provider": "local",
                "model": "gpt-test",
                "template": "{{q}}",
                **target,
            },
            "scorers": [
                {"type": "exact"},
                {
                    "type": "judge",
                    "name": "quality",
                    "provider": "gem",
                    "model": "gemini-test",
                    "rubric": "Correct?",
                },
            ],
        }
    )


class Fakes:
    def __init__(self) -> None:
        verdict = json.dumps({"reasoning": "fine", "score": 5})
        self.openai = FakeOpenAI("http://llm.test/v1", api_key=None)
        self.gemini = FakeGemini(
            "http://gemini.test/v1beta", api_key=None, answer=lambda model, body: verdict
        )
        self.claude = FakeAnthropic(json_reply(message("ok"))).client()

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        fake = self.gemini if request.url.host == "gemini.test" else self.openai
        return fake.handle(request)

    def registry(self, prices: dict[str, Any]) -> ProviderRegistry:
        return ProviderRegistry(
            {"local": LOCAL, "gem": GEM},
            PriceBook(prices),
            claude=self.claude,
            transport=httpx2.MockTransport(self.handle),
        )


PRICES: dict[str, Any] = {
    "local": GPT,
    "gem": {"gemini-test": ModelPrice(input=1.0, output=4.0)},
}


def test_a_run_records_its_prices_and_is_costed_at_them(
    sessions: sessionmaker[Session], dataset: None
) -> None:
    fakes = Fakes()
    runner = Runner(sessions, Settings(), providers=fakes.registry(PRICES))
    run_id = runner.create_run(gpt_config()).id
    with sessions() as s:
        recorded = s.get_one(Run, run_id).prices
    local = PriceBook(PRICES).table("local")
    gem = PriceBook(PRICES).table("gem")
    assert recorded == {
        "local": {"version": local.version, "models": {"gpt-test": {"input": 2.0, "output": 8.0}}},
        "gem": {"version": gem.version, "models": {"gemini-test": {"input": 1.0, "output": 4.0}}},
    }
    assert local.version.startswith("none@none+local.")
    assert asyncio.run(runner.execute(run_id)).status == "completed"
    with sessions() as s:
        summary = summarize(s, run_id)
    assert summary.prices == recorded
    assert summary.cost_usd == pytest.approx(6 * (100 * 2.0 + 10 * 8.0) / 1e6)
    assert summary.judge_cost_usd == pytest.approx(6 * (100 * 1.0 + 10 * 4.0) / 1e6)
    shown = cli._format_summary(summary)
    assert f"prices local {local.version}: gpt-test in 2 out 8 (USD per 1M tokens)" in shown
    assert f"prices gem {gem.version}: gemini-test in 1 out 4 (USD per 1M tokens)" in shown
    assert fakes.openai.rejected == fakes.gemini.rejected == []


@pytest.mark.parametrize("role", ["target", "judge"])
def test_a_process_with_other_prices_will_not_execute_the_run(
    sessions: sessionmaker[Session], dataset: None, role: str
) -> None:
    fakes = Fakes()
    run_id = (
        Runner(sessions, Settings(), providers=fakes.registry(PRICES)).create_run(gpt_config()).id
    )
    changed = dict(PRICES)
    if role == "target":
        changed["local"] = {"gpt-test": ModelPrice(input=2.5, output=8.0)}
        expected = "local/gpt-test: input 2, output 8 in none@none+local."
    else:
        changed["gem"] = {"gemini-test": ModelPrice(input=1.0, output=5.0)}
        expected = "gem/gemini-test: input 1, output 4 in none@none+local."
    other = Runner(sessions, Settings(), providers=fakes.registry(changed))
    with pytest.raises(PreconditionError, match="prices the run's models differently") as raised:
        asyncio.run(other.execute(run_id))
    assert expected in str(raised.value)
    assert fakes.openai.requests == fakes.gemini.requests == []
    # Prices for models the run does not use may differ freely.
    extra = {**PRICES, "local": {**GPT, "gpt-other": ModelPrice(input=9, output=9)}}
    same = Runner(sessions, Settings(), providers=fakes.registry(extra))
    assert asyncio.run(same.execute(run_id)).status == "completed"


def test_runs_from_before_prices_were_recorded_still_execute(
    sessions: sessionmaker[Session], dataset: None
) -> None:
    fakes = Fakes()
    run_id = (
        Runner(sessions, Settings(), providers=fakes.registry(PRICES)).create_run(gpt_config()).id
    )
    with sessions() as s:
        s.get_one(Run, run_id).prices = None
        s.commit()
    changed = {**PRICES, "local": {"gpt-test": ModelPrice(input=3, output=8)}}
    other = Runner(sessions, Settings(), providers=fakes.registry(changed))
    assert asyncio.run(other.execute(run_id)).status == "completed"
    with sessions() as s:
        assert summarize(s, run_id).prices is None
        assert "prices" not in cli._format_summary(summarize(s, run_id))


# Estimates for providers that cannot count


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 8
    assert estimate_tokens("abc") == 1 + 8
    assert estimate_tokens("abcd") == 2 + 8
    assert estimate_tokens("abcd", "sys") == 2 + 8 + 1 + 8
    # Three bytes a character in UTF-8: one token a character.
    assert estimate_tokens("日本語のテキスト") == 8 + 8


@given(st.text(max_size=2000), st.text(max_size=200))
def test_estimates_lean_high(prompt: str, system: str) -> None:
    """Never below a four-bytes-a-token count of the same text (the usual English
    rate), so a budget built on the estimate is not an undercount for prose."""
    floor = math.ceil(len(prompt.encode()) / 4) + math.ceil(len(system.encode()) / 4)
    assert estimate_tokens(prompt, system) >= floor
    assert estimate_tokens(prompt + "x", system) >= estimate_tokens(prompt, system)


def test_estimate_labels_an_uncountable_provider_and_counts_the_others(
    sessions: sessionmaker[Session], dataset: None
) -> None:
    fakes = Fakes()
    providers = fakes.registry(PRICES)
    config = gpt_config()
    result = asyncio.run(estimate(sessions, Settings(), config, fakes.claude, providers=providers))
    prompts = [f"q{i}" for i in range(6)]
    assert result.target_input_estimated
    assert result.target_input_tokens == sum(estimate_tokens(p) for p in prompts)
    assert result.target_input_cost_usd == pytest.approx(result.target_input_tokens * 2.0 / 1e6)
    # Gemini counts on its countTokens endpoint, so the judge is not an estimate.
    assert not result.judge_input_estimated
    counted = [b for b in fakes.gemini.bodies if "generateContentRequest" in b]
    assert len(counted) == 6
    counts = sum(count_text(text_of(b["generateContentRequest"])) for b in counted)
    assert result.judge_input_tokens == counts + 6 * 4096  # plus the target's max_tokens
    assert result.price_versions == {
        "local": PriceBook(PRICES).table("local").version,
        "gem": PriceBook(PRICES).table("gem").version,
    }
    assert result.notes == [
        "the target's provider local cannot count tokens; "
        "its input tokens are estimated from the text's length"
    ]
    shown = cli._format_estimate(result)
    assert f"target input  ~{result.target_input_tokens:,} tokens (estimated)" in shown
    assert f"judge calls 6  input {result.judge_input_tokens:,} tokens  $" in shown
    assert f"at most ~${result.max_total_cost_usd:.4f}" in shown
    assert f"prices gem {result.price_versions['gem']}  local " in shown
    assert fakes.openai.requests == []  # nothing was sent to the server that cannot count
    assert fakes.gemini.rejected == []


def test_a_counted_estimate_is_not_labelled(sessions: sessionmaker[Session], dataset: None) -> None:
    fakes = Fakes()
    providers = fakes.registry(PRICES)
    config = RunConfig.model_validate(
        {
            "dataset": {"name": "qa"},
            "target": {
                "type": "prompt",
                "provider": "gem",
                "model": "gemini-test",
                "template": "{{q}}",
            },
            "scorers": [{"type": "exact"}],
        }
    )
    result = asyncio.run(estimate(sessions, Settings(), config, fakes.claude, providers=providers))
    assert not result.target_input_estimated
    assert not result.judge_input_estimated
    shown = cli._format_estimate(result)
    assert "estimated" not in shown
    assert "~" not in shown
