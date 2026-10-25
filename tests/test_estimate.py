import asyncio
from typing import Any

import httpx2
import pytest
from sqlalchemy.orm import Session, sessionmaker

from evalkit.config import Settings
from evalkit.datasets import Case, import_dataset
from evalkit.errors import InvalidInputError
from evalkit.estimate import estimate
from evalkit.runner import Runner
from evalkit.runspec import RunConfig
from tests.fake_anthropic import FakeAnthropic, json_reply, message


def counter(body: dict[str, Any]) -> httpx2.Response:
    # One "token" per character of the prompt, so totals are predictable.
    return json_reply({"input_tokens": len(body["messages"][0]["content"])})


def cfg(**over: Any) -> RunConfig:
    raw: dict[str, Any] = {
        "dataset": {"name": "qa"},
        "target": {"type": "prompt", "template": "Q: {{q}}", "max_tokens": 100},
        "scorers": [{"type": "exact"}],
    }
    raw.update(over)
    return RunConfig.model_validate(raw)


@pytest.fixture
def data(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        import_dataset(
            s, "qa", [Case(id=str(i), input={"q": "x" * i}, expected="y") for i in range(1, 5)]
        )


def test_counts_uncached_target_prompts(sessions: sessionmaker[Session], data: None) -> None:
    fake = FakeAnthropic()
    fake.default = counter
    result = asyncio.run(estimate(sessions, Settings(), cfg(), fake.client()))
    # Prompts are "Q: x", "Q: xx", ... -> 4, 5, 6, 7 characters.
    assert result.target_input_tokens == 22
    assert result.target_input_cost_usd == pytest.approx(22 * 4 / 1e6)
    assert result.target_max_output_cost_usd == pytest.approx(4 * 100 * 20 / 1e6)
    assert result.max_total_cost_usd == pytest.approx((22 * 4 + 400 * 20) / 1e6)
    assert set(fake.paths) == {"/v1/messages/count_tokens"}
    assert fake.requests[0]["model"] == "claude-opus-5-5"
    batch = asyncio.run(
        estimate(sessions, Settings(), cfg(options={"mode": "batch"}), fake.client())
    )
    assert batch.max_total_cost_usd == pytest.approx(result.max_total_cost_usd / 2)


def test_cached_cases_are_free(sessions: sessionmaker[Session], data: None) -> None:
    run_fake = FakeAnthropic()
    run_fake.default = lambda body: json_reply(message("y"))
    runner = Runner(sessions, Settings(), claude=run_fake.client())
    asyncio.run(runner.execute(runner.create_run(cfg()).id))
    counted = FakeAnthropic()
    counted.default = counter
    result = asyncio.run(estimate(sessions, Settings(), cfg(), counted.client()))
    assert (result.cases, result.cached_cases, result.target_input_tokens) == (4, 4, 0)
    fresh = asyncio.run(
        estimate(sessions, Settings(), cfg(options={"use_cache": False}), counted.client())
    )
    assert fresh.cached_cases == 0


def test_judges_are_counted_per_call(sessions: sessionmaker[Session], data: None) -> None:
    fake = FakeAnthropic()
    fake.default = counter
    scorers = [
        {"type": "judge", "name": "rubric", "rubric": "r", "max_tokens": 500},
        {"type": "judge", "name": "pair", "rubric": "r", "mode": "pairwise", "max_tokens": 300},
    ]
    result = asyncio.run(estimate(sessions, Settings(), cfg(scorers=scorers), fake.client()))
    assert result.judge_calls == 4 + 8
    assert result.judge_input_tokens > 12 * 100  # includes the response bound
    assert result.judge_max_output_cost_usd == pytest.approx((4 * 500 + 8 * 300) * 20 / 1e6)
    systems = [r.get("system", "") for r in fake.requests]
    assert any("compare two responses" in s for s in systems)


def test_other_targets_and_template_misfits(sessions: sessionmaker[Session], data: None) -> None:
    fake = FakeAnthropic()
    fake.default = counter
    other = asyncio.run(estimate(sessions, Settings(), cfg(target={"type": "fake"}), fake.client()))
    assert other.target_input_tokens == 0
    assert other.notes == ["the fake target's own cost is not estimated"]
    misfit = asyncio.run(
        estimate(
            sessions,
            Settings(),
            cfg(target={"type": "prompt", "template": "{{nope}}"}),
            fake.client(),
        )
    )
    assert misfit.notes == ["4 case(s) do not fit the template and will fail"]
    with sessions() as s:
        import_dataset(s, "noref", [Case(id="a", input="q")])
    pair = cfg(
        dataset={"name": "noref"}, scorers=[{"type": "judge", "rubric": "r", "mode": "pairwise"}]
    )
    assert asyncio.run(estimate(sessions, Settings(), pair, fake.client())).judge_calls == 0


def test_empty_filter(sessions: sessionmaker[Session], data: None) -> None:
    with pytest.raises(InvalidInputError, match="selects no cases"):
        asyncio.run(
            estimate(sessions, Settings(), cfg(filter={"split": "x"}), FakeAnthropic().client())
        )
