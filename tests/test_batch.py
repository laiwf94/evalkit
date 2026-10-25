import asyncio
import json
from typing import Any

import anthropic
import httpx2
import pytest
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from evalkit.config import Settings
from evalkit.datasets import Case, import_dataset
from evalkit.db.models import Result, Run
from evalkit.errors import InvalidInputError
from evalkit.llm import ClaudeClient
from evalkit.pricing import PriceTable
from evalkit.report import summarize
from evalkit.runner import Runner, batch_custom_id
from evalkit.runspec import RunConfig
from evalkit.targets.fake import FakeTarget
from tests.fake_anthropic import message
from tests.fakeclock import FakeClock

BATCH = {
    "id": "msgbatch_01",
    "type": "message_batch",
    "processing_status": "in_progress",
    "request_counts": {"processing": 1, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0},
    "created_at": "2026-01-01T00:00:00Z",
    "expires_at": "2026-01-02T00:00:00Z",
    "ended_at": None,
    "archived_at": None,
    "cancel_initiated_at": None,
    "results_url": None,
}
USAGE = {
    "input_tokens": 1000,
    "output_tokens": 100,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
}


class FakeBatchApi:
    def __init__(self, polls_before_end: int = 2, failing: tuple[str, ...] = ()) -> None:
        self.polls_before_end = polls_before_end
        self.failing = {batch_custom_id(c) for c in failing}
        self.created: list[list[dict[str, Any]]] = []
        self.polls = 0
        self.online: list[dict[str, Any]] = []
        self.hold = False

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        if request.method == "POST" and path == "/v1/messages/batches":
            self.created.append(json.loads(request.content)["requests"])
            return httpx2.Response(200, json=BATCH)
        if path == "/v1/messages/batches/msgbatch_01":
            self.polls += 1
            if self.hold or self.polls <= self.polls_before_end:
                return httpx2.Response(200, json=BATCH)
            return httpx2.Response(
                200,
                json={
                    **BATCH,
                    "processing_status": "ended",
                    "results_url": "http://fake-anthropic/v1/messages/batches/msgbatch_01/results",
                },
            )
        if path == "/v1/messages/batches/msgbatch_01/results":
            lines = []
            for req in self.created[-1]:
                cid = req["custom_id"]
                if cid in self.failing:
                    result: dict[str, Any] = {
                        "type": "errored",
                        "error": {
                            "type": "error",
                            "error": {"type": "api_error", "message": "boom"},
                        },
                    }
                else:
                    answer = req["params"]["messages"][0]["content"].split(": ")[-1]
                    result = {"type": "succeeded", "message": message(answer, usage=USAGE)}
                lines.append(json.dumps({"custom_id": cid, "result": result}))
            lines.append(json.dumps({"custom_id": "c-unknown", "result": {"type": "expired"}}))
            return httpx2.Response(200, content="\n".join(lines).encode())
        if request.method == "POST" and path == "/v1/messages":
            body = json.loads(request.content)
            self.online.append(body)
            answer = body["messages"][0]["content"].split(": ")[-1]
            return httpx2.Response(200, json=message(answer, usage=USAGE))
        raise AssertionError(f"unexpected {request.method} {path}")

    def client(self) -> ClaudeClient:
        http = anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(self.handler))
        return ClaudeClient(
            PriceTable(),
            SecretStr("sk-ant-test-0000000000"),
            http_client=http,
            base_url="http://fake-anthropic",
        )


@pytest.fixture
def cases(sessions: sessionmaker[Session]) -> list[Case]:
    items = [Case(id=f"case {i}", input={"q": f"a{i}"}, expected=f"a{i}") for i in range(10)]
    with sessions() as s:
        import_dataset(s, "qa", items)
    return items


def batch_config(template: str = "Echo: {{q}}") -> RunConfig:
    return RunConfig.model_validate(
        {
            "dataset": {"name": "qa"},
            "target": {"type": "prompt", "template": template, "system": "Repeat the value."},
            "scorers": [{"type": "exact"}],
            "options": {"mode": "batch", "batch_poll_seconds": 5, "retry": {"base_delay": 0}},
        }
    )


def test_batch_run_submits_once_and_maps_results(
    sessions: sessionmaker[Session], cases: list[Case]
) -> None:
    api = FakeBatchApi(failing=("case 3",))
    clock = FakeClock()
    runner = Runner(sessions, Settings(), claude=api.client(), clock=clock)
    run = runner.create_run(batch_config())
    report = asyncio.run(runner.execute(run.id))
    assert len(api.created) == 1
    requests = api.created[0]
    assert len(requests) == 10
    first = requests[0]["params"]
    assert first["model"] == "claude-opus-5-5"
    assert "fallbacks" not in first  # not accepted by the Batches API
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert api.polls == 4  # three status polls, and the SDK reads the batch again for results_url
    assert clock.sleeps == [5, 5]
    assert report.batched == 9
    assert report.executed == 1  # the errored entry went through the online path
    assert len(api.online) == 1
    with sessions() as s:
        assert s.get_one(Run, run.id).batch_id == "msgbatch_01"
        summary = summarize(s, run.id, resamples=200)
        costs = sorted(s.scalars(select(Result.cost_usd).where(Result.run_id == run.id)))
    assert summary.metrics[0].mean == 1.0
    online_cost = (1000 * 4 + 100 * 20) / 1e6
    assert costs[0] == pytest.approx(online_cost / 2)  # batch discount
    assert costs[-1] == pytest.approx(online_cost)


def test_resume_polls_the_existing_batch(
    sessions: sessionmaker[Session], cases: list[Case]
) -> None:
    api = FakeBatchApi()
    api.hold = True
    runner = Runner(sessions, Settings(), claude=api.client(), clock=FakeClock())
    run = runner.create_run(batch_config())

    async def crash_while_waiting() -> None:
        task = asyncio.create_task(runner.execute(run.id))
        while True:
            await asyncio.sleep(0.01)
            with sessions() as s:
                if s.get_one(Run, run.id).batch_id:
                    break
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(crash_while_waiting())
    api.hold = False
    report = asyncio.run(
        Runner(sessions, Settings(), claude=api.client(), clock=FakeClock()).execute(run.id)
    )
    assert len(api.created) == 1
    assert report.batched == 10
    with sessions() as s:
        assert summarize(s, run.id, resamples=200).completed == 10


def test_cached_cases_are_not_batched(sessions: sessionmaker[Session], cases: list[Case]) -> None:
    api = FakeBatchApi(polls_before_end=0)
    runner = Runner(sessions, Settings(), claude=api.client(), clock=FakeClock())
    asyncio.run(runner.execute(runner.create_run(batch_config()).id))
    report = asyncio.run(runner.execute(runner.create_run(batch_config()).id))
    assert len(api.created) == 1
    assert (report.cached, report.batched) == (10, 0)


def test_template_errors_skip_the_batch(sessions: sessionmaker[Session], cases: list[Case]) -> None:
    api = FakeBatchApi()
    runner = Runner(sessions, Settings(), claude=api.client(), clock=FakeClock())
    run = runner.create_run(batch_config("{{missing}}"))
    report = asyncio.run(runner.execute(run.id))
    assert api.created == []
    assert (report.errors, report.batched) == (10, 0)
    with sessions() as s:
        errors = set(s.scalars(select(Result.error).where(Result.run_id == run.id)))
    assert errors == {"case input has no field 'missing' used by the template"}


def test_batch_mode_requires_a_prompt_target(
    sessions: sessionmaker[Session], cases: list[Case]
) -> None:
    api = FakeBatchApi()
    runner = Runner(sessions, Settings(), claude=api.client(), target_factory=FakeTarget)
    run = runner.create_run(batch_config())
    with pytest.raises(InvalidInputError, match="only available for prompt targets"):
        asyncio.run(runner.execute(run.id))


def test_custom_ids_fit_the_api_pattern() -> None:
    cid = batch_custom_id("a case id with spaces/and slashes")
    assert len(cid) <= 64
    assert cid.replace("_", "").replace("-", "").isalnum()
