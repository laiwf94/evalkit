import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx2
import pytest
from sqlalchemy.orm import Session, sessionmaker

from evalkit import api
from evalkit.api import BodyLimit, build, create_app
from evalkit.config import ConfigError, Settings
from evalkit.db import make_engine, make_session_factory
from evalkit.targets.factory import build_target
from tests.fake_anthropic import FakeAnthropic

pytestmark = pytest.mark.anyio

TOKEN = "test-token-0123456789"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
CASES = "\n".join(
    json.dumps({"id": f"c{i:02d}", "input": {"q": str(i)}, "expected": str(i)}) for i in range(12)
)


def settings(**kw: Any) -> Settings:
    return Settings.from_env({"EVALKIT_API_TOKENS": TOKEN, **kw})


@asynccontextmanager
async def serving(
    sessions: sessionmaker[Session], **env: str
) -> AsyncIterator[tuple[httpx2.AsyncClient, Any]]:
    """The app with its lifespan running (which runs the embedded worker)."""
    app = create_app(settings(**env), sessions=sessions, claude=FakeAnthropic().client())
    transport = httpx2.ASGITransport(app=BodyLimit(app, 64 * 1024), raise_app_exceptions=False)
    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(transport=transport, base_url="http://test") as c,
    ):
        yield c, app.state.services


@pytest.fixture
async def client(sessions: sessionmaker[Session]) -> AsyncIterator[httpx2.AsyncClient]:
    async with serving(sessions) as (c, _):
        yield c


async def wait_status(
    c: httpx2.AsyncClient, run_id: str, statuses: tuple[str, ...]
) -> dict[str, Any]:
    for _ in range(1000):
        body = (await c.get(f"/v1/runs/{run_id}", headers=AUTH)).json()
        if body["status"] in statuses:
            return dict(body)
        await asyncio.sleep(0.01)
    raise AssertionError(f"run did not reach {statuses}")


async def wait_done(c: httpx2.AsyncClient, run_id: str) -> dict[str, Any]:
    return await wait_status(c, run_id, ("completed", "failed"))


def run_body(**target: Any) -> dict[str, Any]:
    return {
        "label": target.pop("label", "api"),
        "dataset": {"name": "qa"},
        "target": {"type": "fake", **target},
        "scorers": [{"type": "exact"}],
    }


async def test_health_and_auth(client: httpx2.AsyncClient) -> None:
    assert (await client.get("/healthz")).json() == {"status": "ok"}
    missing = await client.get("/v1/datasets")
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert missing.json()["error"]["code"] == "unauthenticated"
    assert missing.json()["error"]["request_id"]
    wrong = await client.get(
        "/v1/datasets", headers={"Authorization": "Bearer nope-nope-nope-nope"}
    )
    assert wrong.status_code == 401
    basic = await client.get("/v1/datasets", headers={"Authorization": f"Basic {TOKEN}"})
    assert basic.status_code == 401
    assert (await client.get("/metrics")).status_code == 401
    metrics = await client.get("/metrics", headers=AUTH)
    assert metrics.status_code == 200
    assert "evalkit_http_requests_total" in metrics.text


async def test_request_ids(client: httpx2.AsyncClient) -> None:
    r = await client.get("/healthz", headers={"X-Request-ID": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123"
    r = await client.get("/healthz", headers={"X-Request-ID": "bad id\n"})
    assert r.headers["x-request-id"] != "bad id\n"
    assert len(r.headers["x-request-id"]) == 32


async def test_dataset_endpoints(client: httpx2.AsyncClient) -> None:
    created = await client.post("/v1/datasets/qa?description=first", content=CASES, headers=AUTH)
    assert created.status_code == 201
    assert created.json()["version"] == 1
    assert created.json()["case_count"] == 12
    again = await client.post("/v1/datasets/qa", content=CASES, headers=AUTH)
    assert again.status_code == 200
    bad = await client.post("/v1/datasets/qa", content='{"id": "x"}', headers=AUTH)
    assert bad.status_code == 400
    assert bad.json()["error"] == {
        "code": "invalid_argument",
        "message": "line 1: input: Field required",
        "request_id": bad.headers["x-request-id"],
    }
    assert (await client.post("/v1/datasets/-bad", content=CASES, headers=AUTH)).status_code == 400
    latin1 = await client.post("/v1/datasets/qa", content=b"\xff\xfe", headers=AUTH)
    assert latin1.status_code == 400
    listing = (await client.get("/v1/datasets", headers=AUTH)).json()
    assert [d["name"] for d in listing["datasets"]] == ["qa"]
    version = await client.get("/v1/datasets/qa/versions/1", headers=AUTH)
    assert version.json()["description"] == "first"
    missing = await client.get("/v1/datasets/qa/versions/9", headers=AUTH)
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"
    export = await client.get("/v1/datasets/qa/versions/1/export", headers=AUTH)
    assert export.headers["content-type"].startswith("application/x-ndjson")
    assert len(export.text.splitlines()) == 12


async def test_run_lifecycle_results_and_compare(client: httpx2.AsyncClient) -> None:
    await client.post("/v1/datasets/qa", content=CASES, headers=AUTH)
    started = await client.post("/v1/runs", json=run_body(label="base"), headers=AUTH)
    assert started.status_code == 202
    base = await wait_done(client, started.json()["run_id"])
    assert base["status"] == "completed"
    assert base["metrics"][0]["mean"] == 1.0
    other = await client.post("/v1/runs", json=run_body(accuracy=0.0, label="cand"), headers=AUTH)
    cand = await wait_done(client, other.json()["run_id"])

    page = (await client.get(f"/v1/runs/{base['run_id']}/results?limit=5", headers=AUTH)).json()
    assert [r["case_id"] for r in page["results"]] == ["c00", "c01", "c02", "c03", "c04"]
    assert page["next"] == "c04"
    rest = (
        await client.get(f"/v1/runs/{base['run_id']}/results?after=c04&limit=100", headers=AUTH)
    ).json()
    assert len(rest["results"]) == 7
    assert rest["next"] is None
    assert (await client.get("/v1/runs/nope/results", headers=AUTH)).status_code == 404

    runs = (await client.get("/v1/runs?label=base", headers=AUTH)).json()["runs"]
    assert [r["run_id"] for r in runs] == [base["run_id"]]
    assert len((await client.get("/v1/runs", headers=AUTH)).json()["runs"]) == 2

    report = await client.post(
        "/v1/compare", json={"baseline": base["run_id"], "candidate": cand["run_id"]}, headers=AUTH
    )
    assert report.status_code == 200
    assert report.json()["passed"] is False
    assert report.json()["comparisons"][0]["verdict"] == "worse"

    resumed = await client.post(f"/v1/runs/{base['run_id']}/resume", headers=AUTH)
    assert resumed.status_code == 202
    assert resumed.json()["status"] == "completed"
    assert (await client.get("/v1/runs/nope", headers=AUTH)).status_code == 404


async def test_run_validation_errors(client: httpx2.AsyncClient) -> None:
    await client.post("/v1/datasets/qa", content=CASES, headers=AUTH)
    plugin = run_body()
    plugin["scorers"] = [{"type": "plugin", "entry": "os:system"}]
    r = await client.post("/v1/runs", json=plugin, headers=AUTH)
    assert r.status_code == 400
    assert "import paths are not accepted" in r.json()["error"]["message"]
    extra = {**run_body(), "surprise": 1}
    r = await client.post("/v1/runs", json=extra, headers=AUTH)
    assert r.status_code == 400
    assert r.json()["error"]["message"].startswith("surprise: Extra inputs")
    two = run_body()
    two["options"] = {"concurrency": 0, "mode": "nope"}
    r = await client.post("/v1/runs", json=two, headers=AUTH)
    assert "(and 1 more)" in r.json()["error"]["message"]
    http = {**run_body(), "target": {"type": "http", "endpoint": "internal"}}
    assert (await client.post("/v1/runs", json=http, headers=AUTH)).status_code == 400
    r = await client.post(
        "/v1/runs", content="not json", headers={**AUTH, "content-type": "application/json"}
    )
    assert r.status_code == 400


async def test_compare_preconditions(
    client: httpx2.AsyncClient, sessions: sessionmaker[Session]
) -> None:
    await client.post("/v1/datasets/qa", content=CASES, headers=AUTH)
    runner = client._transport.app.app.state.services.runner  # type: ignore[attr-defined]
    from evalkit.runspec import RunConfig

    pending = runner.create_run(RunConfig.model_validate(run_body()))
    r = await client.post(
        "/v1/compare", json={"baseline": pending.id, "candidate": pending.id}, headers=AUTH
    )
    assert r.status_code == 412
    assert r.json()["error"]["code"] == "failed_precondition"
    resumed = await client.post(f"/v1/runs/{pending.id}/resume", headers=AUTH)
    assert resumed.status_code == 202
    assert resumed.json()["status"] == "queued"
    assert (await wait_done(client, pending.id))["status"] == "completed"


async def test_body_limit_and_unknown_routes(client: httpx2.AsyncClient) -> None:
    big = await client.post("/v1/datasets/qa", content="x" * (64 * 1024 + 1), headers=AUTH)
    assert big.status_code == 413
    assert big.json()["error"]["code"] == "payload_too_large"
    assert big.json()["error"]["request_id"] == big.headers["x-request-id"]

    async def chunks() -> AsyncIterator[bytes]:
        for _ in range(100):
            yield b"x" * 1024

    streamed = await client.post("/v1/datasets/qa", content=chunks(), headers=AUTH)
    assert streamed.status_code == 413
    nope = await client.get("/nope")
    assert nope.status_code == 404
    assert nope.json()["error"]["code"] == "not_found"
    method = await client.delete("/healthz")
    assert method.json()["error"]["code"] == "method_not_allowed"


async def test_unexpected_errors_are_generic(
    client: httpx2.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(session: Session) -> None:
        raise RuntimeError("secret detail")

    monkeypatch.setattr(api, "list_datasets", boom)
    r = await client.get("/v1/datasets", headers=AUTH)
    assert r.status_code == 500
    assert r.json()["error"]["message"] == "internal error"
    assert "secret" not in r.text


async def test_health_reports_database_down() -> None:
    broken = make_session_factory(make_engine("postgresql+psycopg://u@127.0.0.1:1/none"))
    app = create_app(settings(), sessions=broken, claude=FakeAnthropic().client())
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://t"
    ) as c:
        r = await c.get("/healthz")
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "unavailable"


async def test_background_failures_and_shutdown(sessions: sessionmaker[Session]) -> None:
    async with serving(sessions) as (c, svc):
        await c.post("/v1/datasets/qa", content=CASES, headers=AUTH)
        # A run is "running" before its target is built, so wait for the slow run's
        # target itself before breaking the factory for the next run.
        built = asyncio.Event()

        def recording_target(spec: Any) -> Any:
            target = build_target(spec, endpoints={}, providers=svc.runner.providers)
            built.set()
            return target

        svc.runner._target_factory = recording_target
        slow = await c.post("/v1/runs", json=run_body(latency_ms=300), headers=AUTH)
        await wait_status(c, slow.json()["run_id"], ("running",))
        await asyncio.wait_for(built.wait(), 10)

        def broken_target(spec: Any) -> Any:
            raise RuntimeError("target exploded")

        svc.runner._target_factory = broken_target
        failing = await c.post("/v1/runs", json=run_body(seed=5), headers=AUTH)
        failed = await wait_done(c, failing.json()["run_id"])
        assert "target exploded" in failed["error"]
    # Shutdown drained the embedded worker: the slow run's eight cases in flight
    # finished and were saved, the other four went back, and the run is queued
    # for the next worker.
    async with serving(sessions, EVALKIT_MAX_ACTIVE_RUNS="0") as (c, svc):
        after = (await c.get(f"/v1/runs/{slow.json()['run_id']}", headers=AUTH)).json()
        assert after["status"] == "queued"
        assert after["completed"] == 8
        assert after["workers"] == 0
        assert svc.worker is None


async def test_queue_cancel_and_resume(sessions: sessionmaker[Session]) -> None:
    async with serving(sessions, EVALKIT_MAX_ACTIVE_RUNS="0") as (c, _):
        await c.post("/v1/datasets/qa", content=CASES, headers=AUTH)
        body = run_body() | {"queue": {"priority": 5, "max_workers": 2}}
        queued = (await c.post("/v1/runs", json=body, headers=AUTH)).json()
        assert queued["status"] == "queued"
        assert (queued["priority"], queued["max_workers"]) == (5, 2)
        bad = await c.post(
            "/v1/runs", json=run_body() | {"queue": {"priority": 5000}}, headers=AUTH
        )
        assert bad.status_code == 400
        cancelled = await c.post(f"/v1/runs/{queued['run_id']}/cancel", headers=AUTH)
        assert cancelled.status_code == 202
        # Nobody was executing it, so it is cancelled at once.
        assert cancelled.json()["status"] == "cancelled"
        again = await c.post(f"/v1/runs/{queued['run_id']}/cancel", headers=AUTH)
        assert again.json()["status"] == "cancelled"
        resumed = await c.post(f"/v1/runs/{queued['run_id']}/resume", headers=AUTH)
        assert resumed.status_code == 409
        assert (await c.post("/v1/runs/nope/cancel", headers=AUTH)).status_code == 404
        assert (await c.post("/v1/runs/nope/resume", headers=AUTH)).status_code == 404

    async with serving(sessions) as (c, _):
        slow = (await c.post("/v1/runs", json=run_body(latency_ms=200), headers=AUTH)).json()
        await wait_status(c, slow["run_id"], ("running",))
        stopping = await c.post(f"/v1/runs/{slow['run_id']}/cancel", headers=AUTH)
        assert stopping.json()["status"] in ("cancelling", "cancelled")
        final = await wait_status(c, slow["run_id"], ("cancelled",))
        assert final["finished_at"] is not None
        assert final["completed"] < 12
        done = (await c.post("/v1/runs", json=run_body(), headers=AUTH)).json()
        await wait_done(c, done["run_id"])
        refused = await c.post(f"/v1/runs/{done['run_id']}/cancel", headers=AUTH)
        assert refused.status_code == 409
        assert refused.json()["error"]["code"] == "conflict"


async def test_unauthenticated_mode_on_loopback(sessions: sessionmaker[Session]) -> None:
    app = create_app(Settings(), sessions=sessions, claude=FakeAnthropic().client())
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://t"
    ) as c:
        assert (await c.get("/v1/datasets")).status_code == 200


def test_build_refuses_open_network_without_tokens(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="without EVALKIT_API_TOKENS"):
        build(Settings(host="0.0.0.0"))
    built, engine = build(
        settings(EVALKIT_DATABASE_URL=f"sqlite:///{tmp_path / 'b.db'}", EVALKIT_HOST="0.0.0.0")
    )
    assert isinstance(built, BodyLimit)
    engine.dispose()


def test_serve_starts_uvicorn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr("evalkit.api.uvicorn.run", lambda app, **kw: seen.update(kw))
    api.serve(settings(EVALKIT_DATABASE_URL=f"sqlite:///{tmp_path / 's.db'}", EVALKIT_PORT="9123"))
    assert seen["port"] == 9123
    assert seen["host"] == "127.0.0.1"
    from evalkit.cli import main

    monkeypatch.setenv("EVALKIT_HOST", "0.0.0.0")
    monkeypatch.delenv("EVALKIT_API_TOKENS", raising=False)
    assert main(["serve"]) == 2
    monkeypatch.setenv("EVALKIT_HOST", "127.0.0.1")
    monkeypatch.setenv("EVALKIT_DATABASE_URL", f"sqlite:///{tmp_path / 'c.db'}")
    assert main(["serve"]) == 0


async def test_lifespan_closes_resources(sessions: sessionmaker[Session]) -> None:
    app = create_app(settings(), sessions=sessions, claude=FakeAnthropic().client())
    async with app.router.lifespan_context(app):
        assert app.state.services.worker is not None


async def test_body_limit_passes_other_scopes_and_disconnects() -> None:
    seen: list[Any] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        seen.append((scope["type"], await receive(), await receive()))

    async def disconnect() -> Any:
        return {"type": "http.disconnect"}

    async def send(message: Any) -> None:
        return None

    limit = BodyLimit(app, 10)
    await limit({"type": "lifespan"}, disconnect, send)
    await limit({"type": "http", "headers": []}, disconnect, send)
    await limit(
        {"type": "http", "headers": []},
        _sequence([{"type": "http.request", "body": b"abc", "more_body": False}]),
        send,
    )
    assert seen[0][0] == "lifespan"
    assert seen[1][1:] == ({"type": "http.disconnect"}, {"type": "http.disconnect"})
    assert seen[2][1]["body"] == b"abc"
    assert seen[2][2] == {"type": "http.disconnect"}


def _sequence(messages: list[Any]) -> Any:
    async def receive() -> Any:
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    return receive
