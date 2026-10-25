"""The whole service against PostgreSQL: real server process, real HTTP target."""

import json
import subprocess
import sys
import time
from typing import Any

import httpx2
import pytest

from tests.e2e.conftest import TOKEN, evalkit_env

pytestmark = pytest.mark.e2e
AUTH = {"Authorization": f"Bearer {TOKEN}"}
CASES = "\n".join(
    json.dumps({"id": f"q{n:03d}", "input": {"q": str(n)}, "expected": f"a{n}"})
    for n in range(1, 151)
)


def wait(client: httpx2.Client, run_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        body: dict[str, Any] = client.get(f"/v1/runs/{run_id}").json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.2)
    raise AssertionError(f"run {run_id} did not finish")


def test_import_run_compare_over_http(server: str, postgres_url: str) -> None:
    with httpx2.Client(base_url=server, headers=AUTH, timeout=30) as client:
        assert client.post("/v1/datasets/support", content=CASES).status_code == 201
        assert client.post("/v1/datasets/support", content=CASES).status_code == 200

        http_run = client.post(
            "/v1/runs",
            json={
                "label": "app-v2",
                "dataset": {"name": "support"},
                "target": {"type": "http", "endpoint": "support-bot"},
                "scorers": [{"type": "exact"}, {"type": "token_f1", "gate": False}],
                "options": {"concurrency": 16},
            },
        )
        assert http_run.status_code == 202, http_run.text
        app = wait(client, http_run.json()["run_id"])
        assert app["status"] == "completed"
        assert app["ok"] == 150
        # 21 of 150 questions are multiples of seven, which the app gets wrong.
        assert app["metrics"][0]["mean"] == pytest.approx(129 / 150)

        page = client.get(f"/v1/runs/{app['run_id']}/results?limit=2").json()
        assert len(page["results"]) == 2
        # The secret header reached the app, but nothing secret was stored.
        assert "bot-secret-value" not in json.dumps(page)

        baseline = client.post(
            "/v1/runs",
            json={
                "label": "app-v1",
                "dataset": {"name": "support"},
                "target": {"type": "fake", "accuracy": 1.0},
                "scorers": [{"type": "exact"}],
            },
        )
        base = wait(client, baseline.json()["run_id"])
        report = client.post(
            "/v1/compare", json={"baseline": base["run_id"], "candidate": app["run_id"]}
        ).json()
        assert report["passed"] is False
        assert report["regressions"] == ["exact"]

        metrics = client.get("/metrics").text
        assert 'evalkit_cases_total{source="target",status="ok"}' in metrics

    gate = subprocess.run(
        [sys.executable, "-m", "evalkit", "gate", "label:app-v1", "label:app-v2"],
        env=evalkit_env(postgres_url),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert gate.returncode == 1, gate.stderr
    assert "FAIL: exact" in gate.stdout
    same = subprocess.run(
        [sys.executable, "-m", "evalkit", "gate", "label:app-v2", "label:app-v2"],
        env=evalkit_env(postgres_url),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert same.returncode == 0, same.stderr


def test_unknown_endpoint_and_auth_over_the_wire(server: str) -> None:
    assert httpx2.get(f"{server}/v1/runs").status_code == 401
    with httpx2.Client(base_url=server, headers=AUTH) as client:
        r = client.post(
            "/v1/runs",
            json={
                "dataset": {"name": "support"},
                "target": {"type": "http", "endpoint": "metadata"},
                "scorers": [{"type": "exact"}],
            },
        )
        assert r.status_code == 400
        assert "unknown HTTP endpoint" in r.json()["error"]["message"]
