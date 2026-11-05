"""The shipped image as a fleet: an API container and evalkit worker containers on
PostgreSQL, calling one rate-limited provider with one key, with long cached
outputs in an S3-compatible blob store.

The image is built from the repository's Dockerfile, or taken from
``EVALKIT_E2E_IMAGE`` when that names an image already built from it. Every
container uses the host network, so the database, the store and the provider
below are reached at the same addresses as from the tests.

One test, in three phases on its own database:

1. Eight workers share a 1,600-case run limited to 200 requests per second. The
   provider answers one request with 429 and Retry-After: 1, and no worker
   calls it again until the cool-down is over. The achieved rate is measured at
   the provider and printed.
2. Four of them share an 800-case run; one is killed with SIGKILL partway. The
   other three finish it with every case recorded once, and the run passes
   ``evalkit run verify`` and the blob store passes ``evalkit cache scrub``.
3. A slow run is cancelled while three workers execute it, and every container
   then stops cleanly on SIGTERM.
"""

import json
import os
import subprocess
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx2
import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import make_url

from evalkit import aggregates
from evalkit.blobs import S3BlobStore
from evalkit.db import make_engine, make_session_factory, upgrade
from evalkit.db.models import CachedResponse, CaseLease, Result
from tests.e2e.conftest import ACCESS, SECRET, TOKEN, docker_available, free_port

pytestmark = pytest.mark.e2e

REPO = Path(__file__).resolve().parents[2]
AUTH = {"Authorization": f"Bearer {TOKEN}"}
PROVIDER_KEY = "Bearer provider-key-0123456789"
BUCKET = "evalkit-fleet"
RATE = 200.0
RETRY_AFTER = 1.0
LONG_EVERY = 10  # every tenth answer is long enough for the blob store


def answer(n: int) -> str:
    return f"a{n}" + ("." * 2048 if n % LONG_EVERY == 0 else "")


def until(check: Callable[[], bool], seconds: float, what: str) -> float:
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        if check():
            return time.monotonic() - start
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


# --- the provider -------------------------------------------------------------


@dataclass
class Call:
    at: float
    worker: str
    case: int


@dataclass
class Provider:
    """Records every call; answers call number ``throttle_at`` with a 429."""

    url: str = ""
    calls: list[Call] = field(default_factory=list)
    throttle_at: int | None = None
    throttled: Call | None = None
    refused: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def reset(self, throttle_at: int | None = None) -> None:
        with self.lock:
            self.calls, self.throttled, self.throttle_at = [], None, throttle_at

    def snapshot(self) -> list[Call]:
        with self.lock:
            return list(self.calls)


def provider_handler(state: Provider) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.headers.get("Authorization") != PROVIDER_KEY:
                with state.lock:
                    state.refused += 1
                self._send(401, {"error": "bad key"})
                return
            n = int(body["input"][1:])
            with state.lock:
                call = Call(time.monotonic(), self.headers.get("X-Worker", "?"), n)
                state.calls.append(call)
                throttle = len(state.calls) == state.throttle_at
                if throttle:
                    state.throttled = call
            if throttle:
                self._send(429, {"error": "slow down"}, {"Retry-After": str(int(RETRY_AFTER))})
                return
            time.sleep(0.01)  # a fast provider, but not an instant one
            self._send(200, {"output": answer(n)})

        def _send(self, status: int, payload: Any, headers: dict[str, str] | None = None) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            for name, value in {"Content-Type": "application/json", **(headers or {})}.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


@pytest.fixture
def provider() -> Iterator[Provider]:
    state = Provider()
    server = ThreadingHTTPServer(("127.0.0.1", 0), provider_handler(state))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_address[1]}/v1/generate"
    yield state
    server.shutdown()
    server.server_close()


# --- the image and its containers ---------------------------------------------


def docker(*args: str, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


@pytest.fixture(scope="module")
def image() -> str:
    if not docker_available():
        pytest.skip("Docker is not available")
    prebuilt = os.environ.get("EVALKIT_E2E_IMAGE", "").strip()
    if prebuilt:
        if docker("image", "inspect", prebuilt).returncode != 0:
            pytest.fail(f"EVALKIT_E2E_IMAGE={prebuilt} is not a local image")
        return prebuilt
    tag = "evalkit:e2e"
    built = docker("build", "-t", tag, str(REPO), timeout=900)
    if built.returncode != 0:
        pytest.fail(f"docker build failed:\n{built.stdout[-4000:]}\n{built.stderr[-4000:]}")
    return tag


@pytest.fixture
def fleet_db(postgres_url: str) -> Iterator[str]:
    """A database of its own, so no run left queued by another module is taken."""
    name = f"fleet_{uuid.uuid4().hex[:8]}"
    admin = create_engine(postgres_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    url = make_url(postgres_url).set(database=name).render_as_string(hide_password=False)
    engine = make_engine(url)
    upgrade(engine)
    engine.dispose()
    yield url
    with admin.connect() as conn:
        conn.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
    admin.dispose()


class Fleet:
    """Containers started from one image, removed at teardown."""

    def __init__(self, image: str, workdir: Path, env: dict[str, str], provider_url: str) -> None:
        self.image = image
        self.workdir = workdir
        self.env = env
        self.provider_url = provider_url
        self.prefix = f"evalkit-e2e-{uuid.uuid4().hex[:6]}"
        self.names: list[str] = []
        # The image runs as uid 10001: the mounted files must be readable to it.
        workdir.chmod(0o755)

    def endpoints(self, name: str) -> Path:
        # Each container tags its calls, so the provider can tell the workers apart.
        path = self.workdir / f"{name}.toml"
        path.write_text(
            f'[endpoints.provider]\nurl = "{self.provider_url}"\nallow_private = true\n'
            f'headers = {{ X-Worker = "{name}" }}\n'
            'secret_headers = { Authorization = "PROVIDER_KEY" }\n'
        )
        path.chmod(0o644)
        return path

    def _args(self, name: str, env: dict[str, str]) -> list[str]:
        args = ["--network", "host", "-v", f"{self.endpoints(name)}:/etc/evalkit/endpoints.toml:ro"]
        for key, value in {**self.env, **env}.items():
            args += ["-e", f"{key}={value}"]
        return args

    def start(self, name: str, command: list[str], **env: str) -> str:
        container = f"{self.prefix}-{name}"
        started = docker(
            "run", "-d", "--name", container, *self._args(name, env), self.image, *command
        )
        assert started.returncode == 0, started.stderr
        self.names.append(container)
        return container

    def once(self, command: list[str], **env: str) -> subprocess.CompletedProcess[str]:
        """Runs one evalkit command in a container of its own and waits for it."""
        return docker("run", "--rm", *self._args("cli", env), self.image, *command, timeout=300)

    def stop(self, container: str) -> int:
        """SIGTERM, then the container's exit code."""
        assert docker("stop", "-t", "60", container, timeout=90).returncode == 0
        return int(docker("inspect", "-f", "{{.State.ExitCode}}", container).stdout.strip())

    def kill(self, container: str) -> None:
        assert docker("kill", "--signal", "KILL", container).returncode == 0

    def logs(self, container: str) -> str:
        out = docker("logs", container)
        return out.stdout + out.stderr

    def remove(self) -> None:
        for container in self.names:
            (self.workdir / f"{container}.log").write_text(self.logs(container))
            docker("rm", "-f", container)


@pytest.fixture
def fleet(
    image: str, fleet_db: str, s3_endpoint: str, provider: Provider, tmp_path: Path
) -> Iterator[Fleet]:
    store = S3BlobStore(s3_endpoint, BUCKET, access_key=ACCESS, secret_key=SECRET)
    store.create_bucket()
    env = {
        "EVALKIT_DATABASE_URL": fleet_db,
        "EVALKIT_HTTP_TARGETS_FILE": "/etc/evalkit/endpoints.toml",
        "PROVIDER_KEY": PROVIDER_KEY,
        "EVALKIT_BLOB_URL": f"s3+{s3_endpoint}/{BUCKET}",
        "EVALKIT_BLOB_ACCESS_KEY": ACCESS,
        "EVALKIT_BLOB_SECRET_KEY": SECRET,
        "EVALKIT_BLOB_MIN_BYTES": "1KiB",
        "EVALKIT_LOG_LEVEL": "INFO",
    }
    containers = Fleet(image, tmp_path, env, provider.url)
    yield containers
    containers.remove()


# --- the test -----------------------------------------------------------------


def dataset(prefix: str, cases: int) -> str:
    return "\n".join(
        json.dumps({"id": f"{prefix}{i:04d}", "input": f"{prefix}{i}", "expected": answer(i)})
        for i in range(cases)
    )


def body(name: str, rate: float, **queue: Any) -> dict[str, Any]:
    return {
        "label": name,
        "dataset": {"name": name},
        "target": {"type": "http", "endpoint": "provider"},
        "scorers": [{"type": "exact"}],
        "options": {
            "concurrency": 8,
            "rate_limit": {"requests_per_second": rate, "burst": 1},
            "retry": {"base_delay": 0.05},
        },
        "queue": queue,
    }


def achieved_rate(calls: list[Call]) -> float:
    return (len(calls) - 1) / (calls[-1].at - calls[0].at)


def busiest_second(calls: list[Call]) -> int:
    times = [c.at for c in calls]
    most, j = 0, 0
    for i, t in enumerate(times):
        while times[j] <= t - 1.0:
            j += 1
        most = max(most, i - j + 1)
    return most


@dataclass
class Scene:
    fleet: Fleet
    provider: Provider
    client: httpx2.Client
    sessions: Any
    s3_endpoint: str

    def run(self, run_id: str) -> dict[str, Any]:
        summary: dict[str, Any] = self.client.get(f"/v1/runs/{run_id}").json()
        return summary

    def submit(self, name: str, rate: float) -> str:
        response = self.client.post("/v1/runs", json=body(name, rate))
        assert response.status_code == 202, response.text
        run_id: str = response.json()["run_id"]
        return run_id

    def listeners(self) -> int:
        with self.sessions() as s:
            query = text(
                "SELECT count(*) FROM pg_stat_activity WHERE query = 'LISTEN \"evalkit_runs\"'"
            )
            return int(s.scalar(query) or 0)

    def recorded(self, run_id: str) -> list[str]:
        with self.sessions() as s:
            return list(s.scalars(select(Result.case_id).where(Result.run_id == run_id)))

    def workers_seen(self) -> set[str]:
        return {c.worker for c in self.provider.snapshot()}


def eight_workers_share_one_rate(scene: Scene) -> None:
    """Phase 1: one key, one rate, and one 429 that pauses every worker."""
    scene.provider.reset(throttle_at=800)
    run_id = scene.submit("rate", RATE)
    until(lambda: scene.run(run_id)["status"] == "completed", 120, "the rate-limited run")
    calls = scene.provider.snapshot()
    throttled = scene.provider.throttled
    assert throttled is not None
    assert scene.workers_seen() == {f"w{i}" for i in range(8)}
    ids = scene.recorded(run_id)
    assert len(ids) == len(set(ids)) == 1600
    assert scene.run(run_id)["ok"] == 1600, "the throttled case was retried"
    assert scene.provider.refused == 0
    # Steady state: from the 50th call (every worker has joined) to the 429.
    before = [c for c in calls if c.at < throttled.at][50:]
    pre_rate = achieved_rate(before)
    assert abs(pre_rate - RATE) <= 0.05 * RATE, pre_rate
    # The cool-down reaches every worker shortly after the 429.
    quiet = [c for c in calls if throttled.at + 0.25 <= c.at < throttled.at + RETRY_AFTER - 0.1]
    assert quiet == [], [(c.worker, round(c.at - throttled.at, 3)) for c in quiet]
    resumed = [c for c in calls if c.at > throttled.at + RETRY_AFTER]
    post_rate = achieved_rate(resumed[50:-50])
    assert abs(post_rate - RATE) <= 0.05 * RATE, post_rate
    peak = busiest_second(calls)
    assert peak <= RATE * 1.05 + 1, peak
    print(
        f"\nachieved provider rate, 8 worker containers at {RATE:.0f}/s: "
        f"{pre_rate:.1f}/s before the 429, {post_rate:.1f}/s after it, "
        f"busiest 1 s window {peak} calls; first call after the 429 at "
        f"+{resumed[0].at - throttled.at:.2f} s (Retry-After {RETRY_AFTER:.0f} s)"
    )


def a_killed_worker_loses_nothing(scene: Scene, killed: str) -> None:
    """Phase 2: four workers share a run and one is SIGKILLed partway."""
    scene.provider.reset()
    run_id = scene.submit("fleet", 100)
    until(lambda: scene.run(run_id)["workers"] == 4, 15, "four slots")
    until(lambda: scene.run(run_id)["completed"] >= 200, 60, "progress")
    scene.fleet.kill(killed)
    until(lambda: scene.run(run_id)["status"] == "completed", 120, "the shared run")
    ids = scene.recorded(run_id)
    assert len(ids) == len(set(ids)) == 800
    with scene.sessions() as s:
        leased = select(CaseLease.case_id, CaseLease.fence).where(CaseLease.run_id == run_id)
        fences: dict[str, int] = dict(s.execute(leased).all())
        digests = list(
            s.scalars(
                select(CachedResponse.output_blob).where(CachedResponse.output_blob.is_not(None))
            )
        )
    with scene.sessions() as s:
        check = aggregates.verify(s, run_id)  # its own snapshot: a session of its own
    assert max(fences.values()) == 2, "the killed worker's cases were claimed again"
    assert check.ok, check.problems
    # Every case reached the provider, and only a case claimed again was called twice.
    calls = Counter(f"f{c.case:04d}" for c in scene.provider.snapshot())
    assert set(calls) == set(fences)
    assert {case for case, n in calls.items() if n > 1} <= {
        case for case, fence in fences.items() if fence == 2
    }
    assert scene.workers_seen() == {f"w{i}" for i in range(4)}
    # Long answers went to the blob store from inside the containers. The same
    # answer from both runs is one blob: blobs are named by their hash.
    assert len(digests) == (1600 + 800) // LONG_EVERY
    assert len(set(digests)) == 1600 // LONG_EVERY
    store = S3BlobStore(scene.s3_endpoint, BUCKET, access_key=ACCESS, secret_key=SECRET)
    assert set(digests) <= {b.digest for b in store.list()}
    verified = scene.fleet.once(["run", "verify", run_id, "--json"])
    assert verified.returncode == 0, verified.stdout + verified.stderr
    scrubbed = scene.fleet.once(["cache", "scrub", "--json"])
    assert scrubbed.returncode == 0, scrubbed.stdout + scrubbed.stderr
    assert json.loads(scrubbed.stdout)["corrupt"] == []


def a_cancellation_reaches_the_fleet(scene: Scene) -> None:
    """Phase 3: a slow run is cancelled while three workers execute it."""
    run_id = scene.submit("slow", 20)
    until(lambda: scene.run(run_id)["workers"] == 3, 15, "three slots")
    until(lambda: scene.run(run_id)["completed"] >= 20, 60, "progress")
    assert scene.client.post(f"/v1/runs/{run_id}/cancel").status_code == 202
    took = until(lambda: scene.run(run_id)["status"] == "cancelled", 10, "the cancellation")
    assert took < 5
    print(f"cancelled while three worker containers executed it: {took:.2f} s")
    cancelled = scene.run(run_id)
    assert cancelled["completed"] < 800
    assert cancelled["workers"] == 0


def test_the_image_as_an_api_and_a_fleet_of_workers(
    fleet: Fleet, fleet_db: str, provider: Provider, s3_endpoint: str
) -> None:
    engine = make_engine(fleet_db)
    port = free_port()
    api = fleet.start(
        "api",
        ["serve"],
        EVALKIT_API_TOKENS=TOKEN,
        EVALKIT_PORT=str(port),
        EVALKIT_MAX_ACTIVE_RUNS="0",
    )
    client = httpx2.Client(base_url=f"http://127.0.0.1:{port}", headers=AUTH, timeout=30)
    scene = Scene(fleet, provider, client, make_session_factory(engine), s3_endpoint)

    def healthy() -> bool:
        try:
            return client.get("/healthz").status_code == 200
        except httpx2.TransportError:
            return False

    try:
        until(healthy, 60, "the API container")
        for name, cases in (("rate", 1600), ("fleet", 800), ("slow", 800)):
            upload = client.post(f"/v1/datasets/{name}", content=dataset(name[0], cases))
            assert upload.status_code == 201, upload.text
        workers = [
            fleet.start(
                f"w{i}",
                ["worker", "--runs", "1"],
                EVALKIT_LEASE_SECONDS="3",
                # Polling is slow on purpose: work must arrive by notification.
                EVALKIT_WORKER_POLL_SECONDS="60",
                EVALKIT_DRAIN_SECONDS="20",
            )
            for i in range(8)
        ]
        until(lambda: scene.listeners() >= 8, 90, "eight listening worker containers")

        eight_workers_share_one_rate(scene)
        for worker in workers[4:]:
            assert fleet.stop(worker) == 0
        a_killed_worker_loses_nothing(scene, workers[0])
        a_cancellation_reaches_the_fleet(scene)
        for worker in [*workers[1:4], api]:
            assert fleet.stop(worker) == 0
        assert "worker stopped" in fleet.logs(workers[1])
        assert PROVIDER_KEY not in fleet.logs(workers[1])
    finally:
        client.close()
        engine.dispose()
