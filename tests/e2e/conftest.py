"""PostgreSQL in a container, a real evalkit server and a stand-in app under test."""

import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx2
import pytest

from evalkit.db import make_engine, upgrade

pytestmark = pytest.mark.e2e

TOKEN = "e2e-token-0123456789abcdef"


def docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    if not docker_available():
        pytest.skip("Docker is not available")
    # The reaper container is not needed: the context manager below removes the
    # database container, and some Docker setups cannot publish the reaper's port.
    os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")
    from testcontainers.community.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine", driver="psycopg") as pg:
        url = pg.get_connection_url()
        engine = make_engine(url)
        upgrade(engine)
        engine.dispose()
        yield url


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class AppUnderTest(BaseHTTPRequestHandler):
    """Answers "a<n>" for question "<n>", except multiples of seven: a known 86% accuracy."""

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n = int(body["input"]["q"])
        answer = f"a{n}" if n % 7 else "not sure"
        payload = json.dumps({"output": answer, "auth": self.headers.get("Authorization")}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture(scope="session")
def app_under_test() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), AppUnderTest)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/answer"
    server.shutdown()
    server.server_close()


def evalkit_env(postgres_url: str, **extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("EVALKIT_", "ANTHROPIC_"))}
    env.update({"EVALKIT_DATABASE_URL": postgres_url, "EVALKIT_LOG_LEVEL": "WARNING", **extra})
    return env


@pytest.fixture(scope="session")
def server(
    postgres_url: str, app_under_test: str, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[str]:
    workdir = tmp_path_factory.mktemp("server")
    endpoints = workdir / "endpoints.toml"
    endpoints.write_text(
        f'[endpoints.support-bot]\nurl = "{app_under_test}"\nallow_private = true\n'
        'secret_headers = { Authorization = "BOT_TOKEN" }\n'
    )
    port = free_port()
    env = evalkit_env(
        postgres_url,
        EVALKIT_API_TOKENS=TOKEN,
        EVALKIT_PORT=str(port),
        EVALKIT_HTTP_TARGETS_FILE=str(endpoints),
        BOT_TOKEN="Bearer bot-secret-value",
    )
    log = (workdir / "server.log").open("w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "evalkit", "serve"], env=env, stdout=log, stderr=log
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            if httpx2.get(f"{base}/healthz", timeout=1).status_code == 200:
                break
        except httpx2.TransportError:
            time.sleep(0.2)
    else:
        proc.kill()
        raise AssertionError(Path(log.name).read_text())
    yield base
    proc.terminate()
    proc.wait(timeout=30)
    log.close()
    assert "bot-secret-value" not in Path(log.name).read_text()
