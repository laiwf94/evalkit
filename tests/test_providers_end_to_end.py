"""End to end: a three-provider matrix on the bundled dataset, judged by a fourth
provider, then a prompt version promoted to production and gated.

Everything goes over real sockets to the strict fake servers, and evalkit is
configured only the way an operator configures it: ``ANTHROPIC_BASE_URL``,
``EVALKIT_PROVIDERS_FILE`` (the built-in ``openai`` and ``gemini`` redefined to
point here, and the grader from examples/providers/providers.toml),
``EVALKIT_PRICES_FILE`` from the example and API keys in the environment. The
matrix itself runs as the installed command in a child process. The run configs
and prompts are the files in examples/providers, unchanged.

Each model answers a case correctly for a share of the cases decided by a hash
of the model, the prompt version and the case, so the expected quality of every
run is known exactly. The grader scores 5 when the response is the reference and
1 otherwise, like a perfect judge would.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx2
import pytest

from evalkit import cli
from evalkit.pricing import ANTHROPIC_PRICES, GEMINI_PRICES, OPENAI_PRICES, PriceList
from evalkit.targets.fake import unit
from tests.fake_anthropic import FakeAnthropic, json_reply, message
from tests.fake_gemini import FakeGemini, text_of
from tests.fake_openai import FakeOpenAI

ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "examples" / "providers"
ROWS = [
    json.loads(line) for line in (ROOT / "examples/data/support.jsonl").read_text().splitlines()
]
N = len(ROWS)
# The question each case asks, to its index and reference answer.
EXPECTED = {row["input"]["question"]: (i, row["expected"]) for i, row in enumerate(ROWS)}
WRONG = "Please contact support."

# How often each model answers correctly with each prompt version. The second
# version asks for the menu path in one sentence, which helps every model.
ACCURACY = {
    ("claude-sonnet-5-5", 1): 0.97,
    ("gpt-5-mini", 1): 0.60,
    ("gemini-2.5-flash", 1): 0.80,
    ("gemini-2.5-flash", 2): 0.95,
}
V2_MARKER = "Answer in one sentence that names the menu path"
TAG = re.compile(r"<(reference|response)>\n(.*?)\n</\1>", re.S)

CLAUDE_KEY = "sk-ant-e2e-0000000000"
OPENAI_KEY = "sk-e2e-openai-0000000000"
GEMINI_KEY = "g-e2e-key-0000000000"


def correct(model: str, version: int, index: int) -> bool:
    return unit(f"{model}@{version}", index) < ACCURACY[(model, version)]


def expected_quality(model: str, version: int) -> float:
    return sum(correct(model, version, i) for i in range(N)) / N


def reply(model: str, system: str, question: str) -> str:
    index, answer = EXPECTED[question]
    version = 2 if V2_MARKER in system else 1
    return answer if correct(model, version, index) else WRONG


@dataclass
class Served:
    """One fake behind a real HTTP server on 127.0.0.1."""

    name: str
    server: ThreadingHTTPServer
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"


def serve(name: str, make: Callable[[str], Callable[[httpx2.Request], httpx2.Response]]) -> Served:
    """Start an HTTP server that hands each request to the handler ``make``
    returns for the server's own URL, and record every path and body."""
    holder: dict[str, Any] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            content = self.rfile.read(int(self.headers.get("content-length", 0)))
            request = httpx2.Request(
                "POST",
                f"{served.url}{self.path}",
                headers=dict(self.headers.items()),
                content=content,
            )
            served.calls.append((self.path.split("?")[0], json.loads(content) if content else {}))
            response = holder["handle"](request)
            body = response.read()
            self.send_response(response.status_code)
            for key, value in response.headers.items():
                if key.lower() not in ("content-length", "transfer-encoding", "connection"):
                    self.send_header(key, value)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    served = Served(name, server)
    holder["handle"] = make(served.url)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return served


def claude_answer(body: dict[str, Any]) -> httpx2.Response:
    system = body.get("system") or ""
    if isinstance(system, list):
        system = "".join(block["text"] for block in system)
    question = body["messages"][-1]["content"]
    if isinstance(question, list):
        question = "".join(block["text"] for block in question)
    return json_reply(message(reply(body["model"], system, question), model=body["model"]))


def openai_answer(body: dict[str, Any]) -> str:
    messages = body["messages"]
    system = messages[0]["content"] if messages[0]["role"] == "system" else ""
    return reply(body["model"], system, messages[-1]["content"])


def gemini_answer(model: str, body: dict[str, Any]) -> str:
    system = (
        text_of({"contents": [body["systemInstruction"]]}) if "systemInstruction" in body else ""
    )
    return reply(model, system, body["contents"][-1]["parts"][0]["text"])


def grade(body: dict[str, Any]) -> str:
    """A judge that knows the right answer: 5 for the reference, 1 for anything else."""
    assert body["model"] == "qwen3-32b"
    assert body["response_format"]["type"] == "json_schema"
    found = dict(TAG.findall(body["messages"][-1]["content"]))
    score = 5 if found["response"] == found["reference"] else 1
    return json.dumps({"reasoning": "compared with the reference", "score": score})


@dataclass
class World:
    claude: Served
    openai: Served
    gemini: Served
    grader: Served
    fakes: list[FakeOpenAI | FakeGemini]

    def evalkit(self, *args: str, expect: int = 0, child: bool = False) -> str:
        """Run the CLI with this world's environment: in this process, or as the
        installed command in a child process when ``child`` is set."""
        if child:
            done = subprocess.run(
                [sys.executable, "-m", "evalkit", *args],
                env=dict(os.environ),
                capture_output=True,
                text=True,
                timeout=120,
            )
            code, text, errors = done.returncode, done.stdout, done.stderr
        else:
            out = io.StringIO()
            code, text, errors = cli.main(list(args), out=out), out.getvalue(), ""
        assert code == expect, f"exit {code}\n{text}\n{errors}"
        return text

    def json(self, *args: str, expect: int = 0) -> Any:
        return json.loads(self.evalkit(*args, "--json", expect=expect))

    def targets(self, served: Served) -> list[dict[str, Any]]:
        """Bodies of the answer requests ``served`` received (not counts or embeddings)."""
        return [
            body
            for path, body in served.calls
            if path.endswith(("/v1/messages", "/chat/completions", ":generateContent"))
        ]


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    claude_fake = FakeAnthropic()
    claude_fake.default = claude_answer

    def claude_handler(url: str) -> Callable[[httpx2.Request], httpx2.Response]:
        def handle(request: httpx2.Request) -> httpx2.Response:
            assert request.headers.get("x-api-key") == CLAUDE_KEY
            if request.url.path.endswith("/count_tokens"):
                return json_reply({"input_tokens": 150})
            return claude_fake.handler(request)

        return handle

    fakes: list[FakeOpenAI | FakeGemini] = []

    def openai_handler(url: str) -> Callable[[httpx2.Request], httpx2.Response]:
        fake = FakeOpenAI(
            f"{url}/v1", api_key=OPENAI_KEY, answer=openai_answer, models={"gpt-5-mini"}
        )
        fakes.append(fake)
        return fake.handle

    def gemini_handler(url: str) -> Callable[[httpx2.Request], httpx2.Response]:
        fake = FakeGemini(
            f"{url}/v1beta", api_key=GEMINI_KEY, answer=gemini_answer, models={"gemini-2.5-flash"}
        )
        fakes.append(fake)
        return fake.handle

    def grader_handler(url: str) -> Callable[[httpx2.Request], httpx2.Response]:
        fake = FakeOpenAI(
            f"{url}/v1",
            api_key=None,
            answer=grade,
            models={"qwen3-32b"},
            output_limit_field="max_tokens",
        )
        fakes.append(fake)
        return fake.handle

    servers = [
        serve("claude", claude_handler),
        serve("openai", openai_handler),
        serve("gemini", gemini_handler),
        serve("grader", grader_handler),
    ]
    claude, openai, gemini, grader = servers

    # The example's providers file, with the grader pointed at the fake and the
    # built-in openai and gemini providers redefined to point here as well.
    example = (EXAMPLES / "providers.toml").read_text()
    assert 'base_url = "http://localhost:8000/v1"' in example
    providers = tmp_path / "providers.toml"
    providers.write_text(
        example.replace("http://localhost:8000/v1", f"{grader.url}/v1")
        + f'\n[providers.openai]\nkind = "openai_compat"\nbase_url = "{openai.url}/v1"\n'
        'api_key_env = "OPENAI_API_KEY"\n'
        f'\n[providers.gemini]\nkind = "gemini"\nbase_url = "{gemini.url}/v1beta"\n'
        'api_key_env = "GEMINI_API_KEY"\n'
    )
    for key in list(os.environ):
        if key.upper().endswith("_PROXY") or key.startswith(("EVALKIT_", "ANTHROPIC_")):
            monkeypatch.delenv(key)
    for key, value in {
        "NO_PROXY": "*",
        "EVALKIT_DATABASE_URL": f"sqlite:///{tmp_path / 'e2e.db'}",
        "EVALKIT_PROVIDERS_FILE": str(providers),
        "EVALKIT_PRICES_FILE": str(EXAMPLES / "prices.toml"),
        "ANTHROPIC_BASE_URL": claude.url,
        "ANTHROPIC_API_KEY": CLAUDE_KEY,
        "OPENAI_API_KEY": OPENAI_KEY,
        "GEMINI_API_KEY": GEMINI_KEY,
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(ROOT)
    yield World(claude, openai, gemini, grader, fakes)
    for served in servers:
        served.server.shutdown()
        served.server.server_close()
    for fake in fakes:
        assert fake.rejected == [], fake.rejected


def cost(prices: PriceList, model: str, input_tokens: int, output_tokens: int) -> float:
    price = prices.prices[model]
    return (input_tokens * price.input + output_tokens * price.output) / 1e6


EX = "examples/providers"
NAMES = ["claude-sonnet-5-5", "openai:gpt-5-mini", "gemini:gemini-2.5-flash"]
VARIANTS = [
    # name, provider, model, price list, output tokens the fake reports per answer
    (NAMES[0], "anthropic", "claude-sonnet-5-5", ANTHROPIC_PRICES, 20),
    (NAMES[1], "openai", "gpt-5-mini", OPENAI_PRICES, 10),
    (NAMES[2], "gemini", "gemini-2.5-flash", GEMINI_PRICES, 10),
]


def test_three_providers_judged_by_a_fourth_then_a_prompt_promoted(world: World) -> None:
    # A prompt in the registry, labelled production.
    world.evalkit("prompt", "push", "support", f"{EX}/prompt-v1.toml", "--note", "first")
    world.evalkit("prompt", "label", "support@1", "production")
    check_estimate(world)

    # The matrix: one config, the same dataset on three providers, run by the
    # installed command.
    world.evalkit("run", f"{EX}/matrix.toml", child=True)
    check_requests(world)
    shown = check_accounting(world)
    check_frontier(world, shown)
    check_promotion(world, shown)


def check_estimate(world: World) -> None:
    """Input is counted on Claude and Gemini, and estimated (and labelled so) for
    OpenAI, which has no counting endpoint; nothing is answered."""
    estimate = world.json("estimate", f"{EX}/matrix.toml")
    by_name = {v["name"]: v["estimate"] for v in estimate["variants"]}
    assert list(by_name) == NAMES
    assert by_name[NAMES[0]]["target_input_tokens"] == 150 * N
    assert by_name[NAMES[0]]["target_input_estimated"] is False
    assert by_name[NAMES[1]]["target_input_estimated"] is True
    assert by_name[NAMES[2]]["target_input_estimated"] is False
    assert world.targets(world.claude) == []
    assert world.targets(world.gemini) == []
    assert world.openai.calls == []


def check_requests(world: World) -> None:
    """Each provider answered every case with its own model and the registry
    prompt, and only the grader was asked to judge."""
    for served, (_, _, model, _, _) in zip(
        (world.claude, world.openai, world.gemini), VARIANTS, strict=True
    ):
        bodies = world.targets(served)
        assert len(bodies) == N, served.name
        assert {b.get("model", model) for b in bodies} == {model}
        assert "Help centre:" in json.dumps(bodies[0])
        assert V2_MARKER not in json.dumps(bodies)
        assert "<rubric>" not in json.dumps(bodies)
    # A verdict is cached by judge, case and response, so two models giving the
    # same answer to a case cost one judge call.
    distinct = {(i, correct(v[2], 1, i)) for i in range(N) for v in VARIANTS}
    assert len(world.targets(world.grader)) == len(distinct) < 3 * N


def check_accounting(world: World) -> dict[str, Any]:
    """Tokens and cost are accounted for each provider at its own prices, and the
    judge's at the grader's: each run pays for the verdicts it was first to need.
    Each run records its prompt version, its judge and the prices it used."""
    shown = {}
    seen: set[tuple[int, bool]] = set()
    for name, provider, model, prices, tokens_out in VARIANTS:
        run = world.json("show", f"label:providers[{name}]")
        shown[name] = run
        assert run["variant"] == name
        assert (run["ok"], run["errors"], run["cached"]) == (N, 0, 0)
        assert (run["input_tokens"], run["output_tokens"]) == (N * 100, N * tokens_out)
        assert run["cost_usd"] == pytest.approx(cost(prices, model, N * 100, N * tokens_out))
        judged = {(i, correct(model, 1, i)) for i in range(N)} - seen
        seen |= judged
        assert run["judge_input_tokens"] == 100 * len(judged), name
        assert run["judge_cost_usd"] == pytest.approx(len(judged) * (100 * 0.10 + 10 * 0.30) / 1e6)
        assert run["prompt"]["ref"] == "support@production"
        assert run["prompt"]["version"] == 1
        assert run["judges"]["correct_path"]["provider"] == "grader"
        assert run["judges"]["correct_path"]["endpoint"] == f"openai_compat:{world.grader.url}/v1"
        assert set(run["prices"]) == {provider, "grader"}
        assert set(run["prices"][provider]["models"]) == {model}
        assert run["prices"][provider]["version"] == f"{prices.name}@{prices.version}"
    return shown


def check_frontier(world: World, shown: dict[str, Any]) -> None:
    """Quality is exactly what the fakes were set up to give. GPT-5 mini is the
    cheapest and worst, Claude the dearest and best, and Gemini in between on
    both, so whatever the measured latencies all three are on the frontier, and
    its two steps are paired comparisons over every case, cheapest first."""
    report = world.json("report", "matrix", "label:providers")
    assert report["complete"] is True
    assert report["metric"] == "correct_path"
    rows = {v["name"]: v for v in report["variants"]}
    assert list(rows) == NAMES
    quality = {name: expected_quality(model, 1) for name, _, model, _, _ in VARIANTS}
    for name, value in quality.items():
        assert rows[name]["quality"] == pytest.approx(value)
        assert rows[name]["cost_per_case_usd"] == pytest.approx(shown[name]["cost_usd"] / N)
    claude, gpt, gemini = NAMES
    assert quality[gpt] < quality[gemini] < quality[claude]
    assert rows[gpt]["cost_per_case_usd"] < rows[gemini]["cost_per_case_usd"]
    assert rows[gemini]["cost_per_case_usd"] < rows[claude]["cost_per_case_usd"]
    assert sorted(report["frontier"]) == sorted(NAMES)
    assert [(s["baseline"], s["candidate"]) for s in report["steps"]] == [
        (gpt, gemini),
        (gemini, claude),
    ]
    for step in report["steps"]:
        comparison = step["comparison"]
        assert comparison["n_pairs"] == N
        assert comparison["difference"] == pytest.approx(
            quality[step["candidate"]] - quality[step["baseline"]]
        )
        assert comparison["p_adjusted"] >= comparison["p_value"]
        assert comparison["verdict"] == "better"
    table = world.evalkit("report", "matrix", "label:providers")
    assert "frontier over quality, cost, latency" in table


def check_promotion(world: World, shown: dict[str, Any]) -> None:
    """A second version promoted to production is what the next run uses, the
    matrix runs keep the first, and the gate pairs the two on the same judge."""
    world.evalkit("prompt", "push", "support", f"{EX}/prompt-v2.toml", "--note", "name the path")
    diff = world.evalkit("prompt", "diff", "support@production", "support@2")
    assert f"+{V2_MARKER}" in diff
    world.evalkit("prompt", "label", "support@2", "production")
    assert world.json("prompt", "show", "support@production")["version"] == 2

    before = len(world.targets(world.gemini))
    world.evalkit("run", f"{EX}/promoted.toml")
    promoted = world.json("show", "label:promoted")
    assert promoted["prompt"]["version"] == 2
    assert promoted["prompt"]["label"] == "production"
    assert len(world.targets(world.gemini)) == before + N
    assert V2_MARKER in json.dumps(world.targets(world.gemini)[before:])

    baseline = "label:providers[gemini:gemini-2.5-flash]"
    again = world.json("show", baseline)
    assert again["run_id"] == shown[NAMES[2]]["run_id"]
    assert again["prompt"]["version"] == 1

    gate = world.json("gate", baseline, "label:promoted")
    assert gate["passed"] is True
    assert gate["judge_changes"] == []
    (judged,) = [m for m in gate["comparisons"] if m["metric"] == "correct_path"]
    assert judged["n_pairs"] == N
    assert judged["difference"] == pytest.approx(
        expected_quality("gemini-2.5-flash", 2) - expected_quality("gemini-2.5-flash", 1)
    )
    assert judged["verdict"] == "better"
