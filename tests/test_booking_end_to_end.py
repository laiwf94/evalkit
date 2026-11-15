"""End to end: 50 booking conversations between a scripted simulated user and the
reference agent (examples/agent_server.py) over the session protocol, a run
killed with SIGKILL mid-conversation and resumed, and a gate that catches an
agent which books without looking up the free tables first.

The agents are the reference server, unchanged, on local sockets; the run
configs, the dataset and the sandbox are the files in examples/agents; evalkit
is configured only the way an operator configures it (``EVALKIT_DATABASE_URL``,
``EVALKIT_HTTP_TARGETS_FILE``). The killed run is the installed command in a
child process; everything else goes through the CLI in this process.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from evalkit import cli
from evalkit.db import make_engine, make_session_factory
from evalkit.db.models import Result, ResultStep, ResultTurn, Run

ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "examples" / "agents"
RUN = "examples/agents/booking-run.toml"
BLIND = "examples/agents/booking-blind.toml"
CASES = [json.loads(line) for line in (EXAMPLES / "booking.jsonl").read_text().splitlines()]
KIND = {c["id"]: c["tags"][0] for c in CASES}
CONCURRENCY = 8  # booking-run.toml's options.concurrency


def _load_reference() -> ModuleType:
    spec = importlib.util.spec_from_file_location("agent_server", ROOT / "examples/agent_server.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = _load_reference()


class Recording(ref.Agent):  # type: ignore[name-defined,misc]
    """The reference agent, noting each request it gets (session, step, request
    id) and the sessions whose last answer asked for a tool after an earlier
    answer: evalkit has stored at least one reply of those, and they need more."""

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self.requests: list[tuple[str, int, str]] = []
        self.under_way: set[str] = set()

    def handle(self, raw: bytes, key: str | None) -> tuple[int, bytes]:
        body = json.loads(raw)
        with self.lock:
            self.requests.append((body["session_id"], body["step"], body["request_id"]))
        status, answer = super().handle(raw, key)
        assert status == 200, answer
        calls = json.loads(answer)["message"]["tool_calls"]
        with self.lock:
            if calls and body["step"] >= 1:
                self.under_way.add(body["session_id"])
            else:
                self.under_way.discard(body["session_id"])
        return status, answer


@dataclass
class Served:
    agent: Recording
    url: str
    server: Any


def serve(agent: Recording) -> Served:
    server = ref.make_server("127.0.0.1", 0, agent)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return Served(agent, f"http://127.0.0.1:{server.server_address[1]}/v1/step", server)


class World:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.served: list[Served] = []
        self.engines: list[Any] = []

    def endpoints(self, booker: Recording, blind: Recording) -> None:
        """Serve the two agents and point the operator's endpoints file at them."""
        a, b = serve(booker), serve(blind)
        self.served += [a, b]
        path = self.tmp_path / "endpoints.toml"
        path.write_text(
            f'[endpoints.booker]\nurl = "{a.url}"\nallow_private = true\ntimeout_seconds = 10\n'
            f'[endpoints.booker-blind]\nurl = "{b.url}"\nallow_private = true\n'
            "timeout_seconds = 10\n"
        )
        self.monkeypatch.setenv("EVALKIT_HTTP_TARGETS_FILE", str(path))

    def database(self, name: str) -> sessionmaker[Session]:
        url = f"sqlite:///{self.tmp_path / name}"
        self.monkeypatch.setenv("EVALKIT_DATABASE_URL", url)
        engine = make_engine(url)
        self.engines.append(engine)
        return make_session_factory(engine)

    def evalkit(self, *args: str, expect: int = 0) -> str:
        out = io.StringIO()
        code = cli.main(list(args), out=out)
        assert code == expect, f"exit {code}\n{out.getvalue()}"
        return out.getvalue()

    def json(self, *args: str, expect: int = 0) -> Any:
        return json.loads(self.evalkit(*args, "--json", expect=expect))

    def close(self) -> None:
        for served in self.served:
            served.server.shutdown()
            served.server.server_close()
        for engine in self.engines:
            engine.dispose()


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    for key in list(os.environ):
        if key.upper().endswith("_PROXY") or key.startswith(("EVALKIT_", "ANTHROPIC_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("NO_PROXY", "*")
    # A resume takes a killed process's cases over once their leases lapse.
    monkeypatch.setenv("EVALKIT_LEASE_SECONDS", "2")
    monkeypatch.chdir(ROOT)
    w = World(tmp_path, monkeypatch)
    yield w
    w.close()


def outcome(sessions: sessionmaker[Session], run_id: str) -> dict[str, Any]:
    """What each case of a run came to: everything but timings, cache flags and ids."""
    with sessions() as s:
        found: dict[str, Any] = {
            r.case_id: {
                "output": r.output,
                "stop_reason": r.stop_reason,
                "error": r.error,
                "scores": {k: v["value"] for k, v in (r.scores or {}).items()},
                "steps": [],
                "turns": [],
            }
            for r in s.scalars(select(Result).where(Result.run_id == run_id))
        }
        for step in s.scalars(
            select(ResultStep).where(ResultStep.run_id == run_id).order_by(ResultStep.step)
        ):
            found[step.case_id]["steps"].append(
                (step.kind, step.text, step.tool, step.arguments, step.status, step.result)
            )
        for turn in s.scalars(
            select(ResultTurn).where(ResultTurn.run_id == run_id).order_by(ResultTurn.turn)
        ):
            found[turn.case_id]["turns"].append((turn.role, turn.content, turn.stop_reason))
    return found


def usage(sessions: sessionmaker[Session], run_id: str) -> dict[str, tuple[bool, int]]:
    """Per case: whether any reply came from the step cache, and the tokens the
    case was charged with (replies from the cache are charged nothing)."""
    with sessions() as s:
        cached = dict(
            s.execute(
                select(ResultStep.case_id, func.max(ResultStep.cached))
                .where(ResultStep.run_id == run_id, ResultStep.kind == "model")
                .group_by(ResultStep.case_id)
            ).all()
        )
        return {
            r.case_id: (bool(cached[r.case_id]), r.input_tokens + r.output_tokens)
            for r in s.scalars(select(Result).where(Result.run_id == run_id))
        }


def by_kind(found: dict[str, Any], field: str) -> dict[str, set[Any]]:
    kinds: dict[str, set[Any]] = {}
    for case_id, result in found.items():
        value = result["scores"][field] if field in result["scores"] else result[field]
        kinds.setdefault(KIND[case_id], set()).add(value)
    return kinds


def saved(sessions: sessionmaker[Session], run_id: str) -> int:
    with sessions() as s:
        return int(
            s.scalar(select(func.count()).select_from(Result).where(Result.run_id == run_id)) or 0
        )


def test_the_dataset_covers_every_way_a_booking_dialogue_ends() -> None:
    assert len(CASES) == len({c["id"] for c in CASES}) == 50
    assert Counter(KIND.values()) == {
        "direct": 16,
        "time": 10,
        "ask_date": 8,
        "opened": 4,
        "full": 4,
        "late": 4,
        "undecided": 4,
    }
    for case in CASES:
        assert case["input"]["success"] == ["confirmation BK-"]
        assert case["expected"]["tool_calls"] or KIND[case["id"]] == "undecided"


@dataclass
class Control:
    """The uninterrupted run: what each case came to, what each case was charged
    and how many replies the agent gave."""

    outcome: dict[str, Any]
    tokens: dict[str, int]
    replies: int


def control_run(world: World) -> Control:
    agent = Recording()
    world.endpoints(agent, Recording(skip_lookup=True))
    sessions = world.database("control.db")
    control = world.json("run", RUN)
    assert (control["status"], control["completed"], control["errors"]) == ("completed", 50, 0)
    expected = outcome(sessions, control["run_id"])
    assert by_kind(expected, "stop_reason") == {
        "direct": {"goal_reached"},
        "time": {"goal_reached"},
        "ask_date": {"goal_reached"},
        "opened": {"goal_reached"},
        "late": {"goal_reached"},
        "full": {"gave_up"},
        "undecided": {"max_turns"},
    }
    assert {r["scores"]["tool_calls"] for r in expected.values()} == {1.0}
    assert {r["scores"]["efficiency"] for r in expected.values()} == {1.0}
    assert all(
        r["scores"]["confirmed"] == (r["stop_reason"] == "goal_reached") for r in expected.values()
    )
    # A dialogue with a missing date takes the agent's question and the user's answer.
    ask = next(r for c, r in expected.items() if KIND[c] == "ask_date")
    roles = [role for role, _, _ in ask["turns"]]
    assert roles == ["user", "assistant", "user", "assistant", "user"]
    assert "Which date" in ask["turns"][1][1]
    assert ask["turns"][-1][2] == "goal_reached"
    assert [s[2] for s in ask["steps"] if s[0] == "tool"] == ["find_slots", "book"]
    # The user who never names a day is cut off after max_turns = 3 agent turns.
    undecided = next(r for c, r in expected.items() if KIND[c] == "undecided")
    assert sum(1 for role, _, _ in undecided["turns"] if role == "assistant") == 3
    assert agent.replayed == 0
    replies = sum(1 for r in expected.values() for s in r["steps"] if s[0] == "model")
    assert agent.answered == len(agent.requests) == replies
    tokens = {c: t for c, (_, t) in usage(sessions, control["run_id"]).items()}
    return Control(expected, tokens, replies)


def killed_run(world: World, booker: Recording) -> tuple[sessionmaker[Session], str]:
    """Start the run as a child process and SIGKILL it once 15 cases are saved
    and a dialogue is part way through a turn: its first reply is stored, and its
    next one is at least the agent's 25 ms delay away."""
    sessions = world.database("killed.db")
    child = subprocess.Popen(
        [sys.executable, "-m", "evalkit", "run", RUN],
        env=dict(os.environ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert child.stderr is not None
    run_id = child.stderr.readline().split()[2]
    deadline = time.monotonic() + 60
    while saved(sessions, run_id) < 15 or not booker.under_way:
        assert child.poll() is None, "the run finished before it could be killed"
        assert time.monotonic() < deadline
        time.sleep(0.005)
    os.kill(child.pid, signal.SIGKILL)
    child.wait(timeout=10)
    child.stderr.close()
    before = saved(sessions, run_id)
    with sessions() as s:
        assert s.get_one(Run, run_id).status == "running"  # nobody lived to mark it
        stored_replies = s.scalar(
            select(func.count())
            .select_from(ResultStep)
            .where(ResultStep.run_id == run_id, ResultStep.kind == "model")
        )
    assert 15 <= before < 50
    # The agent had answered more than the finished cases account for: some
    # dialogues were cut off part way, with replies already paid for.
    assert len({rid for _, _, rid in booker.requests}) > (stored_replies or 0)
    assert len({sid for sid, _, _ in booker.requests}) > before
    return sessions, run_id


def check_resumed(
    sessions: sessionmaker[Session], run_id: str, booker: Recording, control: Control
) -> None:
    """The resumed run came to what the uninterrupted one did, and the agent
    answered each step once over the two processes."""
    assert outcome(sessions, run_id) == control.outcome
    assert booker.answered == control.replies
    ids = [rid for _, _, rid in booker.requests]
    assert len(set(ids)) == control.replies
    # A request the kill cut off before its answer was stored is asked again, and
    # the agent replays its stored answer (one at most per running case).
    assert len(ids) - control.replies == booker.replayed <= CONCURRENCY
    # A case finished in one go is charged what it was in the control run; one
    # resumed part way only for the replies asked for after the resume (#83).
    charged = usage(sessions, run_id)
    assert all(
        (tokens < control.tokens[c]) if replayed else (tokens == control.tokens[c])
        for c, (replayed, tokens) in charged.items()
    )
    # The dialogues under way were finished from where the crash left them:
    # their first replies replayed from the cache, the rest asked for after.
    assert any(replayed for replayed, _ in charged.values())
    with sessions() as s:
        split = s.execute(
            select(ResultStep.case_id)
            .where(ResultStep.run_id == run_id, ResultStep.kind == "model")
            .group_by(ResultStep.case_id)
            .having(func.min(ResultStep.cached) != func.max(ResultStep.cached))
        ).all()
    assert split


def check_blind(
    world: World, sessions: sessionmaker[Session], run_id: str, control: Control
) -> None:
    """The agent that books without find_slots: its answers look as good, its
    trajectories do not, and the gate fails."""
    candidate = world.json("run", BLIND)
    assert (candidate["status"], candidate["errors"]) == ("completed", 0)
    found = outcome(sessions, candidate["run_id"])
    assert not any(s[2] == "find_slots" for r in found.values() for s in r["steps"])
    assert by_kind(found, "tool_calls") == {
        "direct": {0.5},
        "time": {0.5},
        "ask_date": {0.5},
        "opened": {0.5},
        "full": {0.0},  # it booked a full night
        "late": {0.0},  # it asked for a time that was gone, twice
        "undecided": {1.0},
    }
    # Booking blind confirms as many dialogues, just the wrong ones.
    assert sum(r["scores"]["confirmed"] for r in found.values()) == sum(
        r["scores"]["confirmed"] for r in control.outcome.values()
    )
    assert by_kind(found, "stop_reason")["full"] == {"goal_reached"}
    assert by_kind(found, "stop_reason")["late"] == {"gave_up"}
    report = world.evalkit("gate", run_id, "label:booker-blind", expect=1)
    assert "FAIL: tool_calls" in report
    gate = world.json("gate", run_id, "label:booker-blind", expect=1)
    assert (gate["passed"], gate["regressions"]) == (False, ["tool_calls"])
    (row,) = gate["comparisons"]
    assert (row["metric"], row["n_pairs"], row["verdict"]) == ("tool_calls", 50, "worse")
    assert row["difference"] == pytest.approx(-0.54)


def test_booking_dialogues_resume_after_a_kill_and_the_gate_catches_a_blind_agent(
    world: World,
) -> None:
    # An uninterrupted run, the reference the resumed run must match.
    control = control_run(world)
    # The same run on fresh agents and a fresh database, killed with SIGKILL
    # while dialogues are under way. The agent is slowed so the kill lands mid-run.
    booker = Recording(delay_ms=25)
    world.endpoints(booker, Recording(skip_lookup=True))
    sessions, run_id = killed_run(world, booker)
    # Resumed from this process: each step the agent answered before the kill
    # comes from the step cache, so no reply is paid for twice.
    resumed = world.json("run", "--resume", run_id)
    assert (resumed["status"], resumed["completed"], resumed["errors"]) == ("completed", 50, 0)
    check_resumed(sessions, run_id, booker, control)
    # A rerun is answered entirely from the cache and asks the agent nothing.
    rerun = world.json("run", RUN, "--label", "booker-rerun")
    assert (rerun["completed"], rerun["cached"]) == (50, 50)
    assert booker.answered == control.replies
    world.evalkit("gate", run_id, "label:booker-rerun")
    check_blind(world, sessions, run_id, control)
