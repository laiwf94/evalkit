import asyncio
import io
import json
from pathlib import Path
from typing import Any

import pytest

from evalkit import cli
from evalkit.cli import main
from tests.fake_anthropic import FakeAnthropic, json_reply, message


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("ANTHROPIC_API_KEY", "EVALKIT_DATABASE_URL", "EVALKIT_PORT"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def db(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'cli.db'}"


def run_cli(db: str, *args: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(["--database-url", db, *args], out=out)
    return code, out.getvalue()


def write_dataset(path: Path, n: int = 120) -> Path:
    lines = [
        json.dumps({"id": f"q{i}", "input": {"q": f"question {i}"}, "expected": f"a{i}"})
        for i in range(n)
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def write_config(tmp_path: Path, name: str, accuracy: float, extra: str = "") -> Path:
    write_dataset(tmp_path / "qa.jsonl")
    cfg = tmp_path / f"{name}.toml"
    cfg.write_text(
        f'label = "{name}"\n'
        '[dataset]\nname = "qa"\npath = "qa.jsonl"\n'
        f'[target]\ntype = "fake"\naccuracy = {accuracy}\nseed = 1\n'
        '[[scorers]]\ntype = "exact"\n'
        '[[scorers]]\ntype = "token_f1"\ngate = false\n'
        f"{extra}"
    )
    return cfg


def test_dataset_commands(db: str, tmp_path: Path) -> None:
    data = write_dataset(tmp_path / "d.jsonl", 3)
    assert run_cli(db, "dataset", "import", "qa", str(data)) == (
        0,
        "qa imported version 1 (3 cases)\n",
    )
    assert (
        run_cli(db, "dataset", "import", "qa", str(data))[1]
        == "qa unchanged, already version 1 (3 cases)\n"
    )
    code, listing = run_cli(db, "dataset", "list")
    assert code == 0
    assert listing.startswith("qa\tv1\t3 cases\t")
    code, exported = run_cli(db, "dataset", "export", "qa")
    assert code == 0
    assert len(exported.splitlines()) == 3
    out = tmp_path / "out.jsonl"
    assert run_cli(db, "dataset", "export", "qa", "--version", "1", "-o", str(out))[0] == 0
    assert out.read_text() == exported


def test_run_show_compare_and_gate(db: str, tmp_path: Path) -> None:
    base = write_config(tmp_path, "main", 0.9)
    good = write_config(tmp_path, "good", 0.9)
    bad = write_config(tmp_path, "bad", 0.6)
    code, text = run_cli(db, "run", str(base))
    assert code == 0
    assert "exact" in text
    assert "completed" in text
    code, raw = run_cli(db, "run", str(good), "--label", "good-branch", "--json")
    summary = json.loads(raw)
    assert code == 0
    assert summary["label"] == "good-branch"
    assert summary["cached"] == 120  # same target as the baseline: served from the cache
    assert run_cli(db, "run", str(bad))[0] == 0

    assert run_cli(db, "show", "label:main")[0] == 0
    code, shown = run_cli(db, "show", summary["run_id"], "--json")
    assert json.loads(shown)["run_id"] == summary["run_id"]

    code, text = run_cli(db, "gate", "label:main", "label:bad")
    assert code == 1
    assert "FAIL: exact" in text
    assert "worse" in text
    code, text = run_cli(db, "gate", "label:main", "label:good-branch")
    assert code == 0
    assert "PASS" in text
    code, raw = run_cli(db, "compare", "label:main", "label:bad", "--json", "--metric", "token_f1")
    report = json.loads(raw)
    assert code == 0
    assert report["passed"] is False
    assert report["comparisons"][0]["metric"] == "token_f1"


def test_resume_and_unusable_runs(db: str, tmp_path: Path) -> None:
    from evalkit.config import Settings
    from evalkit.datasets import import_dataset, parse_jsonl
    from evalkit.db import make_engine, make_session_factory, upgrade
    from evalkit.runner import Runner
    from evalkit.runspec import RunConfig

    engine = make_engine(db)
    upgrade(engine)
    sessions = make_session_factory(engine)
    with sessions() as s:
        import_dataset(s, "qa", parse_jsonl(write_dataset(tmp_path / "x.jsonl", 5).read_text()))
    config = RunConfig.model_validate(
        {"dataset": {"name": "qa"}, "target": {"type": "fake"}, "scorers": [{"type": "exact"}]}
    )
    runner = Runner(sessions, Settings())
    pending = runner.create_run(config)
    done = runner.create_run(config)
    asyncio.run(runner.execute(done.id))
    engine.dispose()
    assert run_cli(db, "gate", done.id, pending.id)[0] == 3
    code, text = run_cli(db, "run", "--resume", pending.id)
    assert code == 0
    assert "5/5" in text
    assert run_cli(db, "gate", done.id, pending.id)[0] == 0


@pytest.mark.parametrize(
    ("setup", "args", "code"),
    [
        (lambda p: None, ["run"], 2),
        (lambda p: None, ["run", "missing.toml"], 2),
        (lambda p: (p / "bad.toml").write_text("not = [toml"), ["run", "{p}/bad.toml"], 2),
        (
            lambda p: (p / "inv.toml").write_text('[dataset]\nname = "qa"\n'),
            ["run", "{p}/inv.toml"],
            2,
        ),
        (lambda p: None, ["show", "nope"], 2),
        (lambda p: None, ["show", "label:nope"], 2),
        (lambda p: None, ["run", "--resume", "nope"], 2),
        (lambda p: None, ["dataset", "export", "nope"], 2),
    ],
)
def test_input_errors_exit_2(
    db: str, tmp_path: Path, setup: Any, args: list[str], code: int
) -> None:
    setup(tmp_path)
    assert run_cli(db, *[a.format(p=tmp_path) for a in args])[0] == code


def test_config_error_from_environment(monkeypatch: pytest.MonkeyPatch, db: str) -> None:
    monkeypatch.setenv("EVALKIT_PORT", "none")
    assert run_cli(db, "dataset", "list")[0] == 2


def test_db_upgrade(db: str) -> None:
    assert run_cli(db, "db", "upgrade") == (0, "database is at the latest schema\n")


def test_interrupts(db: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = write_config(tmp_path, "main", 0.9)

    async def cancelled(self: Any, run_id: str) -> Any:
        raise asyncio.CancelledError

    monkeypatch.setattr("evalkit.runner.Runner.execute", cancelled)
    assert run_cli(db, "run", str(cfg))[0] == 130

    def keyboard(coro: Any) -> Any:
        coro.close()
        raise KeyboardInterrupt

    monkeypatch.setattr("evalkit.cli.asyncio.run", keyboard)
    assert run_cli(db, "dataset", "list")[0] == 130


def test_failed_runs_exit_3_not_1(db: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = write_config(tmp_path, "main", 0.9)

    async def broken(self: Any, run_id: str) -> Any:
        raise RuntimeError("database went away key=sk-ant-abcdefghijkl")

    monkeypatch.setattr("evalkit.runner.Runner.execute", broken)
    assert run_cli(db, "run", str(cfg))[0] == 3


def fake_claude(monkeypatch: pytest.MonkeyPatch, fake: FakeAnthropic) -> None:
    monkeypatch.setattr(cli.App, "claude", lambda self: fake.client())


def test_estimate_command(db: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_dataset(tmp_path / "qa.jsonl", 3)
    cfg = tmp_path / "p.toml"
    cfg.write_text(
        '[dataset]\nname = "qa"\npath = "qa.jsonl"\n'
        '[target]\ntype = "prompt"\ntemplate = "{{q}}"\nmax_tokens = 10\n'
        '[[scorers]]\ntype = "exact"\n'
    )
    fake = FakeAnthropic()
    fake.default = lambda body: json_reply({"input_tokens": 50})
    fake_claude(monkeypatch, fake)
    code, text = run_cli(db, "estimate", str(cfg))
    assert code == 0
    assert "target input  150 tokens" in text
    code, raw = run_cli(db, "estimate", str(cfg), "--json")
    assert json.loads(raw)["max_total_cost_usd"] == pytest.approx((150 * 4 + 30 * 20) / 1e6)


def test_run_with_prompt_target(db: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_dataset(tmp_path / "qa.jsonl", 2)
    cfg = tmp_path / "p.toml"
    cfg.write_text(
        '[dataset]\nname = "qa"\npath = "qa.jsonl"\n'
        '[target]\ntype = "prompt"\ntemplate = "{{q}}"\n'
        '[[scorers]]\ntype = "exact"\n'
    )
    fake = FakeAnthropic()
    fake.default = lambda body: json_reply(message("a0"))
    fake_claude(monkeypatch, fake)
    code, text = run_cli(db, "-v", "run", str(cfg))
    assert code == 0
    assert "cost $" in text
    assert "latency p50" in text


def test_calibrate_command(db: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    judge = tmp_path / "judge.toml"
    judge.write_text('[judge]\nrubric = "Correct and brief."\n')
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "\n".join(
            json.dumps({"id": f"l{i}", "input": "q", "output": "o", "label": lab})
            for i, lab in enumerate([5, 1, 4])
        )
    )
    fake = FakeAnthropic(
        *(json_reply(message(json.dumps({"reasoning": "r", "score": s}))) for s in (5, 2, 4))
    )
    fake_claude(monkeypatch, fake)
    code, text = run_cli(db, "calibrate", "--judge", str(judge), str(labels), "--concurrency", "1")
    assert code == 0
    assert "Cohen's kappa" in text
    assert "disagreements: l1" in text
    for content in ("[other]\n", '[judge]\nmode = "x"\n'):
        judge.write_text(content)
        assert run_cli(db, "calibrate", "--judge", str(judge), str(labels))[0] == 2


def test_large_files_are_refused(db: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    big = tmp_path / "big.toml"
    big.write_bytes(b"#" * (1024 * 1024 + 1))
    assert run_cli(db, "run", str(big))[0] == 2
    monkeypatch.setattr(cli, "MAX_CONFIG_BYTES", 10**9)

    class Huge:
        st_size = 600 * 1024 * 1024

    monkeypatch.setattr(Path, "stat", lambda self, **kw: Huge())
    assert run_cli(db, "dataset", "import", "x", str(big))[0] == 2


def test_formatters_cover_optional_fields() -> None:
    from evalkit.cli import _format_calibration, _format_estimate, _jsonable
    from evalkit.estimate import Estimate
    from evalkit.scorers.calibration import CalibrationReport

    est = Estimate(cases=1, cached_cases=0, batch=True, notes=["n"])
    text = _format_estimate(est)
    assert "batch pricing" in text
    assert "note: n" in text
    cal = CalibrationReport(
        n=1,
        judged=1,
        agreement=1.0,
        kappa=1.0,
        weighted_kappa=None,
        labels=["loss", "tie", "win"],
        confusion=[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
    )
    assert "quadratic" not in _format_calibration(cal)
    assert _jsonable((1, [2])) == [1, [2]]
