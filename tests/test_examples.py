"""The bundled examples stay valid as the config schema changes."""

import tomllib
from pathlib import Path

import pytest

from evalkit.datasets import parse_jsonl
from evalkit.runspec import RunConfig
from evalkit.scorers.calibration import parse_labels
from evalkit.scorers.registry import JudgeSpec

EXAMPLES = Path(__file__).parent.parent / "examples"
CONFIGS = sorted(EXAMPLES.glob("*.toml")) + sorted((EXAMPLES / "live").glob("*.toml"))


@pytest.mark.parametrize(
    "path", [p for p in CONFIGS if p.name != "judge.toml"], ids=lambda p: p.name
)
def test_run_configs_parse(path: Path) -> None:
    raw = tomllib.loads(path.read_text())
    data_file = path.parent / raw["dataset"].pop("path")
    assert parse_jsonl(data_file.read_bytes())
    RunConfig.model_validate(raw)


def test_judge_and_labels_parse() -> None:
    raw = tomllib.loads((EXAMPLES / "live" / "judge.toml").read_text())
    spec = JudgeSpec.model_validate({"type": "judge", **raw["judge"]})
    labels = parse_labels((EXAMPLES / "live" / "labels.jsonl").read_bytes(), spec.mode)
    assert len(labels) == 10


def test_live_configs_differ_only_in_label_and_instruction() -> None:
    base = (EXAMPLES / "live" / "baseline.toml").read_text().splitlines()
    cand = (EXAMPLES / "live" / "candidate.toml").read_text().splitlines()
    assert len(base) - len(cand) == 1
    assert any("Answer briefly." in line for line in cand)
