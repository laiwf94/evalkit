"""The prompt registry: content-hashed versions, compare-and-set labels, the CLI
and API around them, and run configs that name a registry prompt."""

import asyncio
import io
import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx2
import pytest
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from evalkit import cli, prompts
from evalkit.api import BodyLimit, create_app
from evalkit.config import Settings
from evalkit.datasets import import_dataset, parse_jsonl
from evalkit.db import models
from evalkit.db.models import Prompt, PromptVersion, Run
from evalkit.errors import ConflictError, InvalidInputError, NotFoundError
from evalkit.estimate import estimate, estimate_matrix
from evalkit.pricing import PriceTable
from evalkit.prompts import (
    content_hash,
    diff,
    list_prompts,
    move_label,
    parse_ref,
    push_version,
    resolve,
    resolve_config,
)
from evalkit.providers.registry import ProviderRegistry
from evalkit.report import summarize
from evalkit.runner import Runner
from evalkit.runspec import RunConfig
from evalkit.targets.factory import build_target
from evalkit.targets.specs import PromptTargetSpec, target_hash
from tests.fake_anthropic import FakeAnthropic, json_reply, message

TOKEN = "test-token-0123456789"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
V1 = "Answer briefly: {{q}}"
V2 = "Answer in one word: {{q}}"


def push(session: Session, template: str, name: str = "support", **kw: Any) -> PromptVersion:
    return push_version(session, name, template=template, **kw)[0]


# Versions


def test_versions_are_content_hashed_and_immutable(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        first, created = push_version(
            s, "support", template=V1, system="Be kind.", note="first", description="help desk"
        )
        assert created
        assert (first.version, first.note) == (1, "first")
        assert first.content_hash == content_hash("Be kind.", V1)
        # The same content again: the same version, nothing new stored.
        again, created = push_version(s, "support", template=V1, system="Be kind.", note="other")
        assert (again.id, created) == (first.id, False)
        assert again.note == "first"
        # A change to either part is a new version.
        assert push(s, V1).version == 2
        assert push(s, V2, system="Be kind.").version == 3
        # A later description does not overwrite the first.
        push(s, V2, description="changed")
        assert s.scalar(select(Prompt.description)) == "help desk"
        assert s.scalar(select(func.count()).select_from(PromptVersion)) == 4
        # Another prompt numbers its own versions.
        assert push(s, V1, name="billing").version == 1


def test_the_content_hash_separates_system_from_template() -> None:
    assert content_hash("ab", "c") != content_hash("a", "bc")
    assert content_hash("", "x") == content_hash("", "x")
    assert len(content_hash("", "x")) == 64


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"name": "-bad"}, "prompt name"),
        ({"name": "a" * 129}, "prompt name"),
        ({"template": ""}, "template must not be empty"),
        ({"template": "x" * 200_001}, "template is longer"),
        ({"system": "x" * 200_001}, "system is longer"),
        ({"note": "x" * 2001}, "note is longer"),
        ({"description": "x" * 2001}, "description is longer"),
        ({"template": "Hi {{ q"}, "not a valid {{ field }} placeholder"),
    ],
)
def test_bad_pushes_are_refused(
    sessions: sessionmaker[Session], kw: dict[str, Any], message: str
) -> None:
    args: dict[str, Any] = {"name": "support", "template": V1} | kw
    name = args.pop("name")
    with sessions() as s, pytest.raises(InvalidInputError, match=message):
        push_version(s, name, **args)


def _racing(
    monkeypatch: pytest.MonkeyPatch, sessions: sessionmaker[Session], other: Callable[[], None]
) -> None:
    """Run ``other`` in its own session the first time a push stamps a row: after
    this push has read the state and before it commits."""
    calls = {"n": 0}
    real = models.utcnow

    def stamp() -> datetime:
        calls["n"] += 1
        if calls["n"] == 1:
            other()
        return real()

    monkeypatch.setattr("evalkit.prompts.utcnow", stamp)


def test_a_first_push_that_loses_the_race_uses_the_winner(
    sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    def other() -> None:
        with sessions() as s:
            s.add(Prompt(name="support", description="theirs", created_at=datetime.now()))
            s.commit()

    _racing(monkeypatch, sessions, other)
    with sessions() as s:
        version, created = push_version(s, "support", template=V1, description="mine")
        assert (version.version, created) == (1, True)
        assert s.scalar(select(Prompt.description)) == "theirs"


def test_a_push_that_loses_the_version_number_takes_the_next(
    sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    with sessions() as s:
        push(s, V1)

    def other() -> None:
        with sessions() as s:
            push_version(s, "support", template="Someone else: {{q}}")

    _racing(monkeypatch, sessions, other)
    with sessions() as s:
        version, created = push_version(s, "support", template=V2)
        assert (version.version, created) == (3, True)


def test_the_same_content_pushed_twice_at_once_is_one_version(
    sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    with sessions() as s:
        push(s, V1)

    def other() -> None:
        with sessions() as s:
            push_version(s, "support", template=V2)

    _racing(monkeypatch, sessions, other)
    with sessions() as s:
        version, created = push_version(s, "support", template=V2)
        assert (version.version, created) == (2, False)


def test_a_push_that_keeps_losing_gives_up(
    sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    with sessions() as s:
        push(s, V1)
    counter = {"n": 0, "inside": False}
    real = models.utcnow

    def stamp() -> datetime:
        # Every attempt finds its version number taken by someone else.
        if not counter["inside"]:
            counter["inside"] = True
            counter["n"] += 1
            with sessions() as s:
                push_version(s, "support", template=f"Other {counter['n']}: {{{{q}}}}")
            counter["inside"] = False
        return real()

    with sessions() as s:
        monkeypatch.setattr("evalkit.prompts.utcnow", stamp)
        with pytest.raises(ConflictError, match="could not allocate a version"):
            push_version(s, "support", template=V2)


# References


@pytest.mark.parametrize(
    ("ref", "parsed"),
    [
        ("support", ("support", "latest")),
        ("support@latest", ("support", "latest")),
        ("support@12", ("support", 12)),
        ("support@production", ("support", "production")),
        ("a.b_c-d@canary-2", ("a.b_c-d", "canary-2")),
    ],
)
def test_parse_ref(ref: str, parsed: tuple[str, int | str]) -> None:
    assert parse_ref(ref) == parsed


@pytest.mark.parametrize("ref", ["", "@prod", "support@", "support@0", "support@Prod", "a@b@c"])
def test_bad_refs(ref: str) -> None:
    with pytest.raises(InvalidInputError, match="is not NAME"):
        parse_ref(ref)


def test_resolve_by_number_label_and_latest(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        push(s, V1)
        push(s, V2)
        move_label(s, "support", "production", 1, expected=None)
        assert resolve(s, "support").version.version == 2
        assert resolve(s, "support@latest").version.template == V2
        assert resolve(s, "support@1").version.template == V1
        found = resolve(s, "support@production")
        assert (found.version.version, found.label, found.title) == (
            1,
            "production",
            "support@production (v1)",
        )
        assert resolve(s, "support@2").title == "support (v2)"
        with pytest.raises(NotFoundError, match="no version 3"):
            resolve(s, "support@3")
        with pytest.raises(NotFoundError, match=r"no label 'staging' \(labels: production\)"):
            resolve(s, "support@staging")
        with pytest.raises(NotFoundError, match="prompt 'billing' not found"):
            resolve(s, "billing@production")


# Labels


def test_labels_move_only_by_compare_and_set(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        for template in (V1, V2, "Third: {{q}}"):
            push(s, template)
        created = move_label(s, "support", "production", 1, expected=None)
        assert (created.previous, created.version) == (None, 1)
        with pytest.raises(ConflictError, match="is at version 1; the move expected it not to"):
            move_label(s, "support", "production", 2, expected=None)
        moved = move_label(s, "support", "production", 2, expected=1)
        assert (moved.previous, moved.version) == (1, 2)
        # A writer that read version 1 before that move is refused, not applied.
        with pytest.raises(
            ConflictError, match="is at version 2; the move expected it to be at version 1"
        ):
            move_label(s, "support", "production", 3, expected=1)
        assert resolve(s, "support@production").version.version == 2
        with pytest.raises(ConflictError, match="'staging' of prompt 'support' does not exist"):
            move_label(s, "support", "staging", 3, expected=2)
        # Moving back is allowed: a label can point anywhere, an old version included.
        move_label(s, "support", "production", 1, expected=2)
        with pytest.raises(NotFoundError, match="no version 9"):
            move_label(s, "support", "production", 9, expected=1)
        with pytest.raises(NotFoundError):
            move_label(s, "billing", "production", 1, expected=None)
        assert prompts.labels_of(s, prompts.get_prompt(s, "support")) == {"production": 1}


@pytest.mark.parametrize("label", ["latest", "Production", "1", "-x", "a" * 65, "with space"])
def test_bad_labels(sessions: sessionmaker[Session], label: str) -> None:
    with sessions() as s:
        push(s, V1)
        with pytest.raises(InvalidInputError, match="a label is"):
            move_label(s, "support", label, 1, expected=None)


def test_list_prompts(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        push(s, V1)
        push(s, V2)
        push(s, V1, name="billing")
        move_label(s, "support", "production", 1, expected=None)
        move_label(s, "support", "canary", 2, expected=None)
        rows = list_prompts(s)
    assert [(r.name, r.latest_version, r.labels) for r in rows] == [
        ("billing", 1, {}),
        ("support", 2, {"canary": 2, "production": 1}),
    ]


def test_diff(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        push(s, "Line one\nAnswer: {{q}}\n", system="Be kind.")
        push(s, "Line one\nReply: {{q}}", system="Be kind.")
        move_label(s, "support", "production", 1, expected=None)
        text = diff(resolve(s, "support@production"), resolve(s, "support@2"))
        assert diff(resolve(s, "support@1"), resolve(s, "support@production")) == ""
    assert text == (
        "--- support@production (v1) template\n"
        "+++ support (v2) template\n"
        "@@ -1,2 +1,2 @@\n"
        " Line one\n"
        "-Answer: {{q}}\n"
        "+Reply: {{q}}\n"
        "\\ No newline at end of file\n"
    )


# Target specs and runs


def test_a_target_names_a_prompt_or_a_template() -> None:
    spec = PromptTargetSpec.model_validate({"type": "prompt", "prompt": "support@production"})
    assert spec.template == ""
    with pytest.raises(ValidationError, match="give a template"):
        PromptTargetSpec.model_validate({"type": "prompt"})
    with pytest.raises(ValidationError, match="not both"):
        PromptTargetSpec.model_validate({"type": "prompt", "prompt": "a", "template": "x"})
    with pytest.raises(ValidationError, match="not both"):
        PromptTargetSpec.model_validate({"type": "prompt", "prompt": "a", "system": "x"})
    with pytest.raises(ValidationError, match="should match pattern"):
        PromptTargetSpec.model_validate({"type": "prompt", "prompt": "a@B"})


def test_an_unresolved_prompt_never_runs() -> None:
    spec = PromptTargetSpec.model_validate({"type": "prompt", "prompt": "support"})
    providers = ProviderRegistry({}, PriceTable(), claude=FakeAnthropic().client())
    with pytest.raises(InvalidInputError, match="must be resolved"):
        build_target(spec, endpoints={}, providers=providers)
    # Unresolved references never hash alike, whatever they would resolve to.
    other = spec.model_copy(update={"prompt": "support@production"})
    assert target_hash(spec) != target_hash(other)


def test_resolved_content_hashes_like_the_same_content_inline(
    sessions: sessionmaker[Session],
) -> None:
    inline = run_config(system="S", template=V1).target
    config = run_config(prompt="support@production")
    with sessions() as s:
        push(s, V1, system="S")
        move_label(s, "support", "production", 1, expected=None)
        resolved, pinned = resolve_config(s, config)
        assert resolve_config(s, run_config()) == (run_config(), None)
    assert resolved.target == inline
    assert target_hash(resolved.target) == target_hash(inline)
    assert pinned == {
        "ref": "support@production",
        "name": "support",
        "version": 1,
        "label": "production",
        "content_hash": content_hash("S", V1),
    }


def dataset_bytes(n: int = 6) -> bytes:
    return "\n".join(
        json.dumps({"id": f"q{i}", "input": {"q": f"question {i}"}, "expected": f"a{i}"})
        for i in range(n)
    ).encode()


def run_config(**target: Any) -> RunConfig:
    spec = {"type": "prompt", "model": "claude-sonnet-5-5", "max_tokens": 32, **target}
    if "prompt" not in target:
        spec.setdefault("template", V1)
    return RunConfig.model_validate(
        {
            "label": "prompted",
            "dataset": {"name": "qa"},
            "target": spec,
            "scorers": [{"type": "exact"}],
        }
    )


def answering(fake: FakeAnthropic) -> FakeAnthropic:
    """Claude answers "aN" when the prompt asks for one word, "no idea" otherwise."""

    def reply(body: dict[str, Any]) -> httpx2.Response:
        if fake.paths[-1].endswith("/count_tokens"):
            return json_reply({"input_tokens": 40})
        prompt = body["messages"][0]["content"]
        text = prompt if isinstance(prompt, str) else prompt[0]["text"]
        number = text.rsplit(" ", 1)[-1]
        return json_reply(message(f"a{number}" if "one word" in text else "no idea"))

    fake.default = reply
    return fake


@pytest.fixture
def fake() -> FakeAnthropic:
    return answering(FakeAnthropic())


@pytest.fixture
def runner(sessions: sessionmaker[Session], fake: FakeAnthropic) -> Runner:
    with sessions() as s:
        import_dataset(s, "qa", parse_jsonl(dataset_bytes()))
        push(s, V1, note="first")
        push(s, V2, note="one word")
        move_label(s, "support", "production", 1, expected=None)
    providers = ProviderRegistry({}, PriceTable(), claude=fake.client())
    return Runner(sessions, Settings(), providers=providers)


def test_a_run_freezes_the_version_its_label_names(
    runner: Runner, sessions: sessionmaker[Session], fake: FakeAnthropic
) -> None:
    run = runner.create_run(run_config(prompt="support@production"))
    # Promoting a new version after the run was created does not change the run.
    with sessions() as s:
        move_label(s, "support", "production", 2, expected=1)
    assert asyncio.run(runner.execute(run.id)).status == "completed"
    assert {r["messages"][0]["content"] for r in fake.requests} >= {"Answer briefly: question 0"}
    with sessions() as s:
        stored = s.get(Run, run.id)
        assert stored is not None
        assert stored.target_spec["template"] == V1
        assert stored.target_spec["prompt"] is None
        assert stored.prompt == {
            "ref": "support@production",
            "name": "support",
            "version": 1,
            "label": "production",
            "content_hash": content_hash("", V1),
        }
        summary = summarize(s, run.id)
    assert summary.prompt == stored.prompt
    assert summary.metrics[0].mean == 0.0

    # A new run of the same config picks up the promoted version.
    second = runner.create_run(run_config(prompt="support@production"))
    asyncio.run(runner.execute(second.id))
    with sessions() as s:
        assert summarize(s, second.id).metrics[0].mean == 1.0
        assert summarize(s, second.id).prompt["version"] == 2  # type: ignore[index]


def test_the_same_content_inline_comes_from_the_cache(runner: Runner, fake: FakeAnthropic) -> None:
    first = runner.create_run(run_config(prompt="support@1"))
    asyncio.run(runner.execute(first.id))
    calls = len(fake.requests)
    inline = runner.create_run(run_config(template=V1))
    assert inline.target_hash == first.target_hash
    asyncio.run(runner.execute(inline.id))
    assert len(fake.requests) == calls


def test_a_matrix_resolves_its_prompt_once(runner: Runner, sessions: sessionmaker[Session]) -> None:
    config = run_config(prompt="support@production").model_copy(
        update={
            "matrix": RunConfig.model_validate(
                {
                    "dataset": {"name": "qa"},
                    "target": {"type": "fake"},
                    "scorers": [{"type": "exact"}],
                    "matrix": {"effort": ["low", "high"]},
                }
            ).matrix
        }
    )
    group, runs = runner.create_matrix(config)
    with sessions() as s:
        stored = [s.get(Run, r.id) for r in runs]
        assert all(r is not None and r.prompt and r.prompt["version"] == 1 for r in stored)
        assert all(r is not None and r.target_spec["template"] == V1 for r in stored)
    assert group.config["target"]["template"] == V1


def test_unknown_prompts_are_refused_before_a_run_is_recorded(
    runner: Runner, sessions: sessionmaker[Session]
) -> None:
    with pytest.raises(NotFoundError, match="no label 'staging'"):
        runner.create_run(run_config(prompt="support@staging"))
    with sessions() as s:
        assert s.scalar(select(func.count()).select_from(Run)) == 0


def test_estimates_use_the_version_named_now(
    runner: Runner, sessions: sessionmaker[Session], fake: FakeAnthropic
) -> None:
    claude = fake.client()
    found = asyncio.run(
        estimate(
            sessions, Settings(), run_config(prompt="support"), claude, providers=runner.providers
        )
    )
    assert found.target_input_tokens == 40 * 6
    counted = [r for r, p in zip(fake.requests, fake.paths, strict=True) if "count" in p]
    assert counted[0]["messages"][0]["content"] == "Answer in one word: question 0"
    config = run_config(prompt="support@production").model_copy(
        update={
            "matrix": RunConfig.model_validate(
                {
                    "dataset": {"name": "qa"},
                    "target": {"type": "fake"},
                    "scorers": [{"type": "exact"}],
                    "matrix": {"max_tokens": [8, 16]},
                }
            ).matrix
        }
    )
    each = asyncio.run(
        estimate_matrix(sessions, Settings(), config, claude, providers=runner.providers)
    )
    assert [v.estimate.target_input_tokens for v in each.variants] == [240, 240]


# The CLI


def run_cli(db: str, *args: str) -> tuple[int, str]:
    out = io.StringIO()
    code = cli.main(["--database-url", db, *args], out=out)
    return code, out.getvalue()


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: FakeAnthropic) -> str:
    monkeypatch.delenv("EVALKIT_DATABASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cli.App, "claude", lambda self: fake.client())
    return f"sqlite:///{tmp_path / 'cli.db'}"


def test_cli_push_show_list_and_diff(db: str, tmp_path: Path) -> None:
    toml = tmp_path / "support.toml"
    toml.write_text(f'system = "Be kind."\ntemplate = "{V1}"\n')
    code, text = run_cli(db, "prompt", "push", "support", str(toml), "--note", "first")
    assert (code, text) == (
        0,
        f"support pushed version 1 ({content_hash('Be kind.', V1)[:12]})\n",
    )
    code, text = run_cli(db, "prompt", "push", "support", str(toml))
    assert (code, text.split(" (")[0]) == (0, "support unchanged, already version 1")

    template, system = tmp_path / "v2.txt", tmp_path / "system.txt"
    template.write_text(V2 + "\n")
    system.write_text("Be kind.")
    code, text = run_cli(db, "prompt", "push", "support", str(template), "--system", str(system))
    assert code == 0
    assert "pushed version 2" in text

    code, text = run_cli(db, "prompt", "show", "support@1")
    assert code == 0
    assert text.startswith("prompt support version 1  ")
    assert "labels none\nnote first\n\n--- system\nBe kind.\n\n--- template\n" + V1 in text
    code, raw = run_cli(db, "prompt", "show", "support", "--json")
    shown = json.loads(raw)
    assert (shown["version"], shown["template"], shown["system"]) == (2, V2 + "\n", "Be kind.")

    code, text = run_cli(db, "prompt", "diff", "support@1")
    assert code == 0
    assert text.splitlines()[:2] == ["--- support (v1) template", "+++ support (v2) template"]
    assert f"-{V1}" in text
    assert f"+{V2}" in text
    code, text = run_cli(db, "prompt", "diff", "support@2", "support@latest")
    assert (code, text) == (0, "")

    run_cli(db, "prompt", "label", "support@2", "production")
    code, text = run_cli(db, "prompt", "list")
    assert (code, text) == (0, "support\tv2\tproduction=2\n")
    code, raw = run_cli(db, "prompt", "list", "--json")
    assert json.loads(raw)[0]["labels"] == {"production": 2}


def test_cli_label_compare_and_set(db: str, tmp_path: Path) -> None:
    for i, template in enumerate((V1, V2, "Third: {{q}}")):
        path = tmp_path / f"v{i}.txt"
        path.write_text(template)
        assert run_cli(db, "prompt", "push", "support", str(path))[0] == 0

    code, text = run_cli(db, "prompt", "label", "support@1", "production")
    assert (code, text) == (0, "support@production: new -> v1\n")
    # By default the move checks the label is still where it was when read.
    code, text = run_cli(db, "prompt", "label", "support@2", "production")
    assert (code, text) == (0, "support@production: v1 -> v2\n")
    # A move that expects an older position is refused with exit 3.
    code, text = run_cli(db, "prompt", "label", "support@3", "production", "--expect", "1")
    assert (code, text) == (3, "")
    code, raw = run_cli(db, "prompt", "label", "support@3", "production", "--expect", "2", "--json")
    assert json.loads(raw) == {
        "name": "support",
        "label": "production",
        "previous": 2,
        "version": 3,
    }
    # A label can be labelled from another label.
    code, text = run_cli(db, "prompt", "label", "support@production", "canary", "--expect", "none")
    assert (code, text) == (0, "support@canary: new -> v3\n")
    assert run_cli(db, "prompt", "label", "support@1", "canary", "--expect", "none")[0] == 3
    assert run_cli(db, "prompt", "label", "support@1", "canary", "--expect", "x")[0] == 2
    assert run_cli(db, "prompt", "label", "support", "canary")[0] == 2
    assert run_cli(db, "prompt", "label", "support@9", "canary")[0] == 2
    assert run_cli(db, "prompt", "label", "support@1", "latest")[0] == 2


@pytest.mark.parametrize(
    ("content", "extra", "message"),
    [
        ('template = "x"\nmodel = "m"\n', (), "unknown key(s) model"),
        ("template = 3\n", (), "must be strings"),
        ('template = "x"\n', ("--system", "s.txt"), "holds its own system prompt"),
        ('template = ""\n', (), "template must not be empty"),
    ],
)
def test_cli_bad_prompt_files(
    *,
    db: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    content: str,
    extra: tuple[str, ...],
    message: str,
) -> None:
    path = tmp_path / "p.toml"
    path.write_text(content)
    assert run_cli(db, "prompt", "push", "support", str(path), *extra)[0] == 2
    assert message in capsys.readouterr().err


def test_cli_prompt_files_must_be_utf8_and_small(
    db: str, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "p.txt"
    path.write_bytes(b"\xff\xfe")
    assert run_cli(db, "prompt", "push", "support", str(path))[0] == 2
    assert "is not UTF-8" in capsys.readouterr().err
    monkeypatch.setattr(cli, "MAX_CONFIG_BYTES", 4)
    path.write_text("Hello {{q}}")
    assert run_cli(db, "prompt", "push", "support", str(path))[0] == 2
    assert "is larger than 4 bytes" in capsys.readouterr().err


def test_cli_run_with_a_registry_prompt(db: str, tmp_path: Path, fake: FakeAnthropic) -> None:
    (tmp_path / "qa.jsonl").write_bytes(dataset_bytes())
    for i, template in enumerate((V1, V2), start=1):
        (tmp_path / f"v{i}.txt").write_text(template)
        run_cli(db, "prompt", "push", "support", str(tmp_path / f"v{i}.txt"))
    run_cli(db, "prompt", "label", "support@1", "production")
    cfg = tmp_path / "run.toml"
    cfg.write_text(
        'label = "support"\n'
        '[dataset]\nname = "qa"\npath = "qa.jsonl"\n'
        '[target]\ntype = "prompt"\nmodel = "claude-sonnet-5-5"\nprompt = "support@production"\n'
        '[[scorers]]\ntype = "exact"\n'
    )
    code, text = run_cli(db, "run", str(cfg))
    assert code == 0, text
    assert f"prompt support@production: support version 1 ({content_hash('', V1)[:12]})" in text
    run_cli(db, "prompt", "label", "support@2", "production")
    code, raw = run_cli(db, "run", str(cfg), "--json")
    body = json.loads(raw)
    assert body["prompt"]["version"] == 2
    assert body["metrics"][0]["mean"] == 1.0
    code, text = run_cli(db, "estimate", str(cfg))
    assert code == 0, text


# The API


@pytest.fixture
async def client(sessions: sessionmaker[Session], fake: FakeAnthropic) -> Any:
    app = create_app(
        Settings.from_env({"EVALKIT_API_TOKENS": TOKEN}), sessions=sessions, claude=fake.client()
    )
    transport = httpx2.ASGITransport(app=BodyLimit(app, 1024 * 1024), raise_app_exceptions=False)
    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(transport=transport, base_url="http://test", headers=AUTH) as c,
    ):
        yield c


@pytest.mark.anyio
async def test_api_prompts(client: httpx2.AsyncClient) -> None:
    first = await client.post(
        "/v1/prompts/support/versions",
        json={"template": V1, "system": "S", "note": "first", "description": "help desk"},
    )
    assert first.status_code == 201
    body = first.json()
    assert (body["version"], body["content_hash"], body["template"]) == (
        1,
        content_hash("S", V1),
        V1,
    )
    again = await client.post("/v1/prompts/support/versions", json={"template": V1, "system": "S"})
    assert (again.status_code, again.json()["version"]) == (200, 1)
    assert (await client.post("/v1/prompts/support/versions", json={"template": V2})).json()[
        "version"
    ] == 2

    put = await client.put(
        "/v1/prompts/support/labels/production", json={"version": 1, "expected_version": None}
    )
    assert put.json() == {
        "name": "support",
        "label": "production",
        "previous_version": None,
        "version": 1,
    }
    stale = await client.put(
        "/v1/prompts/support/labels/production", json={"version": 2, "expected_version": 2}
    )
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "conflict"
    assert "is at version 1" in stale.json()["error"]["message"]
    # expected_version must be given, even as null: a label never moves unchecked.
    missing = await client.put("/v1/prompts/support/labels/production", json={"version": 2})
    assert missing.status_code == 400
    moved = await client.put(
        "/v1/prompts/support/labels/production", json={"version": 2, "expected_version": 1}
    )
    assert moved.json()["previous_version"] == 1

    listed = (await client.get("/v1/prompts")).json()["prompts"]
    assert [(p["name"], p["latest_version"], p["labels"]) for p in listed] == [
        ("support", 2, {"production": 2})
    ]
    history = (await client.get("/v1/prompts/support")).json()
    assert history["description"] == "help desk"
    assert [v["version"] for v in history["versions"]] == [1, 2]
    assert "template" not in history["versions"][0]
    by_label = (await client.get("/v1/prompts/support/versions/production")).json()
    assert (by_label["version"], by_label["label"], by_label["template"]) == (2, "production", V2)
    assert (await client.get("/v1/prompts/support/versions/1")).json()["system"] == "S"
    assert (await client.get("/v1/prompts/support/versions/latest")).json()["version"] == 2

    patch = await client.get("/v1/prompts/support/diff", params={"from": "1", "to": "production"})
    assert patch.headers["content-type"].startswith("text/x-diff")
    assert f"-{V1}\n" in patch.text
    assert f"+{V2}\n" in patch.text
    assert (await client.get("/v1/prompts/support/diff", params={"from": "2"})).text == ""

    assert (await client.get("/v1/prompts/billing")).status_code == 404
    assert (await client.get("/v1/prompts/support/versions/9")).status_code == 404
    assert (await client.get("/v1/prompts/support/versions/Bad")).status_code == 400
    bad = await client.post("/v1/prompts/support/versions", json={"template": "{{ x"})
    assert bad.status_code == 400
    assert (await client.post("/v1/prompts/-x/versions", json={"template": V1})).status_code == 400
    assert (await client.get("/v1/prompts/support/diff")).status_code == 400


@pytest.mark.anyio
async def test_api_run_with_a_registry_prompt(
    client: httpx2.AsyncClient, sessions: sessionmaker[Session]
) -> None:
    with sessions() as s:
        import_dataset(s, "qa", parse_jsonl(dataset_bytes()))
    await client.post("/v1/prompts/support/versions", json={"template": V2})
    await client.put(
        "/v1/prompts/support/labels/production", json={"version": 1, "expected_version": None}
    )
    body = run_config(prompt="support@production").model_dump(mode="json")
    created = await client.post("/v1/runs", json=body)
    assert created.status_code == 202, created.text
    assert created.json()["prompt"]["version"] == 1
    for _ in range(1000):
        run = (await client.get(f"/v1/runs/{created.json()['run_id']}")).json()
        if run["status"] == "completed":
            break
        await asyncio.sleep(0.01)
    assert run["metrics"][0]["mean"] == 1.0
    missing = await client.post("/v1/runs", json=run_config(prompt="nope").model_dump(mode="json"))
    assert missing.status_code == 404
