import hashlib
from pathlib import Path

import pytest

from evalkit.config import ConfigError, Settings, read_secret


def test_defaults_are_local_and_unauthenticated() -> None:
    s = Settings.from_env({})
    assert s.database_url == "sqlite:///evalkit.db"
    assert (s.host, s.port) == ("127.0.0.1", 8080)
    assert not s.auth_enabled
    assert s.host_is_loopback
    assert s.anthropic_api_key is None
    assert s.secret_values() == []


def test_plain_values_are_parsed() -> None:
    s = Settings.from_env(
        {
            "EVALKIT_DATABASE_URL": "postgresql+psycopg://u@db/evalkit",
            "EVALKIT_HOST": "0.0.0.0",
            "EVALKIT_PORT": "9000",
            "EVALKIT_MAX_BODY_BYTES": "2048",
            "EVALKIT_MAX_ACTIVE_RUNS": "4",
            "EVALKIT_LOG_LEVEL": "debug",
        }
    )
    assert s.database_url.startswith("postgresql")
    assert s.port == 9000
    assert s.max_body_bytes == 2048
    assert s.max_active_runs == 4
    assert s.log_level == "DEBUG"
    assert not s.host_is_loopback


@pytest.mark.parametrize(
    ("host", "loopback"),
    [("localhost", True), ("::1", True), ("127.0.0.2", True), ("10.0.0.1", False), ("db", False)],
)
def test_loopback_detection(host: str, loopback: bool) -> None:
    assert Settings.from_env({"EVALKIT_HOST": host}).host_is_loopback is loopback


@pytest.mark.parametrize(
    "env",
    [
        {"EVALKIT_PORT": "0"},
        {"EVALKIT_PORT": "http"},
        {"EVALKIT_LOG_LEVEL": "loud"},
        {"EVALKIT_MAX_ACTIVE_RUNS": "1000"},
        {"EVALKIT_MAX_BODY_BYTES": "1"},
    ],
)
def test_invalid_values_are_rejected_with_field_name(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError, match="invalid configuration"):
        Settings.from_env(env)


def test_api_tokens_are_stored_as_digests() -> None:
    s = Settings.from_env({"EVALKIT_API_TOKENS": "token-aaaaaaaaaaaa, token-bbbbbbbbbbbb"})
    assert s.auth_enabled
    assert s.api_token_digests == (
        hashlib.sha256(b"token-aaaaaaaaaaaa").digest(),
        hashlib.sha256(b"token-bbbbbbbbbbbb").digest(),
    )
    assert "token-aaaa" not in repr(s)


def test_short_or_empty_tokens_are_rejected() -> None:
    with pytest.raises(ConfigError, match="at least 16"):
        Settings.from_env({"EVALKIT_API_TOKENS": "short"})
    with pytest.raises(ConfigError, match="no tokens"):
        Settings.from_env({"EVALKIT_API_TOKENS": " , ,"})


def test_secret_from_file(tmp_path: Path) -> None:
    key_file = tmp_path / "key"
    key_file.write_text("sk-ant-test-0123456789\n")
    s = Settings.from_env({"ANTHROPIC_API_KEY_FILE": str(key_file)})
    assert s.anthropic_api_key is not None
    assert s.anthropic_api_key.get_secret_value() == "sk-ant-test-0123456789"
    assert s.secret_values() == ["sk-ant-test-0123456789"]
    assert "sk-ant" not in repr(s)


def test_secret_direct_and_file_together_is_an_error(tmp_path: Path) -> None:
    f = tmp_path / "k"
    f.write_text("x" * 20)
    with pytest.raises(ConfigError, match="not both"):
        read_secret({"K": "a", "K_FILE": str(f)}, "K")


def test_secret_file_problems(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        read_secret({"K_FILE": str(tmp_path / "missing")}, "K")
    empty = tmp_path / "empty"
    empty.write_text("  \n")
    with pytest.raises(ConfigError, match="empty"):
        read_secret({"K_FILE": str(empty)}, "K")
    big = tmp_path / "big"
    big.write_bytes(b"x" * (64 * 1024 + 1))
    with pytest.raises(ConfigError, match="larger than"):
        read_secret({"K_FILE": str(big)}, "K")


def test_unset_secret_is_none() -> None:
    assert read_secret({}, "K") is None
    assert read_secret({"K": "  "}, "K") is None


def test_redaction_file(tmp_path: Path) -> None:
    f = tmp_path / "redaction.toml"
    f.write_text(
        '[redaction]\nbuiltins = ["email"]\n'
        '[[redaction.patterns]]\nname = "TICKET"\npattern = "TCK-\\\\d+"\n'
    )
    s = Settings.from_env({"EVALKIT_REDACTION_FILE": str(f)})
    assert s.redaction.builtins == ("email",)
    assert s.redaction.patterns[0].name == "TICKET"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not toml [", "not valid TOML"),
        ("[other]\nx = 1\n", r"no \[redaction\] table"),
        ('[redaction]\nbuiltins = ["dob"]\n', "invalid configuration"),
    ],
)
def test_bad_redaction_file(tmp_path: Path, content: str, message: str) -> None:
    f = tmp_path / "r.toml"
    f.write_text(content)
    with pytest.raises(ConfigError, match=message):
        Settings.from_env({"EVALKIT_REDACTION_FILE": str(f)})


def test_missing_redaction_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        Settings.from_env({"EVALKIT_REDACTION_FILE": str(tmp_path / "none.toml")})
