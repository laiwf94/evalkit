"""Process configuration, read once from the environment at start-up.

Every variable is optional; the defaults run a local, single-user setup on
SQLite. Secrets can be given directly or as a path in ``<NAME>_FILE`` (the
Docker and Kubernetes secret convention); setting both is an error rather than
a silent precedence rule.

    EVALKIT_DATABASE_URL        SQLAlchemy URL (default sqlite:///evalkit.db)
    EVALKIT_HOST, EVALKIT_PORT  API listen address (default 127.0.0.1:8080)
    EVALKIT_API_TOKENS          comma-separated bearer tokens, 16+ chars each (secret)
    EVALKIT_MAX_BODY_BYTES      request body cap for the API (default 16 MiB)
    EVALKIT_MAX_ACTIVE_RUNS     runs the API executes at once (default 2)
    EVALKIT_LOG_LEVEL           DEBUG, INFO, WARNING or ERROR (default INFO)
    ANTHROPIC_API_KEY           Claude API key (secret)
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

MIN_TOKEN_LENGTH = 16
MAX_SECRET_FILE_BYTES = 64 * 1024


class ConfigError(Exception):
    """Raised when the environment does not describe a valid configuration."""


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    database_url: str = "sqlite:///evalkit.db"
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    api_token_digests: tuple[bytes, ...] = ()
    max_body_bytes: int = Field(default=16 * 1024 * 1024, ge=1024, le=256 * 1024 * 1024)
    max_active_runs: int = Field(default=2, ge=1, le=64)
    log_level: LogLevel = "INFO"
    anthropic_api_key: SecretStr | None = None

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_token_digests)

    @property
    def host_is_loopback(self) -> bool:
        if self.host == "localhost":
            return True
        try:
            return ipaddress.ip_address(self.host).is_loopback
        except ValueError:
            return False

    def secret_values(self) -> list[str]:
        """Raw secret strings the log scrubber must never let through."""
        return [self.anthropic_api_key.get_secret_value()] if self.anthropic_api_key else []

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        values: dict[str, object] = {}
        plain = {
            "EVALKIT_DATABASE_URL": "database_url",
            "EVALKIT_HOST": "host",
            "EVALKIT_PORT": "port",
            "EVALKIT_MAX_BODY_BYTES": "max_body_bytes",
            "EVALKIT_MAX_ACTIVE_RUNS": "max_active_runs",
            "EVALKIT_LOG_LEVEL": "log_level",
        }
        for var, field in plain.items():
            raw = env.get(var, "").strip()
            if raw:
                values[field] = raw.upper() if field == "log_level" else raw

        tokens = read_secret(env, "EVALKIT_API_TOKENS")
        if tokens is not None:
            values["api_token_digests"] = _token_digests(tokens)
        key = read_secret(env, "ANTHROPIC_API_KEY")
        if key is not None:
            values["anthropic_api_key"] = SecretStr(key)

        try:
            return cls.model_validate(values)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
            raise ConfigError(f"invalid configuration: {problems}") from None


def read_secret(env: Mapping[str, str], name: str) -> str | None:
    """Return the secret ``name`` from the variable itself or from ``name_FILE``."""
    direct = env.get(name, "").strip()
    path = env.get(f"{name}_FILE", "").strip()
    if direct and path:
        raise ConfigError(f"set {name} or {name}_FILE, not both")
    if path:
        file = Path(path)
        try:
            size = file.stat().st_size
            if size > MAX_SECRET_FILE_BYTES:
                raise ConfigError(f"{name}_FILE is larger than {MAX_SECRET_FILE_BYTES} bytes")
            content = file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            # The path is configuration, not a secret, so it is safe to name.
            raise ConfigError(f"cannot read {name}_FILE {path}: {exc.strerror}") from None
        if not content:
            raise ConfigError(f"{name}_FILE {path} is empty")
        return content
    return direct or None


def _token_digests(raw: str) -> tuple[bytes, ...]:
    tokens = [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip()]
    if not tokens:
        raise ConfigError("EVALKIT_API_TOKENS is set but contains no tokens")
    for token in tokens:
        if len(token) < MIN_TOKEN_LENGTH:
            raise ConfigError(f"every API token must be at least {MIN_TOKEN_LENGTH} characters")
    return tuple(hashlib.sha256(t.encode()).digest() for t in tokens)
