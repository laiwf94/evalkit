"""Process configuration, read once from the environment at start-up.

Every variable is optional; the defaults run a local, single-user setup on
SQLite. Secrets can be given directly or as a path in ``<NAME>_FILE`` (the
Docker and Kubernetes secret convention); setting both is an error rather than
a silent precedence rule.

    EVALKIT_DATABASE_URL        SQLAlchemy URL (default sqlite:///evalkit.db)
    EVALKIT_HOST, EVALKIT_PORT  API listen address (default 127.0.0.1:8080)
    EVALKIT_API_TOKENS          comma-separated bearer tokens, 16+ chars each (secret)
    EVALKIT_MAX_BODY_BYTES      request body cap for the API (default 16 MiB)
    EVALKIT_MAX_ACTIVE_RUNS     runs the API or a worker executes at once (default 2; 0: the
                                API only queues runs, for worker daemons to execute)
    EVALKIT_WORKER_POLL_SECONDS how often an idle worker looks for queued runs (default 5)
    EVALKIT_DRAIN_SECONDS       on SIGTERM, how long in-flight cases may take to finish (default 25)
    EVALKIT_LOG_LEVEL           DEBUG, INFO, WARNING or ERROR (default INFO)
    EVALKIT_LEASE_SECONDS       seconds before a dead runner's run can be resumed (default 30)
    EVALKIT_REDACTION_FILE      TOML file with a [redaction] table (default: all built-ins on)
    EVALKIT_PRICES_FILE         TOML [prices.<model>] tables overriding the built-in prices
    EVALKIT_HTTP_TARGETS_FILE   TOML [endpoints.<name>] tables: the only URLs HTTP targets may call
    EVALKIT_CACHE_MAX_AGE       sweep cache entries unused for this long, e.g. 30d (default: never)
    EVALKIT_CACHE_MAX_BYTES     sweep least recently used entries above this size, e.g. 20GiB
    EVALKIT_CACHE_SWEEP_SECONDS how often worker daemons sweep when a limit is set (default 3600)
    EVALKIT_BLOB_URL            blob store for long cached outputs: file:///path or
                                s3+https://host[:port]/bucket[/prefix] (default: none)
    EVALKIT_BLOB_ACCESS_KEY,    S3 credentials for the blob store (secrets)
    EVALKIT_BLOB_SECRET_KEY
    EVALKIT_BLOB_REGION         S3 signing region (default us-east-1)
    EVALKIT_BLOB_MIN_BYTES      cached outputs this long go to the blob store (default 64 KiB)
    EVALKIT_BLOB_GRACE          garbage collection keeps unreferenced blobs this new (default 1h)
    ANTHROPIC_API_KEY           Claude API key (secret)
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from evalkit.blobs import BlobStore, open_store
from evalkit.pricing import ModelPrice
from evalkit.redaction import RedactionConfig
from evalkit.targets.http import HttpEndpoint

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

MIN_TOKEN_LENGTH = 16
ENDPOINT_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
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
    max_active_runs: int = Field(default=2, ge=0, le=64)
    worker_poll_seconds: float = Field(default=5.0, ge=0.05, le=300)
    drain_seconds: float = Field(default=25.0, ge=0, le=3600)
    log_level: LogLevel = "INFO"
    lease_seconds: float = Field(default=30.0, ge=2, le=3600)
    anthropic_api_key: SecretStr | None = None
    redaction: RedactionConfig = RedactionConfig()
    prices: dict[str, ModelPrice] = Field(default_factory=dict)
    http_endpoints: dict[str, HttpEndpoint] = Field(default_factory=dict)
    cache_max_age_seconds: float | None = Field(default=None, gt=0)
    cache_max_bytes: int | None = Field(default=None, ge=0)
    cache_sweep_seconds: float = Field(default=3600.0, ge=1, le=7 * 86400)
    blob_url: str | None = None
    blob_access_key: SecretStr | None = None
    blob_secret_key: SecretStr | None = None
    blob_region: str = Field(default="us-east-1", pattern=r"^[a-z0-9-]{1,64}$")
    blob_min_bytes: int = Field(default=64 * 1024, ge=0)
    blob_grace_seconds: float = Field(default=3600.0, ge=0)

    @property
    def cache_limited(self) -> bool:
        return self.cache_max_age_seconds is not None or self.cache_max_bytes is not None

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
        values = [self.anthropic_api_key.get_secret_value()] if self.anthropic_api_key else []
        for endpoint in self.http_endpoints.values():
            values.extend(v.get_secret_value() for v in endpoint.secret_headers.values())
        values.extend(
            s.get_secret_value() for s in (self.blob_access_key, self.blob_secret_key) if s
        )
        return values

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
            "EVALKIT_LEASE_SECONDS": "lease_seconds",
            "EVALKIT_WORKER_POLL_SECONDS": "worker_poll_seconds",
            "EVALKIT_DRAIN_SECONDS": "drain_seconds",
            "EVALKIT_CACHE_SWEEP_SECONDS": "cache_sweep_seconds",
            "EVALKIT_BLOB_URL": "blob_url",
            "EVALKIT_BLOB_REGION": "blob_region",
        }
        for var, field in plain.items():
            raw = env.get(var, "").strip()
            if raw:
                values[field] = raw.upper() if field == "log_level" else raw

        for var, field, parse in (
            ("EVALKIT_CACHE_MAX_AGE", "cache_max_age_seconds", parse_duration),
            ("EVALKIT_CACHE_MAX_BYTES", "cache_max_bytes", parse_size),
            ("EVALKIT_BLOB_MIN_BYTES", "blob_min_bytes", parse_size),
            ("EVALKIT_BLOB_GRACE", "blob_grace_seconds", parse_duration),
        ):
            raw = env.get(var, "").strip()
            if raw:
                try:
                    values[field] = parse(raw)
                except ValueError as exc:
                    raise ConfigError(f"{var}: {exc}") from None
        for var, field in (
            ("EVALKIT_BLOB_ACCESS_KEY", "blob_access_key"),
            ("EVALKIT_BLOB_SECRET_KEY", "blob_secret_key"),
        ):
            secret = read_secret(env, var)
            if secret is not None:
                values[field] = SecretStr(secret)

        tokens = read_secret(env, "EVALKIT_API_TOKENS")
        if tokens is not None:
            values["api_token_digests"] = _token_digests(tokens)
        key = read_secret(env, "ANTHROPIC_API_KEY")
        if key is not None:
            values["anthropic_api_key"] = SecretStr(key)

        redaction_file = env.get("EVALKIT_REDACTION_FILE", "").strip()
        if redaction_file:
            values["redaction"] = load_toml_table(redaction_file, "redaction")
        prices_file = env.get("EVALKIT_PRICES_FILE", "").strip()
        if prices_file:
            values["prices"] = load_toml_table(prices_file, "prices")
        endpoints_file = env.get("EVALKIT_HTTP_TARGETS_FILE", "").strip()
        if endpoints_file:
            values["http_endpoints"] = _endpoints(env, load_toml_table(endpoints_file, "endpoints"))

        try:
            settings = cls.model_validate(values)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
            raise ConfigError(f"invalid configuration: {problems}") from None
        if settings.blob_url is not None:
            # Fail at start-up, not at the first long output.
            try:
                store = settings.open_blob_store()
                if store is not None:
                    store.close()
            except ValueError as exc:
                raise ConfigError(f"EVALKIT_BLOB_URL: {exc}") from None
        return settings

    def open_blob_store(self) -> BlobStore | None:
        if self.blob_url is None:
            return None
        return open_store(
            self.blob_url,
            access_key=self.blob_access_key.get_secret_value() if self.blob_access_key else None,
            secret_key=self.blob_secret_key.get_secret_value() if self.blob_secret_key else None,
            region=self.blob_region,
        )


_SIZE = re.compile(r"^(\d+(?:\.\d+)?)\s*([KMGT]i?B|B)?$", re.IGNORECASE)
_SIZE_UNITS = {"": 1, "b": 1, "kb": 10**3, "mb": 10**6, "gb": 10**9, "tb": 10**12}
_SIZE_UNITS.update({"kib": 2**10, "mib": 2**20, "gib": 2**30, "tib": 2**40})
_DURATION = re.compile(r"^(\d+(?:\.\d+)?)\s*([smhdw]?)$", re.IGNORECASE)
_DURATION_UNITS = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}


def parse_size(raw: str) -> int:
    """Bytes from ``1048576``, ``500MB`` or ``20GiB``."""
    match = _SIZE.fullmatch(raw.strip())
    if not match:
        raise ValueError(f"{raw!r} is not a size such as 500MB or 20GiB")
    return int(float(match.group(1)) * _SIZE_UNITS[(match.group(2) or "").lower()])


def parse_duration(raw: str) -> float:
    """Seconds from ``90``, ``45m``, ``12h``, ``30d`` or ``2w``."""
    match = _DURATION.fullmatch(raw.strip())
    if not match or float(match.group(1)) <= 0:
        raise ValueError(f"{raw!r} is not a duration such as 45m, 12h or 30d")
    return float(match.group(1)) * _DURATION_UNITS[match.group(2).lower()]


def load_toml_table(path: str, table: str) -> dict[str, object]:
    try:
        with Path(path).open("rb") as fh:
            data = tomllib.load(fh)
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc.strerror}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from None
    section = data.get(table)
    if not isinstance(section, dict):
        raise ConfigError(f"{path} has no [{table}] table")
    return section


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


def _endpoints(env: Mapping[str, str], table: dict[str, object]) -> dict[str, object]:
    """Endpoint configs with ``secret_headers`` resolved from the variables they name."""
    out: dict[str, object] = {}
    for name, raw in table.items():
        if not ENDPOINT_NAME.fullmatch(name):
            raise ConfigError(f"endpoint name {name!r} must match {ENDPOINT_NAME.pattern}")
        if not isinstance(raw, dict):
            raise ConfigError(f"endpoints.{name} must be a table")
        endpoint = dict(raw)
        secret_vars = endpoint.get("secret_headers", {})
        if not isinstance(secret_vars, dict):
            raise ConfigError(f"endpoints.{name}.secret_headers must map header names to variables")
        resolved = {}
        for header, var in secret_vars.items():
            value = read_secret(env, str(var))
            if value is None:
                raise ConfigError(f"endpoints.{name}: header {header} needs {var} or {var}_FILE")
            resolved[header] = value
        endpoint["secret_headers"] = resolved
        out[name] = endpoint
    return out


def _token_digests(raw: str) -> tuple[bytes, ...]:
    tokens = [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip()]
    if not tokens:
        raise ConfigError("EVALKIT_API_TOKENS is set but contains no tokens")
    for token in tokens:
        if len(token) < MIN_TOKEN_LENGTH:
            raise ConfigError(f"every API token must be at least {MIN_TOKEN_LENGTH} characters")
    return tuple(hashlib.sha256(t.encode()).digest() for t in tokens)
