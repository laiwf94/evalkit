"""JSON logging with a context-local correlation ID and credential scrubbing."""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, TextIO

_context: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "evalkit_log_context", default=None
)

# Shapes of credentials we know about. The scrubber runs on the final rendered
# line, so a key that reaches a log through an exception message is caught too.
_CREDENTIAL_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=\-]{8,}"),
    # key=value and JSON "key": "value", also when the JSON is itself escaped.
    re.compile(
        r"(?i)((?:\\?\")?(?:api[_-]?key|x-api-key|authorization|token|secret|password)"
        r"(?:\\?\")?\s*[:=]\s*(?:\\?\")?)[^\s\"\\,}]{6,}"
    ),
]
_REDACTED = "[REDACTED]"

_RESERVED = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


@contextmanager
def log_context(**fields: str) -> Iterator[None]:
    """Attach fields (request_id, run_id, ...) to every record logged inside the block."""
    current = _context.get() or {}
    token = _context.set({**current, **fields})
    try:
        yield
    finally:
        _context.reset(token)


def current_context() -> dict[str, str]:
    return dict(_context.get() or {})


class Scrubber:
    """Removes known secret values and credential-shaped strings from text."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._secrets: tuple[str, ...] = ()

    def register(self, *secrets: str) -> None:
        with self._lock:
            known = set(self._secrets)
            known.update(s for s in secrets if len(s) >= 6)
            # Longest first so a secret that contains another is removed whole.
            self._secrets = tuple(sorted(known, key=len, reverse=True))

    def scrub(self, text: str) -> str:
        for secret in self._secrets:
            if secret in text:
                text = text.replace(secret, _REDACTED)
        for pattern in _CREDENTIAL_PATTERNS:
            text = pattern.sub(_replacement, text)
        return text


def _replacement(match: re.Match[str]) -> str:
    if match.lastindex:
        return match.group(1) + _REDACTED
    return _REDACTED


scrubber = Scrubber()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        entry.update(_context.get() or {})
        for key, value in vars(record).items():
            if key not in _RESERVED and not key.startswith("_"):
                entry[key] = value
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return scrubber.scrub(json.dumps(entry, default=str, ensure_ascii=False))


def configure_logging(level: str = "INFO", stream: TextIO | None = None) -> None:
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # The SDK and HTTP client log request lines at INFO; keep them out unless debugging.
    for noisy in ("httpx", "httpx2", "httpcore", "httpcore2", "anthropic"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))
