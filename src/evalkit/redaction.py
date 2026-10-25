"""PII redaction for everything we store.

Outputs from the application under test are redacted before they are written
to the results table or the response cache, and scorers (including the LLM
judge) see the redacted text. That keeps personal data out of the database and
away from the judge, and it means a rerun served from the cache scores exactly
like the run that filled it.

Matching runs on the ``regex`` module so every pattern gets a time limit. A
pattern that exceeds it fails closed: the whole text is replaced, because
storing unredacted text is the worse failure.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import regex
from pydantic import BaseModel, ConfigDict, Field, field_validator

BuiltinName = Literal["email", "phone", "card", "ssn", "credential"]

# Bump when a built-in pattern changes, so cache entries redacted under the old
# rules are not reused.
BUILTIN_VERSION = 1

FAILED_PLACEHOLDER = "[REDACTION_FAILED]"

_BUILTINS: dict[str, str] = {
    "email": (
        r"(?<![\w.%+-])[A-Za-z0-9._%+-]{1,64}@"
        r"[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63}){0,8}\.[A-Za-z]{2,63}(?![\w-])"
    ),
    # International numbers with a leading +, and North American numbers written
    # with separators. Bare digit runs are left alone: they are usually amounts or IDs.
    "phone": (
        r"(?<![\w+])(?:\+\d{1,3}[ .-]?(?:\(\d{1,4}\)|\d{1,4})(?:[ .-]?\d{2,4}){2,4}"
        r"|\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4})(?!\w)"
    ),
    "card": r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)",
    "ssn": r"(?<!\d)(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\d)",
    "credential": (
        r"sk-ant-[A-Za-z0-9_-]{8,}"
        r"|\bAKIA[0-9A-Z]{16}\b"
        r"|\bgh[pousr]_[A-Za-z0-9]{36,255}\b"
        r"|\bxox[abprs]-[A-Za-z0-9-]{10,}"
        r"|\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
        r"|-----BEGIN [A-Z ]{0,40}PRIVATE KEY-----[\s\S]{0,8192}?"
        r"-----END [A-Z ]{0,40}PRIVATE KEY-----"
    ),
}


class PatternSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,31}$")
    pattern: str = Field(min_length=1, max_length=1000)

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, value: str) -> str:
        try:
            regex.compile(value)
        except regex.error as exc:
            raise ValueError(f"invalid pattern: {exc}") from None
        return value


class RedactionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    builtins: tuple[BuiltinName, ...] = ("email", "phone", "card", "ssn", "credential")
    patterns: tuple[PatternSpec, ...] = Field(default=(), max_length=64)
    timeout_seconds: float = Field(default=0.25, gt=0, le=5)

    def fingerprint(self) -> str:
        payload = {"v": BUILTIN_VERSION, **self.model_dump(mode="json")}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Redaction:
    text: str
    counts: dict[str, int] = field(default_factory=dict)
    failed: bool = False


def luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


class Redactor:
    def __init__(self, config: RedactionConfig | None = None) -> None:
        self.config = config or RedactionConfig()
        self._rules: list[tuple[str, regex.Pattern[str], Callable[[str], bool] | None]] = []
        if not self.config.enabled:
            return
        for name in self.config.builtins:
            check = _card_check if name == "card" else None
            self._rules.append((name.upper(), regex.compile(_BUILTINS[name]), check))
        for spec in self.config.patterns:
            self._rules.append((spec.name, regex.compile(spec.pattern), None))

    def redact(self, text: str) -> Redaction:
        counts: dict[str, int] = {}
        for label, pattern, check in self._rules:
            placeholder = f"[{label}]"
            hits = 0

            def replace(
                m: regex.Match[str],
                placeholder: str = placeholder,
                check: Callable[[str], bool] | None = check,
            ) -> str:
                nonlocal hits
                if check is not None and not check(m.group(0)):
                    return m.group(0)
                hits += 1
                return placeholder

            try:
                text = pattern.sub(replace, text, timeout=self.config.timeout_seconds)
            except TimeoutError:
                return Redaction(FAILED_PLACEHOLDER, {label: 1}, failed=True)
            if hits:
                counts[label] = counts.get(label, 0) + hits
        return Redaction(text, counts)


def _card_check(candidate: str) -> bool:
    digits = "".join(ch for ch in candidate if ch.isdigit())
    return 13 <= len(digits) <= 19 and luhn_valid(digits)
