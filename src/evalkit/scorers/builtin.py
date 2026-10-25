"""Deterministic scorers: no model calls, same answer every time."""

from __future__ import annotations

import json
import math
import re
import string
from collections import Counter
from typing import Any

import regex
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from referencing.exceptions import Unresolvable

from evalkit.datasets import Case
from evalkit.scorers.base import Scorer, ScoreResult, expected_value

_WS = re.compile(r"\s+")
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)
_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?|[-+]?\.\d+")
_FENCE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL)
MAX_DETAIL = 300


def _short(text: str) -> str:
    return text if len(text) <= MAX_DETAIL else text[: MAX_DETAIL - 3] + "..."


class ExactMatch(Scorer):
    def __init__(
        self,
        name: str,
        *,
        case_sensitive: bool = True,
        normalize_whitespace: bool = False,
        expected_key: str | None = None,
    ) -> None:
        super().__init__(name)
        self.case_sensitive = case_sensitive
        self.normalize_whitespace = normalize_whitespace
        self.expected_key = expected_key

    def _norm(self, text: str) -> str:
        if self.normalize_whitespace:
            text = _WS.sub(" ", text).strip()
        return text if self.case_sensitive else text.casefold()

    async def score(self, case: Case, output: str) -> ScoreResult:
        expected = expected_value(case, self.expected_key)
        if expected is None:
            return ScoreResult.unscorable("case has no expected value")
        if not isinstance(expected, str):
            expected = json.dumps(expected, sort_keys=True)
        return ScoreResult.passed(self._norm(output) == self._norm(expected))


class RegexMatch(Scorer):
    """Matches ``pattern``, or the case's expected value when no pattern is set."""

    def __init__(
        self,
        name: str,
        *,
        pattern: str | None = None,
        ignore_case: bool = False,
        full_match: bool = False,
        timeout_seconds: float = 0.25,
    ) -> None:
        super().__init__(name)
        self.flags = regex.IGNORECASE if ignore_case else 0
        self.full_match = full_match
        self.timeout = timeout_seconds
        self.pattern = regex.compile(pattern, self.flags) if pattern is not None else None

    async def score(self, case: Case, output: str) -> ScoreResult:
        pattern = self.pattern
        if pattern is None:
            raw = case.expected
            if not isinstance(raw, str) or not raw:
                return ScoreResult.unscorable("case has no expected pattern")
            if len(raw) > 1000:
                return ScoreResult.unscorable("expected pattern is longer than 1000 characters")
            try:
                pattern = regex.compile(raw, self.flags)
            except regex.error as exc:
                return ScoreResult.unscorable(f"expected pattern is invalid: {exc}")
        try:
            if self.full_match:
                found = pattern.fullmatch(output, timeout=self.timeout)
            else:
                found = pattern.search(output, timeout=self.timeout)
        except TimeoutError:
            return ScoreResult.unscorable("pattern took longer than the time limit")
        return ScoreResult.passed(found is not None)


class JsonSchemaMatch(Scorer):
    """The output parses as JSON and validates against the schema (Draft 2020-12).

    References are resolved only within the schema; nothing is fetched.
    """

    def __init__(self, name: str, *, schema: dict[str, Any] | None = None) -> None:
        super().__init__(name)
        self.validator: Draft202012Validator | None = None
        if schema is not None:
            Draft202012Validator.check_schema(schema)
            self.validator = Draft202012Validator(schema)

    async def score(self, case: Case, output: str) -> ScoreResult:
        validator = self.validator
        if validator is None:
            if not isinstance(case.expected, dict):
                return ScoreResult.unscorable("case has no expected schema")
            try:
                Draft202012Validator.check_schema(case.expected)
            except SchemaError as exc:
                return ScoreResult.unscorable(_short(f"expected schema is invalid: {exc.message}"))
            validator = Draft202012Validator(case.expected)
        fenced = _FENCE.match(output)
        text = fenced.group(1) if fenced else output
        try:
            document = json.loads(text)
        except ValueError as exc:
            return ScoreResult.passed(ok=False, detail=_short(f"not JSON: {exc}"))
        try:
            error = next(iter(validator.iter_errors(document)), None)
        except Unresolvable as exc:
            return ScoreResult.unscorable(_short(f"schema reference cannot be resolved: {exc}"))
        if error is None:
            return ScoreResult.passed(ok=True)
        where = "/".join(str(p) for p in error.absolute_path) or "(root)"
        return ScoreResult.passed(ok=False, detail=_short(f"{where}: {error.message}"))


class NumericMatch(Scorer):
    """Compares a number in the output to the expected number within tolerance."""

    def __init__(
        self,
        name: str,
        *,
        abs_tol: float = 0.0,
        rel_tol: float = 1e-9,
        extract: str = "last",
        expected_key: str | None = None,
    ) -> None:
        super().__init__(name)
        if extract not in ("first", "last", "whole"):
            raise ValueError("extract must be first, last or whole")
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.extract = extract
        self.expected_key = expected_key

    def _parse_output(self, output: str) -> float | None:
        if self.extract == "whole":
            return _to_float(output.strip())
        found = _NUMBER.findall(output)
        if not found:
            return None
        return _to_float(found[0] if self.extract == "first" else found[-1])

    async def score(self, case: Case, output: str) -> ScoreResult:
        raw = expected_value(case, self.expected_key)
        expected = _to_float(raw) if isinstance(raw, str | int | float) else None
        if expected is None or isinstance(raw, bool):
            return ScoreResult.unscorable("case has no numeric expected value")
        got = self._parse_output(output)
        if got is None:
            return ScoreResult.passed(ok=False, detail="no number in output")
        ok = math.isclose(got, expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        return ScoreResult.passed(ok, "" if ok else f"got {got:g}, expected {expected:g}")


def _to_float(value: str | float) -> float | None:
    if isinstance(value, int | float):
        return float(value) if math.isfinite(value) else None
    try:
        number = float(value.replace(",", ""))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def normalize_answer(text: str) -> list[str]:
    """SQuAD normalisation: lower case, drop punctuation and articles, split on spaces."""
    text = text.lower().translate(_PUNCT)
    return _ARTICLES.sub(" ", text).split()


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = normalize_answer(prediction), normalize_answer(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


class TokenF1(Scorer):
    """Bag-of-tokens F1 against the reference; partial credit without embeddings."""

    def __init__(self, name: str, *, expected_key: str | None = None) -> None:
        super().__init__(name)
        self.expected_key = expected_key

    async def score(self, case: Case, output: str) -> ScoreResult:
        expected = expected_value(case, self.expected_key)
        if not isinstance(expected, str):
            return ScoreResult.unscorable("case has no expected text")
        return ScoreResult(token_f1(output, expected))
