"""``{{ field }}`` substitution for prompt templates.

Deliberately not a template language: no expressions, loops or filters, so a
template from an API caller cannot execute anything. A placeholder names a key
of the case input (dotted for nested objects); ``{{ input }}`` is the whole
input. Anything that looks like a placeholder but is not a valid one is an
error at validation time, so a typo cannot silently send a literal ``{{x}`` to
the model.
"""

from __future__ import annotations

import json
import re
from typing import Any

from evalkit.errors import InvalidInputError

_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)\s*\}\}")


def placeholders(template: str) -> list[str]:
    """The field paths in ``template``; raises on malformed braces."""
    stripped = _PLACEHOLDER.sub("", template)
    if "{{" in stripped or "}}" in stripped:
        raise InvalidInputError(
            "template has '{{' or '}}' that is not a valid {{ field }} placeholder"
        )
    return _PLACEHOLDER.findall(template)


def render(template: str, value: Any) -> str:
    def lookup(match: re.Match[str]) -> str:
        path = match.group(1)
        if path == "input":
            return _text(value)
        current = value
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                raise InvalidInputError(f"case input has no field {path!r} used by the template")
            current = current[part]
        return _text(current)

    return _PLACEHOLDER.sub(lookup, template)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
