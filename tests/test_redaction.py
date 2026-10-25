import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from evalkit.redaction import (
    FAILED_PLACEHOLDER,
    PatternSpec,
    RedactionConfig,
    Redactor,
    luhn_valid,
)

R = Redactor()


@pytest.mark.parametrize(
    ("text", "expected", "label"),
    [
        ("mail alice@example.com now", "mail [EMAIL] now", "EMAIL"),
        ("a.b+c@sub.mail.co.uk.", "[EMAIL].", "EMAIL"),
        ("call +44 20 7946 0958 today", "call [PHONE] today", "PHONE"),
        ("call (415) 555-0132", "call [PHONE]", "PHONE"),
        ("call 415-555-0132.", "call [PHONE].", "PHONE"),
        ("card 4111 1111 1111 1111 ok", "card [CARD] ok", "CARD"),
        ("card 4111-1111-1111-1111", "card [CARD]", "CARD"),
        ("ssn 123-45-6789", "ssn [SSN]", "SSN"),
        ("key sk-ant-api03-abcdefgh_ijk", "key [CREDENTIAL]", "CREDENTIAL"),
        ("aws AKIAIOSFODNN7EXAMPLE", "aws [CREDENTIAL]", "CREDENTIAL"),
        ("jwt eyJhbGciOiJI.eyJzdWIiOiIx.SflKxwRJSMeKKF2QT4", "jwt [CREDENTIAL]", "CREDENTIAL"),
        (
            "-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----",
            "[CREDENTIAL]",
            "CREDENTIAL",
        ),
    ],
)
def test_builtins(text: str, expected: str, label: str) -> None:
    result = R.redact(text)
    assert result.text == expected
    assert result.counts == {label: 1}
    assert not result.failed


@pytest.mark.parametrize(
    "text",
    [
        "order 4111 1111 1111 1112 failed",  # fails the Luhn check
        "the year 2024 and 1,000,000 dollars",
        "ssn 000-12-3456 and 666-12-3456 are invalid",
        "version 1.2.3 at @handle",
        "id 12345678901234567890123",  # too long for a card
    ],
)
def test_lookalikes_are_kept(text: str) -> None:
    assert R.redact(text).text == text


def test_disabled_and_selected_builtins() -> None:
    assert Redactor(RedactionConfig(enabled=False)).redact("alice@example.com").text == (
        "alice@example.com"
    )
    only_ssn = Redactor(RedactionConfig(builtins=("ssn",)))
    assert only_ssn.redact("alice@example.com 123-45-6789").text == "alice@example.com [SSN]"


def test_custom_patterns_and_counts() -> None:
    cfg = RedactionConfig(builtins=(), patterns=(PatternSpec(name="TICKET", pattern=r"TCK-\d+"),))
    result = Redactor(cfg).redact("see TCK-1 and TCK-22")
    assert result.text == "see [TICKET] and [TICKET]"
    assert result.counts == {"TICKET": 2}


def test_invalid_custom_patterns() -> None:
    with pytest.raises(ValidationError, match="invalid pattern"):
        PatternSpec(name="X", pattern="(unclosed")
    with pytest.raises(ValidationError):
        PatternSpec(name="lower", pattern="x")


def test_catastrophic_pattern_fails_closed() -> None:
    cfg = RedactionConfig(
        builtins=(), patterns=(PatternSpec(name="BAD", pattern=r"(a|aa)+$"),), timeout_seconds=0.05
    )
    result = Redactor(cfg).redact("secret alice@example.com " + "a" * 40 + "!")
    assert result.failed
    assert result.text == FAILED_PLACEHOLDER


def test_fingerprint_tracks_config() -> None:
    a = RedactionConfig()
    assert a.fingerprint() == RedactionConfig().fingerprint()
    assert a.fingerprint() != RedactionConfig(builtins=("email",)).fingerprint()
    assert a.fingerprint() != RedactionConfig(enabled=False).fingerprint()


def test_luhn() -> None:
    assert luhn_valid("4111111111111111")
    assert luhn_valid("79927398713")
    assert not luhn_valid("79927398710")


local = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789._+", min_size=1, max_size=20).filter(
    lambda s: s[0].isalnum()
)
domain = st.lists(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=10),
    min_size=1,
    max_size=3,
)
tld = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=2, max_size=6)
emails = st.builds(lambda u, d, t: f"{u}@{'.'.join(d)}.{t}", local, domain, tld)
filler = st.text(alphabet=st.characters(categories=["Ll", "Lu", "Zs", "Po"]), max_size=40)


def _with_check_digit(body: str) -> str:
    for d in "0123456789":
        if luhn_valid(body + d):
            return body + d
    raise AssertionError("unreachable")


cards = st.builds(_with_check_digit, st.text(alphabet="0123456789", min_size=12, max_size=18)).map(
    lambda n: " ".join(n[i : i + 4] for i in range(0, len(n), 4))
)


@given(filler, emails, filler)
@settings(max_examples=300)
def test_generated_emails_are_always_removed(before: str, email: str, after: str) -> None:
    text = f"{before} {email} {after}"
    out = R.redact(text).text
    assert email not in out
    assert "[EMAIL]" in out


@given(filler, cards, filler)
@settings(max_examples=300)
def test_generated_valid_cards_are_always_removed(before: str, card: str, after: str) -> None:
    out = R.redact(f"{before} {card} {after}").text
    assert card not in out
    assert "".join(c for c in card if c.isdigit()) not in "".join(c for c in out if c.isdigit())


@given(st.text(max_size=200) | st.builds(lambda a, e, b: a + e + b, filler, emails, filler))
@settings(max_examples=300)
def test_redaction_is_idempotent(text: str) -> None:
    once = R.redact(text).text
    assert R.redact(once).text == once
