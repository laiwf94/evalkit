import io
import json
import logging

import pytest

from evalkit.logs import Scrubber, configure_logging, current_context, log_context, scrubber


@pytest.fixture
def stream() -> io.StringIO:
    out = io.StringIO()
    configure_logging("DEBUG", out)
    return out


def lines(out: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in out.getvalue().splitlines()]


def test_records_are_json_with_context_and_extras(stream: io.StringIO) -> None:
    log = logging.getLogger("evalkit.test")
    with log_context(request_id="req-1"), log_context(run_id="run-9"):
        assert current_context() == {"request_id": "req-1", "run_id": "run-9"}
        log.info("case finished", extra={"case_id": "c1", "latency_ms": 12})
    log.warning("outside")
    first, second = lines(stream)
    assert first["msg"] == "case finished"
    assert first["request_id"] == "req-1"
    assert first["run_id"] == "run-9"
    assert first["case_id"] == "c1"
    assert first["latency_ms"] == 12
    assert first["level"] == "INFO"
    assert "request_id" not in second
    assert current_context() == {}


def test_exceptions_are_included(stream: io.StringIO) -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("t").exception("failed")
    (entry,) = lines(stream)
    assert "ValueError: boom" in str(entry["exc"])


def test_credentials_never_reach_the_log(stream: io.StringIO) -> None:
    scrubber.register("hunter2-the-registered-secret")
    log = logging.getLogger("t")
    log.error("key sk-ant-api03-abcdefghijklmnop leaked")
    log.error("header Authorization: Bearer abcdefghijklmnopqrstuvwxyz")
    log.error("value hunter2-the-registered-secret in text")
    log.error("payload", extra={"body": '{"api_key": "abcdef123456"}'})
    text = stream.getvalue()
    assert "sk-ant-api03" not in text
    assert "abcdefghijklmnopqrstuvwxyz" not in text
    assert "hunter2-the-registered-secret" not in text
    assert "abcdef123456" not in text
    assert text.count("[REDACTED]") == 4


def test_scrubber_prefers_longest_secret_and_ignores_tiny_ones() -> None:
    s = Scrubber()
    s.register("abcdefgh", "abcdefghijkl", "abc")
    assert s.scrub("x abcdefghijkl y abc") == "x [REDACTED] y abc"


def test_noisy_loggers_are_quieted() -> None:
    configure_logging("DEBUG", io.StringIO())
    assert logging.getLogger("httpx2").level == logging.WARNING
