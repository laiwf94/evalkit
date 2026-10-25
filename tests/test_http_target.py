import asyncio
import ipaddress
import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
from pydantic import ValidationError

from evalkit.config import ConfigError, Settings
from evalkit.datasets import Case
from evalkit.targets import HttpTargetSpec, TargetError
from evalkit.targets.http import HttpEndpoint, HttpTarget, is_public, system_resolver

CASE = Case(id="c1", input={"question": "hi"}, expected="hello")


class Recorder:
    def __init__(self, *responses: httpx2.Response) -> None:
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.responses.pop(0)


def resolver_for(*answers: list[str]) -> Any:
    queue = list(answers)
    calls: list[tuple[str, int]] = []

    async def resolve(host: str, port: int) -> list[str]:
        calls.append((host, port))
        return queue.pop(0)

    resolve.calls = calls  # type: ignore[attr-defined]
    return resolve


def target(
    recorder: Recorder,
    resolver: Any,
    url: str = "https://bot.example.com/api/answer?v=2",
    field: str = "output",
    **endpoint: Any,
) -> HttpTarget:
    return HttpTarget(
        HttpTargetSpec(type="http", endpoint="bot", response_field=field),
        HttpEndpoint(url=url, **endpoint),
        resolver=resolver,
        transport=httpx2.MockTransport(recorder),
    )


def run(t: HttpTarget) -> str:
    async def main() -> str:
        try:
            return (await t.generate(CASE)).text
        finally:
            await t.aclose()

    return asyncio.run(main())


def test_request_is_pinned_to_the_checked_address() -> None:
    rec = Recorder(httpx2.Response(200, json={"output": "hello"}))
    resolve = resolver_for(["93.184.216.34"])
    t = target(
        rec,
        resolve,
        headers={"X-Team": "evals"},
        secret_headers={"Authorization": "Bearer s3cret-value"},
    )
    assert run(t) == "hello"
    (req,) = rec.requests
    assert str(req.url) == "https://93.184.216.34/api/answer?v=2"
    assert req.headers["host"] == "bot.example.com"
    assert req.extensions["sni_hostname"] == "bot.example.com"
    assert req.headers["x-team"] == "evals"
    assert req.headers["authorization"] == "Bearer s3cret-value"
    assert json.loads(req.content) == {"input": {"question": "hi"}}
    assert resolve.calls == [("bot.example.com", 443)]
    assert t.provider == "http:bot"


def test_custom_port_ipv6_and_plain_http() -> None:
    rec = Recorder(httpx2.Response(200, json={"output": "x"}))
    t = target(
        rec,
        resolver_for(["2606:2800:220:1:248:1893:25c8:1946"]),
        url="http://bot.example.com:8080/a",
    )
    run(t)
    req = rec.requests[0]
    assert str(req.url) == "http://[2606:2800:220:1:248:1893:25c8:1946]:8080/a"
    assert req.headers["host"] == "bot.example.com:8080"
    assert "sni_hostname" not in req.extensions


@pytest.mark.parametrize(
    "address",
    [
        "169.254.169.254",
        "127.0.0.1",
        "10.1.2.3",
        "172.16.0.1",
        "192.168.1.1",
        "100.64.0.1",
        "0.0.0.0",
        "224.0.0.1",
        "::1",
        "fe80::1",
        "fc00::1",
        "::ffff:127.0.0.1",
        "::ffff:169.254.169.254",
        "2002:7f00:1::",  # 6to4 wrapping 127.0.0.1
    ],
)
def test_non_public_addresses_are_refused(address: str) -> None:
    rec = Recorder()
    with pytest.raises(TargetError, match="non-public address") as info:
        run(target(rec, resolver_for([address])))
    assert not info.value.retryable
    assert rec.requests == []


def test_one_bad_address_among_good_ones_is_refused() -> None:
    with pytest.raises(TargetError, match=r"10\.0\.0\.1"):
        run(target(Recorder(), resolver_for(["93.184.216.34", "10.0.0.1"])))


def test_rebinding_after_the_check_cannot_redirect() -> None:
    # First call: public answer, request goes to that literal address.
    # Second call: the name now points inside; the check refuses it.
    rec = Recorder(httpx2.Response(200, json={"output": "ok"}))
    resolve = resolver_for(["93.184.216.34"], ["10.0.0.5"])
    t = target(rec, resolve)

    async def main() -> None:
        try:
            assert (await t.generate(CASE)).text == "ok"
            with pytest.raises(TargetError, match="non-public"):
                await t.generate(CASE)
        finally:
            await t.aclose()

    asyncio.run(main())
    assert [r.url.host for r in rec.requests] == ["93.184.216.34"]


def test_allow_private_and_literal_ip_urls() -> None:
    rec = Recorder(httpx2.Response(200, json={"output": "local"}))
    t = target(rec, resolver_for(), url="http://127.0.0.1:9000/x", allow_private=True)
    assert run(t) == "local"
    assert str(rec.requests[0].url) == "http://127.0.0.1:9000/x"
    with pytest.raises(TargetError, match="non-public"):
        run(target(Recorder(), resolver_for(), url="http://127.0.0.1:9000/x"))


def test_resolution_failures_are_retryable() -> None:
    async def broken(host: str, port: int) -> list[str]:
        raise OSError("no such host")

    with pytest.raises(TargetError, match="cannot resolve") as info:
        run(target(Recorder(), broken))
    assert info.value.retryable
    with pytest.raises(TargetError, match="no addresses"):
        run(target(Recorder(), resolver_for([])))


@pytest.mark.parametrize(
    ("response", "message", "retryable", "retry_after"),
    [
        (
            httpx2.Response(302, headers={"location": "http://169.254.169.254/"}),
            "redirect",
            False,
            None,
        ),
        (httpx2.Response(429, headers={"retry-after": "3"}), "returned 429", True, 3.0),
        (httpx2.Response(503), "returned 503", True, None),
        (httpx2.Response(408), "returned 408", True, None),
        (httpx2.Response(404), "returned 404", False, None),
        (httpx2.Response(200, content=b"<html>"), "did not return JSON", False, None),
        (httpx2.Response(200, content=b"[" * 200_000), "did not return JSON", False, None),
        (httpx2.Response(200, json={"answer": "x"}), "no field 'output'", False, None),
        (httpx2.Response(200, json=["output"]), "no field 'output'", False, None),
    ],
)
def test_response_errors(
    response: httpx2.Response, message: str, retryable: bool, retry_after: float | None
) -> None:
    with pytest.raises(TargetError, match=message) as info:
        run(target(Recorder(response), resolver_for(["93.184.216.34"])))
    assert info.value.retryable is retryable
    assert info.value.retry_after == retry_after


def test_nested_field_and_non_string_values() -> None:
    rec = Recorder(
        httpx2.Response(200, json={"data": {"answer": "deep"}}),
        httpx2.Response(200, json={"data": {"answer": {"n": 4}}}),
    )
    resolve = resolver_for(["93.184.216.34"], ["93.184.216.34"])
    t = target(rec, resolve, field="data.answer")

    async def main() -> list[str]:
        try:
            return [(await t.generate(CASE)).text for _ in range(2)]
        finally:
            await t.aclose()

    assert asyncio.run(main()) == ["deep", '{"n": 4}']


def test_response_size_is_capped_before_buffering() -> None:
    big = json.dumps({"output": "x" * 5000}).encode()
    declared = httpx2.Response(200, content=big)
    with pytest.raises(TargetError, match="larger than 1024 bytes"):
        run(target(Recorder(declared), resolver_for(["93.184.216.34"]), max_response_bytes=1024))

    async def chunks() -> Any:
        for _ in range(10):
            yield b"x" * 500

    streamed = httpx2.Response(200, content=chunks())
    with pytest.raises(TargetError, match="larger than 1024 bytes"):
        run(target(Recorder(streamed), resolver_for(["93.184.216.34"]), max_response_bytes=1024))


def test_transport_failures_are_retryable() -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused")

    def slow(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("slow")

    async def broken_body() -> Any:
        yield b'{"out'
        raise httpx2.ReadError("reset")

    for handler, message in ((refuse, "cannot reach"), (slow, "timed out")):
        t = HttpTarget(
            HttpTargetSpec(type="http", endpoint="bot"),
            HttpEndpoint(url="https://bot.example.com/"),
            resolver=resolver_for(["93.184.216.34"]),
            transport=httpx2.MockTransport(handler),
        )
        with pytest.raises(TargetError, match=message) as info:
            run(t)
        assert info.value.retryable
    with pytest.raises(TargetError, match="cut off") as info:
        run(
            target(
                Recorder(httpx2.Response(200, content=broken_body())),
                resolver_for(["93.184.216.34"]),
            )
        )
    assert info.value.retryable


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"url": "ftp://x/"}, "http or https"),
        ({"url": "https:///nohost"}, "must have a host"),
        ({"url": "https://user:pw@x/"}, "secret_headers"),
        ({"url": "https://x/#frag"}, "fragment"),
        ({"url": "https://x/", "headers": {"Host": "evil"}}, "not allowed"),
        ({"url": "https://x/", "headers": {"Bad Header": "v"}}, "not allowed"),
    ],
)
def test_endpoint_validation(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        HttpEndpoint(**kwargs)


def test_fingerprint_hides_secret_values() -> None:
    ep = HttpEndpoint(url="https://x/", secret_headers={"Authorization": "Bearer abc"})
    assert ep.fingerprint() == {
        "url": "https://x/",
        "headers": {},
        "secret_headers": ["Authorization"],
    }


def test_is_public() -> None:
    assert is_public(ipaddress.ip_address("8.8.8.8"))
    assert is_public(ipaddress.ip_address("2001:4860:4860::8888"))
    assert not is_public(ipaddress.ip_address("2001:0:4136:e378:8000:63bf:3fff:fdd2"))  # Teredo


def test_system_resolver_handles_localhost() -> None:
    assert "127.0.0.1" in asyncio.run(system_resolver("localhost", 80)) or "::1" in asyncio.run(
        system_resolver("localhost", 80)
    )


def test_endpoints_file(tmp_path: Path) -> None:
    f = tmp_path / "endpoints.toml"
    f.write_text(
        '[endpoints.support-bot]\nurl = "https://bot.example.com/answer"\n'
        'headers = { X-Team = "evals" }\n'
        'secret_headers = { Authorization = "BOT_TOKEN" }\n'
    )
    s = Settings.from_env({"EVALKIT_HTTP_TARGETS_FILE": str(f), "BOT_TOKEN": "Bearer tok-123456"})
    ep = s.http_endpoints["support-bot"]
    assert ep.secret_headers["Authorization"].get_secret_value() == "Bearer tok-123456"
    assert "Bearer tok-123456" in s.secret_values()
    assert "tok-123456" not in repr(s)


@pytest.mark.parametrize(
    ("content", "env", "message"),
    [
        ('[endpoints.BAD]\nurl = "https://x/"\n', {}, "endpoint name"),
        ("[endpoints]\nbot = 3\n", {}, "must be a table"),
        ('[endpoints.bot]\nurl = "https://x/"\nsecret_headers = 3\n', {}, "map header names"),
        (
            '[endpoints.bot]\nurl = "https://x/"\nsecret_headers = { A = "MISSING" }\n',
            {},
            "needs MISSING",
        ),
        ('[endpoints.bot]\nurl = "ftp://x/"\n', {}, "invalid configuration"),
    ],
)
def test_bad_endpoints_file(
    tmp_path: Path, content: str, env: dict[str, str], message: str
) -> None:
    f = tmp_path / "e.toml"
    f.write_text(content)
    with pytest.raises(ConfigError, match=message):
        Settings.from_env({"EVALKIT_HTTP_TARGETS_FILE": str(f), **env})
