"""Fake retrievers, a fake generator and a fake grader for the RAG demo.

One server with no dependencies beyond the standard library, standing in for
every remote part of a RAG pipeline so that ``examples/rag/demo.sh`` runs
without keys or network:

    python examples/rag/server.py --port 8766

``POST /search/baseline`` and ``POST /search/regressed`` are two retrievers
over ``corpus.jsonl``, in the request and response shape of a ``retrieval``
endpoint: ``{"query", "k", "filters"}`` in, ``{"results": [{"id", "score",
"text"}]}`` out. For the twelve queries of ``retrieval.jsonl`` they return the
rankings recorded in ``rankings.jsonl``; any other query is ranked by the
content words it shares with each document.

``POST /v1/chat/completions`` speaks the OpenAI chat completions API, so an
``openai_compat`` provider in ``EVALKIT_PROVIDERS_FILE`` reaches it. What it
does depends on the system prompt it is sent:

* evalkit's claim extraction: every sentence of the answer with at least two
  content words is a claim, except remarks about the answerer itself
  (sentences starting with "I "); "Yes." and "No." are not claims either;
* evalkit's support check: a claim is supported by a source that has all of
  its numbers and at least 80% of its other content words, contradicted by a
  source that has 60% of those words but other numbers, and unsupported
  otherwise;
* evalkit's context relevance check: a source is relevant when it shares two
  content words with the question;
* anything else is the generator, an extractive reader: it reads the question
  from ``<question>`` and the numbered documents from ``<documents>`` and
  answers with the text of the document sharing the most content words with
  the question (the earlier one on a tie), or says it cannot answer from them
  when none shares any.

Every answer is a pure function of the request, so reruns are identical, and
usage is reported as a quarter of the characters in and out.

``GET /healthz`` answers ``ok``; ``GET /stats`` counts requests per route.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import threading
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
MAX_BODY = 4 * 1024 * 1024
RETRIEVERS = ("baseline", "regressed")
NO_ANSWER = "I cannot answer that from these documents."

# The first words of evalkit's judge system prompts (evalkit.scorers.faithfulness
# and evalkit.scorers.rag); the test of this server checks they still match.
EXTRACT = "You split an answer into the factual claims it makes."
VERIFY = "You check claims against source documents."
RELEVANCE = "You decide whether retrieved documents are relevant to a question."

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "get",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "my",
        "of",
        "on",
        "or",
        "so",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "this",
        "to",
        "under",
        "up",
        "via",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
        "yes",
        "no",
        "not",
    ]
)
WORD = re.compile(r"[a-z0-9]+")
NUMBER = re.compile(r"^\d+$")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
SOURCE = re.compile(r'<source id="([^"]*)">\n(.*?)\n</source>', re.DOTALL)
CLAIM = re.compile(r"^\[(\d+)\] (.*)$", re.MULTILINE)
DOCUMENT = re.compile(r"^\[(\d+)\] (\S+)\n(.*?)(?=\n\n\[\d+\] |\Z)", re.DOTALL | re.MULTILINE)
TAGGED = re.compile(r"<(question|answer|documents|sources|claims)>\n?(.*?)\n?</\1>", re.DOTALL)


class BadRequestError(Exception):
    """A request this server cannot answer; the client gets 400."""


def words(text: str) -> list[str]:
    """Content words, lower case, with a plural ``s`` taken off."""
    out = []
    for found in WORD.findall(text.lower()):
        if found in STOPWORDS:
            continue
        plural = len(found) > 3 and found.endswith("s") and not found.endswith("ss")
        out.append(found[:-1] if plural else found)
    return out


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text.strip()) if s.strip()]


def _tagged(prompt: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in TAGGED.finditer(prompt)}


def _jsonl(name: str) -> list[dict[str, Any]]:
    lines = (HERE / name).read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ---------------------------------------------------------------- the judge


def extract_claims(answer: str) -> list[str]:
    return [s for s in sentences(answer) if len(words(s)) >= 2 and not s.startswith("I ")]


def verdict(claim: str, sources: list[tuple[str, str]]) -> tuple[str, list[str]]:
    """``supported``, ``contradicted`` or ``unsupported``, and the deciding sources."""
    tokens = words(claim)
    numbers = {t for t in tokens if NUMBER.match(t)}
    others = [t for t in tokens if not NUMBER.match(t)]
    supporting, contradicting = [], []
    for source_id, text in sources:
        have = set(words(text))
        share = sum(t in have for t in others) / len(others) if others else 1.0
        if numbers <= have and share >= 0.8:
            supporting.append(source_id)
        elif share >= 0.6 and numbers and any(NUMBER.match(t) for t in have - numbers):
            contradicting.append(source_id)
    if supporting:
        return "supported", supporting
    if contradicting:
        return "contradicted", contradicting
    return "unsupported", []


def verify(prompt: str) -> dict[str, Any]:
    parts = _tagged(prompt)
    sources = [(html.unescape(i), t) for i, t in SOURCE.findall(parts.get("sources", ""))]
    claims = CLAIM.findall(parts.get("claims", ""))
    if not claims:
        raise BadRequestError("the prompt lists no claims")
    out = []
    for number, claim in claims:
        decided, ids = verdict(claim, sources)
        out.append(
            {
                "claim": int(number),
                "reasoning": f"{decided} by word overlap",
                "verdict": decided,
                "sources": ids,
            }
        )
    return {"verdicts": out}


def relevance(prompt: str) -> dict[str, Any]:
    parts = _tagged(prompt)
    question = set(words(parts.get("question", "")))
    found = re.findall(r'<source number="(\d+)"[^>]*>\n(.*?)\n</source>', prompt, re.DOTALL)
    if not found:
        raise BadRequestError("the prompt lists no sources")
    return {
        "sources": [
            {
                "source": int(number),
                "reasoning": "shared words",
                "relevant": len(question & set(words(text))) >= 2,
            }
            for number, text in found
        ]
    }


# ---------------------------------------------------------------- the generator


def read(prompt: str) -> str:
    """The extractive reader's answer to a generation prompt."""
    parts = _tagged(prompt)
    if "question" not in parts or "documents" not in parts:
        raise BadRequestError("a generation prompt needs <question> and <documents>")
    question = set(words(parts["question"]))
    best, best_overlap = NO_ANSWER, 0
    for _, _, text in DOCUMENT.findall(parts["documents"]):
        overlap = len(question & set(words(text)))
        if overlap > best_overlap:
            best, best_overlap = text.strip(), overlap
    return best


# ---------------------------------------------------------------- the retrievers


class Retrievers:
    def __init__(self) -> None:
        self.corpus = {d["id"]: d["text"] for d in _jsonl("corpus.jsonl")}
        queries = {c["id"]: c["input"] for c in _jsonl("retrieval.jsonl")}
        self.recorded = {
            name: {queries[r["id"]]: r[name] for r in _jsonl("rankings.jsonl")}
            for name in RETRIEVERS
        }

    def search(self, name: str, body: Any) -> dict[str, Any]:
        if not isinstance(body, dict) or not isinstance(body.get("query"), str):
            raise BadRequestError("the body must be an object with a query")
        k = body.get("k", 10)
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 1000:
            raise BadRequestError("k must be an integer from 1 to 1000")
        ids = self.recorded[name].get(body["query"])
        if ids is None:
            question = set(words(body["query"]))
            scored = [(len(question & set(words(t))), i) for i, t in self.corpus.items()]
            ids = [i for n, i in sorted(scored, key=lambda s: (-s[0], s[1])) if n > 0]
        top = ids[:k]
        return {
            "results": [
                {"id": i, "score": round(1 / rank, 4), "text": self.corpus[i]}
                for rank, i in enumerate(top, start=1)
            ]
        }


# ---------------------------------------------------------------- HTTP


def chat(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict) or not isinstance(body.get("model"), str):
        raise BadRequestError("the body must be an object with a model")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise BadRequestError("messages must be a non-empty list")
    system = "".join(
        m.get("content", "") for m in messages if isinstance(m, dict) and m.get("role") == "system"
    )
    user = [m.get("content") for m in messages if isinstance(m, dict) and m.get("role") == "user"]
    if not user or not isinstance(user[-1], str):
        raise BadRequestError("the last user message must be text")
    prompt = user[-1]
    structured = body.get("response_format") is not None
    if system.startswith(EXTRACT):
        claims = extract_claims(_tagged(prompt).get("answer", ""))
        content = json.dumps({"claims": claims})
    elif system.startswith(VERIFY):
        content = json.dumps(verify(prompt))
    elif system.startswith(RELEVANCE):
        content = json.dumps(relevance(prompt))
    elif structured:
        raise BadRequestError("this fake grades claims and relevance only")
    else:
        content = read(prompt)
    seen = system + "".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "model": body["model"],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": max(1, len(seen) // 4),
            "completion_tokens": max(1, len(content) // 4),
        },
    }


class Service:
    def __init__(self) -> None:
        self.retrievers = Retrievers()
        self.counts: Counter[str] = Counter()
        self.lock = threading.Lock()

    def handle(self, path: str, raw: bytes) -> tuple[int, bytes]:
        try:
            body = json.loads(raw)
        except ValueError:
            return HTTPStatus.BAD_REQUEST, _error("the body is not JSON")
        with self.lock:
            self.counts[path] += 1
        try:
            if path == "/v1/chat/completions":
                answer = chat(body)
            elif path.startswith("/search/") and path[8:] in RETRIEVERS:
                answer = self.retrievers.search(path[8:], body)
            else:
                return HTTPStatus.NOT_FOUND, _error("not found")
        except BadRequestError as exc:
            return HTTPStatus.BAD_REQUEST, _error(str(exc))
        return HTTPStatus.OK, json.dumps(answer).encode()


def _error(message: str) -> bytes:
    return json.dumps({"error": {"message": message, "type": "invalid_request_error"}}).encode()


def make_server(host: str, port: int, service: Service) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._send(HTTPStatus.OK, b"ok", "text/plain")
            elif self.path == "/stats":
                with service.lock:
                    stats = dict(service.counts)
                self._send(HTTPStatus.OK, json.dumps(stats).encode())
            else:
                self._send(HTTPStatus.NOT_FOUND, _error("not found"))

        def do_POST(self) -> None:
            length = self.headers.get("Content-Length", "")
            if not length.isdigit() or int(length) > MAX_BODY:
                # The body is not read, so close rather than parse it as a request.
                self.close_connection = True
                self._send(HTTPStatus.BAD_REQUEST, _error("bad Content-Length"), close=True)
                return
            status, payload = service.handle(self.path, self.rfile.read(int(length)))
            self._send(status, payload)

        def _send(
            self, status: int, payload: bytes, kind: str = "application/json", close: bool = False
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(payload)))
            if close:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766, help="0 picks a free port")
    parser.add_argument("--port-file", help="write the port here once listening")
    args = parser.parse_args(argv)
    server = make_server(args.host, args.port, Service())
    port = server.server_address[1]
    if args.port_file:
        tmp = Path(args.port_file + ".tmp")
        tmp.write_text(str(port))
        tmp.replace(args.port_file)
    print(f"RAG fakes listening on http://{args.host}:{port}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
