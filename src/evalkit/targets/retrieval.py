"""A retriever behind an HTTP endpoint, and the retrieval case schema.

A retrieval case asks one query and labels documents by how relevant they are:

    {"id": "q1", "input": "how do I reset my password",
     "expected": {"relevance": {"kb-12": 3, "kb-40": 1, "kb-7": 0}}}

``input`` is the query, or ``{"query": ..., "filters": {...}}`` when the
retriever takes filters. ``expected.relevance`` maps document ids to graded
labels from 0 (judged and not relevant) to ``MAX_GRADE``; a document with no
label counts as not relevant. A list of ids is shorthand for grade 1 each.
The ranking scorers (``evalkit.scorers.ranking``) read these labels.

Requests go through ``HttpTarget.post``, so the endpoint is named in operator
config and every call has the same protections as an HTTP target: checked and
pinned addresses, no redirects, a capped response. The request is
``{"query", "k", "filters"}``; the answer's ``results_field`` is a list of ids
or of objects with an id and, optionally, a score and a text. Ids are kept in
the order given, a repeated id keeps its first (best) rank, and the list is
cut to ``k``. The case's output is the ranking as canonical JSON
(``Ranking.to_output``).

Retrieval results are cached on their own, keyed by the retriever (endpoint,
``k`` and the fields read) and the request, in the response cache table the
moment they arrive. Anything that wraps a retriever with the same settings
(a retrieval-then-generation pipeline, a rerun with other scorers) reuses them,
and identical queries running at the same time make one request between them.
Document texts are redacted before they are cached, as every stored output is.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from typing import Annotated, Any

import httpx2
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from evalkit.datasets import Case, canonical_json, sha256_hex
from evalkit.errors import InvalidInputError
from evalkit.ratelimit import RateLimiter
from evalkit.redaction import Redactor
from evalkit.targets.base import Generation, Target, TargetError, Usage
from evalkit.targets.conversation import TurnStore, call_limited
from evalkit.targets.http import HttpEndpoint, HttpTarget, Resolver, system_resolver
from evalkit.targets.specs import HttpTargetSpec, RetrieverSpec

MAX_GRADE = 10
MAX_LABELS = 10_000
MAX_ID_CHARS = 512
MAX_TEXT_CHARS = 100_000
MAX_QUERY_CHARS = 10_000

DocId = Annotated[str, Field(min_length=1, max_length=MAX_ID_CHARS)]


class RetrievalQuery(BaseModel):
    """The input of a retrieval case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    filters: dict[str, JsonValue] = Field(default_factory=dict)


class Document(BaseModel):
    """One retrieved document: its id, and the retriever's score and text if given."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: DocId
    score: float | None = None
    text: str | None = Field(default=None, max_length=MAX_TEXT_CHARS)


class Ranking(BaseModel):
    """What a retriever returned for one query, best first."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    documents: tuple[Document, ...] = ()

    @property
    def ids(self) -> list[str]:
        return [d.id for d in self.documents]

    def to_output(self) -> str:
        return canonical_json(self.model_dump(mode="json", exclude_none=True))


def query_of(case: Case) -> RetrievalQuery:
    """The query a case's input asks, or InvalidInputError naming the case."""
    raw: Any = {"query": case.input} if isinstance(case.input, str) else case.input
    try:
        return RetrievalQuery.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "input"
        raise InvalidInputError(
            f"case {case.id!r} is not a retrieval query: {where}: {first['msg']}"
        ) from None


def parse_labels(value: Any) -> dict[str, int]:
    """Graded relevance labels from a case: ``{id: grade}`` or a list of ids
    (grade 1 each). Raises ValueError saying what is wrong."""
    if isinstance(value, list):
        value = dict.fromkeys(value, 1) if all(isinstance(v, str) for v in value) else None
    if not isinstance(value, dict):
        raise ValueError(  # noqa: TRY004 - a bad label in a dataset, not a programming error
            "relevance labels must be an object of document id to grade"
        )
    if len(value) > MAX_LABELS:
        raise ValueError(f"more than {MAX_LABELS} relevance labels")
    labels: dict[str, int] = {}
    for doc, grade in value.items():
        if not doc or len(doc) > MAX_ID_CHARS:
            raise ValueError(f"document ids must be 1 to {MAX_ID_CHARS} characters")
        if isinstance(grade, bool) or not isinstance(grade, int) or not 0 <= grade <= MAX_GRADE:
            raise ValueError(f"grade of {doc!r} must be an integer from 0 to {MAX_GRADE}")
        labels[doc] = grade
    return labels


def parse_ranking(output: str) -> Ranking:
    """The ranking a retrieval target wrote as its output. Raises ValueError."""
    try:
        return Ranking.model_validate_json(output)
    except ValidationError as exc:
        raise ValueError(f"output is not a ranking: {exc.errors()[0]['msg']}") from None


def retrieval_key(retriever_hash: str, request: Any, fingerprint: str) -> str:
    """The cache key of one retrieval: which retriever, asked what, redacted how."""
    return sha256_hex(canonical_json(["retrieval", retriever_hash, request, fingerprint]))


class RetrievalTarget(Target):
    caches_cases = False

    def __init__(
        self,
        spec: RetrieverSpec,
        endpoint: HttpEndpoint,
        *,
        resolver: Resolver = system_resolver,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self.spec = spec
        self.http = HttpTarget(
            HttpTargetSpec(type="http", endpoint=spec.endpoint),
            endpoint,
            resolver=resolver,
            transport=transport,
        )
        # Requests share rate limits and cool-downs with HTTP targets on the endpoint.
        self.provider = self.http.provider
        self.retriever_hash = sha256_hex(
            canonical_json(
                {
                    "retriever": spec.retriever_part(),
                    "endpoint": endpoint.fingerprint(),
                }
            )
        )
        self.store: TurnStore | None = None
        self.redactor = Redactor()
        self.fingerprint = self.redactor.config.fingerprint()
        self.limiter: RateLimiter | None = None
        self.max_retry_after = 120.0
        self._inflight: dict[str, asyncio.Future[Ranking]] = {}
        # Requests actually sent, for tests and logs.
        self.requests = 0

    def attach(
        self,
        *,
        store: TurnStore | None,
        redactor: Redactor,
        limiter: RateLimiter | None = None,
        max_retry_after: float = 120.0,
    ) -> None:
        """What the run provides: the retrieval cache (None when the run does not
        use the cache), its redaction and the endpoint's limiter, which only
        requests that are actually sent wait on."""
        self.store = store
        self.redactor = redactor
        self.fingerprint = redactor.config.fingerprint()
        self.limiter = limiter
        self.max_retry_after = max_retry_after

    async def generate(self, case: Case) -> Generation:
        ranking, cached = await self.retrieve(query_of(case))
        return Generation(text=ranking.to_output(), cached=cached)

    async def retrieve(self, query: RetrievalQuery) -> tuple[Ranking, bool]:
        """The ranking for ``query`` and whether this call paid nothing for it:
        from the cache, or from an identical request already on its way."""
        request = {"query": query.query, "k": self.spec.k, "filters": query.filters}
        key = retrieval_key(self.retriever_hash, request, self.fingerprint)
        while (pending := self._inflight.get(key)) is not None:
            try:
                return await asyncio.shield(pending), True
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if not pending.cancelled() or (task is not None and task.cancelling()):
                    raise
                # The request we waited on was cancelled with its case; send our own.
        future: asyncio.Future[Ranking] = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            ranking, cached = await self._fetch(key, request)
        except asyncio.CancelledError:
            future.cancel()
            raise
        except Exception as exc:
            future.set_exception(exc)
            # Waiters see the failure; when there are none, it is ours alone to raise.
            future.exception()
            raise
        else:
            future.set_result(ranking)
            return ranking, cached
        finally:
            del self._inflight[key]

    async def _fetch(self, key: str, request: dict[str, Any]) -> tuple[Ranking, bool]:
        hit = await self.store.get(key) if self.store is not None else None
        if hit is not None:
            try:
                return parse_ranking(hit.output), True
            except ValueError:
                pass  # not an entry this version wrote; ask again
        self.requests += 1
        document = await call_limited(
            self.limiter, self.max_retry_after, lambda: self.http.post(request)
        )
        ranking, failed = self._redact(self.read(document))
        # A redaction that timed out left a placeholder: keep it out of the cache.
        if self.store is not None and not failed:
            await self.store.put(key, self.retriever_hash, ranking.to_output(), None, Usage())
        return ranking, False

    def _redact(self, ranking: Ranking) -> tuple[Ranking, bool]:
        """The ranking with each text redacted, and whether any redaction failed."""
        documents = []
        failed = False
        for doc in ranking.documents:
            if doc.text is None:
                documents.append(doc)
                continue
            redacted = self.redactor.redact(doc.text)
            failed = failed or redacted.failed
            documents.append(doc.model_copy(update={"text": redacted.text}))
        return Ranking(documents=tuple(documents)), failed

    def read(self, document: Any) -> Ranking:
        """The top ``k`` of an answer, best first, each id once."""
        spec = self.spec
        value: Any = document
        for part in spec.results_field.split("."):
            if not isinstance(value, Mapping) or part not in value:
                raise TargetError(
                    f"retriever answer has no field {spec.results_field!r}", retryable=False
                )
            value = value[part]
        if not isinstance(value, list):
            raise TargetError(f"{spec.results_field!r} must be a list", retryable=False)
        seen: set[str] = set()
        documents: list[Document] = []
        for position, item in enumerate(value):
            doc = self._document(position, item)
            if doc.id in seen:
                continue
            seen.add(doc.id)
            documents.append(doc)
            if len(documents) == spec.k:
                break
        return Ranking(documents=tuple(documents))

    def _document(self, position: int, item: Any) -> Document:
        spec = self.spec
        where = f"{spec.results_field}[{position}]"
        if isinstance(item, str):
            item = {spec.id_field: item}
        if not isinstance(item, Mapping):
            raise TargetError(f"{where} must be an id or an object", retryable=False)
        doc_id = item.get(spec.id_field)
        if isinstance(doc_id, int) and not isinstance(doc_id, bool):
            doc_id = str(doc_id)
        if not isinstance(doc_id, str) or not 1 <= len(doc_id) <= MAX_ID_CHARS:
            raise TargetError(
                f"{where}.{spec.id_field} must be an id of 1 to {MAX_ID_CHARS} characters",
                retryable=False,
            )
        score = item.get(spec.score_field)
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, int | float)
            or not math.isfinite(score)
        ):
            raise TargetError(f"{where}.{spec.score_field} must be a number", retryable=False)
        text = item.get(spec.text_field)
        if text is not None and not isinstance(text, str):
            raise TargetError(f"{where}.{spec.text_field} must be a string", retryable=False)
        if text is not None and len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS]
        return Document(id=doc_id, score=None if score is None else float(score), text=text)

    async def aclose(self) -> None:
        await self.http.aclose()
