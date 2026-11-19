"""Embeddings for scorers: batched provider calls and a float32 cache table.

``Embedder.embed(texts)`` returns one float32 vector per text. Three things
keep that cheap:

* Batching. Texts that miss the cache join a queue shared by every case of the
  run; the queue is sent as one provider request when it holds ``batch_size``
  texts, or ``linger`` seconds after its first text arrived. Cases that finish
  together therefore share requests, and ``evalkit calibrate`` embeds a whole
  labelled file in ``ceil(texts / batch_size)`` requests. At most
  ``max_requests`` are in flight at once, under the provider's limiter and the
  retry policy.
* Deduplication. A text already on its way (the same reference answer in two
  cases) is awaited, not sent again; the case that sent it pays for it.
* The cache table (``embedding_cache``). Vectors are stored as little-endian
  float32 bytes keyed by SHA-256 over the embedder's fingerprint (provider,
  endpoint, model) and the text. A fresh vector is rounded to float32 before
  it is used, so a rerun from the cache scores bit for bit like the first run.

A batch's usage is shared among its texts in proportion to their length, so
each case is charged for its own texts and the shares add up to what the
provider billed.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session, sessionmaker

from evalkit.cache import TOUCH_INTERVAL
from evalkit.datasets import canonical_json, sha256_hex
from evalkit.db.models import CachedEmbedding, utcnow
from evalkit.providers.base import Embeddings, ErrorKind, Provider, ProviderError
from evalkit.ratelimit import Limiters, RateLimiter, RetryPolicy, RetryStats, call_with_retries
from evalkit.targets.base import Usage

log = logging.getLogger(__name__)

Vector = npt.NDArray[np.float32]

DEFAULT_BATCH = 64
MAX_BATCH = 2048  # the most inputs OpenAI's embeddings endpoint takes in one request
# Longer texts are cut before they are embedded (and keyed); about 5,000 tokens
# of English, inside every provider's input limit.
MAX_TEXT_CHARS = 20_000
_LOOKUP_CHUNK = 500
_DTYPE = np.dtype("<f4")


def clip(text: str) -> str:
    return text[:MAX_TEXT_CHARS]


def pack(vector: Vector) -> bytes:
    return np.asarray(vector, dtype=_DTYPE).tobytes()


def unpack(data: bytes, dims: int) -> Vector:
    """The vector stored in ``data``; ValueError when it is not ``dims`` float32 values."""
    if dims < 1 or len(data) != dims * _DTYPE.itemsize:
        raise ValueError(f"{len(data)} bytes do not hold {dims} float32 values")
    vector: Vector = np.frombuffer(data, dtype=_DTYPE).astype(np.float32)
    return vector


def cosine(a: Vector, b: Vector) -> float | None:
    """Cosine similarity in float64, or None when either vector is zero or the
    dimensions differ."""
    if a.shape != b.shape:
        return None
    x = a.astype(np.float64)
    y = b.astype(np.float64)
    norm = float(np.linalg.norm(x)) * float(np.linalg.norm(y))
    if norm == 0.0:
        return None
    return max(-1.0, min(1.0, float(np.dot(x, y)) / norm))


def embedding_key(embedder_hash: str, text: str) -> str:
    return sha256_hex(canonical_json(["embedding", embedder_hash, text]))


class VectorStore(Protocol):
    """Where vectors are kept between runs (``EmbeddingStore``)."""

    async def get(self, keys: Sequence[str]) -> dict[str, Vector]: ...

    async def put(self, embedder_hash: str, vectors: dict[str, Vector]) -> None: ...


def lookup_vectors(
    session: Session, keys: Sequence[str], *, now: datetime | None = None
) -> dict[str, Vector]:
    """The cached vectors for ``keys``, marking them used at most once per
    ``TOUCH_INTERVAL``. A row whose bytes do not match its dimensions is deleted
    and treated as a miss. The caller commits."""
    now = now or utcnow()
    found: dict[str, Vector] = {}
    broken: list[str] = []
    for start in range(0, len(keys), _LOOKUP_CHUNK):
        chunk = keys[start : start + _LOOKUP_CHUNK]
        rows = session.execute(
            select(CachedEmbedding.key, CachedEmbedding.dims, CachedEmbedding.vector).where(
                CachedEmbedding.key.in_(chunk)
            )
        )
        for key, dims, data in rows:
            try:
                found[key] = unpack(bytes(data), dims)
            except ValueError:
                log.warning("cached embedding is corrupt; dropping it", extra={"key": key})
                broken.append(key)
    hits = list(found)
    for start in range(0, len(hits), _LOOKUP_CHUNK):
        session.execute(
            update(CachedEmbedding)
            .where(
                CachedEmbedding.key.in_(hits[start : start + _LOOKUP_CHUNK]),
                CachedEmbedding.last_used_at < now - TOUCH_INTERVAL,
            )
            .values(last_used_at=now)
            .execution_options(synchronize_session=False)
        )
    if broken:
        session.execute(delete(CachedEmbedding).where(CachedEmbedding.key.in_(broken)))
    return found


def store_vectors(
    session: Session,
    embedder_hash: str,
    vectors: dict[str, Vector],
    *,
    now: datetime | None = None,
) -> None:
    """Insert vectors, keeping the existing row when two runs race on a key."""
    now = now or utcnow()
    rows = [
        {
            "key": key,
            "embedder_hash": embedder_hash,
            "dims": int(vector.shape[0]),
            "vector": pack(vector),
            "created_at": now,
            "last_used_at": now,
        }
        for key, vector in vectors.items()
    ]
    if not rows:
        return
    insert = pg_insert if session.get_bind().dialect.name == "postgresql" else sqlite_insert
    for start in range(0, len(rows), _LOOKUP_CHUNK):
        session.execute(
            insert(CachedEmbedding)
            .values(rows[start : start + _LOOKUP_CHUNK])
            .on_conflict_do_nothing(index_elements=["key"])
        )


class EmbeddingStore:
    """``VectorStore`` on the ``embedding_cache`` table, one short transaction per call."""

    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self.sessions = sessions

    async def get(self, keys: Sequence[str]) -> dict[str, Vector]:
        def load() -> dict[str, Vector]:
            with self.sessions() as session:
                found = lookup_vectors(session, keys)
                session.commit()
                return found

        return await asyncio.to_thread(load)

    async def put(self, embedder_hash: str, vectors: dict[str, Vector]) -> None:
        def save() -> None:
            with self.sessions() as session:
                store_vectors(session, embedder_hash, vectors)
                session.commit()

        await asyncio.to_thread(save)


@dataclass(frozen=True)
class Embedded:
    """``vectors`` in input order; ``usage`` is what this call paid for."""

    vectors: list[Vector]
    usage: Usage
    fresh: int  # texts this call sent to the provider
    cached: int  # texts served from the cache or from another call's request


@dataclass
class _Pending:
    key: str
    text: str
    future: asyncio.Future[tuple[Vector, Usage]] = field(repr=False)


def shares(total: int, weights: Sequence[int]) -> list[int]:
    """``total`` split in proportion to ``weights`` by largest remainder: the
    parts are whole numbers that add up to ``total``."""
    if not weights:
        return []
    weight = sum(weights)
    if weight == 0:
        raise ValueError("the weights must not all be zero")
    exact = [total * w / weight for w in weights]
    parts = [math.floor(x) for x in exact]
    by_remainder = sorted(range(len(exact)), key=lambda i: (parts[i] - exact[i], i))
    for i in by_remainder[: total - sum(parts)]:
        parts[i] += 1
    return parts


class Embedder:
    """One embedding model on one provider, shared by every case of a run."""

    def __init__(
        self,
        client: Provider,
        model: str,
        *,
        endpoint: str | None = None,
        batch_size: int = DEFAULT_BATCH,
        max_requests: int = 4,
        linger: float = 0.005,
        retry: RetryPolicy | None = None,
        limiter: RateLimiter | None = None,
        store: VectorStore | None = None,
    ) -> None:
        if not 1 <= batch_size <= MAX_BATCH:
            raise ValueError(f"batch_size must be between 1 and {MAX_BATCH}")
        # An unknown model would embed with no cost accounting; refuse it up front.
        client.prices.require(model)
        self.client = client
        self.provider = client.name
        self.model = model
        self.endpoint = endpoint
        self.batch_size = batch_size
        self.linger = linger
        self.retry = retry or RetryPolicy()
        self.limiter: RateLimiter = limiter or Limiters().get(f"{self.provider}:embed")
        self.store = store
        self._gate = asyncio.Semaphore(max_requests)
        self._queue: list[_Pending] = []
        self._inflight: dict[str, _Pending] = {}
        self._timer: asyncio.TimerHandle | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        # Provider requests sent and texts they carried, for tests and logs.
        self.requests = 0
        self.texts_sent = 0

    def identity(self) -> dict[str, Any]:
        return {"provider": self.provider, "endpoint": self.endpoint, "model": self.model}

    def fingerprint(self) -> str:
        return sha256_hex(canonical_json(["embedder", self.identity()]))

    async def embed(self, texts: Sequence[str]) -> Embedded:
        """One vector per text (each cut to ``MAX_TEXT_CHARS``). Raises
        ``ProviderError`` when a request fails after its retries."""
        clipped = [clip(t) for t in texts]
        if any(not t for t in clipped):
            raise ValueError("cannot embed an empty text")
        fingerprint = self.fingerprint()
        keys = [embedding_key(fingerprint, t) for t in clipped]
        texts_by_key = dict(zip(keys, clipped, strict=True))
        # Claim every text nobody else is fetching before looking in the cache, so
        # a case that arrives while this one reads the cache waits instead of
        # paying for the same text again.
        loop = asyncio.get_running_loop()
        owned: list[_Pending] = []
        waits: list[_Pending] = []
        for key, text in texts_by_key.items():
            pending = self._inflight.get(key)
            if pending is not None:
                waits.append(pending)
                continue
            pending = _Pending(key, text, loop.create_future())
            self._inflight[key] = pending
            owned.append(pending)
        hits: dict[str, Vector] = {}
        try:
            if self.store is not None and owned:
                hits = await self.store.get([p.key for p in owned])
        except asyncio.CancelledError:
            for pending in owned:
                self._inflight.pop(pending.key, None)
                pending.future.cancel()  # waiters fetch the text themselves
            raise
        except Exception:
            # As with a failed write: the texts are fetched, only not from the cache.
            log.warning("cannot read embeddings from the cache", exc_info=True)
        sent: list[_Pending] = []
        for pending in owned:
            vector = hits.get(pending.key)
            if vector is None:
                sent.append(pending)
                self._enqueue(pending)
            else:
                self._inflight.pop(pending.key, None)
                pending.future.set_result((vector, Usage()))
        found: dict[str, Vector] = {}
        usage = Usage()
        for pending in owned:
            found[pending.key], share = await self._wait(pending)
            usage = usage + share
        for pending in waits:
            found[pending.key], share = await self._wait(pending, waiting=True)
            usage = usage + share
        return Embedded(
            vectors=[found[k] for k in keys],
            usage=usage,
            fresh=len(sent),
            cached=len(texts_by_key) - len(sent),
        )

    async def _wait(self, pending: _Pending, *, waiting: bool = False) -> tuple[Vector, Usage]:
        """The vector ``pending`` resolves to, and what this caller pays for it
        (nothing when it waited on another caller's request). If that request
        was cancelled with its caller's case, and this caller is not being
        cancelled itself, it fetches the text on its own."""
        try:
            vector, share = await asyncio.shield(pending.future)
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if not pending.future.cancelled() or (task is not None and task.cancelling()):
                raise
            again = await self.embed([pending.text])
            return again.vectors[0], again.usage
        return vector, Usage() if waiting else share

    def _enqueue(self, pending: _Pending) -> None:
        self._queue.append(pending)
        if len(self._queue) >= self.batch_size:
            self._start(self._take())
        elif self._timer is None:
            self._timer = asyncio.get_running_loop().call_later(self.linger, self._due)

    def _due(self) -> None:
        self._timer = None
        while self._queue:
            self._start(self._take())

    def _take(self) -> list[_Pending]:
        batch = self._queue[: self.batch_size]
        del self._queue[: self.batch_size]
        return batch

    def _start(self, batch: list[_Pending]) -> None:
        task = asyncio.get_running_loop().create_task(self._send(batch))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _send(self, batch: list[_Pending]) -> None:
        texts = [p.text for p in batch]
        try:
            async with self._gate:
                self.requests += 1
                self.texts_sent += len(texts)
                result = await call_with_retries(
                    lambda: self.client.embed(model=self.model, texts=texts),
                    limiter=self.limiter,
                    policy=self.retry,
                    stats=RetryStats(),
                )
            vectors = self._vectors(result, len(texts))
            if self.store is not None:
                try:
                    await self.store.put(
                        self.fingerprint(), {p.key: v for p, v in zip(batch, vectors, strict=True)}
                    )
                except Exception:
                    # The vectors are good; only the next run pays for them again.
                    log.warning("cannot write embeddings to the cache", exc_info=True)
        except asyncio.CancelledError:
            for pending in batch:
                pending.future.cancel()
            raise
        except Exception as exc:
            for pending in batch:
                pending.future.set_exception(exc)
                pending.future.exception()  # retrieved: waiters re-raise it, or nobody cares
            return
        finally:
            for pending in batch:
                self._inflight.pop(pending.key, None)
        weights = [len(t) for t in texts]
        tokens = shares(result.usage.input_tokens, weights)
        total = sum(weights)
        for pending, vector, weight, used in zip(batch, vectors, weights, tokens, strict=True):
            cost = result.usage.cost_usd * weight / total
            pending.future.set_result((vector, Usage(input_tokens=used, cost_usd=cost)))

    def _vectors(self, result: Embeddings, count: int) -> list[Vector]:
        """The provider's vectors as float32, checked: one per text, all of one
        non-zero length, every value finite."""

        def malformed(why: str) -> ProviderError:
            return ProviderError(
                f"malformed embeddings from {self.provider}: {why}",
                kind=ErrorKind.MALFORMED_RESPONSE,
                provider=self.provider,
            )

        if len(result.vectors) != count:
            raise malformed(f"{len(result.vectors)} vectors for {count} texts")
        with np.errstate(over="ignore"):  # too large for float32: inf, refused below
            vectors = [np.asarray(v, dtype=np.float32) for v in result.vectors]
        dims = {v.shape for v in vectors}
        if len(dims) != 1 or vectors[0].ndim != 1 or vectors[0].shape[0] == 0:
            raise malformed(f"vector shapes {sorted(dims)}")
        if not all(bool(np.isfinite(v).all()) for v in vectors):
            raise malformed("a value is not a finite float32")
        return vectors
