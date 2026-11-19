"""The embedder: batching across callers, deduplication, the float32 cache table,
usage shares, error handling and the cache sweep."""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from evalkit import cache
from evalkit.config import Settings
from evalkit.db.models import CachedEmbedding
from evalkit.embeddings import (
    MAX_TEXT_CHARS,
    Embedder,
    EmbeddingStore,
    Vector,
    clip,
    cosine,
    embedding_key,
    lookup_vectors,
    pack,
    shares,
    store_vectors,
    unpack,
)
from evalkit.errors import InvalidInputError
from evalkit.providers.base import Embeddings, ErrorKind, ProviderError
from evalkit.ratelimit import RetryPolicy
from evalkit.targets.base import Usage
from tests.fake_embeddings import MODEL, FakeEmbedder, bag_of_words, overloaded, refused

PRICES = Settings.from_env({}).price_book().table("openai")
RETRY = RetryPolicy(max_attempts=3, base_delay=0)
T0 = datetime(2026, 11, 1, tzinfo=UTC)


def embedder(fake: FakeEmbedder, **kw: Any) -> Embedder:
    kw.setdefault("retry", RETRY)
    return Embedder(fake, MODEL, **kw)


def vec(*values: float) -> Vector:
    return np.asarray(values, dtype=np.float32)


def test_pack_round_trips_little_endian_float32() -> None:
    v = vec(0.1, -2.5, 3.0)
    data = pack(v)
    assert len(data) == 12
    assert data[:4] == bytes.fromhex("cdcccc3d")  # 0.1 as float32, little end first
    assert np.array_equal(unpack(data, 3), v)
    assert unpack(data, 3).dtype == np.float32
    for dims in (0, 2, 4):
        with pytest.raises(ValueError, match="do not hold"):
            unpack(data, dims)


def test_cosine() -> None:
    assert cosine(vec(1, 0), vec(1, 0)) == 1.0
    assert cosine(vec(1, 0), vec(0, 2)) == 0.0
    assert cosine(vec(1, 0), vec(-3, 0)) == -1.0
    assert cosine(vec(3, 4), vec(4, 3)) == pytest.approx(24 / 25)
    assert cosine(vec(0, 0), vec(1, 0)) is None
    assert cosine(vec(1, 0), vec(1, 0, 0)) is None


@given(st.integers(0, 10_000), st.lists(st.integers(1, 1000), min_size=1, max_size=50))
def test_shares_are_whole_and_add_up(total: int, weights: list[int]) -> None:
    parts = shares(total, weights)
    assert sum(parts) == total
    assert all(p >= 0 for p in parts)
    for part, weight in zip(parts, weights, strict=True):
        assert abs(part - total * weight / sum(weights)) < 1


def test_shares_edge_cases() -> None:
    assert shares(5, []) == []
    assert shares(10, [1, 3]) == [3, 7]
    with pytest.raises(ValueError, match="zero"):
        shares(3, [0, 0])


def test_concurrent_texts_share_batches() -> None:
    fake = FakeEmbedder(PRICES)

    async def go() -> list[Any]:
        e = embedder(fake, batch_size=4)
        results = await asyncio.gather(*(e.embed([f"text number {i}"]) for i in range(10)))
        assert (e.requests, e.texts_sent) == (3, 10)
        return results

    results = asyncio.run(go())
    assert [len(b) for b in fake.batches] == [4, 4, 2]
    for i, got in enumerate(results):
        assert np.array_equal(got.vectors[0], vec(*bag_of_words(f"text number {i}")))
        assert (got.fresh, got.cached) == (1, 0)


def test_one_call_is_split_into_batches_and_keeps_input_order() -> None:
    fake = FakeEmbedder(PRICES)
    texts = [f"t{i}" for i in range(7)]

    async def go() -> Any:
        return await embedder(fake, batch_size=3).embed(texts)

    got = asyncio.run(go())
    assert [len(b) for b in fake.batches] == [3, 3, 1]
    assert [v.tolist() for v in got.vectors] == [vec(*bag_of_words(t)).tolist() for t in texts]


def test_a_text_on_its_way_is_awaited_and_paid_for_once() -> None:
    fake = FakeEmbedder(PRICES)

    async def go() -> list[Any]:
        e = embedder(fake)
        return list(
            await asyncio.gather(
                e.embed(["the same reference", "answer a"]),
                e.embed(["the same reference", "answer b"]),
                e.embed(["the same reference", "the same reference"]),
            )
        )

    first, second, third = asyncio.run(go())
    assert sorted(fake.texts) == ["answer a", "answer b", "the same reference"]
    assert (first.fresh, first.cached) == (2, 0)
    assert (second.fresh, second.cached) == (1, 1)
    assert (third.fresh, third.cached) == (0, 1)
    assert third.usage == Usage()
    assert np.array_equal(first.vectors[0], second.vectors[0])
    # One request for 3 + 2 + 2 words; its tokens go to the two callers that sent texts.
    assert first.usage.input_tokens + second.usage.input_tokens == 7


def test_usage_is_shared_by_length_and_adds_up() -> None:
    fake = FakeEmbedder(PRICES)
    fake.reply = lambda texts: Embeddings(
        [[1.0, 0.0]] * len(texts), Usage(input_tokens=10, cost_usd=0.4), MODEL
    )

    async def go() -> list[Any]:
        e = embedder(fake)
        return list(await asyncio.gather(e.embed(["a"]), e.embed(["bbb"])))

    short, long = asyncio.run(go())
    assert len(fake.batches) == 1
    assert (short.usage.input_tokens, long.usage.input_tokens) == (3, 7)
    assert short.usage.cost_usd == pytest.approx(0.1)
    assert long.usage.cost_usd == pytest.approx(0.3)


def test_fresh_vectors_are_float32_like_cached_ones(sessions: sessionmaker[Session]) -> None:
    fake = FakeEmbedder(PRICES, vectorize=lambda t: [0.1, 1 / 3, float(len(t))])
    store = EmbeddingStore(sessions)

    async def go() -> tuple[Any, Any]:
        first = await embedder(fake, store=store).embed(["alpha", "beta"])
        again = await embedder(fake, store=store).embed(["beta", "alpha"])
        return first, again

    first, again = asyncio.run(go())
    assert len(fake.batches) == 1
    assert (again.fresh, again.cached) == (0, 2)
    assert again.usage == Usage()
    assert first.vectors[0].dtype == np.float32
    assert first.vectors[0].tobytes() == again.vectors[1].tobytes()
    assert first.vectors[1].tobytes() == again.vectors[0].tobytes()
    with sessions() as session:
        rows = session.scalars(select(CachedEmbedding)).all()
    assert {(r.dims, len(r.vector)) for r in rows} == {(3, 12)}


def test_models_and_providers_never_share_vectors(sessions: sessionmaker[Session]) -> None:
    fake = FakeEmbedder(PRICES)
    store = EmbeddingStore(sessions)

    async def go() -> None:
        await embedder(fake, store=store).embed(["hello"])
        await Embedder(fake, "text-embedding-3-large", retry=RETRY, store=store).embed(["hello"])
        await embedder(fake, store=store, endpoint="openai_compat:http://other").embed(["hello"])
        await embedder(fake, store=store).embed(["hello"])

    asyncio.run(go())
    assert len(fake.batches) == 3


def test_a_corrupt_cache_row_is_dropped_and_fetched_again(
    sessions: sessionmaker[Session], caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeEmbedder(PRICES)
    store = EmbeddingStore(sessions)
    asyncio.run(embedder(fake, store=store).embed(["hello"]))
    with sessions() as session:
        session.execute(update(CachedEmbedding).values(vector=b"\x00\x00"))
        session.commit()
    with caplog.at_level(logging.WARNING):
        got = asyncio.run(embedder(fake, store=store).embed(["hello"]))
    assert "corrupt" in caplog.text
    assert got.fresh == 1
    assert len(fake.batches) == 2
    with sessions() as session:
        assert len(session.scalars(select(CachedEmbedding)).all()) == 1


def test_lookup_marks_rows_used_at_most_once_an_hour(sessions: sessionmaker[Session]) -> None:
    with sessions() as session:
        store_vectors(session, "h", {"k": vec(1, 2)}, now=T0)
        session.commit()
        assert set(lookup_vectors(session, ["k", "missing"], now=T0 + timedelta(minutes=30)))
        session.commit()
        assert session.get(CachedEmbedding, "k").last_used_at.replace(tzinfo=UTC) == T0  # type: ignore[union-attr]
        lookup_vectors(session, ["k"], now=T0 + timedelta(hours=2))
        session.commit()
        row = session.get(CachedEmbedding, "k")
        assert row is not None
        assert row.last_used_at.replace(tzinfo=UTC) == T0 + timedelta(hours=2)
        # A second writer of the same key keeps the first row.
        store_vectors(session, "h", {"k": vec(9, 9)}, now=T0)
        session.commit()
        assert np.array_equal(lookup_vectors(session, ["k"])["k"], vec(1, 2))
        store_vectors(session, "h", {})


def test_retryable_failures_are_retried_and_others_fail_every_caller() -> None:
    fake = FakeEmbedder(PRICES)
    fake.failures = [overloaded()]

    async def retried() -> Any:
        return await embedder(fake).embed(["x"])

    assert asyncio.run(retried()).fresh == 1
    assert len(fake.batches) == 2

    fake.failures = [refused()]

    async def refusal() -> list[Any]:
        e = embedder(fake)
        both = e.embed(["y"]), e.embed(["y", "z"])
        return list(await asyncio.gather(*both, return_exceptions=True))

    errors = asyncio.run(refusal())
    assert all(isinstance(e, ProviderError) and e.kind is ErrorKind.INVALID_REQUEST for e in errors)
    assert len(fake.batches) == 3  # one request, failed for both callers


@pytest.mark.parametrize(
    ("vectors", "why"),
    [
        ([[1.0, 0.0]], "1 vectors for 2 texts"),
        ([[1.0, 0.0], [1.0]], "vector shapes"),
        ([[], []], "vector shapes"),
        ([[1.0, float("nan")], [1.0, 0.0]], "finite"),
        ([[1.0, 1e40], [1.0, 0.0]], "finite"),
    ],
)
def test_malformed_embeddings_are_refused(vectors: list[list[float]], why: str) -> None:
    fake = FakeEmbedder(PRICES)
    fake.reply = lambda texts: Embeddings(vectors, Usage(), MODEL)
    with pytest.raises(ProviderError, match=why) as caught:
        asyncio.run(embedder(fake).embed(["a", "b"]))
    assert caught.value.kind is ErrorKind.MALFORMED_RESPONSE
    assert len(fake.batches) == 1  # not retried


def test_a_failed_cache_write_still_returns_vectors(caplog: pytest.LogCaptureFixture) -> None:
    class Broken:
        async def get(self, keys: Any) -> dict[str, Vector]:
            return {}

        async def put(self, embedder_hash: str, vectors: dict[str, Vector]) -> None:
            raise OSError("disk full")

    fake = FakeEmbedder(PRICES)
    with caplog.at_level(logging.WARNING):
        got = asyncio.run(embedder(fake, store=Broken()).embed(["a"]))
    assert got.fresh == 1
    assert "cannot write embeddings" in caplog.text


def test_a_cancelled_caller_does_not_cancel_the_shared_request() -> None:
    fake = FakeEmbedder(PRICES, delay=0.05)

    async def go() -> Any:
        e = embedder(fake, linger=0)
        first = asyncio.create_task(e.embed(["shared"]))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(e.embed(["shared"]))
        await asyncio.sleep(0.01)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        return await second

    got = asyncio.run(go())
    assert (got.fresh, got.cached) == (0, 1)
    assert len(fake.batches) == 1


def test_texts_are_clipped_and_empty_ones_refused() -> None:
    fake = FakeEmbedder(PRICES)
    long = "word " * MAX_TEXT_CHARS
    asyncio.run(embedder(fake).embed([long]))
    assert fake.batches == [[clip(long)]]
    assert len(fake.batches[0][0]) == MAX_TEXT_CHARS
    with pytest.raises(ValueError, match="empty"):
        asyncio.run(embedder(fake).embed(["ok", ""]))


def test_embedder_refuses_unknown_models_and_bad_batches() -> None:
    fake = FakeEmbedder(PRICES)
    with pytest.raises(InvalidInputError, match="unknown model"):
        Embedder(fake, "no-such-model")
    with pytest.raises(ValueError, match="batch_size"):
        Embedder(fake, MODEL, batch_size=0)
    e = Embedder(fake, MODEL)
    assert e.identity() == {"provider": "openai", "endpoint": None, "model": MODEL}
    assert e.fingerprint() != Embedder(fake, MODEL, endpoint="x").fingerprint()
    assert embedding_key(e.fingerprint(), "a") != embedding_key(e.fingerprint(), "b")


def put_aged(sessions: sessionmaker[Session], ages_days: list[int]) -> None:
    with sessions() as session:
        for i, days in enumerate(ages_days):
            store_vectors(session, "h", {f"k{i}": vec(1, 2, 3)}, now=T0 - timedelta(days=days))
        session.commit()


def test_sweep_embeddings_by_age(sessions: sessionmaker[Session]) -> None:
    put_aged(sessions, [1, 10, 40, 50])
    with sessions() as session:
        dry = cache.sweep_embeddings(session, max_age=timedelta(days=30), now=T0, dry_run=True)
        assert (dry.removed, dry.removed_bytes, dry.kept, dry.kept_bytes) == (2, 24, 2, 24)
        done = cache.sweep_embeddings(session, max_age=timedelta(days=30), now=T0, chunk=1)
        assert (done.removed, done.kept) == (2, 2)
        assert sorted(session.scalars(select(CachedEmbedding.key))) == ["k0", "k1"]


def test_maintain_sweeps_embeddings_given_an_age(sessions: sessionmaker[Session]) -> None:
    put_aged(sessions, [1, 40])
    kept = cache.maintain(sessions, None, max_age=None, max_bytes=10**9, grace=timedelta(0), now=T0)
    assert kept.embeddings is None
    done = cache.maintain(
        sessions, None, max_age=timedelta(days=30), max_bytes=None, grace=timedelta(0), now=T0
    )
    assert done.embeddings is not None
    assert (done.embeddings.removed, done.embeddings.kept) == (1, 1)


def test_cli_sweep_reports_embeddings(tmp_path: Any) -> None:
    import io

    from evalkit.cli import main
    from evalkit.db import make_engine, make_session_factory, upgrade

    db = f"sqlite:///{tmp_path / 'cli.db'}"
    engine = make_engine(db)
    upgrade(engine)
    now = datetime.now(UTC)
    with make_session_factory(engine)() as session:
        store_vectors(session, "h", {"old": vec(1, 2)}, now=now - timedelta(days=40))
        store_vectors(session, "h", {"new": vec(1, 2)}, now=now)
        session.commit()
    engine.dispose()
    out = io.StringIO()
    assert main(["--database-url", db, "cache", "sweep", "--max-age", "30d"], out=out) == 0
    assert out.getvalue().splitlines() == [
        "cache: removed 0 entries (0 B), 0 left (0 B)",
        "embeddings: removed 1 (8 B), 1 left (8 B)",
    ]


def test_a_failed_cache_read_fetches_the_texts(caplog: pytest.LogCaptureFixture) -> None:
    class Unreadable:
        async def get(self, keys: Any) -> dict[str, Vector]:
            raise OSError("connection reset")

        async def put(self, embedder_hash: str, vectors: dict[str, Vector]) -> None:
            return

    fake = FakeEmbedder(PRICES)
    with caplog.at_level(logging.WARNING):
        got = asyncio.run(embedder(fake, store=Unreadable()).embed(["a"]))
    assert got.fresh == 1
    assert "cannot read embeddings" in caplog.text


class SlowStore:
    """A cache that takes a while to answer, and never has anything."""

    async def get(self, keys: Any) -> dict[str, Vector]:
        await asyncio.sleep(0.05)
        return {}

    async def put(self, embedder_hash: str, vectors: dict[str, Vector]) -> None:
        return


def test_a_waiter_fetches_for_itself_when_the_owner_is_cancelled() -> None:
    fake = FakeEmbedder(PRICES)

    async def go() -> Any:
        e = embedder(fake, store=SlowStore(), linger=0)
        owner = asyncio.create_task(e.embed(["shared"]))
        await asyncio.sleep(0.01)  # the owner is reading the cache
        waiter = asyncio.create_task(e.embed(["shared"]))
        await asyncio.sleep(0.01)
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner
        return await waiter

    got = asyncio.run(go())
    assert np.array_equal(got.vectors[0], vec(*bag_of_words("shared")))
    assert got.usage.input_tokens == 1  # it paid, since nobody else did
    assert len(fake.batches) == 1


def test_a_request_cancelled_under_a_caller_is_sent_again() -> None:
    fake = FakeEmbedder(PRICES, delay=0.05)

    async def go() -> Any:
        e = embedder(fake, linger=0)
        caller = asyncio.create_task(e.embed(["x"]))
        await asyncio.sleep(0.01)
        for task in list(e._tasks):
            task.cancel()
        return await caller

    got = asyncio.run(go())
    assert got.vectors[0].tolist() == vec(*bag_of_words("x")).tolist()
    assert len(fake.batches) == 2
