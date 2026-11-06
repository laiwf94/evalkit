"""Relational schema. Works on PostgreSQL and SQLite; JSON columns hold the
parts whose shape belongs to the application (case inputs, specs, scores)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Double,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


def as_utc(value: datetime) -> datetime:
    """SQLite returns naive datetimes; everything we store is UTC."""
    return value if value.tzinfo else value.replace(tzinfo=UTC)


class Base(DeclarativeBase):
    type_annotation_map = {  # noqa: RUF012
        dict[str, Any]: JSON,
        list[Any]: JSON,
        datetime: DateTime(timezone=True),
    }


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_datasets_name_version"),
        UniqueConstraint("name", "content_hash", name="uq_datasets_name_hash"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    version: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64))
    case_count: Mapped[int] = mapped_column(Integer)
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    cases: Mapped[list[DatasetCase]] = relationship(
        back_populates="dataset", order_by="DatasetCase.position", cascade="all, delete-orphan"
    )


class DatasetCase(Base):
    __tablename__ = "dataset_cases"
    __table_args__ = (
        UniqueConstraint("dataset_id", "case_id", name="uq_dataset_cases_case"),
        Index("ix_dataset_cases_position", "dataset_id", "position"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"))
    position: Mapped[int] = mapped_column(Integer)
    case_id: Mapped[str] = mapped_column(String(256))
    input: Mapped[Any] = mapped_column(JSON)
    expected: Mapped[Any] = mapped_column(JSON, nullable=True)
    tags: Mapped[list[Any]] = mapped_column(default=list)
    split: Mapped[str | None] = mapped_column(String(64), nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", default=dict)

    dataset: Mapped[Dataset] = relationship(back_populates="cases")


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (
        Index("ix_runs_created_at", "created_at"),
        Index("ix_runs_queue", "status", "priority", "queued_at"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    label: Mapped[str] = mapped_column(String(128), default="")
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="RESTRICT"))
    target_spec: Mapped[dict[str, Any]] = mapped_column()
    target_hash: Mapped[str] = mapped_column(String(64))
    scorers: Mapped[list[Any]] = mapped_column()
    options: Mapped[dict[str, Any]] = mapped_column()
    case_filter: Mapped[dict[str, Any]] = mapped_column(default=dict)
    seed: Mapped[int] = mapped_column(BigInteger)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    error: Mapped[str] = mapped_column(Text, default="")
    batch_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lease_owner: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(nullable=True)
    leases_seeded: Mapped[bool] = mapped_column(Boolean, default=False)
    # Set when the run was submitted to the worker queue; CLI runs leave it empty.
    queued_at: Mapped[datetime | None] = mapped_column(nullable=True)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    # How many worker daemons may execute the run at once; None means no cap.
    max_workers: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # How many cases the run's filter selects; set at creation (older runs: on first read).
    cases_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Exact percentiles of fresh successful latencies, set when the run is compacted.
    latency_p50_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_p95_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)

    dataset: Mapped[Dataset] = relationship()


class Result(Base):
    __tablename__ = "results"
    __table_args__ = (UniqueConstraint("run_id", "case_id", name="uq_results_run_case"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    case_id: Mapped[str] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(16))
    output: Mapped[str | None] = mapped_column(Text, nullable=True)
    stop_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    cached: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_write_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    judge_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    judge_output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    judge_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    scores: Mapped[dict[str, Any]] = mapped_column(default=dict)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class RunAggregate(Base):
    """Running totals of one run's results, kept in the transaction that writes them.
    Each writing process has its own ``shard`` row, so processes never wait on each
    other's aggregate updates; a run's totals are the sum over its shards."""

    __tablename__ = "run_aggregates"

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    shard: Mapped[str] = mapped_column(String(64), primary_key=True)
    completed: Mapped[int] = mapped_column(BigInteger)
    ok: Mapped[int] = mapped_column(BigInteger)
    errors: Mapped[int] = mapped_column(BigInteger)
    cached: Mapped[int] = mapped_column(BigInteger)
    # Successful results that were not served from the response cache.
    fresh: Mapped[int] = mapped_column(BigInteger)
    input_tokens: Mapped[int] = mapped_column(BigInteger)
    output_tokens: Mapped[int] = mapped_column(BigInteger)
    cache_read_tokens: Mapped[int] = mapped_column(BigInteger)
    cache_write_tokens: Mapped[int] = mapped_column(BigInteger)
    cost_usd: Mapped[float] = mapped_column(Double)
    judge_input_tokens: Mapped[int] = mapped_column(BigInteger)
    judge_output_tokens: Mapped[int] = mapped_column(BigInteger)
    judge_cost_usd: Mapped[float] = mapped_column(Double)


class RunMetricAggregate(Base):
    """Running count, sum and sum of squares of one metric's scores, per shard."""

    __tablename__ = "run_metric_aggregates"

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    shard: Mapped[str] = mapped_column(String(64), primary_key=True)
    metric: Mapped[str] = mapped_column(String(128), primary_key=True)
    n: Mapped[int] = mapped_column(BigInteger)
    total: Mapped[float] = mapped_column(Double)
    total_sq: Mapped[float] = mapped_column(Double)
    unscorable: Mapped[int] = mapped_column(BigInteger)


class RunLatencyBucket(Base):
    """How many fresh successful results of one run, written by one process (the
    shard), had a latency in one logarithmic bucket (see ``aggregates.latency_bucket``).
    A running run's latency percentiles come from these rows, not from its results."""

    __tablename__ = "run_latency_buckets"

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    shard: Mapped[str] = mapped_column(String(64), primary_key=True)
    bucket: Mapped[int] = mapped_column(Integer, primary_key=True)
    count: Mapped[int] = mapped_column(BigInteger)


class RunScores(Base):
    """One metric's scores for every case of a completed run, as one array.

    ``scores`` is little-endian float64, one value per case of the run's dataset in
    case-id order, NaN where the case has no score (not selected, or unscorable).
    Two runs over the same dataset content line up element by element, so a paired
    comparison reads one row per metric per run. ``mean``, ``ci_low`` and ``ci_high``
    are the run summary's interval, computed once when the row is written."""

    __tablename__ = "run_scores"

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    metric: Mapped[str] = mapped_column(String(128), primary_key=True)
    case_count: Mapped[int] = mapped_column(Integer)
    n: Mapped[int] = mapped_column(Integer)
    unscorable: Mapped[int] = mapped_column(Integer)
    mean: Mapped[float | None] = mapped_column(Double, nullable=True)
    ci_low: Mapped[float | None] = mapped_column(Double, nullable=True)
    ci_high: Mapped[float | None] = mapped_column(Double, nullable=True)
    scores: Mapped[bytes] = mapped_column(LargeBinary)


class CaseLease(Base):
    """One row per case of a run that still needed executing when the run was first
    joined. A worker owns a case while ``owner`` is set and ``expires_at`` is in the
    future; ``fence`` grows on every claim, so a result written under an older claim
    can be recognised and refused."""

    __tablename__ = "case_leases"
    __table_args__ = (Index("ix_case_leases_claimable", "run_id", "done", "position"),)

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    case_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    position: Mapped[int] = mapped_column(Integer)
    owner: Mapped[str | None] = mapped_column(String(64), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(nullable=True)
    fence: Mapped[int] = mapped_column(BigInteger, default=0)
    done: Mapped[bool] = mapped_column(Boolean, default=False)


class RunWorker(Base):
    """A worker daemon's slot on a run. It counts towards the run's ``max_workers``
    while ``expires_at`` is in the future; the worker renews it with a heartbeat,
    so the slot of a worker that died frees itself."""

    __tablename__ = "run_workers"

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    worker_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    joined_at: Mapped[datetime] = mapped_column(default=utcnow)
    expires_at: Mapped[datetime] = mapped_column()


class CachedResponse(Base):
    """One cached output. ``output`` is empty when the text is in the blob store
    under ``output_blob``; ``size_bytes`` is the output's UTF-8 size either way."""

    __tablename__ = "response_cache"
    __table_args__ = (Index("ix_response_cache_last_used", "last_used_at", "key"),)

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    target_hash: Mapped[str] = mapped_column(String(64), index=True)
    output: Mapped[str] = mapped_column(Text)
    stop_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    usage: Mapped[dict[str, Any]] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    last_used_at: Mapped[datetime] = mapped_column(default=utcnow)
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    output_blob: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)


class RateLimitState(Base):
    """The shared state of one provider's GCRA limiter (see ``ratelimit_pg.py``).
    Both columns are seconds since the epoch on the database server's clock."""

    __tablename__ = "rate_limits"

    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    tat: Mapped[float] = mapped_column(Double)
    cooldown_until: Mapped[float] = mapped_column(Double)


class Calibration(Base):
    __tablename__ = "judge_calibrations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    judge_hash: Mapped[str] = mapped_column(String(64), index=True)
    judge_spec: Mapped[dict[str, Any]] = mapped_column()
    n: Mapped[int] = mapped_column(Integer)
    kappa: Mapped[float] = mapped_column(Float)
    weighted_kappa: Mapped[float | None] = mapped_column(Float, nullable=True)
    agreement: Mapped[float] = mapped_column(Float)
    confusion: Mapped[dict[str, Any]] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
