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
    __table_args__ = (Index("ix_runs_created_at", "created_at"),)

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


class CachedResponse(Base):
    __tablename__ = "response_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    target_hash: Mapped[str] = mapped_column(String(64), index=True)
    output: Mapped[str] = mapped_column(Text)
    stop_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    usage: Mapped[dict[str, Any]] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


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
