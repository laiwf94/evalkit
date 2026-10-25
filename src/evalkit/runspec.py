"""The run configuration, shared by TOML files on the CLI and JSON on the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evalkit.datasets import Label
from evalkit.ratelimit import RateLimit, RetryPolicy
from evalkit.scorers import ScorerSpec
from evalkit.targets import TargetSpec


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DatasetRef(_Model):
    name: str = Field(min_length=1, max_length=128)
    version: int | None = Field(default=None, ge=1)


class CaseFilter(_Model):
    split: Label | None = None
    tags: tuple[Label, ...] = Field(default=(), max_length=32)


class RunOptions(_Model):
    concurrency: int = Field(default=8, ge=1, le=256)
    seed: int = Field(default=0, ge=0, le=2**63 - 1)
    use_cache: bool = True
    mode: Literal["online", "batch"] = "online"
    case_timeout_seconds: float = Field(default=600.0, gt=0, le=3600)
    batch_poll_seconds: float = Field(default=30.0, ge=0.01, le=3600)
    retry: RetryPolicy = RetryPolicy()
    rate_limit: RateLimit = RateLimit()


class RunConfig(_Model):
    label: str = Field(default="", max_length=128)
    dataset: DatasetRef
    target: TargetSpec
    scorers: list[ScorerSpec] = Field(min_length=1, max_length=32)
    filter: CaseFilter = CaseFilter()
    options: RunOptions = RunOptions()
