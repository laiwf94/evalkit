"""The run configuration, shared by TOML files on the CLI and JSON on the API."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from evalkit.datasets import Label
from evalkit.ratelimit import RateLimit, RetryPolicy
from evalkit.scorers import ScorerSpec
from evalkit.targets import TargetSpec
from evalkit.targets.specs import PROVIDER_NAME, Effort

# A matrix expands into at most this many child runs: each is a full run, so a
# typo that multiplies three long axes should fail at once, not after paying.
MAX_MATRIX_VARIANTS = 64


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


class QueueOptions(_Model):
    """How worker daemons schedule a queued run. They do not change its results."""

    # Higher runs first; equal priorities run oldest first.
    priority: int = Field(default=0, ge=-1000, le=1000)
    # Worker daemons that may execute the run at once; None means no cap.
    max_workers: int | None = Field(default=None, ge=1, le=1024)


class MatrixModel(_Model):
    """One value of the model axis: a model, and the provider that serves it."""

    model: str = Field(min_length=1, max_length=64)
    # None keeps the base target's provider.
    provider: str | None = Field(default=None, pattern=PROVIDER_NAME)
    # Short name for the variant label; defaults to ``provider:model`` (or the model).
    name: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")


class MatrixAxes(_Model):
    """Axes a ``[matrix]`` table expands a prompt target over, in this order.

    Every combination becomes one child run of the same dataset, filter and
    scorers. An empty axis keeps the base target's value."""

    model: tuple[MatrixModel | str, ...] = Field(default=(), max_length=MAX_MATRIX_VARIANTS)
    effort: tuple[Effort, ...] = Field(default=(), max_length=5)
    max_tokens: tuple[Annotated[int, Field(ge=1, le=128_000)], ...] = Field(
        default=(), max_length=MAX_MATRIX_VARIANTS
    )


class RunConfig(_Model):
    label: str = Field(default="", max_length=128)
    dataset: DatasetRef
    target: TargetSpec
    scorers: list[ScorerSpec] = Field(min_length=1, max_length=32)
    filter: CaseFilter = CaseFilter()
    options: RunOptions = RunOptions()
    queue: QueueOptions = QueueOptions()
    # Expands the run into grouped child runs, one per combination (``evalkit.matrix``).
    matrix: MatrixAxes | None = None
