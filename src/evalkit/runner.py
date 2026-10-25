"""Executes runs: a bounded pool of workers, a response cache and crash-safe progress.

How a run stays correct when it is killed at any point:

* Each finished case is one row in ``results``, unique on (run, case). On
  resume, the runner skips cases that already have a row, so a case is never
  executed and paid for twice after its result was saved.
* Results are written by a single writer task in short transactions. No
  transaction is open while a worker waits on a model, so a slow target never
  holds a database connection, and a crash loses at most the results still
  queued for the writer.
* A run is executed only by the process holding its lease. The lease is taken
  with a conditional UPDATE and renewed by a heartbeat; a second process sees
  a live lease and refuses. If the holder dies, the lease expires and anyone
  can resume.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import random
import socket
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Any

from pydantic import TypeAdapter
from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from evalkit import cache
from evalkit.config import Settings
from evalkit.datasets import Case, get_dataset, load_cases
from evalkit.db.models import Dataset, Result, Run, utcnow
from evalkit.errors import ConflictError, EvalkitError, InvalidInputError, NotFoundError
from evalkit.llm import ClaudeClient
from evalkit.logs import log_context, scrubber
from evalkit.ratelimit import (
    Clock,
    Limiters,
    ProviderLimiter,
    RealClock,
    RetryStats,
    call_with_retries,
)
from evalkit.redaction import Redactor
from evalkit.runspec import CaseFilter, RunConfig, RunOptions
from evalkit.scorers import Scorer, ScoreResult, ScorerSpec, build_scorers
from evalkit.targets import Generation, Target, TargetError, TargetSpec, Usage
from evalkit.targets.factory import build_target, identity

log = logging.getLogger(__name__)

LEASE_TTL = timedelta(seconds=30)
HEARTBEAT_SECONDS = 10.0
WRITE_BATCH = 50
WRITE_LINGER_SECONDS = 0.2
MAX_ERROR_CHARS = 2000
RESUMABLE = ("pending", "running", "interrupted", "failed")

_TARGET_SPEC: TypeAdapter[Any] = TypeAdapter(TargetSpec)
_SCORER_SPECS: TypeAdapter[list[Any]] = TypeAdapter(list[ScorerSpec])


@dataclass(frozen=True)
class StoredRun:
    """The parts of a run's config that execution needs, as stored on the run row."""

    target: Any
    scorers: list[Any]
    options: RunOptions
    filter: CaseFilter

    @classmethod
    def from_row(cls, run: Run) -> StoredRun:
        return cls(
            target=_TARGET_SPEC.validate_python(run.target_spec),
            scorers=_SCORER_SPECS.validate_python(run.scorers),
            options=RunOptions.model_validate(run.options),
            filter=CaseFilter.model_validate(run.case_filter),
        )


class LeaseLostError(EvalkitError):
    code = "conflict"


@dataclass
class CaseResult:
    case_id: str
    status: str
    output: str | None = None
    stop_reason: str | None = None
    cached: bool = False
    latency_ms: float = 0.0
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    judge_cost_usd: float = 0.0
    scores: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    cache_entry: dict[str, Any] | None = None

    def row(self, run_id: str) -> dict[str, Any]:
        values = asdict(self)
        values.pop("cache_entry")
        return {"run_id": run_id, **values}


@dataclass(frozen=True)
class ExecutionReport:
    run_id: str
    status: str
    executed: int
    cached: int
    errors: int
    skipped: int


def new_owner_id() -> str:
    return f"{socket.gethostname()[:32]}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def clean_error(message: str) -> str:
    message = scrubber.scrub(message)
    return message if len(message) <= MAX_ERROR_CHARS else message[: MAX_ERROR_CHARS - 3] + "..."


@dataclass(frozen=True)
class _Context:
    """Everything a worker needs for one run."""

    run_id: str
    target_hash: str
    config: StoredRun
    target: Target
    scorers: Sequence[Scorer]
    limiter: ProviderLimiter
    rng: random.Random


class Runner:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        settings: Settings,
        *,
        claude: ClaudeClient | None = None,
        clock: Clock | None = None,
        owner: str | None = None,
        allow_import_paths: bool = False,
        target_factory: Callable[[Any], Target] | None = None,
        lease_ttl: timedelta = LEASE_TTL,
        heartbeat_seconds: float = HEARTBEAT_SECONDS,
    ) -> None:
        self.sessions = sessions
        self.settings = settings
        self.claude = claude
        self.clock = clock or RealClock()
        self.owner = owner or new_owner_id()
        self.allow_import_paths = allow_import_paths
        self.redactor = Redactor(settings.redaction)
        self._target_factory = target_factory
        self.lease_ttl = lease_ttl
        self.heartbeat_seconds = heartbeat_seconds

    def create_run(self, config: RunConfig) -> Run:
        """Validate ``config`` against the stored dataset and record a pending run."""
        build_scorers(config.scorers, allow_import_paths=self.allow_import_paths)
        target_id = identity(config.target, self.settings.http_endpoints)
        if config.options.mode == "batch" and config.target.type != "prompt":
            raise InvalidInputError("batch mode is only available for prompt targets")
        with self.sessions() as session:
            dataset = get_dataset(session, config.dataset.name, config.dataset.version)
            cases = load_cases(session, dataset, config.filter.split, config.filter.tags)
            if not cases:
                raise InvalidInputError("the filter selects no cases from the dataset")
            run = Run(
                id=uuid.uuid4().hex,
                label=config.label,
                dataset_id=dataset.id,
                target_spec=config.target.model_dump(mode="json"),
                target_hash=target_id,
                scorers=[s.model_dump(mode="json", by_alias=True) for s in config.scorers],
                options=config.options.model_dump(mode="json"),
                case_filter=config.filter.model_dump(mode="json"),
                seed=config.options.seed,
                status="pending",
            )
            session.add(run)
            session.commit()
            return run

    async def execute(self, run_id: str) -> ExecutionReport:
        with log_context(run_id=run_id):
            run = await asyncio.to_thread(self._claim, run_id)
            if run.status == "completed":
                return ExecutionReport(run_id, "completed", 0, 0, 0, 0)
            try:
                report = await self._execute_claimed(run)
            except LeaseLostError:
                log.warning("lease lost; another process now owns the run")
                raise
            except (asyncio.CancelledError, KeyboardInterrupt):
                await asyncio.shield(asyncio.to_thread(self._finish, run_id, "interrupted", ""))
                raise
            except Exception as exc:
                message = clean_error(f"{type(exc).__name__}: {exc}")
                await asyncio.to_thread(self._finish, run_id, "failed", message)
                log.exception("run failed")
                raise
            await asyncio.to_thread(self._finish, run_id, "completed", "")
            log.info("run completed", extra=asdict(report))
            return report

    async def _execute_claimed(self, run: Run) -> ExecutionReport:
        config = StoredRun.from_row(run)
        cases, done = await asyncio.to_thread(self._load_cases, run, config)
        pending = [c for c in cases if c.id not in done]
        scorers = build_scorers(config.scorers, allow_import_paths=self.allow_import_paths)
        target = self._build_target(config)
        limiter = Limiters({target.provider: config.options.rate_limit}, self.clock).get(
            target.provider
        )
        fingerprint = self.settings.redaction.fingerprint()
        keys = {c.id: cache.cache_key(run.target_hash, c.input, fingerprint) for c in pending}
        hits: dict[str, Any] = {}
        if config.options.use_cache and pending:
            hits = await asyncio.to_thread(self._cache_lookup, list(keys.values()))
        log.info(
            "run started",
            extra={"cases": len(cases), "already_done": len(done), "cache_hits": len(hits)},
        )

        results: asyncio.Queue[CaseResult | None] = asyncio.Queue(maxsize=WRITE_BATCH * 4)
        work: asyncio.Queue[Case] = asyncio.Queue()
        for c in pending:
            work.put_nowait(c)
        counts = {"executed": 0, "cached": 0, "errors": 0}
        ctx = _Context(
            run_id=run.id,
            target_hash=run.target_hash,
            config=config,
            target=target,
            scorers=scorers,
            limiter=limiter,
            rng=random.Random(run.seed),  # noqa: S311 - retry jitter
        )

        async def worker() -> None:
            while True:
                try:
                    case = work.get_nowait()
                except asyncio.QueueEmpty:
                    return
                hit = hits.get(keys[case.id])
                if hit is not None:
                    result = await self._from_cache(case, hit, scorers)
                    counts["cached"] += 1
                else:
                    result = await self._run_case(ctx, case, keys[case.id])
                    counts["executed"] += 1
                counts["errors"] += result.status == "error"
                await results.put(result)

        unsaved: list[CaseResult] = []
        try:
            async with asyncio.TaskGroup() as group:
                writer = group.create_task(self._writer(run.id, results, unsaved))
                heartbeat = group.create_task(self._heartbeat(run.id))
                workers = [group.create_task(worker()) for _ in range(config.options.concurrency)]
                await asyncio.gather(*workers)
                await results.put(None)
                await writer
                heartbeat.cancel()
        except ExceptionGroup as group:
            # Report the first real failure (often the only one), not the TaskGroup wrapper.
            raise group.exceptions[0] from None
        except asyncio.CancelledError:
            await self._save_unwritten(run.id, results, unsaved)
            raise
        finally:
            await target.aclose()
        return ExecutionReport(
            run.id, "completed", counts["executed"], counts["cached"], counts["errors"], len(done)
        )

    async def _save_unwritten(
        self, run_id: str, queue: asyncio.Queue[CaseResult | None], unsaved: list[CaseResult]
    ) -> None:
        """On interrupt (Ctrl-C, shutdown), save results that are finished but not yet
        written, so a resume does not pay for those cases again."""
        while not queue.empty():
            item = queue.get_nowait()
            if item is not None:
                unsaved.append(item)
        if unsaved:
            await asyncio.shield(asyncio.to_thread(self._write, run_id, list(unsaved)))

    def _build_target(self, config: StoredRun) -> Target:
        if self._target_factory is not None:
            return self._target_factory(config.target)
        return build_target(
            config.target, endpoints=self.settings.http_endpoints, claude=self.claude
        )

    async def _run_case(self, ctx: _Context, case: Case, key: str) -> CaseResult:
        stats = RetryStats()
        start = time.perf_counter()
        options = ctx.config.options
        try:
            async with asyncio.timeout(options.case_timeout_seconds):
                generation: Generation = await call_with_retries(
                    lambda: ctx.target.generate(case),
                    limiter=ctx.limiter,
                    policy=options.retry,
                    stats=stats,
                    rng=ctx.rng,
                )
        except TargetError as exc:
            return self._error(case, exc.message, stats, start)
        except InvalidInputError as exc:
            return self._error(case, exc.message, stats, start)
        except TimeoutError:
            limit = options.case_timeout_seconds
            return self._error(case, f"case took longer than {limit:g} s", stats, start)
        latency = (time.perf_counter() - start) * 1000
        return await self._finish_case(
            ctx, case, key, generation, latency_ms=latency, attempts=stats.attempts
        )

    async def _finish_case(
        self,
        ctx: _Context,
        case: Case,
        key: str,
        generation: Generation,
        *,
        latency_ms: float,
        attempts: int,
    ) -> CaseResult:
        redacted = self.redactor.redact(generation.text)
        result = CaseResult(
            case_id=case.id,
            status="ok",
            output=redacted.text,
            stop_reason=generation.stop_reason,
            latency_ms=latency_ms,
            attempts=attempts,
            **_usage_fields(generation.usage),
        )
        if redacted.failed:
            # Keep the placeholder out of the cache so a rerun gets a fresh output.
            log.warning("redaction timed out; output replaced", extra={"case_id": case.id})
        else:
            result.cache_entry = {
                "key": key,
                "target_hash": ctx.target_hash,
                "output": redacted.text,
                "stop_reason": generation.stop_reason,
                "usage": asdict(generation.usage),
            }
        await self._score(case, result, ctx.scorers)
        return result

    def _error(self, case: Case, message: str, stats: RetryStats, start: float) -> CaseResult:
        log.info("case failed", extra={"case_id": case.id, "attempts": stats.attempts})
        return CaseResult(
            case_id=case.id,
            status="error",
            attempts=stats.attempts,
            latency_ms=(time.perf_counter() - start) * 1000,
            error=clean_error(message),
        )

    async def _from_cache(self, case: Case, hit: Any, scorers: Sequence[Scorer]) -> CaseResult:
        # A cache hit costs nothing in this run; the original cost stays in the cache row.
        result = CaseResult(
            case_id=case.id,
            status="ok",
            output=hit.output,
            stop_reason=hit.stop_reason,
            cached=True,
        )
        await self._score(case, result, scorers)
        return result

    async def _score(self, case: Case, result: CaseResult, scorers: Sequence[Scorer]) -> None:
        output = result.output or ""
        for scorer in scorers:
            try:
                outcome = await scorer.score(case, output)
            except Exception as exc:  # a plugin bug must not take the run down
                log.warning(
                    "scorer raised",
                    extra={"case_id": case.id, "scorer": scorer.name},
                    exc_info=True,
                )
                outcome = ScoreResult.unscorable(clean_error(f"scorer failed: {exc}"))
            value = outcome.value
            detail = outcome.detail
            if value is not None and not (math.isfinite(value) and 0.0 <= value <= 1.0):
                detail = f"scorer returned {value!r}, outside [0, 1]"
                value = None
            result.scores[scorer.name] = {"value": value, "detail": clean_error(detail)}
            result.judge_input_tokens += outcome.input_tokens
            result.judge_output_tokens += outcome.output_tokens
            result.judge_cost_usd += outcome.cost_usd

    async def _writer(
        self, run_id: str, queue: asyncio.Queue[CaseResult | None], batch: list[CaseResult]
    ) -> None:
        """Write results in batches. ``batch`` is shared so an interrupted run can
        save what the writer had collected but not yet written."""
        finished = False
        while not finished:
            item = await queue.get()
            if item is None:
                finished = True
            else:
                batch.append(item)
                deadline = time.monotonic() + WRITE_LINGER_SECONDS
                while len(batch) < WRITE_BATCH:
                    try:
                        nxt = await asyncio.wait_for(
                            queue.get(), max(0.0, deadline - time.monotonic())
                        )
                    except TimeoutError:
                        break
                    if nxt is None:
                        finished = True
                        break
                    batch.append(nxt)
            if batch:
                await asyncio.to_thread(self._write, run_id, list(batch))
                batch.clear()

    async def _heartbeat(self, run_id: str) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_seconds)
            renewed = await asyncio.to_thread(self._renew, run_id)
            if not renewed:
                raise LeaseLostError(f"run {run_id} is now owned by another process")

    def _claim(self, run_id: str) -> Run:
        now = utcnow()
        with self.sessions() as session:
            claimed = session.execute(
                update(Run)
                .where(
                    Run.id == run_id,
                    Run.status.in_(RESUMABLE),
                    or_(
                        Run.lease_owner.is_(None),
                        Run.lease_owner == self.owner,
                        Run.lease_expires_at < now,
                    ),
                )
                .values(
                    lease_owner=self.owner,
                    lease_expires_at=now + self.lease_ttl,
                    status="running",
                    error="",
                )
                .execution_options(synchronize_session=False)
            )
            session.commit()
            run = session.get(Run, run_id)
            if run is None:
                raise NotFoundError(f"run {run_id} not found")
            if claimed.rowcount == 0 and run.status != "completed":  # type: ignore[attr-defined]
                raise ConflictError(f"run {run_id} is being executed by another process")
            if run.started_at is None:
                run.started_at = now
                session.commit()
            return run

    def _renew(self, run_id: str) -> bool:
        with self.sessions() as session:
            renewed = session.execute(
                update(Run)
                .where(Run.id == run_id, Run.lease_owner == self.owner)
                .values(lease_expires_at=utcnow() + self.lease_ttl)
                .execution_options(synchronize_session=False)
            )
            session.commit()
            return bool(renewed.rowcount)  # type: ignore[attr-defined]

    def _finish(self, run_id: str, status: str, error: str) -> None:
        with self.sessions() as session:
            values: dict[str, Any] = {
                "status": status,
                "error": error,
                "lease_owner": None,
                "lease_expires_at": None,
            }
            if status == "completed":
                values["finished_at"] = utcnow()
            session.execute(
                update(Run)
                .where(Run.id == run_id, Run.lease_owner == self.owner)
                .values(**values)
                .execution_options(synchronize_session=False)
            )
            session.commit()

    def _load_cases(self, run: Run, config: StoredRun) -> tuple[list[Case], set[str]]:
        with self.sessions() as session:
            dataset = session.get_one(Dataset, run.dataset_id)
            cases = load_cases(session, dataset, config.filter.split, config.filter.tags)
            done = set(session.scalars(select(Result.case_id).where(Result.run_id == run.id)))
            return cases, done

    def _cache_lookup(self, keys: list[str]) -> dict[str, Any]:
        with self.sessions() as session:
            return cache.lookup(session, keys)

    def _write(self, run_id: str, batch: list[CaseResult]) -> None:
        entries = [r.cache_entry for r in batch if r.cache_entry is not None]
        with self.sessions() as session:
            try:
                session.add_all(Result(**r.row(run_id)) for r in batch)
                cache.store(session, entries)
                session.commit()
            except IntegrityError:
                # Only possible if a result for one of these cases already exists;
                # write the rest one by one and keep the stored rows.
                session.rollback()
                for r in batch:
                    try:
                        session.add(Result(**r.row(run_id)))
                        session.commit()
                    except IntegrityError:
                        session.rollback()
                        log.warning("duplicate result skipped", extra={"case_id": r.case_id})
                cache.store(session, entries)
                session.commit()


def _usage_fields(usage: Usage) -> dict[str, Any]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "cost_usd": usage.cost_usd,
    }
