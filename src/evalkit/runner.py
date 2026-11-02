"""Executes runs: a bounded pool of workers, a response cache and crash-safe progress.

How a run stays correct when it is killed at any point:

* Each finished case is one row in ``results``, unique on (run, case). On
  resume, the runner skips cases that already have a row, so a case is never
  executed and paid for twice after its result was saved.
* Results are written by a single writer task in short transactions. No
  transaction is open while a worker waits on a model, so a slow target never
  holds a database connection, and a crash loses at most the results still
  queued for the writer.
* Any number of processes can execute one run. Each claims batches of cases
  through ``case_leases`` (see ``leases.py``), renews its leases with a
  heartbeat, and writes a result only while its lease on that case is still
  current. If a process dies, its leases expire and the others take the cases
  over. Whoever finds no unfinished case left marks the run completed with a
  compare-and-set, so exactly one process does it.
* Batch mode is the exception: a Message Batch is one paid request for the
  whole run, so a batch-mode run is executed by one process at a time, which
  holds a lease on the run row.
* A run can be cancelled (``runqueue.cancel``) or a process asked to drain
  (``request_drain``, on SIGTERM). A cancelled process saves the results it
  has and stops at once; a draining one stops claiming, gives back the cases
  it has not started and finishes the ones in flight, so no paid work is lost.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import math
import os
import random
import socket
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Any

from pydantic import TypeAdapter
from sqlalchemy import ColumnElement, case, exists, false, func, or_, select, true, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from evalkit import aggregates, cache, leases, metrics, runqueue
from evalkit.config import Settings
from evalkit.datasets import Case, get_dataset, load_cases
from evalkit.db.models import CaseLease, Dataset, Result, Run, as_utc, utcnow
from evalkit.errors import ConflictError, EvalkitError, InvalidInputError, NotFoundError
from evalkit.llm import ClaudeClient
from evalkit.logs import log_context, scrubber
from evalkit.ratelimit import (
    Clock,
    Limiters,
    RateLimit,
    RateLimiter,
    RealClock,
    RetryStats,
    call_with_retries,
)
from evalkit.ratelimit_pg import PgRateLimiter
from evalkit.redaction import Redactor
from evalkit.runspec import CaseFilter, RunConfig, RunOptions
from evalkit.scorers import Scorer, ScoreResult, ScorerSpec, build_scorers
from evalkit.scorers.judge import LLMJudge
from evalkit.targets import Generation, Target, TargetError, TargetSpec, Usage
from evalkit.targets.factory import build_target, identity
from evalkit.targets.prompt import PromptTarget

log = logging.getLogger(__name__)

WRITE_BATCH = 50
WRITE_LINGER_SECONDS = 0.2
MAX_ERROR_CHARS = 2000
RESUMABLE = ("pending", "queued", "running", "interrupted", "failed")
STOPPED = ("cancelling", "cancelled")
MIN_CLAIM = 64
MAX_CLAIM = 500
# How soon to look again when unfinished cases are unowned but were not claimable
# (another worker had them locked, or was giving them back).
RECHECK_SECONDS = 0.05


def claim_size(concurrency: int) -> int:
    """Cases claimed per round trip. Each claim is a write, and on SQLite it waits
    for the result writer's lock, so claims are kept few; a dead process strands
    at most about twice this many cases, for one lease length."""
    return min(max(4 * concurrency, MIN_CLAIM), MAX_CLAIM)


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
    fence: int = 0

    def row(self, run_id: str) -> dict[str, Any]:
        values = asdict(self)
        values.pop("cache_entry")
        values.pop("fence")
        return {"run_id": run_id, **values}


@dataclass(frozen=True)
class ExecutionReport:
    run_id: str
    status: str
    executed: int
    cached: int
    errors: int
    skipped: int
    batched: int = 0
    refused: int = 0


def batch_custom_id(case_id: str) -> str:
    """Batch custom ids allow only [A-Za-z0-9_-]{1,64}; case ids allow more."""
    return "c" + hashlib.sha256(case_id.encode()).hexdigest()[:40]


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
    limiter: RateLimiter
    rng: random.Random
    pending: dict[str, Case]
    keys: dict[str, str]
    hits: dict[str, Any]
    # Set by the writer after every write, for a feeder waiting on its own results.
    written: asyncio.Event = field(default_factory=asyncio.Event)
    # Set by a worker when the work queue falls to ``low_water``, to wake the feeder.
    drained: asyncio.Event = field(default_factory=asyncio.Event)
    low_water: int = 0
    counts: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(
            ("executed", "cached", "errors", "batched", "refused"), 0
        )
    )


_WorkQueue = asyncio.Queue[tuple[Case, int] | None]


@dataclass
class _Control:
    """How other code reaches a run this process is executing."""

    task: asyncio.Task[Any] | None
    # Set when the run was cancelled, so the CancelledError means "stop for good".
    cancelled: bool = False
    exclusive: bool = False
    # Set to drain: stop claiming, give back unstarted cases, finish the rest.
    draining: asyncio.Event = field(default_factory=asyncio.Event)


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
        lease_ttl: timedelta | None = None,
        heartbeat_seconds: float | None = None,
        shared_limits: bool | None = None,
    ) -> None:
        self.sessions = sessions
        self.settings = settings
        self.claude = claude
        self.clock = clock or RealClock()
        self.owner = owner or new_owner_id()
        self.allow_import_paths = allow_import_paths
        self.redactor = Redactor(settings.redaction)
        self._target_factory = target_factory
        self.lease_ttl = lease_ttl or timedelta(seconds=settings.lease_seconds)
        # Renew well before expiry so one slow database round trip cannot lose the lease.
        self.heartbeat_seconds = heartbeat_seconds or self.lease_ttl.total_seconds() / 3
        self._active: dict[str, _Control] = {}
        # Each run's metric names, for the aggregates written with its results.
        self._metrics: dict[str, list[str]] = {}
        # On PostgreSQL several processes may execute one run, so rate limits and
        # cool-downs live in the database where they all see them.
        self.shared_limits = (
            _dialect(sessions) == "postgresql" if shared_limits is None else shared_limits
        )
        # SQLite has one write lock per database, and a connection that finds it
        # taken backs off in sleeps of up to 100 ms. Claims and result writes from
        # this process take turns on a thread lock instead, which hands over at once.
        self._write_turn: contextlib.AbstractContextManager[Any] = (
            threading.Lock() if _dialect(sessions) == "sqlite" else contextlib.nullcontext()
        )

    def create_run(self, config: RunConfig, *, queued: bool = False) -> Run:
        """Validate ``config`` against the stored dataset and record the run:
        ``pending`` for this process to execute, or ``queued`` for worker daemons."""
        build_scorers(
            config.scorers, allow_import_paths=self.allow_import_paths, claude=self.claude
        )
        target_id = identity(config.target, self.settings.http_endpoints)
        if config.options.mode == "batch" and config.target.type != "prompt":
            raise InvalidInputError("batch mode is only available for prompt targets")
        with self.sessions() as session:
            dataset = get_dataset(session, config.dataset.name, config.dataset.version)
            cases = load_cases(session, dataset, config.filter.split, config.filter.tags)
            if not cases:
                raise InvalidInputError("the filter selects no cases from the dataset")
            now = utcnow()
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
                status="queued" if queued else "pending",
                queued_at=now if queued else None,
                priority=config.queue.priority,
                # A batch is one paid request for the whole run: one worker at a time.
                max_workers=1 if config.options.mode == "batch" else config.queue.max_workers,
                cases_total=len(cases),
                created_at=now,
            )
            session.add(run)
            session.flush()
            if queued:
                runqueue.notify(session, "queued", run.id)
            session.commit()
            return run

    @property
    def executing(self) -> set[str]:
        """Runs this process is executing now."""
        return set(self._active)

    def request_cancel(self, run_id: str) -> bool:
        """Stop executing ``run_id`` because it was cancelled: results already
        finished are saved, unfinished cases are given back. Returns whether this
        process was executing it."""
        control = self._active.get(run_id)
        if control is None or control.task is None:
            return False
        if not control.cancelled:
            control.cancelled = True
            control.task.cancel()
        return True

    def request_drain(self, run_id: str) -> bool:
        """Stop executing ``run_id`` without losing paid work: claim nothing more,
        give back cases no worker has started, finish and save the ones in flight.
        A batch-mode run stops at once (its batch id is saved, so a resume polls
        the same batch). Returns whether this process was executing it."""
        control = self._active.get(run_id)
        if control is None:
            return False
        control.draining.set()
        if control.exclusive and control.task is not None:
            control.task.cancel()
        return True

    async def execute(self, run_id: str) -> ExecutionReport:
        # The lease lets its owner re-claim a run, which is how a process resumes
        # its own failed run; this set stops it from executing one run twice at once.
        if run_id in self._active:
            raise ConflictError(f"run {run_id} is already executing in this process")
        control = _Control(asyncio.current_task())
        self._active[run_id] = control
        try:
            return await self._execute(run_id, control)
        finally:
            del self._active[run_id]

    async def _execute(self, run_id: str, control: _Control) -> ExecutionReport:
        with log_context(run_id=run_id):
            run, exclusive = await asyncio.to_thread(self._start, run_id)
            control.exclusive = exclusive
            if run.status == "completed":
                return ExecutionReport(run_id, "completed", 0, 0, 0, 0)
            if run.status in STOPPED:
                final = await asyncio.to_thread(self._finish, run_id, "cancelled", "", exclusive)
                return ExecutionReport(run_id, final, 0, 0, 0, 0)
            metrics.RUNS_ACTIVE.inc()
            status = "failed"
            try:
                report = await self._execute_claimed(run, exclusive, control)
                status = report.status
                if status != "completed":
                    await asyncio.to_thread(self._finish, run_id, "interrupted", "", exclusive)
            except LeaseLostError:
                status = "lease_lost"
                log.warning("lease lost; another process now owns the run")
                raise
            except (asyncio.CancelledError, KeyboardInterrupt):
                status = "interrupted"
                await asyncio.shield(
                    asyncio.to_thread(self._finish, run_id, "interrupted", "", exclusive)
                )
                if not control.cancelled:
                    raise
                # Cancelled on request: that is an outcome, not an interruption.
                if control.task is not None:
                    control.task.uncancel()
                status = "cancelled"
                log.info("run cancelled")
                return ExecutionReport(run_id, "cancelled", 0, 0, 0, 0)
            except Exception as exc:
                message = clean_error(f"{type(exc).__name__}: {exc}")
                await asyncio.to_thread(self._finish, run_id, "failed", message, exclusive)
                log.exception("run failed")
                raise
            finally:
                metrics.RUNS_ACTIVE.dec()
                metrics.RUNS_FINISHED.labels(status).inc()
            log.info("run stopped", extra=asdict(report))
            return report

    async def _execute_claimed(
        self, run: Run, exclusive: bool, control: _Control
    ) -> ExecutionReport:
        ctx, done = await self._prepare(run)
        concurrency = ctx.config.options.concurrency
        results: asyncio.Queue[CaseResult | None] = asyncio.Queue(maxsize=WRITE_BATCH * 4)
        work: _WorkQueue = asyncio.Queue()
        unsaved: list[CaseResult] = []
        try:
            async with asyncio.TaskGroup() as group:
                writer = group.create_task(self._writer(ctx, results, unsaved))
                heartbeat = group.create_task(self._heartbeat(run.id, exclusive))
                feeder = group.create_task(
                    self._feed(ctx, run, exclusive, work, results, stop=control.draining)
                )
                await asyncio.gather(
                    *(
                        group.create_task(self._worker(ctx, work, results))
                        for _ in range(concurrency)
                    )
                )
                await feeder
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
            await ctx.target.aclose()
        status = await asyncio.to_thread(self._complete, run.id, exclusive)
        if status != "completed" and not control.draining.is_set():
            raise EvalkitError(f"run {run.id} still has unfinished cases")
        counts = ctx.counts
        return ExecutionReport(
            run.id,
            "completed" if status == "completed" else "interrupted",
            counts["executed"],
            counts["cached"],
            counts["errors"],
            len(done),
            counts["batched"],
            counts["refused"],
        )

    async def _worker(
        self, ctx: _Context, work: _WorkQueue, results: asyncio.Queue[CaseResult | None]
    ) -> None:
        counts = ctx.counts
        while True:
            item = await work.get()
            if work.qsize() <= ctx.low_water:
                ctx.drained.set()
            if item is None:
                return
            case, fence = item
            hit = ctx.hits.get(ctx.keys[case.id])
            if hit is not None:
                result = await self._from_cache(case, hit, ctx.scorers)
                counts["cached"] += 1
            else:
                result = await self._run_case(ctx, case, ctx.keys[case.id])
                counts["executed"] += 1
            result.fence = fence
            counts["errors"] += result.status == "error"
            _record(result, "cache" if result.cached else "target", ctx.target.provider)
            await results.put(result)

    async def _feed(
        self,
        ctx: _Context,
        run: Run,
        exclusive: bool,
        work: _WorkQueue,
        results: asyncio.Queue[CaseResult | None],
        *,
        stop: asyncio.Event,
    ) -> None:
        """Claim cases and queue them for the workers until none is left, or until
        ``stop`` is set (drain), when the cases no worker has started go back."""
        if exclusive:
            # One process holds the run, so it takes every case in one claim.
            claims = await asyncio.to_thread(self._claim_cases, run.id, len(ctx.pending) + 1)
            fences = {c.case_id: c.fence for c in claims}
            claimed = [ctx.pending[c.case_id] for c in claims]
            uncached = [c for c in claimed if ctx.keys[c.id] not in ctx.hits]
            handled, failed = await self._run_batch(
                ctx, run.batch_id, uncached, fences=fences, results=results
            )
            ctx.counts["batched"] = len(handled) - failed
            ctx.counts["errors"] += failed
            for case in claimed:
                if case.id not in handled:
                    work.put_nowait((case, fences[case.id]))
        else:
            size = claim_size(ctx.config.options.concurrency)
            while not stop.is_set():
                claims = await asyncio.to_thread(self._claim_cases, run.id, size)
                for c in claims:
                    work.put_nowait((ctx.pending[c.case_id], c.fence))
                if claims:
                    # Claim again once the queue is half drained: the next claim's
                    # round trip overlaps the rest, and a process never holds
                    # much more than one claim's worth of queued cases.
                    ctx.drained.clear()
                    if work.qsize() > ctx.low_water:
                        await wait_for_any(ctx.drained, stop)
                    continue
                ctx.written.clear()
                left = await asyncio.to_thread(self._outstanding, run.id)
                if left.total:
                    await wait_for_any(None, stop, self._recheck_after(left))
                elif left.mine:
                    # Only this process's own cases are unfinished. Keep claiming
                    # until they are written: if one was taken over by a process
                    # that then died, it comes back here once that lease expires.
                    await wait_for_any(ctx.written, stop, self.heartbeat_seconds)
                else:
                    break
            if stop.is_set():
                unstarted = []
                while not work.empty():
                    item = work.get_nowait()
                    if item is not None:
                        unstarted.append(item[0].id)
                await asyncio.to_thread(self._release_cases, run.id, unstarted)
                log.info("draining", extra={"returned": len(unstarted)})
        for _ in range(ctx.config.options.concurrency):
            work.put_nowait(None)

    async def _prepare(self, run: Run) -> tuple[_Context, set[str]]:
        config = StoredRun.from_row(run)
        cases, done = await asyncio.to_thread(self._load_cases, run, config)
        pending = {c.id: c for c in cases if c.id not in done}
        scorers = build_scorers(
            config.scorers, allow_import_paths=self.allow_import_paths, claude=self.claude
        )
        if config.options.use_cache:
            for scorer in scorers:
                if isinstance(scorer, LLMJudge):
                    scorer.cache = cache.VerdictStore(self.sessions)
        for scorer in scorers:
            if isinstance(scorer, LLMJudge):
                scorer.limiter = self._limiter("anthropic:judge", RateLimit())
        target = self._build_target(config)
        fingerprint = self.settings.redaction.fingerprint()
        keys = {
            c.id: cache.cache_key(run.target_hash, c.input, fingerprint) for c in pending.values()
        }
        hits: dict[str, Any] = {}
        if config.options.use_cache and pending:
            hits = await asyncio.to_thread(self._cache_lookup, list(keys.values()))
        log.info(
            "run started",
            extra={"cases": len(cases), "already_done": len(done), "cache_hits": len(hits)},
        )
        ctx = _Context(
            run_id=run.id,
            target_hash=run.target_hash,
            config=config,
            target=target,
            scorers=scorers,
            limiter=self._limiter(target.provider, config.options.rate_limit),
            rng=random.Random(run.seed),  # noqa: S311 - retry jitter
            pending=pending,
            keys=keys,
            hits=hits,
            low_water=claim_size(config.options.concurrency) // 2,
        )
        return ctx, done

    def _limiter(self, provider: str, limit: RateLimit) -> RateLimiter:
        if self.shared_limits:
            return PgRateLimiter(self.sessions, provider, limit, self.clock)
        return Limiters({provider: limit}, self.clock).get(provider)

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

    async def _run_batch(
        self,
        ctx: _Context,
        batch_id: str | None,
        cases: list[Case],
        *,
        fences: dict[str, int],
        results: asyncio.Queue[CaseResult | None],
    ) -> tuple[set[str], int]:
        """Send ``cases`` through the Message Batches API.

        Returns the ids that need no online call (answered, or failed before
        submission) and how many of those failed.

        The batch id is saved on the run as soon as the batch exists, so a resumed
        run polls the same batch instead of paying for a second one. Entries that
        errored, expired or were cancelled are left to the online workers.
        """
        target = ctx.target
        if not isinstance(target, PromptTarget):
            raise InvalidInputError("batch mode is only available for prompt targets")
        client, spec = target.client, target.spec
        handled: set[str] = set()
        failed = 0
        by_custom_id = {batch_custom_id(c.id): c for c in cases}

        async def retried(call: Callable[[], Any]) -> Any:
            return await call_with_retries(
                call, limiter=ctx.limiter, policy=ctx.config.options.retry, stats=RetryStats()
            )

        if batch_id is None:
            requests = []
            for custom_id, case in by_custom_id.items():
                try:
                    prompt = target.prompt_for(case)
                except InvalidInputError as exc:
                    failure = self._error(case, exc.message, RetryStats(), time.perf_counter())
                    failure.fence = fences[case.id]
                    await results.put(failure)
                    handled.add(case.id)
                    failed += 1
                    continue
                params = client.request_params(
                    model=spec.model,
                    prompt=prompt,
                    system=spec.system,
                    max_tokens=spec.max_tokens,
                    effort=spec.effort,
                )
                requests.append({"custom_id": custom_id, "params": params})
            if not requests:
                return handled, failed
            batch = await retried(lambda: client.create_batch(requests))
            batch_id = str(batch.id)
            await asyncio.to_thread(self._set_batch_id, ctx.run_id, batch_id)
            log.info("batch submitted", extra={"batch_id": batch_id, "requests": len(requests)})
        while True:
            batch = await retried(lambda: client.get_batch(batch_id))
            if batch.processing_status == "ended":
                break
            await self.clock.sleep(ctx.config.options.batch_poll_seconds)
        entries = await retried(lambda: client.batch_results(batch_id))
        for entry in entries:
            answered = by_custom_id.get(entry.custom_id)
            if answered is None or answered.id in handled or entry.result.type != "succeeded":
                continue
            completion = client.completion_from_message(
                entry.result.message, requested_model=spec.model, batch=True
            )
            generation = Generation(completion.text, completion.stop_reason, completion.usage)
            result = await self._finish_case(
                ctx, answered, ctx.keys[answered.id], generation, latency_ms=0.0, attempts=1
            )
            result.fence = fences[answered.id]
            _record(result, "batch", target.provider)
            await results.put(result)
            handled.add(answered.id)
        log.info("batch results saved", extra={"batch_id": batch_id, "answered": len(handled)})
        return handled, failed

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
        self, ctx: _Context, queue: asyncio.Queue[CaseResult | None], batch: list[CaseResult]
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
                refused = await asyncio.to_thread(self._write, ctx.run_id, list(batch))
                ctx.counts["refused"] += refused
                batch.clear()
                ctx.written.set()

    async def _heartbeat(self, run_id: str, exclusive: bool) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_seconds)
            renewed, status = await asyncio.to_thread(self._renew, run_id, exclusive)
            if status in STOPPED:
                # Cancelled from elsewhere (the API, another host): this is how a
                # process without a NOTIFY listener finds out.
                self.request_cancel(run_id)
                return
            if not renewed:
                raise LeaseLostError(f"run {run_id} is now owned by another process")

    def _start(self, run_id: str) -> tuple[Run, bool]:
        """Join the run, or claim it outright in batch mode. Returns the run and
        whether this process holds it exclusively."""
        with self.sessions() as session:
            run = session.get(Run, run_id)
            if run is None:
                raise NotFoundError(f"run {run_id} not found")
            exclusive = run.options.get("mode") == "batch"
        return (self._claim(run_id) if exclusive else self._join(run_id)), exclusive

    def _join(self, run_id: str) -> Run:
        now = utcnow()
        with self.sessions() as session:
            session.execute(
                update(Run)
                .where(Run.id == run_id, Run.status.in_(RESUMABLE))
                .values(status="running", error="", started_at=func.coalesce(Run.started_at, now))
                .execution_options(synchronize_session=False)
            )
            session.commit()
            return session.get_one(Run, run_id)

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
            run = session.get_one(Run, run_id)
            if claimed.rowcount == 0 and run.status != "completed":  # type: ignore[attr-defined]
                raise ConflictError(f"run {run_id} is being executed by another process")
            if run.started_at is None:
                run.started_at = now
                session.commit()
            return run

    def _claim_cases(self, run_id: str, limit: int) -> list[leases.Claim]:
        with self._write_turn, self.sessions() as session:
            claims = leases.claim(
                session, run_id, self.owner, limit, ttl=self.lease_ttl, now=utcnow()
            )
            session.commit()
            return claims

    def _release_cases(self, run_id: str, case_ids: list[str]) -> None:
        with self._write_turn, self.sessions() as session:
            leases.release(session, run_id, self.owner, case_ids)
            session.commit()

    def _outstanding(self, run_id: str) -> leases.Outstanding:
        with self.sessions() as session:
            return leases.outstanding(session, run_id, self.owner)

    def _recheck_after(self, left: leases.Outstanding) -> float:
        """Nothing was claimable but other processes' cases are unfinished: how
        long to wait before trying again."""
        if left.unowned or left.next_expiry is None:
            return RECHECK_SECONDS
        # Another worker holds cases: look again when the first lease could have
        # expired (its holder died), or after one heartbeat if it keeps renewing.
        until = (as_utc(left.next_expiry) - utcnow()).total_seconds()
        return min(max(until, RECHECK_SECONDS), self.heartbeat_seconds)

    def _set_batch_id(self, run_id: str, batch_id: str) -> None:
        with self.sessions() as session:
            session.execute(
                update(Run)
                .where(Run.id == run_id)
                .values(batch_id=batch_id)
                .execution_options(synchronize_session=False)
            )
            session.commit()

    def _renew(self, run_id: str, exclusive: bool) -> tuple[bool, str]:
        """Extend this process's leases; returns whether it still holds the run
        and the run's status, so a cancellation is noticed within one heartbeat."""
        now = utcnow()
        with self.sessions() as session:
            leases.renew(session, run_id, self.owner, ttl=self.lease_ttl, now=now)
            renewed = True
            if exclusive:
                result = session.execute(
                    update(Run)
                    .where(Run.id == run_id, Run.lease_owner == self.owner)
                    .values(lease_expires_at=now + self.lease_ttl)
                    .execution_options(synchronize_session=False)
                )
                renewed = bool(result.rowcount)  # type: ignore[attr-defined]
            session.commit()
            status = session.scalars(select(Run.status).where(Run.id == run_id)).one()
            return renewed, status

    def _complete(self, run_id: str, exclusive: bool = False) -> str:
        """Mark the run completed if no case is left unfinished, and return its
        status. The condition and the update are one statement, so when several
        processes finish together exactly one of them completes the run."""
        with self.sessions() as session:
            unfinished = exists().where(CaseLease.run_id == run_id, CaseLease.done == false())
            completed = session.execute(
                update(Run)
                .where(
                    Run.id == run_id,
                    Run.status.not_in(runqueue.FINAL),
                    self._holds_run(exclusive),
                    ~unfinished,
                )
                .values(
                    status="completed",
                    error="",
                    finished_at=utcnow(),
                    lease_owner=None,
                    lease_expires_at=None,
                )
                .execution_options(synchronize_session=False)
            )
            session.commit()
            if completed.rowcount:  # type: ignore[attr-defined]
                # This process completed the run: write its score arrays now, so the
                # first summary or comparison does not have to. A failure here is
                # not the run's failure; the next reader compacts it instead.
                try:
                    aggregates.compact(session, session.get_one(Run, run_id))
                except Exception:
                    session.rollback()
                    log.exception("compacting run scores failed", extra={"run_id": run_id})
            return session.scalars(select(Run.status).where(Run.id == run_id)).one()

    def _holds_run(self, exclusive: bool) -> ColumnElement[bool]:
        # Only a batch-mode run has a run-level lease. An online run ignores the
        # column, so a lease left by a process from before case leases cannot
        # keep the run from finishing.
        return Run.lease_owner == self.owner if exclusive else true()

    def _finish(self, run_id: str, status: str, error: str, exclusive: bool = False) -> str:
        """Stop this process's part in the run: give its unfinished cases back and,
        unless another process is still working on the run, record ``status`` (or
        ``cancelled``, if the run was being cancelled). Returns the run's status."""
        now = utcnow()
        with self.sessions() as session:
            leases.release(session, run_id, self.owner)
            others_working = exists().where(
                CaseLease.run_id == run_id,
                CaseLease.done == false(),
                CaseLease.owner.is_not(None),
                CaseLease.expires_at >= now,
            )
            cancelling = Run.status.in_(STOPPED)
            session.execute(
                update(Run)
                .where(
                    Run.id == run_id,
                    Run.status.not_in(runqueue.FINAL),
                    self._holds_run(exclusive),
                    ~others_working,
                )
                .values(
                    status=case((cancelling, "cancelled"), else_=status),
                    error=case((cancelling, ""), else_=error),
                    finished_at=case((cancelling, now), else_=Run.finished_at),
                    lease_owner=None,
                    lease_expires_at=None,
                )
                .execution_options(synchronize_session=False)
            )
            session.commit()
            return session.scalars(select(Run.status).where(Run.id == run_id)).one()

    def _load_cases(self, run: Run, config: StoredRun) -> tuple[list[Case], set[str]]:
        with self.sessions() as session:
            dataset = session.get_one(Dataset, run.dataset_id)
            cases = load_cases(session, dataset, config.filter.split, config.filter.tags)
            done = set(session.scalars(select(Result.case_id).where(Result.run_id == run.id)))
            if leases.seed(session, run.id, [c.id for c in cases if c.id not in done]):
                session.commit()
            return cases, done

    def _cache_lookup(self, keys: list[str]) -> dict[str, Any]:
        with self.sessions() as session:
            return cache.lookup(session, keys)

    def _write(self, run_id: str, batch: list[CaseResult]) -> int:
        """Save results whose lease this process still holds, in the transaction
        that marks those leases done and adds them to the run's aggregates. Returns
        how many were refused because the case had been taken over by another
        process."""
        names = self._metric_names(run_id)
        entries = [r.cache_entry for r in batch if r.cache_entry is not None]
        with self._write_turn, self.sessions() as session:
            try:
                accepted = leases.finish(
                    session, run_id, self.owner, [(r.case_id, r.fence) for r in batch]
                )
                saved = [r for r in batch if r.case_id in accepted]
                session.add_all(Result(**r.row(run_id)) for r in saved)
                aggregates.add(session, run_id, self.owner, aggregates.fold(saved, names))
                cache.store(session, entries)
                session.commit()
            except IntegrityError:
                # A result for one of these cases already exists (written outside
                # the lease protocol). Save the rest one by one and keep stored rows.
                session.rollback()
                accepted = {r.case_id for r in batch if self._write_one(session, run_id, names, r)}
                cache.store(session, entries)
                session.commit()
        refused = len(batch) - len(accepted)
        if refused:
            metrics.RESULTS_REFUSED.inc(refused)
            log.warning(
                "results refused: another process took the cases over", extra={"refused": refused}
            )
        return refused

    def _metric_names(self, run_id: str) -> list[str]:
        names = self._metrics.get(run_id)
        if names is None:
            with self.sessions() as session:
                names = aggregates.metric_names(session.get_one(Run, run_id))
            self._metrics[run_id] = names
        return names

    def _write_one(
        self, session: Session, run_id: str, names: list[str], result: CaseResult
    ) -> bool:
        claim = [(result.case_id, result.fence)]
        if not leases.finish(session, run_id, self.owner, claim):
            session.rollback()
            return False
        try:
            session.add(Result(**result.row(run_id)))
            session.flush()
            aggregates.add(session, run_id, self.owner, aggregates.fold([result], names))
            session.commit()
        except IntegrityError:
            session.rollback()
            # The case has a result either way, so its lease is finished.
            leases.finish(session, run_id, self.owner, claim)
            session.commit()
            log.warning("duplicate result skipped", extra={"case_id": result.case_id})
            return False
        return True


async def wait_for_any(
    event: asyncio.Event | None, stop: asyncio.Event, within: float | None = None
) -> None:
    """Wait for ``event`` (or only the time limit, when it is None), ``stop`` or
    ``within`` seconds, whichever comes first."""
    waiters = [asyncio.ensure_future(stop.wait())]
    if event is not None:
        waiters.append(asyncio.ensure_future(event.wait()))
    try:
        await asyncio.wait(waiters, timeout=within, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for waiter in waiters:
            waiter.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)


def _dialect(sessions: sessionmaker[Session]) -> str | None:
    bind = sessions.kw.get("bind")
    return None if bind is None else str(bind.dialect.name)


def _record(result: CaseResult, source: str, provider: str) -> None:
    metrics.CASES.labels(result.status, source).inc()
    if result.attempts:
        metrics.TARGET_ATTEMPTS.labels(provider).inc(result.attempts)
    metrics.TOKENS.labels("input").inc(result.input_tokens + result.judge_input_tokens)
    metrics.TOKENS.labels("output").inc(result.output_tokens + result.judge_output_tokens)
    metrics.TOKENS.labels("cache_read").inc(result.cache_read_tokens)
    metrics.TOKENS.labels("cache_write").inc(result.cache_write_tokens)
    metrics.COST.labels("target").inc(result.cost_usd)
    metrics.COST.labels("judge").inc(result.judge_cost_usd)


def _usage_fields(usage: Usage) -> dict[str, Any]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "cost_usd": usage.cost_usd,
    }
