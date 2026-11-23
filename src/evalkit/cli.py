"""Command line: run evaluations, compare runs and gate CI on regressions.

Exit codes are part of the interface, so CI can tell a regression from a
broken job:

    0  success; for ``gate``, no regression
    1  ``gate`` found a regression; ``agent conformance`` found a failing check
    2  bad input: config, arguments, unknown dataset or run
    3  the run cannot be used: it failed, is incomplete, is being executed by
       another process, or cannot be paired; also any unexpected error
    130 interrupted (the run is saved and can be resumed with --resume)

Configuration comes from the environment (see ``evalkit.config``); the
database URL can also be given with --database-url. SQLite databases are
migrated automatically; for PostgreSQL run ``evalkit db upgrade`` once.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import signal
import sys
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TextIO

from pydantic import TypeAdapter, ValidationError
from sqlalchemy import select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from evalkit import __version__, aggregates, blobs, cache, runqueue
from evalkit.api import serve
from evalkit.cache import VerdictStore
from evalkit.config import (
    ConfigError,
    GateRules,
    Settings,
    load_gate_rules,
    parse_duration,
    parse_size,
)
from evalkit.conformance import AgentConformance, ConformanceReport
from evalkit.dataset_edits import (
    DatasetEdit,
    apply_edit,
    dataset_log,
    diff_versions,
    lineage,
)
from evalkit.datasets import (
    get_dataset,
    import_dataset,
    list_datasets,
    load_cases,
    parse_jsonl,
    to_jsonl,
)
from evalkit.db import make_engine, make_session_factory, upgrade
from evalkit.db.models import Matrix, Run, utcnow
from evalkit.embeddings import EmbeddingStore
from evalkit.errors import ConflictError, EvalkitError, NotFoundError, PreconditionError
from evalkit.estimate import MatrixEstimate, estimate, estimate_matrix
from evalkit.frontier import MatrixReport, matrix_report
from evalkit.generate import (
    ApprovalResult,
    GenerateSpec,
    GenerationInfo,
    GenerationReport,
    approve,
    generate,
    generation_info,
    generator_from_spec,
    list_drafts,
    list_generations,
    load_documents,
    reject,
)
from evalkit.logs import configure_logging, scrubber
from evalkit.matrix import child_runs, resolve_matrix
from evalkit.prompts import (
    LabelMove,
    diff,
    get_prompt,
    labels_of,
    list_prompts,
    move_label,
    parse_ref,
    push_version,
    resolve,
)
from evalkit.providers.anthropic import ClaudeClient
from evalkit.providers.registry import ANTHROPIC, ProviderRegistry
from evalkit.rag_report import RagReport, format_rag, parse_thresholds, rag_report
from evalkit.report import CompareReport, RunSummary, compare_runs, summarize
from evalkit.runner import Runner
from evalkit.runspec import RunConfig
from evalkit.scorers.calibration import (
    DEFAULT_FOLDS,
    ThresholdReport,
    calibrate,
    calibrate_threshold,
    label_mode,
    parse_labels,
    save_calibration,
    save_threshold,
)
from evalkit.scorers.registry import (
    AnyJudgeSpec,
    SemanticSimilaritySpec,
    judge_from_spec,
    similarity_from_spec,
)
from evalkit.slices import SliceReport
from evalkit.targets.base import TargetError
from evalkit.targets.http import HttpTarget
from evalkit.targets.specs import HttpTargetSpec
from evalkit.tools.sandbox import (
    Fixtures,
    RecordReport,
    fixtures_path,
    load_sandbox,
    parse_calls,
    record,
    sandbox_identity,
)
from evalkit.worker import Worker, listen_url

VERIFY = "verify"
EXIT_OK, EXIT_REGRESSION, EXIT_INPUT, EXIT_UNUSABLE, EXIT_INTERRUPTED = 0, 1, 2, 3, 130
MAX_CONFIG_BYTES = 1024 * 1024
JUDGE_SPEC: TypeAdapter[AnyJudgeSpec] = TypeAdapter(AnyJudgeSpec)
SIMILARITY_SPEC: TypeAdapter[SemanticSimilaritySpec] = TypeAdapter(SemanticSimilaritySpec)


class CliError(Exception):
    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class App:
    """Settings, database and on-demand provider clients for one command."""

    def __init__(self, settings: Settings, out: TextIO) -> None:
        self.settings = settings
        self.out = out
        self.engine = make_engine(settings.database_url)
        if make_url(settings.database_url).get_backend_name() == "sqlite":
            upgrade(self.engine)
        self.sessions: sessionmaker[Session] = make_session_factory(self.engine)
        self._claude: ClaudeClient | None = None
        self._providers: ProviderRegistry | None = None

    def providers(self) -> ProviderRegistry:
        """Every configured provider. The Claude client is made only if a target or
        a judge uses it, so a run with no Claude model does not need ANTHROPIC_API_KEY."""
        if self._providers is None:
            self._providers = ProviderRegistry(
                self.settings.provider_configs(),
                self.settings.price_book(),
                claude_factory=self.claude,
            )
        return self._providers

    def claude(self) -> ClaudeClient:
        if self._claude is None:
            self._claude = ClaudeClient(
                self.settings.price_book().table(ANTHROPIC), self.settings.anthropic_api_key
            )
        return self._claude

    def runner(self, *, allow_import_paths: bool = True) -> Runner:
        return Runner(
            self.sessions,
            self.settings,
            providers=self.providers(),
            allow_import_paths=allow_import_paths,
        )

    async def aclose(self) -> None:
        if self._providers is not None:
            await self._providers.aclose()
        if self._claude is not None:
            await self._claude.aclose()
        self.engine.dispose()


def load_run_config(path: Path, app: App) -> RunConfig:
    """Read a TOML run config. ``[dataset] path = ...`` imports that file first,
    and an agent target's ``sandbox = "FILE.toml"`` is read in with its fixtures."""
    raw = _read_toml(path)
    target = raw.get("target")
    if isinstance(target, dict) and isinstance(target.get("sandbox"), str):
        inline_sandbox(target, path.parent)
    dataset = raw.get("dataset")
    if isinstance(dataset, dict) and "path" in dataset:
        data_path = (path.parent / str(dataset.pop("path"))).resolve()
        name = str(dataset.get("name", ""))
        with app.sessions() as session:
            ds, _ = import_dataset(session, name, parse_jsonl(_read_bytes(data_path)))
            dataset.setdefault("version", ds.version)
    try:
        return RunConfig.model_validate(raw)
    except ValidationError as exc:
        raise CliError(f"{path}: {_describe(exc)}", EXIT_INPUT) from None


def inline_sandbox(target: dict[str, Any], base: Path) -> None:
    """Replace ``target["sandbox"]``, a sandbox file relative to ``base``, by its
    table, and its fixture file by ``target["fixtures"]``: a run carries its
    sandbox, so any worker can answer its tool calls."""
    if "fixtures" in target:
        raise CliError("give fixtures in the sandbox file, not next to sandbox = FILE", EXIT_INPUT)
    spec, fixtures = load_sandbox(base / str(target["sandbox"]))
    target["sandbox"] = spec.model_dump(mode="json", exclude={"fixtures"})
    if fixtures is not None:
        target["fixtures"] = [json.loads(r.to_json()) for r in fixtures.records]


def resolve_run(session: Session, ref: str) -> str:
    """A run id, or ``label:NAME`` for the newest completed run with that label."""
    if ref.startswith("label:"):
        label = ref.removeprefix("label:")
        run_id = session.scalar(
            select(Run.id)
            .where(Run.label == label, Run.status == "completed")
            .order_by(Run.created_at.desc())
            .limit(1)
        )
        if run_id is None:
            raise NotFoundError(f"no completed run labelled {label!r}")
        return run_id
    return ref


async def cmd_run(args: argparse.Namespace, app: App) -> int:
    if args.config == VERIFY:
        return await cmd_verify(args, app)
    if args.run_id is not None:
        raise CliError(f"unexpected argument {args.run_id!r}", EXIT_INPUT)
    if args.repair:
        raise CliError("--repair belongs to: evalkit run verify RUN_ID", EXIT_INPUT)
    if args.resume:
        with app.sessions() as session:
            group = session.get(Matrix, args.resume)
            if group is not None:
                ids = [r.id for r in child_runs(session, group.id) if r.status != "completed"]
            elif session.get(Run, args.resume) is None:
                raise NotFoundError(f"run {args.resume} not found")
        if group is not None:
            return await _run_matrix(args, app, app.runner(), group.id, ids)
        run_id = args.resume
        runner = app.runner()
    else:
        if args.config is None:
            raise CliError("give a run config, or --resume RUN_ID", EXIT_INPUT)
        config = load_run_config(Path(args.config), app)
        if args.label is not None:
            config = config.model_copy(update={"label": args.label})
        # Worker daemons import scorers only from installed entry points.
        runner = app.runner(allow_import_paths=not args.queue)
        if config.matrix is not None:
            group, runs = runner.create_matrix(config, queued=args.queue)
            _err(f"matrix {group.id}: {len(runs)} runs {'queued' if args.queue else 'started'}")
            if args.queue:
                with app.sessions() as session:
                    report = matrix_report(session, group.id)
                _emit(app.out, report, args.json, _format_matrix)
                return EXIT_OK
            return await _run_matrix(args, app, runner, group.id, [r.id for r in runs])
        if args.queue:
            queued = runner.create_run(config, queued=True)
            _err(f"run {queued.id} queued")
            with app.sessions() as session:
                _emit(app.out, summarize(session, queued.id), args.json, _format_summary)
            return EXIT_OK
        run_id = runner.create_run(config).id
        _err(f"run {run_id} started")
    try:
        await runner.execute(run_id)
    except asyncio.CancelledError:
        _err(f"interrupted; resume with: evalkit run --resume {run_id}")
        return EXIT_INTERRUPTED
    with app.sessions() as session:
        summary = summarize(session, run_id)
    _emit(app.out, summary, args.json, _format_summary)
    return EXIT_OK if summary.status == "completed" else EXIT_UNUSABLE


async def _run_matrix(
    args: argparse.Namespace, app: App, runner: Runner, matrix_id: str, run_ids: list[str]
) -> int:
    """Execute a matrix's runs one after another, then print its report. Each run
    is an ordinary run, so an interrupted matrix resumes where it stopped."""
    for run_id in run_ids:
        try:
            await runner.execute(run_id)
        except asyncio.CancelledError:
            _err(f"interrupted; resume with: evalkit run --resume {matrix_id}")
            return EXIT_INTERRUPTED
    with app.sessions() as session:
        report = matrix_report(session, matrix_id)
    _emit(app.out, report, args.json, _format_matrix)
    return EXIT_OK if report.complete else EXIT_UNUSABLE


async def cmd_report(args: argparse.Namespace, app: App) -> int:
    """``evalkit report matrix MATRIX``: a matrix's variants and their frontier."""
    with app.sessions() as session:
        report = matrix_report(
            session,
            resolve_matrix(session, args.matrix),
            metric=args.metric,
            latency=args.latency,
            alpha=args.alpha,
            min_effect=args.min_effect,
        )
    _emit(app.out, report, args.json, _format_matrix)
    return EXIT_OK


async def cmd_report_rag(args: argparse.Namespace, app: App) -> int:
    """``evalkit report rag RUN``: each failing case of a RAG run and its cause."""
    default, per_metric = parse_thresholds(args.threshold or [])
    with app.sessions() as session:
        report = rag_report(
            session,
            resolve_run(session, args.run),
            threshold=default,
            thresholds=per_metric,
            limit=args.limit,
        )
    _emit(app.out, report, args.json, _format_rag)
    return EXIT_OK


async def cmd_verify(args: argparse.Namespace, app: App) -> int:
    """``evalkit run verify RUN_ID [--repair]``: recompute a run's aggregates and
    score arrays from its results and report any difference (exit 3)."""
    if args.run_id is None:
        raise CliError("give the run to verify: evalkit run verify RUN_ID", EXIT_INPUT)
    with app.sessions() as session:
        report = aggregates.verify(session, args.run_id, repair=args.repair)
    _emit(app.out, report, args.json, _format_verification)
    return EXIT_OK if report.ok or report.repaired else EXIT_UNUSABLE


async def cmd_cancel(args: argparse.Namespace, app: App) -> int:
    with app.sessions() as session:
        status = runqueue.cancel(session, args.run, utcnow())
        session.commit()
    app.out.write(f"run {args.run} {status}\n")
    return EXIT_OK


async def cmd_worker(args: argparse.Namespace, app: App) -> int:
    slots = app.settings.max_active_runs if args.runs is None else args.runs
    if slots < 1:
        raise CliError("a worker needs at least one run slot (--runs)", EXIT_INPUT)
    runner = app.runner(allow_import_paths=False)
    worker = Worker(
        app.sessions,
        runner,
        slots=slots,
        poll_seconds=app.settings.worker_poll_seconds,
        drain_seconds=app.settings.drain_seconds,
        listen=listen_url(app.settings.database_url),
    )
    stop = asyncio.Event()

    def on_signal() -> None:
        # The first signal drains; a second one stops waiting for cases in flight.
        if stop.is_set():
            worker.abort()
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, on_signal)
    _err(f"worker {worker.id} started with {slots} run slot(s)")
    try:
        await worker.serve(stop)
    finally:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)
    _err(f"worker {worker.id} stopped")
    return EXIT_OK


async def cmd_cache(args: argparse.Namespace, app: App) -> int:
    """``evalkit cache sweep`` bounds the response cache and collects blobs;
    ``evalkit cache scrub`` reads every blob and checks it against its digest."""
    settings = app.settings
    store = settings.open_blob_store()
    try:
        if args.cache_command == "scrub":
            if store is None:
                raise CliError("no blob store is configured (EVALKIT_BLOB_URL)", EXIT_INPUT)
            scrubbed = await asyncio.to_thread(blobs.scrub, store)
            _emit(app.out, scrubbed, args.json, _format_scrub)
            return EXIT_UNUSABLE if scrubbed.corrupt else EXIT_OK
        max_age = _option(args.max_age, parse_duration, "--max-age")
        max_bytes = _option(args.max_size, parse_size, "--max-size")
        grace = _option(args.blob_grace, parse_duration, "--blob-grace")
        if max_age is None:
            max_age = settings.cache_max_age_seconds
        if max_bytes is None:
            max_bytes = settings.cache_max_bytes
        if max_age is None and max_bytes is None and store is None:
            raise CliError(
                "nothing to sweep: give --max-age or --max-size "
                "(or set EVALKIT_CACHE_MAX_AGE, EVALKIT_CACHE_MAX_BYTES or EVALKIT_BLOB_URL)",
                EXIT_INPUT,
            )
        done = await asyncio.to_thread(
            cache.maintain,
            app.sessions,
            store,
            max_age=None if max_age is None else timedelta(seconds=max_age),
            max_bytes=max_bytes,
            grace=timedelta(seconds=settings.blob_grace_seconds if grace is None else grace),
            dry_run=args.dry_run,
        )
        if done.skipped:
            raise CliError("another process is sweeping the cache; try again later", EXIT_UNUSABLE)
        _emit(app.out, done, args.json, _format_maintenance)
        return EXIT_OK
    finally:
        if store is not None:
            store.close()


def _option[T](raw: str | None, parse: Callable[[str], T], flag: str) -> T | None:
    if raw is None:
        return None
    try:
        return parse(raw)
    except ValueError as exc:
        raise CliError(f"{flag}: {exc}", EXIT_INPUT) from None


def _format_maintenance(done: cache.Maintenance) -> str:
    s = done.sweep
    verb = "would remove" if s.dry_run else "removed"
    lines = [
        f"cache: {verb} {s.removed:,} entries ({_bytes(s.removed_bytes)}), "
        f"{s.kept:,} left ({_bytes(s.kept_bytes)})"
    ]
    if done.embeddings is not None and (done.embeddings.removed or done.embeddings.kept):
        e = done.embeddings
        lines.append(
            f"embeddings: {verb} {e.removed:,} ({_bytes(e.removed_bytes)}), "
            f"{e.kept:,} left ({_bytes(e.kept_bytes)})"
        )
    if done.blobs is not None:
        b = done.blobs
        lines.append(
            f"blobs: scanned {b.scanned:,}, {verb} {b.removed:,} ({_bytes(b.removed_bytes)}), "
            f"kept {b.referenced:,} referenced and {b.recent:,} within the grace period"
            + (
                f", removed {b.incomplete_removed} unfinished writes"
                if b.incomplete_removed
                else ""
            )
        )
    return "\n".join(lines) + "\n"


def _format_scrub(report: blobs.ScrubReport) -> str:
    lines = [f"blobs: checked {report.scanned:,}, corrupt {len(report.corrupt):,}"]
    lines.extend(f"  corrupt {digest}" for digest in report.corrupt)
    return "\n".join(lines) + "\n"


def _bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    value, unit = n / 1024, "KiB"
    for bigger in ("MiB", "GiB", "TiB"):
        if value < 1024:
            break
        value, unit = value / 1024, bigger
    return f"{value:.1f} {unit}"


async def cmd_show(args: argparse.Namespace, app: App) -> int:
    with app.sessions() as session:
        summary = summarize(session, resolve_run(session, args.run))
    _emit(app.out, summary, args.json, _format_summary)
    return EXIT_OK


async def cmd_compare(args: argparse.Namespace, app: App) -> int:
    report = _compare(args, app)
    _emit(app.out, report, args.json, _format_compare)
    return EXIT_OK


async def cmd_gate(args: argparse.Namespace, app: App) -> int:
    report = _compare(args, app)
    _emit(app.out, report, args.json, _format_compare)
    if not report.passed:
        if report.regressions:
            _err(f"regression in: {', '.join(report.regressions)}")
        for failure in report.slices.failures if report.slices else []:
            _err(failure)
        return EXIT_REGRESSION
    return EXIT_OK


def _compare(args: argparse.Namespace, app: App) -> CompareReport:
    rules = _gate_rules(args)
    with app.sessions() as session:
        return compare_runs(
            session,
            resolve_run(session, args.baseline),
            resolve_run(session, args.candidate),
            alpha=args.alpha,
            min_effect=args.min_effect,
            metrics=args.metric or None,
            allow_judge_change=args.allow_judge_change,
            slices=rules,
        )


def _gate_rules(args: argparse.Namespace) -> GateRules | None:
    """Slice rules from ``--rules FILE``, with the slice flags over them."""
    try:
        rules = load_gate_rules(args.rules) if args.rules else GateRules()
    except ConfigError as exc:
        raise CliError(str(exc), EXIT_INPUT) from None
    fields = rules.model_dump()
    if args.by is not None:
        fields["by"] = args.by
    if args.min_slice_size is not None:
        fields["min_slice_size"] = args.min_slice_size
    if args.require_slice:
        fields["require"] = [*fields["require"], *args.require_slice]
    if args.any_slice:
        fields["any_slice"] = True
    try:
        rules = GateRules.model_validate(fields)
    except ValidationError as exc:
        raise CliError(_describe(exc), EXIT_INPUT) from None
    return None if rules.by is None else rules


async def cmd_estimate(args: argparse.Namespace, app: App) -> int:
    config = load_run_config(Path(args.config), app)
    if config.matrix is not None:
        each = await estimate_matrix(
            app.sessions, app.settings, config, app.claude(), providers=app.providers()
        )
        _emit(app.out, each, args.json, _format_matrix_estimate)
        return EXIT_OK
    result = await estimate(
        app.sessions, app.settings, config, app.claude(), providers=app.providers()
    )
    _emit(app.out, result, args.json, _format_estimate)
    return EXIT_OK


async def cmd_calibrate(args: argparse.Namespace, app: App) -> int:
    if args.scorer is not None:
        return await _calibrate_threshold(args, app)
    raw = _read_toml(Path(args.judge))
    table = raw.get("judge")
    if not isinstance(table, dict):
        raise CliError(f"{args.judge} has no [judge] table", EXIT_INPUT)
    try:
        spec = JUDGE_SPEC.validate_python({"type": "judge", **table})
    except ValidationError as exc:
        raise CliError(f"{args.judge}: {_describe(exc)}", EXIT_INPUT) from None
    judge = judge_from_spec(spec, app.providers())
    items = parse_labels(_read_bytes(Path(args.labels)), label_mode(judge))
    report = await calibrate(judge, items, concurrency=args.concurrency)
    with app.sessions() as session:
        save_calibration(session, judge, report)
    _emit(app.out, report, args.json, _format_calibration)
    return EXIT_OK


async def _calibrate_threshold(args: argparse.Namespace, app: App) -> int:
    """``evalkit calibrate --scorer``: the semantic_similarity threshold that agrees
    best with 0/1 labels. Vectors go through the embedding cache, so a rerun on
    the same file is free."""
    raw = _read_toml(Path(args.scorer))
    table = raw.get("scorer")
    if not isinstance(table, dict):
        raise CliError(f"{args.scorer} has no [scorer] table", EXIT_INPUT)
    try:
        spec = SIMILARITY_SPEC.validate_python({"type": "semantic_similarity", **table})
    except ValidationError as exc:
        raise CliError(f"{args.scorer}: {_describe(exc)}", EXIT_INPUT) from None
    scorer = similarity_from_spec(spec, app.providers())
    scorer.embedder.store = EmbeddingStore(app.sessions)
    items = parse_labels(_read_bytes(Path(args.labels)), "binary")
    try:
        report = await calibrate_threshold(scorer, items, folds=args.folds)
    except TargetError as exc:
        raise CliError(f"embedding call failed: {exc.message}", EXIT_UNUSABLE) from None
    with app.sessions() as session:
        save_threshold(session, scorer, report)
    _emit(app.out, report, args.json, _format_threshold)
    return EXIT_OK


async def cmd_dataset(args: argparse.Namespace, app: App) -> int:
    with app.sessions() as session:
        if args.dataset_command == "import":
            cases = parse_jsonl(_read_bytes(Path(args.file)))
            ds, created = import_dataset(session, args.name, cases, args.description)
            verb = "imported" if created else "unchanged, already"
            app.out.write(f"{ds.name} {verb} version {ds.version} ({ds.case_count} cases)\n")
        elif args.dataset_command == "export":
            ds = get_dataset(session, args.name, args.version)
            text = to_jsonl(load_cases(session, ds))
            if args.output:
                await asyncio.to_thread(Path(args.output).write_text, text, encoding="utf-8")
            else:
                app.out.write(text)
        elif args.dataset_command == "edit":
            edit = _read_edit(Path(args.file), args.base, session, args.name)
            result = apply_edit(session, args.name, edit)
            ds = result.dataset
            if result.created:
                app.out.write(
                    f"{ds.name} edited: version {ds.version} ({ds.content_hash[:12]}), "
                    f"{_changes(ds.cases_added, ds.cases_changed, ds.cases_removed)}\n"
                )
            elif result.existing:
                app.out.write(
                    f"{ds.name} unchanged: the edit gives version {ds.version} "
                    f"({ds.content_hash[:12]}), which already exists\n"
                )
            else:
                app.out.write(f"{ds.name} unchanged: the edit changes nothing\n")
        elif args.dataset_command == "diff":
            found = diff_versions(session, args.name, args.old, args.new)
            _emit(app.out, found, args.json, lambda d: _format_dataset_diff(d, args.limit))
        elif args.dataset_command == "log":
            rows = (
                lineage(session, args.name, args.version)
                if args.version is not None
                else dataset_log(session, args.name)
            )
            _emit(app.out, rows, args.json, _format_dataset_log)
        else:
            for ds in list_datasets(session):
                app.out.write(
                    f"{ds.name}\tv{ds.version}\t{ds.case_count} cases\t{ds.content_hash[:12]}\n"
                )
    return EXIT_OK


async def cmd_generate(args: argparse.Namespace, app: App) -> int:
    """``evalkit generate CONFIG``: draft cases from seed documents and examples.
    Exits 3 when no call gave a usable reply."""
    path = Path(args.config)
    table = _read_toml(path).get("generate")
    if not isinstance(table, dict):
        raise CliError(f"{path} has no [generate] table", EXIT_INPUT)
    try:
        spec = GenerateSpec.model_validate(table)
    except ValidationError as exc:
        raise CliError(f"{path}: {_describe(exc)}", EXIT_INPUT) from None
    base = path.parent
    documents = (
        await asyncio.to_thread(load_documents, spec.documents, base) if spec.documents else []
    )
    examples = parse_jsonl(_read_bytes(base / spec.examples)) if spec.examples else []
    generator = generator_from_spec(spec, app.providers(), cache=VerdictStore(app.sessions))
    report = await generate(app.sessions, spec, generator, documents=documents, examples=examples)
    _emit(app.out, report, args.json, _format_generation)
    return (
        EXIT_UNUSABLE if report.calls == 0 or (report.errors and not report.generated) else EXIT_OK
    )


async def cmd_drafts(args: argparse.Namespace, app: App) -> int:
    """``evalkit drafts list|show|approve|reject``: review generated cases."""
    with app.sessions() as session:
        if args.drafts_command == "list":
            rows = list_generations(session, args.dataset)
            _emit(app.out, rows, args.json, _format_generations)
        elif args.drafts_command == "show":
            info = generation_info(session, args.generation)
            drafts = list_drafts(
                session, info.id, args.status, limit=args.limit, offset=args.offset
            )
            _emit(
                app.out,
                {"generation": info, "drafts": drafts},
                args.json,
                _format_drafts,
            )
        elif args.drafts_command == "approve":
            if not args.all and not args.id:
                raise CliError(
                    "name drafts with --id, or approve every pending one with --all", EXIT_INPUT
                )
            result = approve(
                session, args.generation, None if args.all else args.id, note=args.note
            )
            _emit(app.out, result, args.json, _format_approval)
        else:
            rejected = reject(session, args.generation, args.id, note=args.note)
            app.out.write(f"rejected {len(rejected)} drafts\n")
    return EXIT_OK


def _read_edit(path: Path, base: str | None, session: Session, name: str) -> DatasetEdit:
    """An edit file (JSON, the body of PATCH /v1/datasets/NAME). ``--base`` gives or
    overrides its base: a content hash, or a version number to take the hash of."""
    data = _read_bytes(path)
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise CliError(f"{path} is not valid JSON: {exc}", EXIT_INPUT) from None
    if not isinstance(raw, dict):
        raise CliError(f"{path} must hold a JSON object", EXIT_INPUT)
    if base is not None:
        version = re.fullmatch(r"v?(\d{1,9})", base)
        raw["base"] = (
            base if version is None else get_dataset(session, name, int(version[1])).content_hash
        )
    try:
        return DatasetEdit.model_validate(raw)
    except ValidationError as exc:
        raise CliError(f"{path}: {_describe(exc)}", EXIT_INPUT) from None


def _changes(added: int | None, changed: int | None, removed: int | None) -> str:
    if added is None:
        return "first version"
    return f"+{added} added, ~{changed} changed, -{removed} removed"


def _short(value: Any, width: int = 60) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= width else text[: width - 3] + "..."


def _format_field_change(c: Any) -> str:
    if c.kind == "added":
        return f"{c.path}: added {_short(c.after)}"
    if c.kind == "removed":
        return f"{c.path}: removed (was {_short(c.before)})"
    if c.path == "tags" and isinstance(c.before, list) and isinstance(c.after, list):
        plus = [f"+{t}" for t in c.after if t not in c.before]
        minus = [f"-{t}" for t in c.before if t not in c.after]
        if plus or minus:
            return "tags: " + " ".join(plus + minus)
        return "tags: reordered"
    return f"{c.path}: {_short(c.before)} -> {_short(c.after)}"


def _format_dataset_diff(d: Any, limit: int) -> str:
    lines = [
        f"{d.name} v{d.from_version} ({d.from_hash[:12]}) -> v{d.to_version} "
        f"({d.to_hash[:12]}): {len(d.added)} added, {len(d.changed)} changed, "
        f"{len(d.removed)} removed, {d.unchanged} unchanged"
    ]
    entries = (
        [f"+ {case_id}" for case_id in d.added]
        + [f"- {case_id}" for case_id in d.removed]
        + [
            f"~ {c.id}" + "".join(f"\n    {_format_field_change(f)}" for f in c.changes)
            for c in d.changed
        ]
    )
    lines += entries[:limit]
    if len(entries) > limit:
        lines.append(f"... and {len(entries) - limit} more (--limit, or --json for all)")
    return "\n".join(lines) + "\n"


def _format_dataset_log(rows: list[Any]) -> str:
    lines = []
    for r in rows:
        parent = "-" if r.parent_version is None else f"v{r.parent_version}"
        note = f"  {r.note}" if r.note else ""
        lines.append(
            f"v{r.version}\t{r.content_hash[:12]}\t{r.origin}\tfrom {parent}\t"
            f"{r.case_count} cases\t{_changes(r.added, r.changed, r.removed)}\t"
            f"{r.created_at.isoformat(timespec='seconds')}{note}"
        )
    return "\n".join(lines) + "\n"


async def cmd_prompt(args: argparse.Namespace, app: App) -> int:
    """``evalkit prompt push|show|list|diff|label``: the prompt registry."""
    with app.sessions() as session:
        if args.prompt_command == "push":
            system, template = _read_prompt(Path(args.file), args.system)
            version, created = push_version(
                session,
                args.name,
                system=system,
                template=template,
                note=args.note,
                description=args.description,
            )
            verb = "pushed" if created else "unchanged, already"
            app.out.write(
                f"{args.name} {verb} version {version.version} ({version.content_hash[:12]})\n"
            )
        elif args.prompt_command == "show":
            found = resolve(session, args.ref)
            prompt = get_prompt(session, found.name)
            info = {
                "name": found.name,
                "version": found.version.version,
                "label": found.label,
                "content_hash": found.version.content_hash,
                "note": found.version.note,
                "created_at": found.version.created_at,
                "labels": labels_of(session, prompt),
                "system": found.version.system,
                "template": found.version.template,
            }
            _emit(app.out, info, args.json, _format_prompt)
        elif args.prompt_command == "list":
            _emit(app.out, list_prompts(session), args.json, _format_prompt_list)
        elif args.prompt_command == "diff":
            new_ref = args.new or f"{parse_ref(args.old)[0]}@latest"
            app.out.write(diff(resolve(session, args.old), resolve(session, new_ref)))
        else:
            name, _ = parse_ref(args.ref)
            if "@" not in args.ref:
                raise CliError(
                    f"name the version to label: {args.ref}@VERSION or {args.ref}@LABEL",
                    EXIT_INPUT,
                )
            target = resolve(session, args.ref).version.version
            if args.expect is None:
                # Compare-and-set against what the label is at now: a move made
                # by someone else between this read and the write is refused.
                expected = labels_of(session, get_prompt(session, name)).get(args.label)
            elif args.expect == "none":
                expected = None
            elif args.expect.isdigit():
                expected = int(args.expect)
            else:
                raise CliError("--expect takes a version number or 'none'", EXIT_INPUT)
            moved = move_label(session, name, args.label, target, expected=expected)
            _emit(app.out, moved, args.json, _format_label_move)
    return EXIT_OK


def _read_prompt(path: Path, system_path: str | None) -> tuple[str, str]:
    """A prompt file: TOML with ``template`` and an optional ``system``, or any
    other file as the template itself, with ``--system FILE`` beside it."""
    if path.suffix == ".toml":
        if system_path is not None:
            raise CliError("a .toml prompt file holds its own system prompt", EXIT_INPUT)
        raw = _read_toml(path)
        unknown = sorted(set(raw) - {"system", "template"})
        if unknown:
            raise CliError(f"{path}: unknown key(s) {', '.join(unknown)}", EXIT_INPUT)
        system, template = raw.get("system", ""), raw.get("template", "")
        if not isinstance(system, str) or not isinstance(template, str):
            raise CliError(f"{path}: system and template must be strings", EXIT_INPUT)
        return system, template
    system = "" if system_path is None else _read_text(Path(system_path))
    return system, _read_text(path)


def _read_text(path: Path) -> str:
    data = _read_bytes(path)
    if len(data) > MAX_CONFIG_BYTES:
        raise CliError(f"{path} is larger than {MAX_CONFIG_BYTES} bytes", EXIT_INPUT)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise CliError(f"{path} is not UTF-8", EXIT_INPUT) from None


async def cmd_db(args: argparse.Namespace, app: App) -> int:
    upgrade(app.engine)
    app.out.write("database is at the latest schema\n")
    return EXIT_OK


async def cmd_tools(args: argparse.Namespace, settings: Settings, out: TextIO) -> int:
    """``evalkit tools check|record``: tool sandboxes and their fixtures. Needs no database."""
    sandbox_file = Path(args.sandbox)
    if args.tools_command == "check":
        spec, fixtures = load_sandbox(sandbox_file)
        info = {
            "tools": [t.name for t in spec.tools],
            "fixtures": 0 if fixtures is None else len(fixtures),
            "identity": sandbox_identity(spec, fixtures),
        }
        _emit(out, info, args.json, _format_sandbox)
        return EXIT_OK
    spec, _ = load_sandbox(sandbox_file, fixtures=False)
    calls = parse_calls(_read_bytes(Path(args.calls)), spec)
    output = Path(args.output) if args.output else fixtures_path(sandbox_file, spec)
    existing = Fixtures.parse(_read_bytes(output), spec) if output.exists() else None
    targets: dict[str, HttpTarget] = {}
    for tool in spec.tools:
        if tool.endpoint is None:
            continue
        endpoint = settings.http_endpoints.get(tool.endpoint)
        if endpoint is None:
            raise CliError(
                f"tool {tool.name} records from endpoint {tool.endpoint!r}, which is not "
                "configured in EVALKIT_HTTP_TARGETS_FILE",
                EXIT_INPUT,
            )
        targets[tool.name] = HttpTarget(
            HttpTargetSpec(type="http", endpoint=tool.endpoint), endpoint
        )

    def backend(target: HttpTarget) -> Callable[[str, Any], Any]:
        async def send(name: str, arguments: Any) -> Any:
            return await target.post({"tool": name, "arguments": arguments})

        return send

    try:
        fixtures, report = await record(
            spec,
            calls,
            {name: backend(t) for name, t in targets.items()},
            existing,
            refresh=args.refresh,
        )
    finally:
        for target in targets.values():
            await target.aclose()
    # Write what was recorded even when some calls failed, so a rerun only makes those.
    tmp = output.with_name(output.name + ".tmp")
    await asyncio.to_thread(tmp.write_text, fixtures.to_jsonl(), encoding="utf-8")
    await asyncio.to_thread(tmp.replace, output)
    _emit(out, report, args.json, _format_record)
    return EXIT_UNUSABLE if report.failed else EXIT_OK


async def cmd_agent(args: argparse.Namespace, settings: Settings, out: TextIO) -> int:
    """``evalkit agent conformance ENDPOINT``: probe an agent on the session protocol
    (``evalkit.conformance``). Exits 1 when a check fails. Needs no database."""
    endpoint = settings.http_endpoints.get(args.endpoint)
    if endpoint is None:
        raise CliError(
            f"endpoint {args.endpoint!r} is not configured in EVALKIT_HTTP_TARGETS_FILE",
            EXIT_INPUT,
        )
    sandbox_file = Path(args.sandbox)
    spec, fixtures = load_sandbox(sandbox_file)
    tasks: list[str] = list(args.task or [])
    if not tasks:
        table = _read_toml(sandbox_file).get("conformance", {})
        listed = table.get("tasks") if isinstance(table, dict) else None
        if not isinstance(listed, list) or not all(isinstance(t, str) and t for t in listed):
            raise CliError(
                f"give --task, or tasks = [...] in the [conformance] table of {sandbox_file}",
                EXIT_INPUT,
            )
        tasks = listed
    try:
        checker = AgentConformance(
            args.endpoint,
            endpoint,
            spec,
            tasks,
            fixtures=fixtures,
            max_steps=args.max_steps,
        )
    except ValueError as exc:
        raise CliError(str(exc), EXIT_INPUT) from None
    report = await checker.run()
    if args.json:
        out.write(json.dumps(report.to_dict(), indent=2) + "\n")
    else:
        out.write(_format_conformance(report))
    return EXIT_OK if report.passed(strict=args.strict) else EXIT_REGRESSION


def cmd_serve(settings: Settings) -> int:
    serve(settings)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalkit", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"evalkit {__version__}")
    parser.add_argument("--database-url", help="overrides EVALKIT_DATABASE_URL")
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress to stderr")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_json(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--json", action="store_true", help="print JSON instead of a table")
        return p

    run = with_json(
        sub.add_parser(
            "run",
            help="run an evaluation from a TOML config, or check a run (run verify RUN_ID)",
        )
    )
    run.add_argument("config", nargs="?", help="run config (TOML), or 'verify'")
    run.add_argument("run_id", nargs="?", help="with 'verify': the run to check")
    run.add_argument(
        "--repair", action="store_true", help="with 'verify': rewrite what differs from results"
    )
    run.add_argument("--label", help="label the run, e.g. a branch or commit")
    run.add_argument(
        "--resume", metavar="RUN_ID", help="continue an interrupted run, or a matrix's runs"
    )
    run.add_argument(
        "--queue", action="store_true", help="queue the run for worker daemons and exit"
    )
    run.set_defaults(handler=cmd_run)

    cancel = sub.add_parser("cancel", help="cancel a queued or running run")
    cancel.add_argument("run", help="run id")
    cancel.set_defaults(handler=cmd_cancel)

    work = sub.add_parser("worker", help="execute queued runs until SIGTERM, then drain")
    work.add_argument(
        "--runs", type=int, help="runs to execute at once (default EVALKIT_MAX_ACTIVE_RUNS)"
    )
    work.set_defaults(handler=cmd_worker)

    show = with_json(sub.add_parser("show", help="summarise a run"))
    show.add_argument("run", help="run id or label:NAME")
    show.set_defaults(handler=cmd_show)

    _add_compare_commands(sub, with_json)

    _add_report_commands(sub)

    est = with_json(sub.add_parser("estimate", help="estimate the cost of a run config"))
    est.add_argument("config")
    est.set_defaults(handler=cmd_estimate)

    cal = with_json(sub.add_parser("calibrate", help="measure judge agreement with human labels"))
    what = cal.add_mutually_exclusive_group(required=True)
    what.add_argument("--judge", help="TOML file with a [judge] table")
    what.add_argument(
        "--scorer",
        help="TOML file with a [scorer] table (semantic_similarity): sweep its threshold",
    )
    cal.add_argument("labels", help="JSONL file of labelled outputs")
    cal.add_argument("--concurrency", type=int, default=8)
    cal.add_argument(
        "--folds",
        type=int,
        default=DEFAULT_FOLDS,
        help="cross-validation folds for a threshold sweep (0 for none)",
    )
    cal.set_defaults(handler=cmd_calibrate)

    _add_prompt_commands(sub)
    _add_generate_commands(sub)
    _add_maintenance_commands(sub)
    _add_tools_commands(sub)
    _add_agent_commands(sub)
    sub.add_parser("serve", help="run the HTTP API (EVALKIT_HOST, EVALKIT_PORT)")
    return parser


def _add_compare_commands(
    sub: Any, with_json: Callable[[argparse.ArgumentParser], argparse.ArgumentParser]
) -> None:
    for name, handler, text in (
        ("compare", cmd_compare, "compare two runs"),
        ("gate", cmd_gate, "exit 1 if the candidate is worse than the baseline"),
    ):
        p = with_json(sub.add_parser(name, help=text))
        p.add_argument("baseline", help="run id or label:NAME")
        p.add_argument("candidate", help="run id or label:NAME")
        p.add_argument("--alpha", type=float, default=0.05, help="significance level")
        p.add_argument(
            "--min-effect", type=float, default=0.0, help="ignore differences smaller than this"
        )
        p.add_argument("--metric", action="append", help="metric to compare (repeatable)")
        p.add_argument(
            "--allow-judge-change",
            action="store_true",
            help="compare even if a metric's judge (provider, model, effort, rubric) changed",
        )
        p.add_argument(
            "--by",
            help="also compare inside each slice: tag, split or metadata.KEY",
        )
        p.add_argument(
            "--min-slice-size",
            type=int,
            help="test a slice only with this many paired cases (default 20)",
        )
        p.add_argument(
            "--require-slice",
            action="append",
            metavar="NAME",
            help="fail if this slice regresses or is too small to test (repeatable)",
        )
        p.add_argument(
            "--any-slice", action="store_true", help="fail if any tested slice regresses"
        )
        p.add_argument("--rules", metavar="FILE", help="TOML file with a [gate] table")
        p.set_defaults(handler=handler)


def _add_report_commands(sub: Any) -> None:
    report = sub.add_parser("report", help="reports over groups of runs")
    report_sub = report.add_subparsers(dest="report_command", required=True)
    mat = report_sub.add_parser(
        "matrix", help="a matrix's variants, their cost/quality/latency frontier"
    )
    mat.add_argument("matrix", help="matrix id or label:NAME")
    mat.add_argument("--metric", help="quality metric (default: the first gated one)")
    mat.add_argument("--latency", choices=["p50", "p95"], default="p50")
    mat.add_argument("--alpha", type=float, default=0.05, help="significance level")
    mat.add_argument(
        "--min-effect", type=float, default=0.0, help="ignore differences smaller than this"
    )
    mat.set_defaults(handler=cmd_report)
    mat.add_argument("--json", action="store_true", help="print JSON instead of a table")
    rag = report_sub.add_parser(
        "rag", help="why each failing case of a RAG run failed: retrieval, grounding or answer"
    )
    rag.add_argument("run", help="run id or label:NAME")
    rag.add_argument(
        "--threshold",
        action="append",
        metavar="[NAME=]VALUE",
        help="a case fails below this score (default 0.5); NAME=VALUE for one metric; repeatable",
    )
    rag.add_argument("--limit", type=int, default=20, help="failing cases to list (default 20)")
    rag.add_argument("--json", action="store_true", help="print JSON instead of a table")
    rag.set_defaults(handler=cmd_report_rag)


def _add_prompt_commands(sub: Any) -> None:
    def with_json(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--json", action="store_true", help="print JSON instead of a table")
        return p

    prompt = sub.add_parser("prompt", help="push, label, diff and show registry prompts")
    prompt_sub = prompt.add_subparsers(dest="prompt_command", required=True)
    push = prompt_sub.add_parser(
        "push", help="store a prompt file as a new version (unchanged content is a no-op)"
    )
    push.add_argument("name")
    push.add_argument("file", help="TOML with template and system, or a template file")
    push.add_argument("--system", metavar="FILE", help="the system prompt, beside a template file")
    push.add_argument("--note", default="", help="what changed in this version")
    push.add_argument("--description", default="", help="the prompt's description, on first push")
    show_p = with_json(prompt_sub.add_parser("show", help="print a version: NAME[@VERSION|LABEL]"))
    show_p.add_argument("ref")
    with_json(prompt_sub.add_parser("list", help="every prompt, its newest version and labels"))
    dif = prompt_sub.add_parser("diff", help="unified diff of two versions")
    dif.add_argument("old", help="NAME@VERSION or NAME@LABEL")
    dif.add_argument("new", nargs="?", help="another reference (default: NAME@latest)")
    lab = with_json(
        prompt_sub.add_parser("label", help="point a label at a version, by compare-and-set")
    )
    lab.add_argument("ref", help="the version to label: NAME@VERSION or NAME@LABEL")
    lab.add_argument("label", help="the label to move, e.g. production")
    lab.add_argument(
        "--expect",
        metavar="VERSION",
        help="refuse unless the label is at this version now ('none': must not exist yet); "
        "default: wherever it is when read",
    )
    prompt.set_defaults(handler=cmd_prompt)


def _add_tools_commands(sub: Any) -> None:
    tools = sub.add_parser("tools", help="check tool sandboxes and record their fixtures")
    tools_sub = tools.add_subparsers(dest="tools_command", required=True)
    check = tools_sub.add_parser("check", help="validate a sandbox file and its fixtures")
    check.add_argument("sandbox", help="TOML file with a [sandbox] table")
    check.add_argument("--json", action="store_true", help="print JSON instead of a table")
    rec = tools_sub.add_parser(
        "record", help="send a JSONL file of calls to the tools' endpoints and keep the answers"
    )
    rec.add_argument("sandbox", help="TOML file with a [sandbox] table")
    rec.add_argument("calls", help='JSONL of {"tool": ..., "arguments": {...}}')
    rec.add_argument("-o", "--output", help="fixture file (default: the sandbox's fixtures)")
    rec.add_argument("--refresh", action="store_true", help="call again for calls already recorded")
    rec.add_argument("--json", action="store_true", help="print JSON instead of a table")
    tools.set_defaults(handler=cmd_tools)


def _add_agent_commands(sub: Any) -> None:
    agent = sub.add_parser("agent", help="check agents that run behind an endpoint")
    agent_sub = agent.add_subparsers(dest="agent_command", required=True)
    conf = agent_sub.add_parser(
        "conformance",
        help="probe an endpoint's agent on the session protocol; exit 1 when a check fails",
    )
    conf.add_argument("endpoint", help="endpoint name in EVALKIT_HTTP_TARGETS_FILE")
    conf.add_argument("--sandbox", required=True, help="TOML file with the [sandbox] of tools")
    conf.add_argument(
        "--task",
        action="append",
        help="a user's message to start a session with (repeatable; default: the "
        "sandbox file's [conformance] tasks)",
    )
    conf.add_argument("--max-steps", type=int, default=20, help="replies per session")
    conf.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    conf.add_argument("--json", action="store_true", help="print JSON instead of a table")


def _add_generate_commands(sub: Any) -> None:
    gen = sub.add_parser("generate", help="draft cases from seed documents and examples")
    gen.add_argument("config", help="TOML file with a [generate] table")
    gen.add_argument("--json", action="store_true", help="print JSON instead of a summary")
    gen.set_defaults(handler=cmd_generate)

    drafts = sub.add_parser("drafts", help="review generated cases: list, show, approve, reject")
    drafts_sub = drafts.add_subparsers(dest="drafts_command", required=True)
    lst = drafts_sub.add_parser("list", help="every generation and its draft counts")
    lst.add_argument("--dataset", help="only generations for this dataset")
    lst.add_argument("--json", action="store_true", help="print JSON instead of a table")
    show = drafts_sub.add_parser("show", help="a generation's drafts")
    show.add_argument("generation", help="generation id (or a prefix of 6 or more characters)")
    show.add_argument("--status", choices=["pending", "duplicate", "approved", "rejected"])
    show.add_argument("--limit", type=int, default=50, help="drafts to print (default 50)")
    show.add_argument("--offset", type=int, default=0)
    show.add_argument("--json", action="store_true", help="print JSON instead of text")
    app_ = drafts_sub.add_parser("approve", help="add drafts to the dataset as a new version")
    app_.add_argument("generation")
    which = app_.add_mutually_exclusive_group()
    which.add_argument("--id", action="append", default=[], help="a draft id (repeatable)")
    which.add_argument(
        "--all", action="store_true", help="every pending draft not repeating a dataset case"
    )
    app_.add_argument("--note", default="", help="the dataset version's note")
    app_.add_argument("--json", action="store_true", help="print JSON instead of a summary")
    rej = drafts_sub.add_parser("reject", help="mark drafts rejected")
    rej.add_argument("generation")
    rej.add_argument("--id", action="append", required=True, help="a draft id (repeatable)")
    rej.add_argument("--note", default="", help="why")
    drafts.set_defaults(handler=cmd_drafts)


def _add_maintenance_commands(sub: Any) -> None:
    ds = sub.add_parser("dataset", help="import, edit, diff, log, export or list datasets")
    ds_sub = ds.add_subparsers(dest="dataset_command", required=True)
    imp = ds_sub.add_parser("import")
    imp.add_argument("name")
    imp.add_argument("file")
    imp.add_argument("--description", default="")
    exp = ds_sub.add_parser("export")
    exp.add_argument("name")
    exp.add_argument("--version", type=int)
    exp.add_argument("-o", "--output")
    ds_sub.add_parser("list")
    edit = ds_sub.add_parser(
        "edit", help="apply a JSON edit (upsert, update, delete) as the next version"
    )
    edit.add_argument("name")
    edit.add_argument("file", help="JSON: {base, upsert, update, delete, note}")
    edit.add_argument(
        "--base", help="the edited version: a content hash or a version number (sets base)"
    )
    dif = ds_sub.add_parser("diff", help="field-level changes between two versions")
    dif.add_argument("name")
    dif.add_argument("old", type=int, help="version to diff from")
    dif.add_argument("new", type=int, nargs="?", help="version to diff to (default: newest)")
    dif.add_argument("--limit", type=int, default=50, help="cases to print (default 50)")
    dif.add_argument("--json", action="store_true", help="print JSON instead of text")
    log = ds_sub.add_parser("log", help="every version with its parent and changes")
    log.add_argument("name")
    log.add_argument("--version", type=int, help="only this version and its ancestors")
    log.add_argument("--json", action="store_true", help="print JSON instead of a table")
    ds.set_defaults(handler=cmd_dataset)

    cache_cmd = sub.add_parser("cache", help="bound the response cache and check blobs")
    cache_sub = cache_cmd.add_subparsers(dest="cache_command", required=True)
    sweep = cache_sub.add_parser(
        "sweep", help="delete old and least recently used entries, then unreferenced blobs"
    )
    sweep.add_argument("--max-age", help="remove entries unused this long, e.g. 30d")
    sweep.add_argument("--max-size", help="then keep at most this much, e.g. 20GiB")
    sweep.add_argument(
        "--blob-grace", help="keep unreferenced blobs newer than this (default EVALKIT_BLOB_GRACE)"
    )
    sweep.add_argument("--dry-run", action="store_true", help="report without deleting")
    sweep.add_argument("--json", action="store_true", help="print JSON instead of a table")
    scrub = cache_sub.add_parser("scrub", help="read every blob and check its hash")
    scrub.add_argument("--json", action="store_true", help="print JSON instead of a table")
    cache_cmd.set_defaults(handler=cmd_cache)

    db = sub.add_parser("db", help="database maintenance")
    db.add_argument("db_command", choices=["upgrade"])
    db.set_defaults(handler=cmd_db)


def main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    out = out or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        if args.database_url:
            settings = settings.model_copy(update={"database_url": args.database_url})
    except ConfigError as exc:
        _err(str(exc))
        return EXIT_INPUT
    if args.command == "serve":
        try:
            return cmd_serve(settings)
        except ConfigError as exc:
            _err(str(exc))
            return EXIT_INPUT
    # A daemon's log is its record; one-shot commands stay quiet unless asked.
    daemon = args.command == "worker"
    configure_logging("INFO" if args.verbose else settings.log_level if daemon else "WARNING")
    scrubber.register(*settings.secret_values())
    try:
        return asyncio.run(_dispatch(args, settings, out))
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    except CliError as exc:
        _err(exc.message)
        return exc.code
    except (PreconditionError, ConflictError) as exc:
        _err(exc.message)
        return EXIT_UNUSABLE
    except EvalkitError as exc:
        _err(exc.message)
        return EXIT_INPUT
    except Exception as exc:
        # A failed run, an unreachable database: never exit 1, which means "regression".
        _err(scrubber.scrub(f"{type(exc).__name__}: {exc}"))
        return EXIT_UNUSABLE


async def _dispatch(args: argparse.Namespace, settings: Settings, out: TextIO) -> int:
    if args.command == "tools":
        return await cmd_tools(args, settings, out)
    if args.command == "agent":
        return await cmd_agent(args, settings, out)
    app = App(settings, out)
    handler: Callable[[argparse.Namespace, App], Any] = args.handler
    try:
        code: int = await handler(args, app)
        return code
    finally:
        await app.aclose()


def _read_bytes(path: Path) -> bytes:
    try:
        if path.stat().st_size > 512 * 1024 * 1024:
            raise CliError(f"{path} is larger than 512 MiB", EXIT_INPUT)
        return path.read_bytes()
    except OSError as exc:
        raise CliError(f"cannot read {path}: {exc.strerror}", EXIT_INPUT) from None


def _read_toml(path: Path) -> dict[str, Any]:
    data = _read_bytes(path)
    if len(data) > MAX_CONFIG_BYTES:
        raise CliError(f"{path} is larger than {MAX_CONFIG_BYTES} bytes", EXIT_INPUT)
    try:
        return tomllib.loads(data.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise CliError(f"{path} is not valid TOML: {exc}", EXIT_INPUT) from None


def _describe(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5])


def _err(message: str) -> None:
    sys.stderr.write(f"evalkit: {message}\n")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in asdict(value).items()} | _extra(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _extra(value: Any) -> dict[str, Any]:
    if isinstance(value, CompareReport):
        return {"passed": value.passed, "failures": value.failures}
    if isinstance(value, MatrixReport):
        return {"complete": value.complete, "frontier": value.frontier}
    if hasattr(value, "max_total_cost_usd"):
        return {"max_total_cost_usd": value.max_total_cost_usd}
    return {}


def _emit(out: TextIO, value: Any, as_json: bool, formatter: Callable[[Any], str]) -> None:
    if as_json:
        out.write(json.dumps(_jsonable(value), indent=2) + "\n")
    else:
        out.write(formatter(value))


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _format_summary(s: RunSummary) -> str:
    lines = [
        f"run {s.run_id}  {s.label or '(no label)'}  {s.status}",
        f"dataset {s.dataset} v{s.dataset_version}  target {s.target_type}",
        f"cases {s.completed}/{s.cases_total}  ok {s.ok}  errors {s.errors}  cached {s.cached}",
        "",
        f"{'metric':<20}{'mean':>9}{'95% CI':>20}{'n':>7}{'unscorable':>12}",
    ]
    for m in s.metrics:
        ci = "-" if m.ci_low is None else f"[{_pct(m.ci_low)}, {_pct(m.ci_high)}]"
        gate = "" if m.gate else "  (not gated)"
        lines.append(f"{m.name:<20}{_pct(m.mean):>9}{ci:>20}{m.n:>7}{m.unscorable:>12}{gate}")
    lines += [
        "",
        f"tokens in {s.input_tokens:,}  out {s.output_tokens:,}  "
        f"cache read {s.cache_read_tokens:,}  cache write {s.cache_write_tokens:,}",
        f"cost ${s.cost_usd:.4f} target + ${s.judge_cost_usd:.4f} judge",
    ]
    if s.matrix_id is not None:
        lines.append(f"variant {s.variant} of matrix {s.matrix_id}")
    if s.prompt is not None:
        p = s.prompt
        lines.append(
            f"prompt {p['ref']}: {p['name']} version {p['version']} ({p['content_hash'][:12]})"
        )
    for name, j in s.judges.items():
        where = j["provider"] if j.get("endpoint") is None else f"{j['provider']} ({j['endpoint']})"
        if j["mode"] == "semantic_similarity":
            lines.append(f"embeddings {name}: {j['model']} on {where}")
        else:
            lines.append(
                f"judge {name}: {j['model']} on {where}, {j['mode']}, effort {j['effort']}"
            )
    for provider, entry in sorted((s.prices or {}).items()):
        models = "; ".join(
            f"{m} in {p['input']:g} out {p['output']:g}" for m, p in sorted(entry["models"].items())
        )
        lines.append(f"prices {provider} {entry['version']}: {models} (USD per 1M tokens)")
    if s.latency_p50_ms is not None:
        lines.append(f"latency p50 {s.latency_p50_ms:.0f} ms  p95 {s.latency_p95_ms:.0f} ms")
    if s.error:
        lines.append(f"error: {s.error}")
    return "\n".join(lines) + "\n"


def _format_verification(v: aggregates.Verification) -> str:
    arrays = "score arrays checked" if v.compacted else "not compacted yet"
    lines = [f"run {v.run_id}  {v.status}  {v.results} results  {arrays}"]
    if v.ok:
        lines.append("OK: aggregates match the results")
    else:
        lines.append(f"{len(v.problems)} difference(s):")
        lines += [f"  {p}" for p in v.problems]
        if v.repaired:
            lines.append("repaired: aggregates rewritten from the results")
    return "\n".join(lines) + "\n"


def _format_compare(r: CompareReport) -> str:
    lines = [
        f"baseline {r.baseline}  candidate {r.candidate}  alpha {r.alpha}",
        "",
        f"{'metric':<16}{'base':>8}{'cand':>8}{'diff':>9}{'95% CI':>22}{'p (adj)':>10}"
        f"{'d_z':>7}  verdict",
    ]
    for c in r.comparisons:
        ci = f"[{c.ci_low * 100:+.1f}, {c.ci_high * 100:+.1f}]"
        dz = "-" if c.effect_size is None else f"{c.effect_size:+.2f}"
        lines.append(
            f"{c.metric:<16}{_pct(c.baseline_mean):>8}{_pct(c.candidate_mean):>8}"
            f"{c.difference * 100:>+8.1f}%{ci:>22}{c.p_adjusted:>10.4f}{dz:>7}  {c.verdict}"
        )
    for change in r.judge_changes:
        lines.append(f"warning: judge changed for {change.metric}: {', '.join(change.changes)}")
    if r.dataset_change is not None:
        d = r.dataset_change
        left_out = d.changed + d.only_baseline + d.only_candidate
        lines.append(
            f"note: {d.dataset} v{d.baseline_version} vs v{d.candidate_version}: compared on "
            f"the {d.shared} cases both have unchanged; {left_out} left out ({d.changed} "
            f"changed, {d.only_baseline} only in the baseline, {d.only_candidate} only in "
            "the candidate)"
        )
    if r.slices is not None:
        lines += _format_slices(r.slices)
    if r.passed:
        lines += ["", "PASS: no regression"]
    else:
        lines += ["", f"FAIL: {', '.join(r.regressions) or 'slice rules'}"]
        lines += [f"  {f}" for f in (r.slices.failures if r.slices else [])]
    return "\n".join(lines) + "\n"


def _format_slices(r: SliceReport) -> list[str]:
    lines = [
        "",
        f"slices by {r.by}: {len(r.slices)} slice(s), {r.tests} test(s), p adjusted by Holm "
        f"across them; tested from {r.min_size} paired cases"
        + (f"; {r.unsliced} case(s) in no slice" if r.unsliced else ""),
        f"{'slice':<20}{'metric':<16}{'n':>6}{'base':>8}{'cand':>8}{'diff':>9}"
        f"{'p (adj)':>10}  verdict",
    ]
    for s in r.slices:
        rows = [
            f"{c.metric:<16}{c.n_pairs:>6}{_pct(c.baseline_mean):>8}{_pct(c.candidate_mean):>8}"
            f"{c.difference * 100:>+8.1f}%{c.p_adjusted:>10.4f}  {c.verdict}"
            for c in s.comparisons
        ] + [
            f"{metric:<16}{n:>6}{'-':>8}{'-':>8}{'-':>9}{'-':>10}  too small to test"
            for metric, n in s.too_small.items()
        ]
        name = s.name if len(s.name) < 20 else s.name[:18] + "…"
        lines += [f"{name if i == 0 else '':<20}{row}" for i, row in enumerate(rows)]
    return lines


def _format_estimate(e: Any) -> str:
    lines = [
        f"cases {e.cases} ({e.cached_cases} cached){'  batch pricing' if e.batch else ''}",
        f"target input  {_approx(e.target_input_estimated)}{e.target_input_tokens:,} tokens"
        f"{_estimated(e.target_input_estimated)}  ${e.target_input_cost_usd:.4f}",
        f"target output up to ${e.target_max_output_cost_usd:.4f}",
        f"judge calls {e.judge_calls}  input {_approx(e.judge_input_estimated)}"
        f"{e.judge_input_tokens:,} tokens{_estimated(e.judge_input_estimated)}  "
        f"${e.judge_input_cost_usd:.4f}  output up to ${e.judge_max_output_cost_usd:.4f}",
        f"at most {_approx(e.target_input_estimated or e.judge_input_estimated)}"
        f"${e.max_total_cost_usd:.4f}",
    ]
    if e.price_versions:
        lines.append("prices " + "  ".join(f"{p} {v}" for p, v in sorted(e.price_versions.items())))
    lines += [f"note: {n}" for n in e.notes]
    return "\n".join(lines) + "\n"


def _approx(estimated: bool) -> str:
    return "~" if estimated else ""


def _estimated(estimated: bool) -> str:
    return " (estimated)" if estimated else ""


def _format_matrix_estimate(m: MatrixEstimate) -> str:
    lines = []
    for v in m.variants:
        lines.append(f"variant {v.name}")
        lines += [f"  {line}" for line in _format_estimate(v.estimate).splitlines()]
    lines.append(f"matrix of {len(m.variants)} runs: at most ${m.max_total_cost_usd:.4f}")
    return "\n".join(lines) + "\n"


def _money(value: float | None) -> str:
    if value is None:
        return "-"
    return f"${value:.6f}" if value < 0.01 else f"${value:.4f}"


def _format_matrix(r: MatrixReport) -> str:
    width = max([len("variant"), *(len(v.name) for v in r.variants)]) + 2
    lines = [
        f"matrix {r.matrix_id}  {r.label or '(no label)'}  "
        f"dataset {r.dataset} v{r.dataset_version}",
        f"quality {r.metric}  latency {r.latency}  frontier over {', '.join(r.dimensions)}",
        "",
        f"{'variant':<{width}}{'status':<12}{r.metric[:10]:>10}{'95% CI':>18}{'cost/case':>12}"
        f"{r.latency + ' ms':>9}  frontier",
    ]
    for v in r.variants:
        ci = "-" if v.ci_low is None else f"[{_pct(v.ci_low)}, {_pct(v.ci_high)}]"
        lat = "-" if v.latency_ms is None else f"{v.latency_ms:.0f}"
        mark = "*" if v.on_frontier else ""
        if v.dominated_by:
            mark = f"dominated by {', '.join(v.dominated_by)}"
        lines.append(
            f"{v.name:<{width}}{v.status:<12}{_pct(v.quality):>10}{ci:>18}"
            f"{_money(v.cost_per_case_usd):>12}{lat:>9}  {mark}"
        )
    if r.steps:
        lines += [
            "",
            f"along the frontier, cheapest first (paired; p adjusted by Holm over "
            f"{len(r.steps)} step(s), alpha {r.alpha}):",
        ]
        for step in r.steps:
            c = step.comparison
            ci = f"[{c.ci_low * 100:+.1f}, {c.ci_high * 100:+.1f}]"
            lines.append(
                f"  {step.baseline} -> {step.candidate}: {c.difference * 100:+.1f}% {ci}  "
                f"p {c.p_adjusted:.4f}  n {c.n_pairs}  {c.verdict}"
            )
    lines += [f"note: {n}" for n in r.notes]
    return "\n".join(lines) + "\n"


def _format_rag(r: RagReport) -> str:
    return format_rag(r)


def _format_generation(r: GenerationReport) -> str:
    cached = f", {r.cached_calls} from the cache" if r.cached_calls else ""
    lines = [
        f"generation {r.id} for {r.dataset}: {r.generated} drafts of {r.requested} asked "
        f"({r.pending} pending, {r.duplicates} near-duplicates, {r.invalid} invalid)",
        f"{r.calls} calls{cached}, {r.input_tokens} input and {r.output_tokens} output tokens, "
        f"{_money(r.cost_usd)}",
    ]
    lines += [f"error: {e['source']}: {e['error']}" for e in r.errors[:10]]
    if len(r.errors) > 10:
        lines.append(f"... and {len(r.errors) - 10} more errors (--json for all)")
    lines.append(f"review with: evalkit drafts show {r.id}")
    return "\n".join(lines) + "\n"


def _format_generations(rows: list[GenerationInfo]) -> str:
    lines = []
    for g in rows:
        counts = ", ".join(f"{n} {s}" for s, n in g.statuses.items() if n)
        lines.append(
            f"{g.id}\t{g.dataset}\t{g.created_at.isoformat(timespec='seconds')}\t"
            f"{g.generated} drafts\t{counts or 'none'}"
        )
    return "\n".join(lines) + "\n" if lines else "no generations\n"


def _format_drafts(value: dict[str, Any]) -> str:
    g: GenerationInfo = value["generation"]
    counts = ", ".join(f"{n} {s}" for s, n in g.statuses.items())
    lines = [f"generation {g.id} for {g.dataset}: {counts}"]
    for d in value["drafts"]:
        case = d.case
        source = f" from {d.source}" if d.source else ""
        lines.append(f"[{d.status}] {d.id}{source}")
        lines.append(f"    input:    {_short(case['input'], 90)}")
        if case.get("expected") is not None:
            lines.append(f"    expected: {_short(case['expected'], 90)}")
        if d.duplicate_of is not None:
            lines.append(f"    repeats {d.duplicate_of} (similarity {d.similarity:.2f})")
        if d.dataset_version is not None:
            lines.append(f"    in version {d.dataset_version}")
    return "\n".join(lines) + "\n"


def _format_approval(r: ApprovalResult) -> str:
    lines = []
    if r.approved:
        verb = "now at" if r.created else "already had them at"
        lines.append(
            f"approved {len(r.approved)} drafts: {r.dataset} {verb} version {r.version} "
            f"({(r.content_hash or '')[:12]})"
        )
    else:
        lines.append("nothing to approve")
    if r.now_duplicates:
        lines.append(
            f"{len(r.now_duplicates)} drafts left out: they repeat cases now in {r.dataset}"
        )
        lines += [f"    {k} repeats {v}" for k, v in list(r.now_duplicates.items())[:10]]
    if r.already_approved:
        lines.append(f"{len(r.already_approved)} drafts were already approved")
    return "\n".join(lines) + "\n"


def _format_sandbox(info: dict[str, Any]) -> str:
    return (
        f"{len(info['tools'])} tools ({', '.join(info['tools'])}), "
        f"{info['fixtures']} fixtures, identity {info['identity'][:12]}\n"
    )


def _format_conformance(r: ConformanceReport) -> str:
    counts = {s: sum(c.status == s for c in r.checks) for s in ("pass", "warn", "fail", "skip")}
    summary = ", ".join(f"{n} {s}" for s, n in counts.items() if n)
    lines = [f"agent {r.endpoint} on {r.protocol}: {summary}"]
    width = max(len(c.name) for c in r.checks)
    lines += [f"  {c.status:<4}  {c.name:<{width}}  {c.detail}" for c in r.checks]
    lines.append(
        f"{r.requests} requests, latency p50 {r.latency_ms_p50:.0f} ms, "
        f"max {r.latency_ms_max:.0f} ms (timeout {r.timeout_seconds:g} s)"
    )
    return "\n".join(lines) + "\n"


def _format_record(r: RecordReport) -> str:
    lines = [f"recorded {r.recorded}, already recorded {r.skipped}, failed {len(r.failed)}"]
    lines += [f"  failed: {f}" for f in r.failed]
    return "\n".join(lines) + "\n"


def _format_prompt(p: dict[str, Any]) -> str:
    via = f"  via {p['label']}" if p["label"] else ""
    labels = ", ".join(f"{k}={v}" for k, v in p["labels"].items()) or "none"
    lines = [
        f"prompt {p['name']} version {p['version']}{via}  {p['content_hash'][:12]}",
        f"labels {labels}",
    ]
    if p["note"]:
        lines.append(f"note {p['note']}")
    if p["system"]:
        lines += ["", "--- system", p["system"].rstrip("\n")]
    lines += ["", "--- template", p["template"].rstrip("\n")]
    return "\n".join(lines) + "\n"


def _format_prompt_list(rows: list[Any]) -> str:
    return "".join(
        f"{r.name}\tv{r.latest_version}\t"
        f"{', '.join(f'{k}={v}' for k, v in r.labels.items()) or '-'}\n"
        for r in rows
    )


def _format_label_move(m: LabelMove) -> str:
    before = "new" if m.previous is None else f"v{m.previous}"
    return f"{m.name}@{m.label}: {before} -> v{m.version}\n"


def _format_calibration(r: Any) -> str:
    lines = [
        f"labelled {r.n}  judged {r.judged}  agreement {_pct(r.agreement)}",
        f"Cohen's kappa {r.kappa:.3f}"
        + ("" if r.weighted_kappa is None else f"  quadratic-weighted {r.weighted_kappa:.3f}"),
        "",
        "human \\ judge " + " ".join(f"{label:>6}" for label in r.labels),
    ]
    for label, row in zip(r.labels, r.confusion, strict=True):
        lines.append(f"{label:>13} " + " ".join(f"{n:>6}" for n in row))
    if r.disagreements:
        lines.append(f"\ndisagreements: {', '.join(r.disagreements)}")
    lines.append(f"cost ${r.cost_usd:.4f}")
    return "\n".join(lines) + "\n"


def _format_threshold(r: ThresholdReport) -> str:
    best = r.best
    lines = [
        f"labelled {r.n}  scored {r.scored}  labelled 1: {r.positives}  "
        f"cosines {r.min_cosine:.4f} to {r.max_cosine:.4f}",
        f"best threshold {best.threshold:.4f}: Cohen's kappa {best.kappa:.3f}  "
        f"agreement {_pct(best.agreement)}  precision {_pct(best.precision)}  "
        f"recall {_pct(best.recall)}",
    ]
    if r.cv_kappa is not None:
        lines.append(
            f"cross-validated ({r.folds} folds): kappa {r.cv_kappa:.3f}  "
            f"agreement {_pct(r.cv_agreement)}"
        )
    else:
        lines.append("cross-validated: too few items (two per fold needed)")
    lines += ["", f"{'threshold':>9}{'kappa':>8}{'agree':>8}{'precision':>11}{'recall':>8}"]
    for p in r.sweep:
        lines.append(
            f"{p.threshold:>9.2f}{p.kappa:>8.3f}{_pct(p.agreement):>8}"
            f"{_pct(p.precision):>11}{_pct(p.recall):>8}"
        )
    (tn, fp), (fn, tp) = best.confusion
    lines.append(
        f"\nat {best.threshold:.4f}: label 0 -> fail {tn} pass {fp}; label 1 -> fail {fn} pass {tp}"
    )
    if r.disagreements:
        lines.append(f"disagreements: {', '.join(r.disagreements)}")
    lines.append(f"cost ${r.cost_usd:.4f}")
    lines.append(f"set threshold = {best.threshold:.4f} in the scorer to gate on it")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
