"""Command line: run evaluations, compare runs and gate CI on regressions.

Exit codes are part of the interface, so CI can tell a regression from a
broken job:

    0  success; for ``gate``, no regression
    1  ``gate`` found a regression
    2  bad input: config, arguments, unknown dataset or run
    3  the run cannot be used: it failed, is incomplete, or cannot be paired
    130 interrupted (the run is saved and can be resumed with --resume)

Configuration comes from the environment (see ``evalkit.config``); the
database URL can also be given with --database-url. SQLite databases are
migrated automatically; for PostgreSQL run ``evalkit db upgrade`` once.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from evalkit import __version__
from evalkit.api import serve
from evalkit.config import ConfigError, Settings
from evalkit.datasets import (
    get_dataset,
    import_dataset,
    list_datasets,
    load_cases,
    parse_jsonl,
    to_jsonl,
)
from evalkit.db import make_engine, make_session_factory, upgrade
from evalkit.db.models import Run
from evalkit.errors import EvalkitError, NotFoundError, PreconditionError
from evalkit.estimate import estimate
from evalkit.llm import ClaudeClient
from evalkit.logs import configure_logging, scrubber
from evalkit.pricing import PriceTable
from evalkit.report import CompareReport, RunSummary, compare_runs, summarize
from evalkit.runner import Runner
from evalkit.runspec import RunConfig
from evalkit.scorers.calibration import calibrate, parse_labels, save_calibration
from evalkit.scorers.judge import LLMJudge
from evalkit.scorers.registry import JudgeSpec

EXIT_OK, EXIT_REGRESSION, EXIT_INPUT, EXIT_UNUSABLE, EXIT_INTERRUPTED = 0, 1, 2, 3, 130
MAX_CONFIG_BYTES = 1024 * 1024


class CliError(Exception):
    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class App:
    """Settings, database and an on-demand Claude client for one command."""

    def __init__(self, settings: Settings, out: TextIO) -> None:
        self.settings = settings
        self.out = out
        self.engine = make_engine(settings.database_url)
        if make_url(settings.database_url).get_backend_name() == "sqlite":
            upgrade(self.engine)
        self.sessions: sessionmaker[Session] = make_session_factory(self.engine)
        self._claude: ClaudeClient | None = None

    def claude(self) -> ClaudeClient:
        if self._claude is None:
            self._claude = ClaudeClient(
                PriceTable(self.settings.prices), self.settings.anthropic_api_key
            )
        return self._claude

    def runner(self, needs_claude: bool) -> Runner:
        return Runner(
            self.sessions,
            self.settings,
            claude=self.claude() if needs_claude else None,
            allow_import_paths=True,
        )

    async def aclose(self) -> None:
        if self._claude is not None:
            await self._claude.aclose()
        self.engine.dispose()


def load_run_config(path: Path, app: App) -> RunConfig:
    """Read a TOML run config. ``[dataset] path = ...`` imports that file first."""
    raw = _read_toml(path)
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


def needs_claude(config: RunConfig) -> bool:
    return config.target.type == "prompt" or any(s.type == "judge" for s in config.scorers)


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
    if args.resume:
        with app.sessions() as session:
            run = session.get(Run, args.resume)
            if run is None:
                raise NotFoundError(f"run {args.resume} not found")
            claude = run.target_spec.get("type") == "prompt" or any(
                s.get("type") == "judge" for s in run.scorers
            )
        run_id = args.resume
        runner = app.runner(claude)
    else:
        if args.config is None:
            raise CliError("give a run config, or --resume RUN_ID", EXIT_INPUT)
        config = load_run_config(Path(args.config), app)
        if args.label is not None:
            config = config.model_copy(update={"label": args.label})
        runner = app.runner(needs_claude(config))
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
        _err(f"regression in: {', '.join(report.regressions)}")
        return EXIT_REGRESSION
    return EXIT_OK


def _compare(args: argparse.Namespace, app: App) -> CompareReport:
    with app.sessions() as session:
        return compare_runs(
            session,
            resolve_run(session, args.baseline),
            resolve_run(session, args.candidate),
            alpha=args.alpha,
            min_effect=args.min_effect,
            metrics=args.metric or None,
        )


async def cmd_estimate(args: argparse.Namespace, app: App) -> int:
    config = load_run_config(Path(args.config), app)
    result = await estimate(app.sessions, app.settings, config, app.claude())
    _emit(app.out, result, args.json, _format_estimate)
    return EXIT_OK


async def cmd_calibrate(args: argparse.Namespace, app: App) -> int:
    raw = _read_toml(Path(args.judge))
    table = raw.get("judge")
    if not isinstance(table, dict):
        raise CliError(f"{args.judge} has no [judge] table", EXIT_INPUT)
    try:
        spec = JudgeSpec.model_validate({"type": "judge", **table})
    except ValidationError as exc:
        raise CliError(f"{args.judge}: {_describe(exc)}", EXIT_INPUT) from None
    judge = LLMJudge(
        spec.resolved_name(),
        client=app.claude(),
        rubric=spec.rubric,
        mode=spec.mode,
        model=spec.model,
        effort=spec.effort,
        max_tokens=spec.max_tokens,
    )
    items = parse_labels(_read_bytes(Path(args.labels)), spec.mode)
    report = await calibrate(judge, items, concurrency=args.concurrency)
    with app.sessions() as session:
        save_calibration(session, judge, report)
    _emit(app.out, report, args.json, _format_calibration)
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
        else:
            for ds in list_datasets(session):
                app.out.write(
                    f"{ds.name}\tv{ds.version}\t{ds.case_count} cases\t{ds.content_hash[:12]}\n"
                )
    return EXIT_OK


async def cmd_db(args: argparse.Namespace, app: App) -> int:
    upgrade(app.engine)
    app.out.write("database is at the latest schema\n")
    return EXIT_OK


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

    run = with_json(sub.add_parser("run", help="run an evaluation from a TOML config"))
    run.add_argument("config", nargs="?")
    run.add_argument("--label", help="label the run, e.g. a branch or commit")
    run.add_argument("--resume", metavar="RUN_ID", help="continue an interrupted run")
    run.set_defaults(handler=cmd_run)

    show = with_json(sub.add_parser("show", help="summarise a run"))
    show.add_argument("run", help="run id or label:NAME")
    show.set_defaults(handler=cmd_show)

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
        p.set_defaults(handler=handler)

    est = with_json(sub.add_parser("estimate", help="estimate the cost of a run config"))
    est.add_argument("config")
    est.set_defaults(handler=cmd_estimate)

    cal = with_json(sub.add_parser("calibrate", help="measure judge agreement with human labels"))
    cal.add_argument("--judge", required=True, help="TOML file with a [judge] table")
    cal.add_argument("labels", help="JSONL file of labelled outputs")
    cal.add_argument("--concurrency", type=int, default=8)
    cal.set_defaults(handler=cmd_calibrate)

    ds = sub.add_parser("dataset", help="import, export or list datasets")
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
    ds.set_defaults(handler=cmd_dataset)

    db = sub.add_parser("db", help="database maintenance")
    db.add_argument("db_command", choices=["upgrade"])
    db.set_defaults(handler=cmd_db)

    sub.add_parser("serve", help="run the HTTP API (EVALKIT_HOST, EVALKIT_PORT)")
    return parser


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
    configure_logging("INFO" if args.verbose else "WARNING")
    scrubber.register(*settings.secret_values())
    try:
        return asyncio.run(_dispatch(args, settings, out))
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    except CliError as exc:
        _err(exc.message)
        return exc.code
    except PreconditionError as exc:
        _err(exc.message)
        return EXIT_UNUSABLE
    except EvalkitError as exc:
        _err(exc.message)
        return EXIT_INPUT


async def _dispatch(args: argparse.Namespace, settings: Settings, out: TextIO) -> int:
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
        return {"passed": value.passed}
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
    if s.latency_p50_ms is not None:
        lines.append(f"latency p50 {s.latency_p50_ms:.0f} ms  p95 {s.latency_p95_ms:.0f} ms")
    if s.error:
        lines.append(f"error: {s.error}")
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
    lines += ["", "PASS: no regression" if r.passed else f"FAIL: {', '.join(r.regressions)}"]
    return "\n".join(lines) + "\n"


def _format_estimate(e: Any) -> str:
    lines = [
        f"cases {e.cases} ({e.cached_cases} cached){'  batch pricing' if e.batch else ''}",
        f"target input  {e.target_input_tokens:,} tokens  ${e.target_input_cost_usd:.4f}",
        f"target output up to ${e.target_max_output_cost_usd:.4f}",
        f"judge calls {e.judge_calls}  input {e.judge_input_tokens:,} tokens  "
        f"${e.judge_input_cost_usd:.4f}  output up to ${e.judge_max_output_cost_usd:.4f}",
        f"at most ${e.max_total_cost_usd:.4f}",
    ]
    lines += [f"note: {n}" for n in e.notes]
    return "\n".join(lines) + "\n"


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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
