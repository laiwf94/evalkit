"""HTTP API: datasets, prompts, runs and comparisons for a team sharing one
evaluation store.

Every ``/v1`` route takes ``Authorization: Bearer <token>`` when tokens are
configured. Without tokens the server refuses to listen on anything but a
loopback address. Errors have one shape with stable codes:

    {"error": {"code": "invalid_argument", "message": "...", "request_id": "..."}}

invalid_argument 400, unauthenticated 401, not_found 404, conflict 409,
failed_precondition 412, payload_too_large 413, internal 500, unavailable 503.

Submitted runs go into the run queue as ``queued``. Worker daemons
(``evalkit worker``) execute them, and so does a worker inside this process
with ``EVALKIT_MAX_ACTIVE_RUNS`` slots (0 makes this process only accept runs).
On shutdown the embedded worker drains: its runs save their finished cases and
go back to the queue, where another worker, or this one after a restart, picks
them up.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Annotated, Any, Literal, cast

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session, sessionmaker
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from evalkit import metrics, runqueue
from evalkit.config import ConfigError, Settings
from evalkit.datasets import (
    get_dataset,
    import_dataset,
    list_datasets,
    load_cases,
    parse_jsonl,
    to_jsonl,
    validate_name,
)
from evalkit.db import make_engine, make_session_factory, upgrade
from evalkit.db.models import Dataset, Result, ResultStep, ResultTurn, Run, as_utc, utcnow
from evalkit.errors import EvalkitError, InvalidInputError, NotFoundError
from evalkit.frontier import matrix_report
from evalkit.logs import configure_logging, log_context, scrubber
from evalkit.prompts import (
    Resolved,
    diff,
    get_prompt,
    labels_of,
    list_prompts,
    move_label,
    push_version,
    resolve,
    validate_prompt_name,
    versions_of,
)
from evalkit.providers.anthropic import ClaudeClient
from evalkit.providers.registry import ANTHROPIC, ProviderRegistry
from evalkit.report import compare_runs, summarize
from evalkit.runner import Runner
from evalkit.runspec import RunConfig
from evalkit.worker import Worker, listen_url

log = logging.getLogger("evalkit.api")

REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
STATUS_OF = {
    "invalid_argument": 400,
    "not_found": 404,
    "conflict": 409,
    "failed_precondition": 412,
}


def _error(status: int, code: str, message: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": message, "request_id": request_id}},
        status_code=status,
    )


def _request_id(scope: Scope) -> str:
    return str(scope.get("state", {}).get("request_id", ""))


class BodyLimit:
    """Rejects bodies over the limit before the app sees them: by Content-Length
    when it is given, otherwise by counting while reading. The body is read here
    and replayed to the app, so at most ``limit`` bytes are ever held."""

    def __init__(self, app: ASGIApp, limit: int) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        incoming = headers.get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
        too_large = _error(
            413, "payload_too_large", f"body is larger than {self.limit} bytes", request_id
        )
        too_large.headers["X-Request-ID"] = request_id
        declared = headers.get(b"content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > self.limit):
            await too_large(scope, receive, send)
            return
        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                await self.app(scope, _replay([message]), send)
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.limit:
                await too_large(scope, receive, send)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = {"type": "http.request", "body": b"".join(chunks), "more_body": False}
        await self.app(scope, _replay([body], then=receive), send)


def _replay(messages: list[Message], then: Receive | None = None) -> Receive:
    pending = list(messages)

    async def receive() -> Message:
        if pending:
            return pending.pop(0)
        if then is not None:
            return await then()
        return {"type": "http.disconnect"}

    return receive


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline: str = Field(min_length=1, max_length=64)
    candidate: str = Field(min_length=1, max_length=64)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    min_effect: float = Field(default=0.0, ge=0, le=1)
    metrics: list[str] | None = Field(default=None, max_length=32)
    # Compare even when a metric's judge differs between the runs.
    allow_judge_change: bool = False


def _iso(value: datetime | None) -> str | None:
    return as_utc(value).isoformat() if value else None


def _dataset_json(ds: Dataset) -> dict[str, Any]:
    return {
        "name": ds.name,
        "version": ds.version,
        "content_hash": ds.content_hash,
        "case_count": ds.case_count,
        "description": ds.description,
        "created_at": _iso(ds.created_at),
    }


def _summary_json(session: Session, run_id: str) -> dict[str, Any]:
    summary = asdict(summarize(session, run_id))
    for key in ("created_at", "finished_at"):
        summary[key] = summary[key].isoformat() if summary[key] else None
    return summary


class _UnauthenticatedError(Exception):
    pass


@dataclass(frozen=True)
class Services:
    settings: Settings
    sessions: sessionmaker[Session]
    runner: Runner
    worker: Worker | None

    def wake(self) -> None:
        if self.worker is not None:
            self.worker.wake()


def services(request: Request) -> Services:
    svc: Services = request.app.state.services
    return svc


async def authenticate(svc: Annotated[Services, Depends(services)], request: Request) -> None:
    if not svc.settings.auth_enabled:
        return
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    digest = hashlib.sha256(token.strip().encode()).digest()
    # Compare against every key, without stopping early, so timing says nothing.
    matched = False
    for known in svc.settings.api_token_digests:
        matched |= hmac.compare_digest(digest, known)
    if scheme.lower() != "bearer" or not matched:
        raise _UnauthenticatedError


Svc = Annotated[Services, Depends(services)]
router = APIRouter(prefix="/v1", dependencies=[Depends(authenticate)])


def _in_session[T](svc: Services, fn: Callable[[Session], T]) -> Awaitable[T]:
    def call() -> T:
        with svc.sessions() as session:
            return fn(session)

    return asyncio.to_thread(call)


@router.post("/datasets/{name}", status_code=201)
async def put_dataset(
    name: str,
    request: Request,
    response: Response,
    svc: Svc,
    description: Annotated[str, Query(max_length=2000)] = "",
) -> dict[str, Any]:
    validate_name(name)
    cases = parse_jsonl(await request.body())

    def store(session: Session) -> tuple[dict[str, Any], bool]:
        ds, created = import_dataset(session, name, cases, description)
        return _dataset_json(ds), created

    body, created = await _in_session(svc, store)
    if not created:
        response.status_code = 200
    return body


@router.get("/datasets")
async def get_datasets(svc: Svc) -> dict[str, Any]:
    rows = await _in_session(svc, lambda s: [_dataset_json(d) for d in list_datasets(s)])
    return {"datasets": rows}


@router.get("/datasets/{name}/versions/{version}")
async def get_dataset_version(name: str, version: int, svc: Svc) -> dict[str, Any]:
    return await _in_session(svc, lambda s: _dataset_json(get_dataset(s, name, version)))


@router.get("/datasets/{name}/versions/{version}/export")
async def export_dataset(name: str, version: int, svc: Svc) -> PlainTextResponse:
    text_body = await _in_session(
        svc, lambda s: to_jsonl(load_cases(s, get_dataset(s, name, version)))
    )
    return PlainTextResponse(text_body, media_type="application/x-ndjson")


class PromptPush(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template: str = Field(min_length=1, max_length=200_000)
    system: str = Field(default="", max_length=200_000)
    note: str = Field(default="", max_length=2000)
    # Kept from the first push; later pushes do not change it.
    description: str = Field(default="", max_length=2000)


class LabelMoveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    # Required, and may be null: the version the label must be at now, or null
    # when the label must not exist yet. A label never moves without a check.
    expected_version: int | None = Field(ge=1)


def _version_json(found: Resolved, *, content: bool = True) -> dict[str, Any]:
    v = found.version
    body: dict[str, Any] = {
        "name": found.name,
        "version": v.version,
        "label": found.label,
        "content_hash": v.content_hash,
        "note": v.note,
        "created_at": _iso(v.created_at),
    }
    if content:
        body |= {"system": v.system, "template": v.template}
    return body


@router.get("/prompts")
async def get_prompts(svc: Svc) -> dict[str, Any]:
    def load(session: Session) -> list[dict[str, Any]]:
        return [
            {
                "name": p.name,
                "description": p.description,
                "latest_version": p.latest_version,
                "labels": p.labels,
                "created_at": _iso(p.created_at),
            }
            for p in list_prompts(session)
        ]

    return {"prompts": await _in_session(svc, load)}


@router.get("/prompts/{name}")
async def get_prompt_history(name: str, svc: Svc) -> dict[str, Any]:
    """A prompt, its labels and every version, oldest first, without their content."""

    def load(session: Session) -> dict[str, Any]:
        prompt = get_prompt(session, name)
        return {
            "name": prompt.name,
            "description": prompt.description,
            "created_at": _iso(prompt.created_at),
            "labels": labels_of(session, prompt),
            "versions": [
                _version_json(Resolved(name, v), content=False)
                for v in versions_of(session, prompt)
            ],
        }

    return await _in_session(svc, load)


@router.post("/prompts/{name}/versions", status_code=201)
async def push_prompt(name: str, body: PromptPush, response: Response, svc: Svc) -> dict[str, Any]:
    validate_prompt_name(name)

    def store(session: Session) -> tuple[dict[str, Any], bool]:
        version, created = push_version(
            session,
            name,
            template=body.template,
            system=body.system,
            note=body.note,
            description=body.description,
        )
        return _version_json(Resolved(name, version)), created

    out, created = await _in_session(svc, store)
    if not created:
        response.status_code = 200
    return out


@router.get("/prompts/{name}/versions/{selector}")
async def get_prompt_version(name: str, selector: str, svc: Svc) -> dict[str, Any]:
    """A version by number, by label, or ``latest``."""
    ref = f"{validate_prompt_name(name)}@{selector}"
    return await _in_session(svc, lambda s: _version_json(resolve(s, ref)))


@router.get("/prompts/{name}/diff")
async def diff_prompt(
    name: str,
    svc: Svc,
    from_: Annotated[str, Query(alias="from", min_length=1, max_length=64)],
    to: Annotated[str, Query(min_length=1, max_length=64)] = "latest",
) -> PlainTextResponse:
    validate_prompt_name(name)
    text_body = await _in_session(
        svc, lambda s: diff(resolve(s, f"{name}@{from_}"), resolve(s, f"{name}@{to}"))
    )
    return PlainTextResponse(text_body, media_type="text/x-diff")


@router.put("/prompts/{name}/labels/{label}")
async def put_prompt_label(
    name: str, label: str, body: LabelMoveRequest, svc: Svc
) -> dict[str, Any]:
    """Move a label by compare-and-set; 409 with where the label is now if it is
    not at ``expected_version``."""

    def store(session: Session) -> dict[str, Any]:
        moved = move_label(session, name, label, body.version, expected=body.expected_version)
        return {
            "name": moved.name,
            "label": moved.label,
            "previous_version": moved.previous,
            "version": moved.version,
        }

    return await _in_session(svc, store)


@router.post("/runs", status_code=202)
async def create_run(config: RunConfig, svc: Svc) -> dict[str, Any]:
    run = await asyncio.to_thread(svc.runner.create_run, config, queued=True)
    svc.wake()
    return await _in_session(svc, lambda s: _summary_json(s, run.id))


@router.get("/runs")
async def list_runs(
    svc: Svc,
    label: Annotated[str | None, Query(max_length=128)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    query = select(Run).order_by(Run.created_at.desc()).limit(limit)
    if label is not None:
        query = query.where(Run.label == label)

    def load(session: Session) -> list[dict[str, Any]]:
        return [
            {
                "run_id": r.id,
                "label": r.label,
                "status": r.status,
                "target_type": r.target_spec.get("type"),
                "created_at": _iso(r.created_at),
                "finished_at": _iso(r.finished_at),
            }
            for r in session.scalars(query)
        ]

    return {"runs": await _in_session(svc, load)}


@router.get("/runs/{run_id}")
async def get_run(run_id: str, svc: Svc) -> dict[str, Any]:
    return await _in_session(svc, lambda s: _summary_json(s, run_id))


@router.get("/runs/{run_id}/results")
async def get_results(
    run_id: str,
    svc: Svc,
    after: Annotated[str | None, Query(max_length=256)] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> dict[str, Any]:
    def load(session: Session) -> dict[str, Any]:
        if session.get(Run, run_id) is None:
            raise NotFoundError(f"run {run_id} not found")
        query = select(Result).where(Result.run_id == run_id).order_by(Result.case_id)
        if after is not None:
            query = query.where(Result.case_id > after)
        rows = list(session.scalars(query.limit(limit + 1)))
        page = rows[:limit]
        return {
            "results": [_result_json(r) for r in page],
            "next": page[-1].case_id if len(rows) > limit else None,
        }

    return await _in_session(svc, load)


@router.get("/runs/{run_id}/results/{case_id:path}/trajectory")
async def get_trajectory(run_id: str, case_id: str, svc: Svc) -> dict[str, Any]:
    """One case's whole trajectory: an agent's model replies and tool calls
    (``steps``), or a conversation's generated turns (``turns``), in order."""

    def load(session: Session) -> dict[str, Any]:
        if session.get(Run, run_id) is None:
            raise NotFoundError(f"run {run_id} not found")
        result = session.scalar(
            select(Result).where(Result.run_id == run_id, Result.case_id == case_id)
        )
        if result is None:
            raise NotFoundError(f"run {run_id} has no result for case {case_id!r}")
        steps = session.scalars(
            select(ResultStep)
            .where(ResultStep.run_id == run_id, ResultStep.case_id == case_id)
            .order_by(ResultStep.step)
        )
        turns = session.scalars(
            select(ResultTurn)
            .where(ResultTurn.run_id == run_id, ResultTurn.case_id == case_id)
            .order_by(ResultTurn.turn)
        )
        return {
            "run_id": run_id,
            **_result_json(result),
            "steps": [_step_json(s) for s in steps],
            "turns": [_turn_json(t) for t in turns],
        }

    return await _in_session(svc, load)


@router.post("/runs/{run_id}/resume", status_code=202)
async def resume_run(run_id: str, svc: Svc) -> dict[str, Any]:
    def requeue(session: Session) -> dict[str, Any]:
        runqueue.enqueue(session, run_id, utcnow())
        session.commit()
        return _summary_json(session, run_id)

    summary = await _in_session(svc, requeue)
    svc.wake()
    return summary


@router.post("/runs/{run_id}/cancel", status_code=202)
async def cancel_run(run_id: str, svc: Svc) -> dict[str, Any]:
    def request(session: Session) -> dict[str, Any]:
        runqueue.cancel(session, run_id, utcnow())
        session.commit()
        return _summary_json(session, run_id)

    summary = await _in_session(svc, request)
    metrics.RUNS_CANCELLED.inc()
    if svc.worker is not None:
        svc.worker.cancel(run_id)
    return summary


@router.post("/matrices", status_code=202)
async def create_matrix(config: RunConfig, svc: Svc) -> dict[str, Any]:
    """Queue every run of a ``[matrix]`` config ("matrix" in the body)."""
    if config.matrix is None:
        raise InvalidInputError("the body has no matrix; queue a single run with POST /v1/runs")
    group, runs = await asyncio.to_thread(svc.runner.create_matrix, config, queued=True)
    svc.wake()

    def load(session: Session) -> dict[str, Any]:
        return {
            "matrix_id": group.id,
            "label": group.label,
            "runs": [_summary_json(session, r.id) for r in runs],
        }

    return await _in_session(svc, load)


@router.get("/matrices/{matrix_id}")
async def get_matrix_report(
    matrix_id: str,
    svc: Svc,
    *,
    metric: Annotated[str | None, Query(max_length=128)] = None,
    latency: Annotated[Literal["p50", "p95"], Query()] = "p50",
    alpha: Annotated[float, Query(gt=0, lt=1)] = 0.05,
    min_effect: Annotated[float, Query(ge=0, le=1)] = 0.0,
) -> dict[str, Any]:
    """Every variant with its quality, cost and latency, the Pareto frontier, and
    paired comparisons along it."""

    def load(session: Session) -> dict[str, Any]:
        report = matrix_report(
            session, matrix_id, metric=metric, latency=latency, alpha=alpha, min_effect=min_effect
        )
        return {**asdict(report), "complete": report.complete, "frontier": report.frontier}

    return await _in_session(svc, load)


@router.post("/compare")
async def compare(body: CompareRequest, svc: Svc) -> dict[str, Any]:
    def run(session: Session) -> dict[str, Any]:
        report = compare_runs(
            session,
            body.baseline,
            body.candidate,
            alpha=body.alpha,
            min_effect=body.min_effect,
            metrics=body.metrics,
            allow_judge_change=body.allow_judge_change,
        )
        return {**asdict(report), "passed": report.passed}

    return await _in_session(svc, run)


def _result_json(r: Result) -> dict[str, Any]:
    return {
        "case_id": r.case_id,
        "status": r.status,
        "output": r.output,
        "stop_reason": r.stop_reason,
        "scores": r.scores,
        "error": r.error,
        "cached": r.cached,
        "attempts": r.attempts,
        "latency_ms": r.latency_ms,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "cost_usd": r.cost_usd + r.judge_cost_usd,
    }


def _usage_json(row: ResultStep | ResultTurn) -> dict[str, Any]:
    return {
        "input_tokens": row.input_tokens,
        "output_tokens": row.output_tokens,
        "cache_read_tokens": row.cache_read_tokens,
        "cache_write_tokens": row.cache_write_tokens,
        "cost_usd": row.cost_usd,
    }


def _step_json(s: ResultStep) -> dict[str, Any]:
    out: dict[str, Any] = {"step": s.step, "kind": s.kind}
    if s.kind == "model":
        out |= {"text": s.text, "stop_reason": s.stop_reason}
    else:
        out |= {
            "tool": s.tool,
            "call_id": s.call_id,
            "arguments": s.arguments,
            "status": s.status,
            "result": s.result,
            "error": s.error,
            "source": s.source,
        }
    return {**out, "cached": s.cached, "latency_ms": s.latency_ms, **_usage_json(s)}


def _turn_json(t: ResultTurn) -> dict[str, Any]:
    return {
        "turn": t.turn,
        "role": t.role,
        "content": t.content,
        "stop_reason": t.stop_reason,
        "cached": t.cached,
        "latency_ms": t.latency_ms,
        **_usage_json(t),
    }


async def healthz(svc: Svc) -> JSONResponse:
    def ping() -> None:
        with svc.sessions() as session:
            session.execute(text("SELECT 1"))

    try:
        await asyncio.to_thread(ping)
    except Exception:
        log.warning("health check failed", exc_info=True)
        return _error(503, "unavailable", "database unreachable", "")
    return JSONResponse({"status": "ok"})


async def prometheus() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(metrics.REGISTRY), media_type=CONTENT_TYPE_LATEST)


async def request_context(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    incoming = request.headers.get("x-request-id", "")
    request_id = incoming if REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
    request.state.request_id = request_id
    start = time.perf_counter()
    with log_context(request_id=request_id):
        response = await call_next(request)
        path = getattr(request.scope.get("route"), "path", "unmatched")
        elapsed = time.perf_counter() - start
        metrics.HTTP_REQUESTS.labels(request.method, path, str(response.status_code)).inc()
        metrics.HTTP_LATENCY.labels(path).observe(elapsed)
        log.info(
            "request",
            extra={
                "method": request.method,
                "route": path,
                "status": response.status_code,
                "duration_ms": round(elapsed * 1000, 2),
            },
        )
    response.headers["X-Request-ID"] = request_id
    return response


async def _evalkit_error(request: Request, exc: Exception) -> JSONResponse:
    err = cast(EvalkitError, exc)
    return _error(STATUS_OF.get(err.code, 500), err.code, err.message, _request_id(request.scope))


async def _unauthenticated(request: Request, exc: Exception) -> JSONResponse:
    response = _error(
        401, "unauthenticated", "missing or invalid bearer token", _request_id(request.scope)
    )
    response.headers["WWW-Authenticate"] = "Bearer"
    return response


async def _invalid(request: Request, exc: Exception) -> JSONResponse:
    errors = cast(RequestValidationError, exc).errors()
    first = errors[0]
    where = ".".join(str(p) for p in first.get("loc", ()) if p != "body") or "body"
    more = f" (and {len(errors) - 1} more)" if len(errors) > 1 else ""
    message = f"{where}: {first.get('msg', 'invalid')}{more}"
    return _error(400, "invalid_argument", message, _request_id(request.scope))


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    http = cast(StarletteHTTPException, exc)
    codes = {404: "not_found", 405: "method_not_allowed", 413: "payload_too_large"}
    code = codes.get(http.status_code, "invalid_argument")
    return _error(http.status_code, code, str(http.detail), _request_id(request.scope))


async def _internal(request: Request, exc: Exception) -> JSONResponse:
    log.error("unhandled error", exc_info=exc)
    return _error(500, "internal", "internal error", _request_id(request.scope))


def create_app(
    settings: Settings,
    *,
    sessions: sessionmaker[Session] | None = None,
    claude: ClaudeClient | None = None,
) -> FastAPI:
    if sessions is None:
        sessions = make_session_factory(make_engine(settings.database_url))
    book = settings.price_book()
    claude_client = claude or ClaudeClient(book.table(ANTHROPIC), settings.anthropic_api_key)
    providers = ProviderRegistry(settings.provider_configs(), book, claude=claude_client)
    runner = Runner(
        sessions, settings, claude=claude_client, providers=providers, allow_import_paths=False
    )
    worker = None
    if settings.max_active_runs:
        worker = Worker(
            sessions,
            runner,
            slots=settings.max_active_runs,
            poll_seconds=settings.worker_poll_seconds,
            drain_seconds=settings.drain_seconds,
            listen=listen_url(settings.database_url),
        )
    svc = Services(settings, sessions, runner, worker)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        stop = asyncio.Event()
        serving = None if worker is None else asyncio.create_task(worker.serve(stop))
        try:
            yield
        finally:
            stop.set()
            if serving is not None:
                await serving
            await providers.aclose()
            await claude_client.aclose()

    app = FastAPI(title="evalkit", version="1", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.services = svc
    app.middleware("http")(request_context)
    app.add_exception_handler(EvalkitError, _evalkit_error)
    app.add_exception_handler(_UnauthenticatedError, _unauthenticated)
    app.add_exception_handler(RequestValidationError, _invalid)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(Exception, _internal)
    app.add_api_route("/healthz", healthz, methods=["GET"])
    app.add_api_route("/metrics", prometheus, methods=["GET"], dependencies=[Depends(authenticate)])
    app.include_router(router)
    return app


def build(settings: Settings) -> tuple[ASGIApp, Engine]:
    """The production app: migrations applied, body limit in front."""
    if not settings.auth_enabled and not settings.host_is_loopback:
        raise ConfigError("refusing to listen on a non-loopback address without EVALKIT_API_TOKENS")
    engine = make_engine(settings.database_url)
    upgrade(engine)
    app = create_app(settings, sessions=make_session_factory(engine))
    return BodyLimit(app, settings.max_body_bytes), engine


def serve(settings: Settings) -> None:
    configure_logging(settings.log_level)
    scrubber.register(*settings.secret_values())
    asgi, engine = build(settings)
    log.info("listening", extra={"host": settings.host, "port": settings.port})
    try:
        uvicorn.run(
            asgi,
            host=settings.host,
            port=settings.port,
            log_config=None,
            access_log=False,
            server_header=False,
            timeout_graceful_shutdown=30,
        )
    finally:
        engine.dispose()
