"""HTTP API: datasets, runs and comparisons for a team sharing one evaluation store.

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
from typing import Annotated, Any, cast

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
from evalkit.db.models import Dataset, Result, Run, as_utc, utcnow
from evalkit.errors import EvalkitError, NotFoundError
from evalkit.llm import ClaudeClient
from evalkit.logs import configure_logging, log_context, scrubber
from evalkit.pricing import PriceTable
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
    claude_client = claude or ClaudeClient(PriceTable(settings.prices), settings.anthropic_api_key)
    runner = Runner(sessions, settings, claude=claude_client, allow_import_paths=False)
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
