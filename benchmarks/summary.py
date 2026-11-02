"""How long a run summary takes on a very large run, through the HTTP API.

    uv run python benchmarks/summary.py                       # SQLite, 100,000 cases
    EVALKIT_BENCH_DATABASE_URL=postgresql+psycopg://... \\
        uv run python benchmarks/summary.py --cases 1000000

Writes a dataset and a run with one result per case straight into the database
(executing a million cases is not what is being measured), builds the run's
aggregates, then times ``GET /v1/runs/{id}`` while the run is running and after
it completed, and times reading one metric's score array, which is what a
comparison reads per metric.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import tempfile
import time
import uuid
from pathlib import Path

import httpx2
import numpy as np
from sqlalchemy import insert, select, update
from sqlalchemy.orm import Session, sessionmaker

from evalkit import aggregates
from evalkit.api import create_app
from evalkit.config import Settings
from evalkit.db import make_engine, make_session_factory, upgrade
from evalkit.db.models import Dataset, DatasetCase, Result, Run, utcnow

CHUNK = 20_000


def populate(sessions: sessionmaker[Session], cases: int) -> str:
    rng = random.Random(7)
    run_id = uuid.uuid4().hex
    with sessions() as s:
        dataset = Dataset(
            name=f"bench-{run_id[:8]}",
            version=1,
            content_hash=run_id + run_id,
            case_count=cases,
        )
        s.add(dataset)
        s.flush()
        s.add(
            Run(
                id=run_id,
                label="summary-bench",
                dataset_id=dataset.id,
                target_spec={"type": "fake"},
                target_hash="0" * 64,
                scorers=[{"type": "exact"}, {"type": "token_f1", "gate": False}],
                options={},
                case_filter={},
                seed=1,
                status="running",
                cases_total=cases,
                started_at=utcnow(),
            )
        )
        s.commit()
        for start in range(0, cases, CHUNK):
            ids = [f"case-{i:07d}" for i in range(start, min(cases, start + CHUNK))]
            s.execute(
                insert(DatasetCase),
                [
                    {"dataset_id": dataset.id, "position": start + j, "case_id": case_id,
                     "input": {"q": case_id}, "expected": "a", "tags": [], "meta": {}}
                    for j, case_id in enumerate(ids)
                ],
            )  # fmt: skip
            rows = []
            for case_id in ids:
                ok = rng.random() > 0.02
                rows.append(
                    {
                        "run_id": run_id,
                        "case_id": case_id,
                        "status": "ok" if ok else "error",
                        "output": "a" if ok else None,
                        "cached": rng.random() < 0.1,
                        "latency_ms": rng.lognormvariate(5, 0.5),
                        "attempts": 1,
                        "input_tokens": rng.randint(50, 500),
                        "output_tokens": rng.randint(5, 100),
                        "cost_usd": 1e-4,
                        "scores": {
                            "exact": {"value": float(rng.random() < 0.8)},
                            "token_f1": {"value": rng.random()} if rng.random() > 0.05 else {},
                        }
                        if ok
                        else {},
                    }
                )
            s.execute(insert(Result), rows)
            s.commit()
    return run_id


async def time_get(
    app: object, run_id: str, requests: int
) -> tuple[float, float, dict[str, object]]:
    transport = httpx2.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx2.AsyncClient(transport=transport, base_url="http://bench") as client:
        body: dict[str, object] = (await client.get(f"/v1/runs/{run_id}")).json()  # warm-up
        times = []
        for _ in range(requests):
            start = time.perf_counter()
            response = await client.get(f"/v1/runs/{run_id}")
            times.append((time.perf_counter() - start) * 1000)
            response.raise_for_status()
    p50, p99 = np.percentile(times, [50, 99]).tolist()
    return float(p50), float(p99), body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=100_000)
    parser.add_argument("--requests", type=int, default=50)
    args = parser.parse_args()
    url = os.environ.get("EVALKIT_BENCH_DATABASE_URL")
    tmp = None
    if url is None:
        tmp = tempfile.TemporaryDirectory()
        url = f"sqlite:///{Path(tmp.name) / 'bench.db'}"
    engine = make_engine(url)
    upgrade(engine)
    sessions = make_session_factory(engine)
    settings = Settings.from_env({"EVALKIT_DATABASE_URL": url, "EVALKIT_MAX_ACTIVE_RUNS": "0"})
    app = create_app(settings, sessions=sessions)
    print(f"database {engine.dialect.name}, {args.cases:,} cases")

    start = time.perf_counter()
    run_id = populate(sessions, args.cases)
    print(f"wrote dataset and results in {time.perf_counter() - start:.1f} s")

    with sessions() as s:
        start = time.perf_counter()
        aggregates.rebuild(s, s.get_one(Run, run_id))
        s.commit()
    print(f"aggregates from results (verify --repair): {time.perf_counter() - start:.1f} s")

    p50, p99, body = asyncio.run(time_get(app, run_id, args.requests))
    print(f"GET /v1/runs/{{id}} while running:   p50 {p50:.1f} ms  p99 {p99:.1f} ms")
    if body["completed"] != args.cases:
        raise SystemExit(f"summary counted {body['completed']} results, not {args.cases}")

    with sessions() as s:
        s.execute(update(Run).where(Run.id == run_id).values(status="completed"))
        s.commit()
        start = time.perf_counter()
        aggregates.compact(s, s.get_one(Run, run_id))
    print(f"compaction at completion: {time.perf_counter() - start:.1f} s")

    p50, p99, body = asyncio.run(time_get(app, run_id, args.requests))
    print(f"GET /v1/runs/{{id}} when completed:  p50 {p50:.1f} ms  p99 {p99:.1f} ms")
    print(json.dumps(body["metrics"], indent=None))

    with sessions() as s:
        run = s.get_one(Run, run_id)
        times = []
        for _ in range(10):
            start = time.perf_counter()
            arrays = aggregates.load_scores(s, run, ["exact"])
            times.append((time.perf_counter() - start) * 1000)
        size = s.scalar(select(Run.cases_total).where(Run.id == run_id))
    print(
        f"one metric's score array ({arrays['exact'].nbytes / 1e6:.1f} MB, {size:,} cases): "
        f"median {float(np.median(times)):.1f} ms"
    )
    engine.dispose()
    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":
    main()
