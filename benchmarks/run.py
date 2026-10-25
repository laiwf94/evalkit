"""Benchmarks for the harness itself. Prints the tables used in the README.

    uv run python benchmarks/run.py            # SQLite in a temp directory
    EVALKIT_BENCH_DATABASE_URL=postgresql+psycopg://... uv run python benchmarks/run.py

Every target is the fake target, so the numbers are the harness: scheduling,
scoring, redaction, cache lookups and result writes. Nothing calls a model.
"""

from __future__ import annotations

import asyncio
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from evalkit.config import Settings
from evalkit.datasets import Case, import_dataset
from evalkit.db import make_engine, make_session_factory, upgrade
from evalkit.runner import Runner
from evalkit.runspec import RunConfig
from evalkit.stats import bootstrap_ci, compare_many


def config(
    dataset: str, concurrency: int, latency_ms: float, use_cache: bool, seed: int
) -> RunConfig:
    return RunConfig.model_validate(
        {
            "dataset": {"name": dataset},
            "target": {"type": "fake", "latency_ms": latency_ms, "seed": seed, "accuracy": 0.9},
            "scorers": [
                {"type": "exact"},
                {"type": "token_f1"},
                {"type": "regex", "pattern": r"\d"},
            ],
            "options": {"concurrency": concurrency, "use_cache": use_cache},
        }
    )


class Bench:
    def __init__(self, url: str) -> None:
        self.engine = make_engine(url)
        upgrade(self.engine)
        self.sessions = make_session_factory(self.engine)
        self.runner = Runner(self.sessions, Settings())
        self._seed = 0

    def dataset(self, name: str, n: int) -> str:
        cases = [
            Case(
                id=f"{name}-{i}",
                input={"q": f"question {i} about item {i * 7}"},
                expected=f"answer {i}",
            )
            for i in range(n)
        ]
        with self.sessions() as s:
            import_dataset(s, name, cases)
        return name

    def time_run(self, cfg: RunConfig) -> float:
        run = self.runner.create_run(cfg)
        start = time.perf_counter()
        asyncio.run(self.runner.execute(run.id))
        return time.perf_counter() - start

    def fresh_seed(self) -> int:
        self._seed += 1
        return self._seed


def table(title: str, header: list[str], rows: list[list[Any]]) -> str:
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def cpu_name() -> str:
    if platform.system() == "Darwin":
        out = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        if out.stdout.strip():
            return out.stdout.strip()
    return platform.processor() or platform.machine()


def main() -> None:
    url = os.environ.get("EVALKIT_BENCH_DATABASE_URL")
    tmp = tempfile.TemporaryDirectory()
    if url is None:
        url = f"sqlite:///{Path(tmp.name) / 'bench.db'}"
    bench = Bench(url)
    backend = url.split(":", 1)[0]
    print(f"{cpu_name()}, Python {platform.python_version()}, {backend}\n")

    # 1. Overhead per case: a target that answers instantly.
    n = 2000
    ds = bench.dataset("overhead", n)
    rows = []
    for workers in (1, 16):
        bench.time_run(config(ds, workers, 0, False, bench.fresh_seed()))  # warm up
        elapsed = min(
            bench.time_run(config(ds, workers, 0, False, bench.fresh_seed())) for _ in range(3)
        )
        rows.append([workers, n, f"{elapsed * 1000:.0f} ms", f"{elapsed / n * 1e6:.0f} µs"])
    print(
        table(
            "Harness overhead (zero-latency target, 3 scorers)",
            ["workers", "cases", "run time", "per case"],
            rows,
        )
    )

    # 2. Throughput against a target with 50 ms latency.
    n = 400
    ds = bench.dataset("throughput", n)
    rows = []
    for workers in (1, 4, 16, 64):
        elapsed = bench.time_run(config(ds, workers, 50, False, bench.fresh_seed()))
        rate = n / elapsed
        ideal = workers / 0.05
        rows.append([workers, f"{rate:.0f}", f"{ideal:.0f}", f"{rate / ideal * 100:.0f}%"])
    print(
        table(
            "Throughput, 50 ms target, 400 cases",
            ["workers", "cases/s", "ideal", "efficiency"],
            rows,
        )
    )

    # 3. A rerun served from the cache.
    ds = bench.dataset("cache", 400)
    seed = bench.fresh_seed()
    cold = bench.time_run(config(ds, 8, 50, True, seed))
    warm = bench.time_run(config(ds, 8, 50, True, seed))
    print(
        table(
            "Cache: 400 cases, 50 ms target, 8 workers",
            ["run", "time", "speedup"],
            [
                ["first run", f"{cold:.2f} s", "1x"],
                ["rerun (all cached)", f"{warm * 1000:.0f} ms", f"{cold / warm:.0f}x"],
            ],
        )
    )

    # 4. Statistics on 1,000 paired cases, 10,000 resamples.
    rng = np.random.default_rng(0)
    base = {str(i): float(v) for i, v in enumerate(rng.random(1000) < 0.8)}
    cand = {str(i): float(v) for i, v in enumerate(rng.random(1000) < 0.78)}
    start = time.perf_counter()
    bootstrap_ci(list(base.values()))
    ci_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    compare_many({"m": (base, cand)})
    cmp_ms = (time.perf_counter() - start) * 1000
    print(
        table(
            "Statistics, 1,000 cases, 10,000 resamples",
            ["operation", "time"],
            [
                ["bootstrap CI for one run", f"{ci_ms:.0f} ms"],
                ["paired comparison (bootstrap + permutation)", f"{cmp_ms:.0f} ms"],
            ],
        )
    )
    bench.engine.dispose()
    tmp.cleanup()


if __name__ == "__main__":
    main()
