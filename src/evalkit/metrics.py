"""Prometheus metrics. A private registry keeps /metrics to what evalkit defines."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

HTTP_REQUESTS = Counter(
    "evalkit_http_requests_total",
    "HTTP requests by route and status.",
    ["method", "route", "status"],
    registry=REGISTRY,
)
HTTP_LATENCY = Histogram(
    "evalkit_http_request_duration_seconds",
    "HTTP request latency by route.",
    ["route"],
    registry=REGISTRY,
)
RUNS_ACTIVE = Gauge("evalkit_runs_active", "Runs executing in this process.", registry=REGISTRY)
RUNS_FINISHED = Counter(
    "evalkit_runs_finished_total",
    "Runs that stopped, by final status.",
    ["status"],
    registry=REGISTRY,
)
CASES = Counter(
    "evalkit_cases_total",
    "Cases finished, by outcome (ok, error) and source (target, cache, batch).",
    ["status", "source"],
    registry=REGISTRY,
)
TARGET_ATTEMPTS = Counter(
    "evalkit_target_attempts_total",
    "Calls made to targets, including retries.",
    ["provider"],
    registry=REGISTRY,
)
TOKENS = Counter(
    "evalkit_tokens_total",
    "Tokens billed by kind (input, output, cache_read, cache_write) for targets and judges.",
    ["kind"],
    registry=REGISTRY,
)
COST = Counter("evalkit_cost_usd_total", "Spend in US dollars.", ["role"], registry=REGISTRY)
