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
RESULTS_REFUSED = Counter(
    "evalkit_results_refused_total",
    "Results not saved because another process had taken the case over (its lease expired).",
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
RATE_LIMIT_REFUNDS = Counter(
    "evalkit_rate_limit_refunds_total",
    "Reserved rate-limit slots handed back unused (caller cancelled, or a cool-down began).",
    registry=REGISTRY,
)
WORKER_RUNS = Gauge(
    "evalkit_worker_runs", "Runs this worker daemon holds a slot on.", registry=REGISTRY
)
RUNS_CANCELLED = Counter(
    "evalkit_runs_cancelled_total",
    "Cancellations requested through the API or the CLI.",
    registry=REGISTRY,
)
CACHE_SWEPT = Counter(
    "evalkit_cache_swept_total",
    "Response cache entries deleted by a sweep (too old, or over the size limit).",
    registry=REGISTRY,
)
BLOBS_COLLECTED = Counter(
    "evalkit_blobs_collected_total",
    "Unreferenced blobs deleted by garbage collection.",
    registry=REGISTRY,
)
BLOB_FAILURES = Counter(
    "evalkit_blob_failures_total",
    "Blob store problems: write (output cached inline), unreachable (read later), "
    "unusable (missing or corrupt blob; entry dropped).",
    ["reason"],
    registry=REGISTRY,
)
