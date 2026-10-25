# evalkit

An evaluation and regression-testing harness for LLM applications: you keep a
dataset of test cases, run a prompt, a model or your deployed app against it,
and score every answer. Two runs are then compared case by case with a paired
statistical test, and the CLI exits non-zero when a change made things worse,
so a pull request that breaks your prompt fails CI like any other test.

Written in Python 3.12+ with FastAPI, SQLAlchemy on PostgreSQL (or SQLite for
local runs) and the official `anthropic` SDK.

```console
$ evalkit run examples/baseline.toml
$ evalkit run examples/candidate-regressed.toml
$ evalkit gate label:baseline label:candidate-regressed
metric              base    cand     diff                95% CI   p (adj)    d_z  verdict
exact              89.2%   65.8%   -23.3%        [-33.3, -13.3]    0.0001  -0.43  worse

FAIL: exact
$ echo $?
1
```

A run config is a small TOML file:

```toml
label = "prompt-v2"

[dataset]
name = "support"
path = "data/support.jsonl"     # imported (and versioned) on first use

[target]
type = "prompt"
model = "claude-opus-5-5"
system = "You answer questions about the acme web app..."
template = "{{question}}"

[[scorers]]
type = "token_f1"

[[scorers]]
type = "judge"
rubric = "5: names the same menu path as the reference ..."
```

## What it does

- **Datasets**: JSONL test cases with ids, inputs, expected values, tags and
  splits. Each import is versioned and identified by a SHA-256 of its content,
  so re-importing the same file is a no-op and two runs can only be compared if
  they used exactly the same cases.
- **Targets**: a prompt template on a Claude model, your application behind an
  HTTP endpoint named in server config (never a URL from the request), or a fake
  target with known accuracy for demos and tests.
- **Scorers**: exact match, regex, JSON Schema, numeric tolerance, token F1, and
  an LLM judge with a rubric. The pairwise judge runs every comparison twice
  with the answers swapped, so a judge that prefers a position cannot decide the
  result. Judges are calibrated against human labels with Cohen's kappa. Your
  own scorers plug in through an entry point.
- **Runs**: a bounded worker pool, per-provider rate limits, retries that honour
  Retry-After across all workers, and a response cache that makes reruns free.
  A run killed at any point resumes without repeating or re-paying for finished
  cases. Prompt targets can go through the Message Batches API at half price.
- **Statistics**: means with bootstrap confidence intervals, paired comparison
  of two runs (paired bootstrap interval and a sign-flip permutation test) with
  a better / worse / no detectable change verdict, effect size, and Holm
  correction when several metrics are gated together.
- **Cost**: tokens and dollars per case and per run, including prompt cache
  reads and writes and judge calls, and an estimate before you run, from the
  token counting endpoint.
- **Safety**: outputs are redacted (emails, phone numbers, card numbers, keys
  and your own patterns) before they are stored or shown to the judge. Keys come
  from the environment or secret files and are scrubbed from logs.

## Quick start

```console
$ uv sync
$ ./examples/demo.sh                 # no API key needed: fake targets, throwaway SQLite
```

The demo runs a baseline and two candidates on a bundled 120-case dataset. The
candidate of similar quality passes the gate; the broken one fails it with exit
code 1.

With a key, the same comparison on Claude, scored by a judge:

```console
$ export ANTHROPIC_API_KEY=...
$ evalkit estimate examples/live/baseline.toml      # input cost exactly, output at worst case
$ evalkit run examples/live/baseline.toml
$ evalkit run examples/live/candidate.toml
$ evalkit gate label:live-baseline label:live-candidate
$ evalkit calibrate --judge examples/live/judge.toml examples/live/labels.jsonl
```

As a service:

```console
$ export EVALKIT_DATABASE_URL=postgresql+psycopg://evalkit@localhost/evalkit
$ export EVALKIT_API_TOKENS=$(openssl rand -hex 24)
$ evalkit serve                                     # migrates, then listens on 127.0.0.1:8080
```

Or with Docker:

```console
$ docker build -t evalkit .
$ docker run -p 8080:8080 -e EVALKIT_API_TOKENS=... -e EVALKIT_DATABASE_URL=... evalkit
```

[examples/github-action/eval-gate.yml](examples/github-action/eval-gate.yml)
is a workflow that evaluates the base branch and the pull request and fails the
job on a regression.

## CLI

| Command | Purpose |
|---|---|
| `evalkit run CONFIG [--label L]` | Run an evaluation; prints the summary. |
| `evalkit run --resume RUN_ID` | Continue an interrupted or crashed run. |
| `evalkit show RUN` | Summary of a run. |
| `evalkit compare BASE CAND` | Paired comparison per metric. |
| `evalkit gate BASE CAND` | Same, and exit 1 if any gated metric is worse. |
| `evalkit estimate CONFIG` | Cost estimate from token counts. |
| `evalkit calibrate --judge J LABELS` | Judge agreement with human labels. |
| `evalkit dataset import/export/list` | Manage datasets. |
| `evalkit serve`, `evalkit db upgrade` | Run the API; migrate PostgreSQL. |

`RUN` is a run id or `label:NAME` for the newest completed run with that
label. Exit codes: 0 success, 1 regression, 2 bad input, 3 the run cannot be
used (failed, incomplete, different datasets), 130 interrupted. Commands that
print a report take `--json`.

## API

Every `/v1` route takes `Authorization: Bearer <token>`.

| Endpoint | Purpose |
|---|---|
| `POST /v1/datasets/{name}` | Import JSONL (body). 201 for a new version, 200 if the content already exists. |
| `GET /v1/datasets` | All datasets and versions. |
| `GET /v1/datasets/{name}/versions/{v}[/export]` | Metadata, or the cases as JSONL. |
| `POST /v1/runs` | Start a run from a JSON run config. 202 with the run summary. |
| `GET /v1/runs[?label=]` | Recent runs. |
| `GET /v1/runs/{id}` | Summary: progress, metrics with intervals, tokens, cost, latency. |
| `GET /v1/runs/{id}/results?after=&limit=` | Per-case outputs and scores, paged. |
| `POST /v1/runs/{id}/resume` | Continue an interrupted run. |
| `POST /v1/compare` | Paired comparison of two runs. |
| `GET /healthz`, `GET /metrics` | Health (no auth, checks the database) and Prometheus metrics. |

Errors look like `{"error":{"code":"invalid_argument","message":"...","request_id":"..."}}`
with stable codes: `invalid_argument` 400, `unauthenticated` 401, `not_found`
404, `conflict` 409, `failed_precondition` 412, `payload_too_large` 413,
`internal` 500, `unavailable` 503.

## How it is tested

No test calls the real API. The Claude client is exercised through the real SDK
over a fake HTTP transport that checks the exact request body; one opt-in test
(`pytest -m live`) makes a real call when `ANTHROPIC_API_KEY` is set.

| What | How |
|---|---|
| Statistics | Monte Carlo on synthetic data with a known answer: 95% intervals cover the true mean 93.5% (normal) and 95.3% (Bernoulli) of 1,000 trials; the permutation test rejects 4.2% of true nulls at alpha 0.05; a 0.5 SD drop on 100 pairs is caught in at least 95% of trials. Hypothesis tests for invariants (shift equivariance, sign symmetry, Holm monotonicity). Breaking the methods on purpose (one-sided test, unpaired differences, wrong interval tails) makes these tests fail. |
| Hashing and redaction | Hypothesis: the dataset hash ignores key order, whitespace and line order but changes with any field; generated emails and Luhn-valid card numbers are always removed; redaction is idempotent. A catastrophic regex is stopped by its time limit. |
| Crash safety | A CLI run on PostgreSQL is killed with SIGKILL mid-run; a resume is refused while the dead process's lease is live, then completes with every case recorded exactly once. In-process tests interrupt runs and check that finished work is saved and not repeated. |
| SSRF | Loopback, private, link-local, metadata, CGNAT and IPv4-in-IPv6 addresses are refused; a DNS answer that changes after the check cannot redirect the request. |
| End to end | Testcontainers PostgreSQL 16, the real server as a subprocess, a stand-in app over HTTP, the CLI gate against the same database. |
| API and CLI | Every endpoint, error code and exit code, over ASGI and over the wire. |

```console
$ ./scripts/check.sh                 # format, lint, mypy --strict, tests with coverage
$ uv run pip-audit
$ uv run python benchmarks/run.py
```

## Performance

On an Apple M4 Max, SQLite, fake targets (so this is the harness, not a model).
Repeated runs on a busy machine vary by up to a factor of two for the shortest
timings, so ranges are given.

| Case | Result |
|---|---|
| Overhead per case, zero-latency target, 3 scorers | 170-220 µs |
| Throughput, 50 ms target, 1 / 4 / 16 / 64 workers | 19 / 77 / 305 / 1,072 cases/s (97% / 97% / 95% / 84% of ideal) |
| 400-case rerun served from the cache | 34-71 ms vs 2.6 s cold (37-77x) |
| Paired comparison, 1,000 cases, 10,000 resamples | 47-50 ms |

At 64 workers 400 cases are only six rounds deep, so the last partial round
and result writes show up; model latencies of seconds hide both.

## Design

[docs/DESIGN.md](docs/DESIGN.md) covers the data model, the statistics and why
they are paired, the judge and its calibration, caching and resume, rate limits,
failure modes, limits, security and what would change to run it at larger
scale.

## Layout

```text
src/evalkit/datasets.py    JSONL parsing, content hashing, versioned storage
src/evalkit/targets/       prompt, HTTP (SSRF-safe) and fake targets; templates
src/evalkit/scorers/       built-in scorers, plugin registry, LLM judge, calibration
src/evalkit/llm.py         the Claude client: caching, structured output, fallback, cost
src/evalkit/runner.py      worker pool, cache, leases, resume, batch mode
src/evalkit/stats.py       bootstrap, permutation test, effect size, Holm
src/evalkit/report.py      run summaries and comparisons
src/evalkit/estimate.py    cost estimates from token counts
src/evalkit/ratelimit.py   token buckets, cool-downs, retries
src/evalkit/redaction.py   PII redaction
src/evalkit/api.py         HTTP API
src/evalkit/cli.py         command line
src/evalkit/db/            SQLAlchemy models and Alembic migrations
examples/                  demo, bundled dataset, live configs, CI workflow
benchmarks/                harness benchmarks
```

## License

MIT
