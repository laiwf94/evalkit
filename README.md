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

## Trying it

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

[examples/github-action/eval-gate.yml](examples/github-action/eval-gate.yml)
is a workflow that evaluates the base branch and the pull request and fails the
job on a regression.

## Run configs: dataset, target, scorers

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

The dataset is a JSONL file of test cases with ids, inputs, expected values,
tags and splits. Each import is versioned and identified by a SHA-256 of its
content, so re-importing the same file is a no-op, and two runs can only be
compared if they used exactly the same cases.

The target is the thing under test: a prompt template on a Claude model, your
application behind an HTTP endpoint named in server config (never a URL from
the request), or a fake target with known accuracy for demos and tests.

Scorers grade each answer by exact match, regex, JSON Schema, numeric
tolerance, token F1, or an LLM judge with a rubric. The pairwise judge runs
every comparison twice with the answers swapped, so a judge that prefers a
position cannot decide the result, and judges are calibrated against human
labels with Cohen's kappa. Your own scorers plug in through an entry point.

## During a run

A run uses a bounded worker pool, per-provider rate limits, retries that honour
Retry-After across all workers, and a response cache that makes reruns free.
On PostgreSQL the rate limits and Retry-After cool-downs are shared by every
process, so a fleet of workers on one API key stays within that key's limit.
A run killed at any point resumes without repeating or re-paying for finished
cases. Runs submitted to the server wait in a queue, highest priority first, for
`evalkit worker` daemons on any number of machines; several workers share one
run (up to a per-run cap), a cancel stops every one of them within moments, and
a worker sent SIGTERM finishes the cases it has in flight, hands the rest back
and exits. Prompt targets can go through the Message Batches API at half price.

The response cache stays bounded: entries record when they were last used and
how big they are, and `evalkit cache sweep` (or every worker, when a limit is
set) removes entries unused for longer than a maximum age and then the least
recently used ones above a maximum size. Long outputs can be kept in a
content-addressed blob store, a directory or any S3-compatible bucket, with
only the digest in the database; every read checks the bytes against it.

Tokens and dollars are counted per case and per run, including prompt cache
reads and writes and judge calls, and `evalkit estimate` prices a run before
you start it, using the token counting endpoint.

Outputs are redacted (emails, phone numbers, card numbers, keys and your own
patterns) before they are stored or shown to the judge. Keys come from the
environment or secret files and are scrubbed from logs.

## Reading the verdict

Each run reports means with bootstrap confidence intervals (normal intervals
above 20,000 scored cases, where the two agree). Summaries come from per-run
aggregates kept with every result write and from one compact score array per
metric written when the run completes, so reading a run's summary costs the
same at a hundred cases or a million. Two runs are
compared pairwise, with a paired bootstrap interval and a sign-flip permutation
test, giving a better / worse / no detectable change verdict and an effect
size; Holm correction applies when several metrics are gated together.

## Command line

| Command | Purpose |
|---|---|
| `evalkit run CONFIG [--label L]` | Run an evaluation; prints the summary. |
| `evalkit run --resume RUN_ID` | Continue an interrupted or crashed run. |
| `evalkit run CONFIG --queue` | Queue the run for worker daemons and exit. |
| `evalkit worker [--runs N]` | Execute queued runs until SIGTERM, then drain. |
| `evalkit cancel RUN_ID` | Cancel a queued or running run. |
| `evalkit show RUN` | Summary of a run. |
| `evalkit run verify RUN_ID [--repair]` | Recompute a run's aggregates and score arrays from its results; exit 3 on any difference, or rewrite them with `--repair`. |
| `evalkit compare BASE CAND` | Paired comparison per metric. |
| `evalkit gate BASE CAND` | Same, and exit 1 if any gated metric is worse. |
| `evalkit estimate CONFIG` | Cost estimate from token counts. |
| `evalkit calibrate --judge J LABELS` | Judge agreement with human labels. |
| `evalkit dataset import/export/list` | Manage datasets. |
| `evalkit cache sweep [--max-age 30d] [--max-size 20GiB] [--dry-run]` | Delete old and least recently used cache entries, then unreferenced blobs. |
| `evalkit cache scrub` | Read every blob and check it against its digest; exit 3 if any is corrupt. |
| `evalkit serve`, `evalkit db upgrade` | Run the API; migrate PostgreSQL. |

`RUN` is a run id or `label:NAME` for the newest completed run with that
label. Exit codes: 0 success, 1 regression, 2 bad input, 3 the run cannot be
used (failed, incomplete, a batch-mode run held by another process, different
datasets) or any
other error, 130 interrupted. Commands that print a report take `--json`.

## Running it as a service

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

The server executes up to `EVALKIT_MAX_ACTIVE_RUNS` runs itself (default 2).
To spread runs over a fleet, set it to 0 on the server and start workers
against the same PostgreSQL database:

```console
$ evalkit worker --runs 4                 # or: docker run ... evalkit worker --runs 4
```

Workers wake on PostgreSQL notifications, so a run starts as soon as it is
submitted, and they poll every `EVALKIT_WORKER_POLL_SECONDS` (5) as a fallback.
On SIGTERM a worker drains for up to `EVALKIT_DRAIN_SECONDS` (25), which fits
the default Kubernetes grace period of 30 seconds. A run's `[queue]` table (or
`"queue"` in the API body) sets `priority` (higher first, -1000 to 1000) and
`max_workers` (how many workers may share it).

To bound the response cache, set `EVALKIT_CACHE_MAX_AGE` (for example `30d`)
and/or `EVALKIT_CACHE_MAX_BYTES` (`20GiB`); workers then sweep every
`EVALKIT_CACHE_SWEEP_SECONDS` (3600), one at a time. To keep long outputs out
of the database, point `EVALKIT_BLOB_URL` at a directory (`file:///var/lib/evalkit/blobs`)
or a bucket (`s3+https://minio.internal:9000/evalkit/blobs`, with
`EVALKIT_BLOB_ACCESS_KEY` and `EVALKIT_BLOB_SECRET_KEY`); cached outputs of
`EVALKIT_BLOB_MIN_BYTES` (64 KiB) or more go there. Unreferenced blobs are
collected once they are older than `EVALKIT_BLOB_GRACE` (1h).

Every `/v1` route takes `Authorization: Bearer <token>`.

| Endpoint | Purpose |
|---|---|
| `POST /v1/datasets/{name}` | Import JSONL (body). 201 for a new version, 200 if the content already exists. |
| `GET /v1/datasets` | All datasets and versions. |
| `GET /v1/datasets/{name}/versions/{v}[/export]` | Metadata, or the cases as JSONL. |
| `POST /v1/runs` | Queue a run from a JSON run config. 202 with the run summary. |
| `GET /v1/runs[?label=]` | Recent runs. |
| `GET /v1/runs/{id}` | Summary: progress, metrics with intervals, tokens, cost, latency. |
| `GET /v1/runs/{id}/results?after=&limit=` | Per-case outputs and scores, paged. |
| `POST /v1/runs/{id}/resume` | Put an interrupted or failed run back in the queue. |
| `POST /v1/runs/{id}/cancel` | Cancel a run: `cancelled` at once if nobody is executing it, else `cancelling` until every worker has stopped. 409 for a completed run. |
| `POST /v1/compare` | Paired comparison of two runs. |
| `GET /healthz`, `GET /metrics` | Health (no auth, checks the database) and Prometheus metrics. |

Errors look like `{"error":{"code":"invalid_argument","message":"...","request_id":"..."}}`
with stable codes: `invalid_argument` 400, `unauthenticated` 401, `not_found`
404, `conflict` 409, `failed_precondition` 412, `payload_too_large` 413,
`internal` 500, `unavailable` 503.

## Tests

No test calls the real API. The Claude client is exercised through the real SDK
over a fake HTTP transport that checks the exact request body; one opt-in test
(`pytest -m live`) makes a real call when `ANTHROPIC_API_KEY` is set.

| What | How |
|---|---|
| Statistics | Monte Carlo on synthetic data with a known answer: 95% intervals cover the true mean 93.5% (normal) and 95.3% (Bernoulli) of 1,000 trials; the permutation test rejects 4.2% of true nulls at alpha 0.05; a 0.5 SD drop on 100 pairs is caught in at least 95% of trials. Hypothesis tests for invariants (shift equivariance, sign symmetry, Holm monotonicity). Breaking the methods on purpose (one-sided test, unpaired differences, wrong interval tails) makes these tests fail. |
| Hashing and redaction | Hypothesis: the dataset hash ignores key order, whitespace and line order but changes with any field; generated emails and Luhn-valid card numbers are always removed; redaction is idempotent. A catastrophic regex is stopped by its time limit. |
| Worker daemons | On PostgreSQL, an API that only queues and four `evalkit worker` processes share an 800-case run; one is SIGKILLed and the others finish it with every case recorded once. A cancel reaches workers by notification in well under the 20 s their heartbeat would take, a SIGTERMed worker exits 0 after handing its cases to the other, and eight threads taking slots at once on a run capped at three get exactly three. In-process tests cover priority order, caps, drains that finish in-flight cases and drains that time out. |
| Crash safety | A CLI run on PostgreSQL is killed with SIGKILL mid-run; a resume joins at once and takes the dead process's cases over when their leases expire, with every case recorded exactly once. Four processes share one run on PostgreSQL and one is SIGKILLed: the other three finish it, each case recorded and paid for once. Eight threads claiming at once get disjoint batches. In-process tests interrupt runs, take cases over from under a running process (its late results are refused) and check that finished work is saved and not repeated. |
| Claim races | On PostgreSQL each race is forced into a chosen order by holding one transaction open while another statement runs: a claim never waits for, or takes, rows another uncommitted claim holds; a stalled owner's late result waits for an uncommitted takeover and is then refused; a takeover passes over cases whose results are being saved; a renewal cannot win taken cases back; two first joiners seed the leases once; three processes finishing together complete the run and compact it once. On SQLite a Hypothesis state machine drives three runners through random interleavings of claims, fenced writes (some stale), renewals, releases, stalls and completions, and after every step checks the leases, the results and the aggregates against a model: each case is saved once, by the holder of its latest fence, and the aggregates equal a recomputation. Weakening the fence check or dropping SKIP LOCKED makes these tests fail. |
| Rate limits | Hypothesis drives random admissions, refunds and cool-downs through the in-memory limiter, through `acquire()` with concurrent callers and cancellations, and through the shared limiter's SQL on both SQLite and PostgreSQL, and checks that no window of w seconds sees more than burst + w × rate starts and none falls inside a cool-down. Eight workers on one PostgreSQL key at 200 requests/s stay within 5% of it, and a 429 on one pauses all of them. |
| SSRF | Loopback, private, link-local, metadata, CGNAT and IPv4-in-IPv6 addresses are refused; a DNS answer that changes after the check cannot redirect the request. |
| End to end | Testcontainers PostgreSQL 16, the real server as a subprocess, a stand-in app over HTTP, the CLI gate against the same database. |
| Aggregates | Summaries and comparisons from aggregates and score arrays equal those computed from the results; a Hypothesis test folds random result batches into random shards and checks the totals equal one recomputation; `run verify` finds tampered counts, sums, arrays and percentiles and `--repair` fixes them; on PostgreSQL the four-worker run with one SIGKILLed worker verifies clean. |
| Cache and blobs | A Hypothesis test sweeps random caches by age and size and checks the dry run reports exactly what the real sweep removes, that what is left fits the limit and is always the most recently used. An entry used between the sweep's read and its delete survives. Both blob backends round-trip random bytes; tampered blobs are refused; garbage collection never removes a referenced or recent blob (Hypothesis) and spares one written again mid-scan. Requests are signed in our own code and checked by an in-memory S3 server that recomputes every signature, and on a real S3-compatible server (versitygw in Testcontainers) together with a sweep and collection on PostgreSQL. |
| API and CLI | Every endpoint, error code and exit code, over ASGI and over the wire. |

```console
$ ./scripts/check.sh                 # format, lint, mypy --strict, tests with coverage
$ uv run pip-audit
$ uv run python benchmarks/run.py
$ uv run python benchmarks/summary.py --cases 1000000   # summary on a huge run
```

## Speed

On an Apple M4 Max, SQLite, fake targets (so this is the harness, not a model).
Repeated runs on a busy machine vary by up to a factor of two for the shortest
timings, so ranges are given.

| Case | Result |
|---|---|
| Overhead per case, zero-latency target, 3 scorers | 170-220 µs |
| Throughput, 50 ms target, 1 / 4 / 16 / 64 workers | 19 / 77 / 305 / 1,072 cases/s (97% / 97% / 95% / 84% of ideal) |
| 400-case rerun served from the cache | 34-71 ms vs 2.6 s cold (37-77x) |
| Paired comparison, 1,000 cases, 10,000 resamples | 47-50 ms |

On PostgreSQL 16 in Docker on the same machine the overhead is 250-300 µs per
case and a cached rerun takes 98 ms (27x); throughput is the same.

At 64 workers 400 cases are only six rounds deep, so the last partial round
and result writes show up; model latencies of seconds hide both.

## Inside the package

The reasoning behind the data model, the paired statistics, the judge and its
calibration, caching and resume, rate limits, failure modes, limits, security
and what would change at larger scale is in [docs/DESIGN.md](docs/DESIGN.md).

```text
src/evalkit/datasets.py    JSONL parsing, content hashing, versioned storage
src/evalkit/targets/       prompt, HTTP (SSRF-safe) and fake targets; templates
src/evalkit/scorers/       built-in scorers, plugin registry, LLM judge, calibration
src/evalkit/llm.py         the Claude client: caching, structured output, fallback, cost
src/evalkit/runner.py      worker pool, cache, resume, run completion, batch mode, cancel, drain
src/evalkit/cache.py       response cache: keys, last use, sweep by age and size, blob offload
src/evalkit/blobs.py       content-addressed blobs: filesystem and S3 backends, signing, GC
src/evalkit/leases.py      case leases: SKIP LOCKED claims and fenced result writes
src/evalkit/runqueue.py    the run queue: states, priorities, worker slots and caps, cancel
src/evalkit/worker.py      the worker daemon: LISTEN/NOTIFY wake-up, heartbeats, SIGTERM drain
src/evalkit/stats.py       bootstrap, permutation test, effect size, Holm
src/evalkit/aggregates.py  per-run aggregates, compact score arrays, run verify
src/evalkit/report.py      run summaries and comparisons
src/evalkit/estimate.py    cost estimates from token counts
src/evalkit/ratelimit.py   GCRA limiter, cool-downs, retries
src/evalkit/ratelimit_pg.py  the same limiter shared through PostgreSQL
src/evalkit/redaction.py   PII redaction
src/evalkit/api.py         HTTP API
src/evalkit/cli.py         command line
src/evalkit/db/            SQLAlchemy models and Alembic migrations
examples/                  demo, bundled dataset, live configs, CI workflow
benchmarks/                harness benchmarks
```

## License

MIT
