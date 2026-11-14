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

The target is the thing under test: a prompt template on a model (Claude by
default, or `provider = "openai"`, `"gemini"` or a server named in
`EVALKIT_PROVIDERS_FILE`), your application behind an HTTP endpoint named in
server config (never a URL from the request), or a fake target with known
accuracy for demos and tests.

Scorers grade each answer by exact match, regex, JSON Schema, numeric
tolerance, token F1, or an LLM judge with a rubric, and an agent's tool calls
by the trajectory scorers below. The pairwise judge runs
every comparison twice with the answers swapped, so a judge that prefers a
position cannot decide the result, and judges are calibrated against human
labels with Cohen's kappa. A judge can run on Claude, OpenAI, Gemini or any
OpenAI-compatible server (`provider = "openai"`, keys in `OPENAI_API_KEY` and
`GEMINI_API_KEY`, other servers named in `EVALKIT_PROVIDERS_FILE`), so a model
need not grade its own family. Each run records which judge graded it, and the
gate refuses to compare runs whose judges differ unless given
`--allow-judge-change`. Your own scorers plug in through an entry point.

## Model matrices

A `[matrix]` table runs the same dataset and scorers over several models,
efforts and output limits, one child run per combination:

```toml
label = "support"
# [dataset], [target] (a prompt) and [[scorers]] as above, then:

[matrix]
model = [
  "claude-opus-5-5",
  { provider = "openai", model = "gpt-5.1" },
  { provider = "gemini", model = "gemini-3-pro", name = "gemini" },
]
effort = ["low", "high"]        # optional; so is max_tokens = [1024, 4096]
```

`evalkit run` executes the runs and prints the matrix report. On the stand-in
providers of `tests/test_matrix.py` (three models, 120 cases, latency fixed):

```console
$ evalkit report matrix label:models
matrix cb7291fe08cc4362a4b484b9949e7494  models  dataset qa v1
quality exact  latency p50  frontier over quality, cost, latency

variant            status           exact            95% CI   cost/case   p50 ms  frontier
claude-sonnet-5-5  completed        96.7%    [93.3%, 99.2%]   $0.000400      800  *
local:gpt-test     completed        64.2%    [55.8%, 72.5%]   $0.000280      500  dominated by gemini
gemini             completed        86.7%    [80.0%, 92.5%]   $0.000140      300  *

along the frontier, cheapest first (paired; p adjusted by Holm over 1 step(s), alpha 0.05):
  gemini -> claude-sonnet-5-5: +10.0% [+3.3, +16.7]  p 0.0120  n 120  better
```

The report marks the Pareto frontier over quality (the first gated metric, or
`--metric`), the target's cost per case and latency (`--latency p50|p95`): a
variant is dominated when another is at least as good on all three and better
on one. Neighbours on the frontier are compared case by case with the same
paired test as `evalkit gate`, so you can see whether the dearer model's gain is
real. Each child is an ordinary run labelled `support[gemini,effort=low]`, with
its own tokens and cost; `--queue` hands all of them to worker daemons and
`evalkit run --resume MATRIX_ID` finishes an interrupted matrix.

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
you start it, using the provider's token counting endpoint. OpenAI-compatible
servers have none, so their input tokens are estimated from the text's length
and the estimate says so (`~1,234 tokens (estimated)`).

Each provider prices from its own versioned list (`anthropic@2026-11-09`,
`openai@2026-11-09`, `gemini@2026-11-09`). `EVALKIT_PRICES_FILE` overrides or
extends them, with `[prices.<model>]` for every provider or
`[prices.<provider>.<model>]` for one; a self-hosted server prices nothing until
it gets a table there or names a list with `price_list = "openai"` in
`EVALKIT_PROVIDERS_FILE`:

```toml
[prices.local."llama-3.3-70b"]   # a server named local in EVALKIT_PROVIDERS_FILE
input = 0.0
output = 0.0

[prices.openai."gpt-5"]          # a negotiated price
input = 1.0
output = 8.0
cache_read = 0.1
```

An override changes the table's version (`openai@2026-11-09+local.1a2b3c4d`).
A run records the version and the price of every model it uses, `evalkit show`
prints them, and a worker whose prices for those models differ refuses to
execute the run, so one run's costs never mix two price lists.

Outputs are redacted (emails, phone numbers, card numbers, keys and your own
patterns) before they are stored or shown to the judge. Keys come from the
environment or secret files and are scrubbed from logs.

## Conversations

A `conversation` target holds a dialogue with a simulated user, one case per
dialogue. The case input names the goal, and optionally a persona, the opening
messages, a turn limit and, for the scripted user, what the user says and the
phrases that mean the goal is met:

```json
{"id": "b001", "input": {"goal": "Book a table for two at 7pm", "persona": "In a hurry",
  "user_turns": ["Hi", "A table for two", "7pm, book it"], "success": ["confirmation"],
  "max_turns": 6}}
```

```toml
[target]
type = "conversation"
model = "claude-opus-5-5"            # the assistant, on any provider
system = "You take restaurant bookings..."

[target.simulator]
type = "llm"                         # or "scripted": no calls, deterministic
provider = "openai"
model = "gpt-5.1"
```

A dialogue ends when the simulated user says the goal is reached, gives up, or
the assistant has replied `max_turns` times. The result's output is the
assistant's last reply and its stop reason is how the dialogue ended
(`goal_reached`, `gave_up`, `max_turns`, or `no_reply` for an empty answer),
so existing scorers apply. Every generated turn is stored as its own row with
its tokens, cost and latency. Each turn is cached by a hash of the dialogue so
far as soon as it is paid for, so a run killed mid-dialogue resumes from the
turns it has, and a rerun of an unchanged target costs nothing.

### Tool sandboxes

Agents call tools, and an evaluation must not book real flights or depend on
live data, so a tool sandbox answers instead. A sandbox file lists each tool
with a JSON Schema for its arguments (and optionally its result), scripted
responses matched on a subset of the arguments, call limits, and injected
latency and failures:

```toml
[sandbox]
max_calls = 20                       # per session (case), answered or not
fixtures = "fixtures/booking.jsonl"  # recorded answers, beside this file

[[sandbox.tools]]
name = "search_flights"
description = "Find flights between two airports on a date."
input_schema = { type = "object", required = ["origin", "destination", "date"], properties = { origin = { type = "string" }, destination = { type = "string" }, date = { type = "string" } } }
max_calls = 3
latency_ms = 80                      # plus up to jitter_ms
failure_rate = 0.05                  # or fail_calls = [2]: the second call fails
endpoint = "flights"                 # where `evalkit tools record` gets real answers

[[sandbox.tools.responses]]
when = { origin = "LHR" }            # a subset of the arguments
result = { flights = [{ id = "BA117", price = 420 }] }
times = 1                            # then the next matching rule
```

A call is answered by the first scripted rule that matches, else by a recorded
fixture for exactly those arguments, else with a `no_response` error. Unknown
tools, invalid arguments (with the schema's reason), calls over a limit and
injected failures are error results the agent can react to, never exceptions,
and every call is kept in order as the case's trajectory. Failures and jitter
are seeded per case, so a rerun sees the same ones.

`evalkit tools record SANDBOX CALLS` sends each `{"tool", "arguments"}` line of
CALLS to the tool's endpoint (one from `EVALKIT_HTTP_TARGETS_FILE`, through the
SSRF-checked client) as `{"tool": ..., "arguments": ...}`, expects
`{"result": ...}` or `{"error": "..."}` back, and writes the answers as
fixtures. Calls already recorded are skipped unless `--refresh`, so an
interrupted recording resumes; a call that fails is reported (exit 3) and the
rest are still written. `evalkit tools check SANDBOX` validates the file and
its fixtures and prints the sandbox's identity hash.

### Agent targets

An `agent` target gives a model on any provider (Claude, an OpenAI-compatible
server or Gemini) the case as a task and the sandbox's tools, and runs the same
tool loop on all of them: the model replies, every tool it asks for is answered
by the sandbox and sent back, until it answers without asking for a tool or has
replied `max_steps` times.

```toml
[target]
type = "agent"
provider = "anthropic"               # or any provider in EVALKIT_PROVIDERS_FILE
model = "claude-opus-5-5"
system = "You book restaurant tables with the tools you have."
template = "Book a table: {{ request }}"
max_steps = 12                       # model replies per case (stop reason max_steps)
sandbox = "booking.toml"             # read in, with its fixtures, by evalkit run
```

The output is the model's last reply and the stop reason is the provider's
own, or `max_steps`. The whole trajectory is stored per case in order, one row
per model reply (text, stop reason, tokens, cost, latency) and one per tool
call (arguments, the sandbox's status, the result or error the model got, and
whether a script or a fixture answered), and served at
`GET /v1/runs/{id}/results/{case}/trajectory`. Each model reply is cached by a
hash of the transcript so far as soon as it is paid for, so a run killed or a
case retried mid-trajectory replays the replies it has for nothing, and changing
one tool's answer reuses every reply before the model first saw it.

### Trajectory scorers

Four scorers grade what the agent did rather than only what it said. They read
the stored trajectory; for a target that is not an agent they are unscorable,
never zero. The reference comes from the config or from each case's `expected`
(`tool_calls`, `optimal_steps` and `forbidden_tools` keys by default):

```json
{"id": "b001", "input": {"request": "two at 7pm on Friday"},
 "expected": {"tool_calls": [{"tool": "find_slots", "arguments": {"party": 2}}, "book"],
              "optimal_steps": 2, "forbidden_tools": ["cancel_*"]}}
```

```toml
[[scorers]]
type = "tool_calls"                  # the matched fraction of expected calls
order = "ordered"                    # or "unordered"
arguments = "subset"                 # "exact", "schema" (a JSON Schema per call) or "ignore"
allow_extra = true                   # false: every unexpected call counts against it

[[scorers]]
type = "step_efficiency"             # min(1, budget / used)
count = "tool_calls"                 # or "model_steps"; budget = 4 to fix it in the config

[[scorers]]
type = "forbidden_tools"             # 0 if any call matches, answered or not
tools = ["delete_*"]

[[scorers]]
type = "trajectory_judge"            # an LLM judge of the whole trajectory, 1-5
provider = "openai"
model = "gpt-5.1"
rubric = "Looks availability up before booking and never books twice."
```

Ordered matching finds the longest run of expected calls the agent made in
order, so one wrong call early does not void every later one; unordered
matching pairs each expected call with a distinct actual call, as many as
possible. The trajectory judge is a judge like any other: its verdicts are
cached, its identity is recorded with the run and checked by the gate, and
`evalkit calibrate` measures it against people's labels when each labelled line
carries the `trajectory` the trajectory endpoint served (its `steps`).

## Reading the verdict

Each run reports means with bootstrap confidence intervals (normal intervals
above 20,000 scored cases, where the two agree). Summaries come from per-run
aggregates kept with every result write and from one compact score array per
metric written when the run completes, so reading a run's summary costs the
same at a hundred cases or a million. Two runs are
compared pairwise, with a paired bootstrap interval and a sign-flip permutation
test, giving a better / worse / no detectable change verdict and an effect
size; Holm correction applies when several metrics are gated together.

## Prompt registry

Prompts can live in evalkit instead of in each config. Every push of a prompt
file is a version, identified by the SHA-256 of its system prompt and template,
so pushing unchanged content (from CI on every commit, say) is a no-op:

```console
$ cat support.toml
system = "You answer questions about the acme web app..."
template = "{{question}}"
$ evalkit prompt push support support.toml --note "shorter system prompt"
support pushed version 4 (9c1f04e2b7aa)
$ evalkit prompt diff support@production support@4    # a unified diff
$ evalkit prompt label support@4 production
support@production: v3 -> v4
```

A run config names a version instead of giving `system` and `template`:

```toml
[target]
type = "prompt"
model = "claude-opus-5-5"
prompt = "support@production"     # or support@4, or support for the newest
```

The reference is resolved when the run is created and frozen into it: the run
stores the content, and `evalkit show` prints `prompt support@production:
support version 4 (9c1f04e2b7aa)`. Promoting another version afterwards does
not change that run, and a resume uses what it recorded. Content from the
registry hashes like the same text written inline, so the two share cache
entries.

Labels move only by compare-and-set. `evalkit prompt label` moves a label from
wherever it was when read and refuses (exit 3) if someone else moved it in the
meantime; `--expect 3` insists on a version, `--expect none` that the label is
new. Versions are never changed or deleted.

## Three providers, a judge from elsewhere and a promotion

[examples/providers](examples/providers) puts the pieces together: the bundled
120 support questions on Claude, OpenAI and Gemini from one config, graded by
a fourth model on a self-hosted OpenAI-compatible server (so no variant is
graded by its own family), with the prompt taken from the registry.

```console
$ export EVALKIT_PROVIDERS_FILE=examples/providers/providers.toml  # adds the grader
$ export EVALKIT_PRICES_FILE=examples/providers/prices.toml        # and what it costs you
$ export ANTHROPIC_API_KEY=... OPENAI_API_KEY=... GEMINI_API_KEY=...
$ evalkit prompt push support examples/providers/prompt-v1.toml
$ evalkit prompt label support@1 production
$ evalkit estimate examples/providers/matrix.toml    # OpenAI input marked (estimated)
$ evalkit run examples/providers/matrix.toml
```

`tests/test_providers_end_to_end.py` runs exactly these steps on strict fake
servers over real sockets, configured only through those variables (plus
`ANTHROPIC_BASE_URL`, and `openai` and `gemini` redefined in the providers
file), with models that answer a known share of cases correctly:

```console
$ evalkit report matrix label:providers
quality correct_path  latency p50  frontier over quality, cost, latency

variant                  status      correct_pa            95% CI   cost/case   p50 ms  frontier
claude-sonnet-5-5        completed        96.7%    [93.3%, 99.2%]   $0.000400       17  *
openai:gpt-5-mini        completed        64.2%    [55.8%, 72.5%]   $0.000045       14  *
gemini:gemini-2.5-flash  completed        82.5%    [75.8%, 89.2%]   $0.000055       15  *

along the frontier, cheapest first (paired; p adjusted by Holm over 2 step(s), alpha 0.05):
  openai:gpt-5-mini -> gemini:gemini-2.5-flash: +18.3% [+7.5, +29.2]  p 0.0027  n 120  better
  gemini:gemini-2.5-flash -> claude-sonnet-5-5: +14.2% [+6.7, +21.7]  p 0.0020  n 120  better
```

Each child run is billed from its own provider's price list and the judge from
the grader's, and records its prompt version, judge and prices. A verdict is
cached by judge, case and answer, so when two models give the same answer to a
case the second costs no judge call. Then a second prompt version is promoted
and the chosen model run on production again:

```console
$ evalkit prompt push support examples/providers/prompt-v2.toml --note "name the path"
$ evalkit prompt label support@2 production
support@production: v1 -> v2
$ evalkit run examples/providers/promoted.toml
$ evalkit gate "label:providers[gemini:gemini-2.5-flash]" label:promoted
metric              base    cand     diff                95% CI   p (adj)    d_z  verdict
correct_path       82.5%   93.3%   +10.8%         [+2.5, +19.2]    0.0182  +0.23  better

PASS: no regression
```

The matrix runs keep version 1 and the new run has version 2, and both were
graded by the same judge, so the gate pairs them case by case; with a
different judge it would refuse (exit 3) unless `--allow-judge-change` is given.

## Command line

| Command | Purpose |
|---|---|
| `evalkit run CONFIG [--label L]` | Run an evaluation; prints the summary. |
| `evalkit run --resume RUN_ID` | Continue an interrupted or crashed run, or the unfinished runs of a matrix. |
| `evalkit run CONFIG --queue` | Queue the run for worker daemons and exit. |
| `evalkit worker [--runs N]` | Execute queued runs until SIGTERM, then drain. |
| `evalkit cancel RUN_ID` | Cancel a queued or running run. |
| `evalkit show RUN` | Summary of a run. |
| `evalkit run verify RUN_ID [--repair]` | Recompute a run's aggregates and score arrays from its results; exit 3 on any difference, or rewrite them with `--repair`. |
| `evalkit compare BASE CAND` | Paired comparison per metric. |
| `evalkit gate BASE CAND [--allow-judge-change]` | Same, and exit 1 if any gated metric is worse; exit 3 if a metric's judge changed, unless allowed. |
| `evalkit report matrix MATRIX [--metric M] [--latency p50\|p95]` | A matrix's variants, their cost/quality/latency frontier and paired steps along it. |
| `evalkit estimate CONFIG` | Cost estimate from token counts (per variant for a matrix). |
| `evalkit calibrate --judge J LABELS` | Judge agreement with human labels. |
| `evalkit dataset import/export/list` | Manage datasets. |
| `evalkit prompt push NAME FILE [--system FILE] [--note N]` | Store a prompt file (TOML with `system` and `template`, or a template file) as a new version; unchanged content is a no-op. |
| `evalkit prompt show REF`, `evalkit prompt list` | A version's content and labels; every prompt with its newest version and labels. |
| `evalkit prompt diff REF [REF]` | Unified diff of two versions (the second defaults to the newest). |
| `evalkit prompt label REF LABEL [--expect V\|none]` | Point a label at a version by compare-and-set; exit 3 if it moved. |
| `evalkit tools check SANDBOX` | Validate a tool sandbox and its fixtures; print its identity. |
| `evalkit tools record SANDBOX CALLS [-o FILE] [--refresh]` | Record the tools' real answers to a JSONL list of calls as fixtures; exit 3 if any call failed. |
| `evalkit cache sweep [--max-age 30d] [--max-size 20GiB] [--dry-run]` | Delete old and least recently used cache entries, then unreferenced blobs. |
| `evalkit cache scrub` | Read every blob and check it against its digest; exit 3 if any is corrupt. |
| `evalkit serve`, `evalkit db upgrade` | Run the API; migrate PostgreSQL. |

`RUN` is a run id or `label:NAME` for the newest completed run with that
label; `MATRIX` is a matrix id or `label:NAME` for the newest matrix; `REF` is
`NAME`, `NAME@VERSION` or `NAME@LABEL`. Exit codes: 0 success, 1 regression, 2 bad input, 3 the run cannot be
used (failed, incomplete, a batch-mode run held by another process, different
datasets, a changed judge) or any
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
| `POST /v1/prompts/{name}/versions` | Push `{"template", "system", "note", "description"}`. 201 for a new version, 200 if the content already exists. |
| `GET /v1/prompts`, `GET /v1/prompts/{name}` | All prompts with their labels; one prompt's labels and versions. |
| `GET /v1/prompts/{name}/versions/{version\|label\|latest}` | A version and its content. |
| `GET /v1/prompts/{name}/diff?from=&to=` | Unified diff of two versions (`to` defaults to `latest`). |
| `PUT /v1/prompts/{name}/labels/{label}` | Move a label: `{"version": 4, "expected_version": 3}` (`null`: the label must be new). 409 with where it is now if it moved. |
| `POST /v1/runs` | Queue a run from a JSON run config. 202 with the run summary. |
| `GET /v1/runs[?label=]` | Recent runs. |
| `GET /v1/runs/{id}` | Summary: progress, metrics with intervals, tokens, cost, latency. |
| `GET /v1/runs/{id}/results?after=&limit=` | Per-case outputs and scores, paged. |
| `GET /v1/runs/{id}/results/{case}/trajectory` | One case's result with its agent steps (model replies and tool calls) or conversation turns, in order. |
| `POST /v1/runs/{id}/resume` | Put an interrupted or failed run back in the queue. |
| `POST /v1/runs/{id}/cancel` | Cancel a run: `cancelled` at once if nobody is executing it, else `cancelling` until every worker has stopped. 409 for a completed run. |
| `POST /v1/matrices` | Queue every run of a config with a `matrix`. 202 with the matrix id and run summaries. |
| `GET /v1/matrices/{id}[?metric=&latency=p95]` | The matrix report: variants, frontier, paired steps. |
| `POST /v1/compare` | Paired comparison of two runs. |
| `GET /healthz`, `GET /metrics` | Health (no auth, checks the database) and Prometheus metrics. |

Errors look like `{"error":{"code":"invalid_argument","message":"...","request_id":"..."}}`
with stable codes: `invalid_argument` 400, `unauthenticated` 401, `not_found`
404, `conflict` 409, `failed_precondition` 412, `payload_too_large` 413,
`internal` 500, `unavailable` 503.

## Tests

No test calls the real API. The Claude client is exercised through the real SDK
over a fake HTTP transport that checks the exact request body; one opt-in test
(`pytest -m live`) makes a real call when `ANTHROPIC_API_KEY` is set. The
OpenAI-compatible and Gemini clients run against `tests/fake_openai.py` and
`tests/fake_gemini.py`, which refuse any request a strict server would (an
unknown field, a second output limit, a system message out of place, a JSON
schema without the JSON MIME type, a wrong key) with the API's own error, and
tests compare the bodies they received with the exact bodies expected.

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
| The image as a fleet | `tests/e2e/test_containers.py` builds the image from the Dockerfile (or uses `EVALKIT_E2E_IMAGE`) and runs it as an API container and eight `evalkit worker` containers on Testcontainers PostgreSQL, with an S3-compatible blob store and a recording provider behind one key. At 200 requests/s the provider measures 199.8 to 201.1 a second across the eight (the test allows 5%), no second holds more than 209 calls, and after one 429 with Retry-After: 1 no worker calls again for a second. Four of them then share an 800-case run and one is killed with SIGKILL: every case is recorded once, only cases claimed again are called twice, `evalkit run verify` and `evalkit cache scrub` pass inside a container, and a cancelled run stops on all three remaining workers, which then exit 0 on SIGTERM. |
| Aggregates | Summaries and comparisons from aggregates and score arrays equal those computed from the results; a Hypothesis test folds random result batches into random shards and checks the totals equal one recomputation; latency percentiles from the histogram stay within 0.99% of the exact ones (Hypothesis); `run verify` finds tampered counts, sums, arrays, histograms and percentiles and `--repair` fixes them; on PostgreSQL the four-worker run with one SIGKILLed worker verifies clean. |
| Cache and blobs | A Hypothesis test sweeps random caches by age and size and checks the dry run reports exactly what the real sweep removes, that what is left fits the limit and is always the most recently used. An entry used between the sweep's read and its delete survives. Both blob backends round-trip random bytes; tampered blobs are refused; garbage collection never removes a referenced or recent blob (Hypothesis) and spares one written again mid-scan. Requests are signed in our own code and checked by an in-memory S3 server that recomputes every signature, and on a real S3-compatible server (versitygw in Testcontainers) together with a sweep and collection on PostgreSQL. |
| Three providers end to end | `tests/test_providers_end_to_end.py` serves strict fake Claude, OpenAI, Gemini and grader APIs on local sockets and drives the CLI (the matrix run as a child process) through `examples/providers` unchanged: each provider gets exactly its 120 requests with the registry prompt, token counts and costs match each provider's price list, the judge is called once per distinct answer and only on the grader, the frontier and both paired steps match the known accuracies, and a promoted prompt version runs, leaves the matrix runs on version 1 and passes the gate as better. |
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
src/evalkit/targets/       prompt, HTTP (SSRF-safe), conversation, agent and fake targets; templates
src/evalkit/simuser.py     simulated users: scripted, or played by a model in JSON
src/evalkit/tools/         the tool sandbox: schemas, scripted and recorded answers, limits,
                           latency and failure injection; recording fixtures; the tool loop
src/evalkit/scorers/       built-in and trajectory scorers, plugin registry, LLM judges, calibration
src/evalkit/providers/     the Provider protocol and error kinds; the Claude client (caching,
                           structured output, fallback, cost); raw HTTP OpenAI-compatible
                           and Gemini clients (JSON schema output, usage, safety blocks);
                           the registry of named providers targets and judges use; tool
                           use in one provider-neutral shape (tools.py)
src/evalkit/runner.py      worker pool, cache, resume, run completion, batch mode, cancel, drain
src/evalkit/matrix.py      [matrix] configs expanded into grouped child runs
src/evalkit/frontier.py    cost/quality/latency Pareto frontier and paired steps along it
src/evalkit/prompts.py     prompt registry: content-hashed versions, compare-and-set labels
src/evalkit/cache.py       response cache: keys, last use, sweep by age and size, blob offload
src/evalkit/blobs.py       content-addressed blobs: filesystem and S3 backends, signing, GC
src/evalkit/leases.py      case leases: SKIP LOCKED claims and fenced result writes
src/evalkit/runqueue.py    the run queue: states, priorities, worker slots and caps, cancel
src/evalkit/worker.py      the worker daemon: LISTEN/NOTIFY wake-up, heartbeats, SIGTERM drain
src/evalkit/stats.py       bootstrap, permutation test, effect size, Holm
src/evalkit/aggregates.py  per-run aggregates, latency histograms, score arrays, run verify
src/evalkit/report.py      run summaries and comparisons
src/evalkit/pricing.py     versioned per-provider price lists, overrides, run price records,
                           token estimates for providers that cannot count
src/evalkit/estimate.py    cost estimates from token counts
src/evalkit/ratelimit.py   GCRA limiter, cool-downs, retries
src/evalkit/ratelimit_pg.py  the same limiter shared through PostgreSQL
src/evalkit/redaction.py   PII redaction
src/evalkit/api.py         HTTP API
src/evalkit/cli.py         command line
src/evalkit/db/            SQLAlchemy models and Alembic migrations
examples/                  demo, bundled dataset, live and three-provider configs, CI workflow
benchmarks/                harness benchmarks
```

## License

MIT
