# evalkit design

"The candidate scored 2 points lower" is not a decision. evalkit exists to say
whether a difference is larger than the noise for this many cases, and to exit
non-zero only when it is. That verdict is only worth something if the two runs
are comparable, so runs are compared only when they used the same cases, byte
for byte, and every output is tied to the exact target that produced it.
Evaluations run on every pull request, which means an unchanged target has to
cost nothing the second time and a crash must not throw away (or re-bill)
finished work. And because the harness also runs as a shared service, inputs
are bounded, it makes no requests to arbitrary URLs, and neither the store nor
the logs hold personal data or keys.

It does not evaluate model weights at training time, and it is not a human
annotation tool (calibration labels are imported, not collected). Several
processes, on several machines, can share one run; what else changes at scale
is discussed under [Bigger runs](#bigger-runs).

## What is stored

```text
datasets        (name, version) unique, (name, content_hash) unique
dataset_cases   (dataset_id, case_id) unique; input, expected, tags, split, metadata (JSON)
runs            target spec + target hash, scorer specs, options, filter, seed,
                status, lease (owner, expires_at; batch mode only), batch_id,
                queued_at, priority, max_workers
case_leases     (run_id, case_id) key; position, owner, expires_at, fence, done
run_workers     (run_id, worker_id) key; joined_at, expires_at: a worker daemon's slot
results         (run_id, case_id) unique; redacted output, scores (JSON), tokens,
                cost, judge tokens and cost, attempts, latency, error
response_cache  key = sha256(target hash, input, redaction fingerprint)
judge_calibrations  judge hash, kappa, weighted kappa, agreement, confusion matrix
```

### How a dataset is identified

The content hash is SHA-256 over the cases in canonical
JSON (sorted keys, no whitespace), sorted by case id. Key order, whitespace and
line order do not change it; any change to an input, expected value, tag, split
or metadata does. Importing content that already exists under the name returns
the existing version, so a CI job can import on every run without minting
versions. Concurrent imports race on the unique constraints and the loser
re-reads, so the same content never gets two versions.

### How a target is identified

The target hash covers every field that can change an
output: model, system prompt, template, `max_tokens`, effort, fallback setting,
and for HTTP targets the resolved URL and header names (not the secret values).
It keys the response cache, so a cache hit is only possible when nothing that
could change the answer changed.

### Runs keep their own config

The target spec, scorer specs, options and
case filter are copied onto the run row when it is created. Resuming a run
uses the stored config, never the current file, so a run cannot change
definition halfway through.

## How a run executes

```text
work queue -> N workers -> (cache hit? -> score) | (rate limit -> call -> retry) -> redact -> score
                                                                                          |
                                                    results queue -> one writer task -> short transactions
```

Workers pull cases from an in-memory queue. A worker never holds a database
connection: cache entries for the whole run are loaded in one query up front,
and results go to a single writer task that commits in batches of up to 50 or
every 200 ms. A slow model call therefore never pins a pooled connection, and
the bounded results queue (200 entries) gives backpressure if the database is
slower than the target.

### Resuming

A result row is unique on (run, case). On resume the runner loads
the case ids that already have a row and only executes the rest. A crash loses
at most the results still in memory (the in-flight calls plus whatever the
writer had not committed); on Ctrl-C or shutdown even those are written before
the process stops. The kill -9 test in `tests/e2e` checks the exactly-once
property against PostgreSQL.

### Case leases

Any number of processes can execute one run, and a case must never be
executed (and paid for) twice. When a run is first joined, every case still to
do gets a row in `case_leases`. A process claims a batch of rows (four per
worker, at least 64) with one UPDATE whose candidates come from
`SELECT ... FOR UPDATE SKIP LOCKED`, so processes claiming at the same moment
take disjoint batches without waiting for each other. A claim sets an owner, an
expiry and a fence token one higher than the last. The process renews its
leases every third of the lease length (30 s by default) and claims again when
its queue is half empty.

A result is written in the same transaction that marks its lease done, and only
if the lease still has the owner and token of the claim. A process that stalls
past its expiry can lose a case to another process, which claims it with a
higher token; the stalled process's late result is then refused, so the case is
recorded once. If a process dies, its leases expire and the others take the
cases over: a crashed run waits up to one lease length for the cases that were
in flight, and none for the rest. On Ctrl-C a process writes its finished
results and gives its other leases back at once.

A process that finds nothing to claim waits while other processes still hold
cases (until their first lease could expire, or one heartbeat), and keeps
claiming until every case is finished, including its own. Then it marks the run
completed with a compare-and-set (`status <> 'completed' AND NOT EXISTS
(unfinished lease)`), so exactly one of the processes that finish together
completes the run. Worker clocks must agree to well within one lease length.

On SQLite there are no row locks: the claiming UPDATE takes the database write
lock, which serialises claims, and within one process claims and result writes
take turns on a thread lock rather than SQLite's busy-wait. Claims, and the
lease update in each result write, add 20-30% to the harness overhead on a
target that answers instantly (about 100 µs per case on SQLite, 160 µs on
PostgreSQL, on the benchmark machine) and under 5% of throughput on a 50 ms
target.

A batch-mode run is the exception. One Message Batch is one paid request for
the whole run, so a batch-mode run is held by one process at a time with a
lease on the run row (a conditional UPDATE that succeeds only if the lease is
free, its own, or expired); a second process gets `conflict`, and a process
that finds the lease taken stops.

### The run queue and worker daemons

A run submitted through the API, or with `evalkit run --queue`, is `queued`.
`evalkit worker` daemons execute queued runs, and so does a worker inside the
API process with `EVALKIT_MAX_ACTIVE_RUNS` slots (0 turns it off, so the API
only accepts runs). A run created by a plain `evalkit run` is `pending` and
belongs to the process that created it; workers never pick it up.

```text
pending --(CLI executes)--------------------------> running --> completed
queued  --(a worker takes a slot)-----------------> running --> completed
running --(every process stops, cases left)-------> interrupted --(queued run)--> queued
running --(cancel)--> cancelling --(last process stops)--> cancelled
queued, pending, interrupted, failed --(cancel)--> cancelled
interrupted, failed --(resume)--> queued
```

A worker has a number of run slots. When one is free it asks for the best
run it can join: `queued`, or `running` with cases nobody holds (a dead
worker's cases once their leases expire), under the run's `max_workers`,
highest `priority` first and oldest first within a priority. A worker does not
join a running run whose cases are all held by others: it would only wait.
Taking a slot locks the run row with an UPDATE, counts the live slots and adds
a `run_workers` row only if the cap allows. On PostgreSQL a second worker
waits for the row lock and then, under READ COMMITTED, counts again with the
first worker's slot in view, so eight workers racing for a run capped at three
get exactly three. Slots carry an expiry that the worker renews every third of
a lease length, so a dead worker's slot frees itself. A batch-mode run is
capped at one worker, because its batch is one paid request.

Inside a slot the worker calls the same runner as the CLI, so workers share a
run through case leases. Workers wake on PostgreSQL `LISTEN evalkit_runs`:
queuing a run, freeing a slot and requesting a cancel each send a NOTIFY in
the transaction that makes the change, so it is delivered on commit and never
before the change is visible. Workers also poll (every 5 s by default), which
covers SQLite, a dropped notification connection (reconnected with backoff)
and anything missed while disconnected.

Cancelling marks the run `cancelling` and notifies. A worker executing it
cancels its execution at once: results already finished are written, its
leases are released, and the last process to stop marks the run `cancelled`.
A process that gets no notification (a CLI process, or SQLite) sees the status
at its next heartbeat, within a third of a lease length. If nobody is
executing the run, the cancel makes it `cancelled` straight away; if the
processes executing it died, or two stopped at the same moment and each saw
the other still working, a worker's periodic sweep finishes the cancellation
once no slot, case lease or run lease on it is live.

On SIGTERM or SIGINT a worker drains. It takes no new slots; each of its runs
stops claiming, gives back the cases still in its queue so other workers can
claim them at once, and lets the cases in flight finish and be written, for up
to `EVALKIT_DRAIN_SECONDS` (25 s, inside Kubernetes' default 30 s grace
period). Calls still running after that are abandoned, and their cases are
executed again by whoever takes them over. A run the worker leaves
`interrupted` goes back to `queued`; if other processes still execute it, it
stays `running`. A second signal skips the wait. The API's embedded worker
drains the same way on shutdown, so a restarted single server picks its runs
up again by itself.

### The response cache

Outputs are cached after redaction, keyed by (target hash,
canonical input, redaction fingerprint). Cache hits cost nothing in the new run;
the original usage stays on the cache row. Errors are never cached, and neither
is an output whose redaction timed out, so a rerun retries both. Judge verdicts
go in the same table, keyed by the judge configuration and the exact prompt, so
a rerun of an unchanged target with an unchanged judge makes no API calls at
all. `use_cache = false` turns both off.

### Rate limits and retries

Each provider (the Claude API, each HTTP endpoint,
the judge) has a GCRA limiter: one number per provider, the theoretical arrival
time, which a call advances by one interval (1 / rate) and which may run up to
`burst - 1` intervals ahead. A call reserves a slot and sleeps until it, so in
any window of `w` seconds at most `burst + w * rate` calls start. When any call
gets a Retry-After, the whole provider enters a cool-down that every worker
waits out; without this, 32 workers would each spend an attempt on the same
429. The arrival time is also pushed past the end of the cool-down, so the
burst does not all go out the moment it ends. A reserved slot that goes unused,
because its caller was cancelled or a cool-down began while it slept, is handed
back, but only while it is still the latest slot: handing back an earlier one
would let a later caller share a slot that is already taken.

On SQLite the limiter lives in memory. On PostgreSQL, where several processes
may execute one run (and several runs one provider), it lives in a row of
`rate_limits` (`ratelimit_pg.py`), so the rate and the cool-downs are fleet-wide:

- An admission is one `UPDATE ... RETURNING` in autocommit. Callers in one
  process that queue up while a statement is in flight are admitted together by
  the next, which advances the row by one interval each, so one process can
  admit about 50,000 calls a second (measured against PostgreSQL 16 in Docker)
  rather than one per round trip.
- Times are the database server's clock, and callers only sleep for the
  difference between two readings of it, so worker clocks need not agree.
- Every admission returns the current cool-down. A caller whose knowledge of it
  is older than 100 ms when its slot arrives reads it again, so a 429 on one
  worker pauses the others within about 100 ms. A provider with no configured
  rate takes no slots and only shares cool-downs, with one primary-key read per
  100 ms per process made in the background.
- With 8 workers of 4 callers each on one key at 200 requests a second, the
  end-to-end test measures 200.2 a second, and no 100 ms window exceeds the
  limit. Failures without
Retry-After back off exponentially with full jitter (base 0.5 s, cap 30 s).
Retry-After is honoured up to 120 s by default so a misbehaving server cannot
park a run for an hour. The SDK's own retries are turned off so there is
exactly one retry policy.

### Batch mode

Prompt targets can send their uncached cases as one Message
Batch at half price. The batch id is written to the run before polling starts,
so a resumed run polls the same batch rather than paying for a second one. The
window between the batch being created and its id being committed is the one
place a crash can cost a duplicate batch. Entries that errored, expired or were
cancelled go through the normal online path.

## Calling Claude

All model calls go through one client (`llm.py`):

- The system prompt carries `cache_control`. It is the same for every case of a
  run (the prompt target's system prompt, the judge's rubric), so after the
  first request it is read from the prompt cache at a fraction of the input
  price. Prompts shorter than the model's minimum cacheable length are simply
  not cached.
- Structured output (`output_config.format` with a JSON schema) constrains the
  judge's verdict. Current models do not accept forced tool choice, and a schema
  is what is actually wanted.
- Effort is set explicitly (default `medium`). Current models have no
  temperature, top-p or seed parameter, so a model's output cannot be made
  deterministic from the client. Reproducibility comes from the response cache
  instead: a rerun of the same target returns the stored outputs. Everything
  evalkit itself randomises (fake targets, bootstrap and permutation resampling,
  retry jitter) is seeded from the run's seed.
- Server-side refusal fallback (`fallbacks: "default"`) is on by default where
  the model supports it. A refusal that survives the fallback is recorded as an
  empty answer with `stop_reason = refusal`, and the partial text before a
  refusal is discarded.
- Cost is computed from the response's usage. With a fallback, each iteration is
  priced by the model that ran it. Cache reads and writes have their own rates,
  and batch requests are priced at half.
- Requests with `max_tokens` above 16,000 are streamed, because the SDK refuses
  non-streaming requests that may take longer than ten minutes.

The pinned SDK (1.x) sends API version `2023-06-01`. No request sends the
sampling parameters that the newer API version removes, so moving to the 2.x SDK
is a version bump.

`evalkit estimate` counts input tokens with the token counting endpoint for
every uncached target prompt and every judge call, and prices output at the
worst case of `max_tokens` per call. Never a tokenizer from another vendor.

## Scoring

Scores are numbers in [0, 1]. A scorer can also return "unscorable" (for
example, the case has no expected value), which is different from 0.

| Scorer | Score |
|---|---|
| `exact` | 1 if equal after optional case folding and whitespace normalisation. |
| `regex` | 1 if the pattern (from the config or the case) matches. 250 ms limit per match. |
| `json_schema` | 1 if the output is JSON that validates (Draft 2020-12). References are never fetched. |
| `numeric` | 1 if the first, last or whole number is within absolute/relative tolerance. |
| `token_f1` | Bag-of-tokens F1 after SQuAD normalisation (Rajpurkar et al.): partial credit without embeddings. |
| `judge` | Rubric 1-5 mapped to (s-1)/4, or pairwise win/tie/loss against the reference. |
| `plugin` | Any `Scorer` subclass from an `evalkit.scorers` entry point. |

A failed call scores 0 and an unscorable case is left out. If the target
errored after all retries, every metric gets 0 for that case: the system did
not produce an answer, and the gate must see that. Unscorable cases are counted
separately and excluded from that metric.

Scorers see redacted text, because redaction happens before scoring. The judge
therefore never receives personal data, and a cached rerun (which only has
the redacted text) scores exactly like the original. The cost is that an
expected answer which itself contains, say, an email address will not match;
disable that pattern for such datasets.

## The judge

The judge prompt puts the rubric in the system prompt and the task, reference
and response in tagged sections, and tells the judge the response is data to be
graded. Outputs of the system under test can contain instructions; the judge is
asked to ignore them, and structured output means it can only answer with a
verdict.

### Position bias

LLM judges prefer the first (or second) answer more often
than content warrants (Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and
Chatbot Arena*). Pairwise judging runs every comparison twice with the order
swapped and averages the two outcomes from the response's point of view (win 1,
tie 0.5, loss 0). A judge that always picks "A" scores every response 0.5, a
tie, instead of deciding the result by position. The disagreement is noted in
the score detail. The cost is two calls per case.

### Calibration

`evalkit calibrate` runs a judge over outputs that people have
labelled and reports raw agreement, Cohen's kappa (Cohen, *A Coefficient of
Agreement for Nominal Scales*) and, for the ordinal 1-5 scale, quadratic
weighted kappa (Cohen, *Weighted kappa*), which counts a 4-vs-5 disagreement as
less serious than 1-vs-5. Kappa corrects for chance: a judge that always says 5
can agree 80% of the time with people who give 5 to 80% of cases, with kappa 0.
As a rough guide, below 0.4 a judge should not gate a merge; 0.6-0.8 is
substantial agreement (Landis & Koch, *The Measurement of Observer Agreement for
Categorical Data*). Results are stored per judge configuration hash (mode,
model, effort, rubric), so a rubric edit is visibly uncalibrated.

## Deciding whether a change is real

Evaluation datasets are small (tens to a few hundred cases) and scores are
often binary, so a raw difference of means is mostly noise. The comparison is
paired: the same case is scored in both runs, and the test works on per-case
differences. Case difficulty is shared between the two runs and cancels out,
which is why a paired test detects a regression with a fraction of the cases an
unpaired test would need (Berg-Kirkpatrick et al., *An Empirical Investigation
of Statistical Significance in NLP*).

The interval for one run is a percentile bootstrap of the mean with 10,000
resamples (Efron & Tibshirani, *An Introduction to the Bootstrap*): simple and
distribution-free. It runs slightly narrow for small n; the coverage test
measures 93.5% for a nominal 95% at n = 100. BCa would correct that at the cost
of a jackknife pass, but the gate does not use this interval, so the simpler
method was kept. The interval for a difference is the same bootstrap on paired
differences.

The test is a sign-flip permutation test on paired differences. Under "no
difference" each difference is equally likely to have either sign. It is exact
(all 2^n sign patterns) for n <= 16 and uses 10,000 random flips above that,
with the +1 correction so the p-value is never 0 (Phipson & Smyth, *Permutation
P-values Should Never Be Zero*). It makes no normality assumption, which
matters for 0/1 scores, where a t-test is shaky at small n.

Effect size is reported two ways: the mean difference itself, in score units,
and Cohen's d_z, the mean difference over the standard deviation of the
differences. When several metrics are gated, p-values are adjusted with Holm's
step-down method (Holm, *A Simple Sequentially Rejective Multiple Test
Procedure*), so gating on five metrics does not give five chances of a false
alarm.

The verdict is `worse` or `better` when the adjusted p-value is below alpha
(0.05) and the mean difference is larger than `--min-effect` in size, and
`no detectable change` otherwise. That phrase is deliberate: with 30 cases a
3-point drop may be real and still undetectable. `--min-effect` is how a team
says which differences it does not care about.

The property tests check the methods, not one lucky draw: 1,000 simulated
datasets with a known answer per test, checked against a binomial tolerance
band.

## Interrupted and failed runs

| Failure | What happens |
|---|---|
| Process killed mid-run | Finished, committed cases are kept. Other processes on the run, or a `run --resume`, take its cases over when its leases expire (30 s). |
| A process stalls past its lease | Another process takes its cases over; the stalled process's results for them are refused. |
| Ctrl-C | Queued results are written and the process's other leases released; the run is marked `interrupted` unless another process is still working on it. |
| SIGTERM to a worker, or server shutdown | The worker drains: cases in flight finish and are written (up to 25 s), unstarted cases are released, and the run goes back to `queued` unless another process is still working on it. |
| A worker is killed | Its slots expire after one lease length, its cases are taken over when their leases expire, and the run stays joinable by other workers. |
| Run cancelled | Every process executing it stops within moments on PostgreSQL (a third of a lease without notifications); finished results are kept and the run ends `cancelled`. |
| Two processes resume the same run | They share it. For a batch-mode run, the second fails with `conflict` while the first holds the lease. |
| 429 or 529 from the API | Provider-wide cool-down for Retry-After, shared by every process on PostgreSQL, then retry; after the attempt limit the case is recorded as an error and scores 0. |
| Target returns 4xx, redirect, non-JSON or an oversized body | Recorded as an error for that case, not retried. |
| A scorer plugin raises | That score is `unscorable` with the message; the run continues. |
| Redaction pattern times out | The stored output is replaced by `[REDACTION_FAILED]` and not cached. |
| Judge refuses, runs out of tokens or returns bad JSON | That score is `unscorable` with the reason. |
| Database unavailable | `/healthz` returns 503; a running run fails and can be resumed. |
| Runs compared over different dataset contents | `failed_precondition` (CLI exit 3), never a misleading verdict. |

## Limits

| Limit | Value | Why |
|---|---|---|
| Cases per dataset | 100,000 | Bounds import memory and summary cost. |
| JSONL line | 1 MiB | Checked before parsing. |
| API request body | 16 MiB (configurable, max 256 MiB) | Checked by Content-Length and while reading. |
| Workers per run | 1-256 | |
| Concurrent runs per server or worker | 2 (configurable, 0-64) | Later runs wait as `queued` for a free slot on any worker. |
| Workers per run | `max_workers`, unlimited by default | Batch-mode runs: 1. |
| Run priority | -1000 to 1000 | |
| Worker poll interval | 5 s default | Fallback when no notification arrives. |
| Drain on SIGTERM | 25 s default | Inside Kubernetes' default 30 s grace period. |
| Scorers per run | 32 | |
| Template, system prompt | 200,000 characters | |
| Regex patterns | 1,000 characters, 250 ms per match | Patterns can come from case data. |
| Redaction patterns | 64, 1,000 characters each, 250 ms each | |
| HTTP target response | 1 MiB default, 64 MiB max | Checked before buffering. |
| HTTP target timeout | 30 s default | |
| Case timeout | 600 s default, retries included | |
| Retry attempts | 5 default, 20 max | |
| Retry-After honoured | up to 120 s by default | |
| Judge `max_tokens` | 256-16,000 | |
| Calibration items | 5,000 | |
| Result page | 500 | |
| Case lease | 30 s default | Time before a dead process's cases are taken over. |
| Cases per claim | 4 per worker, 64-500 | Bounds what a dead process strands for one lease length. |

## Running it as a shared service

The main risk is server-side request forgery. HTTP targets name an endpoint
from the operator's config file; a run config cannot contain a URL. On each
call the host is resolved and every address must be public: loopback, private,
link-local (which includes the cloud metadata address 169.254.169.254), CGNAT,
multicast and reserved ranges are refused, including IPv4 addresses embedded in
IPv6 (mapped, 6to4, Teredo). The request then goes to the checked IP literal
with the original Host header and TLS server name, so a DNS answer that changes
between check and connect cannot redirect it. Proxy environment variables are
ignored, redirects are not followed, and the response size is capped while
streaming. `allow_private` exists per endpoint for local development.

Nothing a caller sends is executed. Templates only substitute `{{ field }}`;
there is no expression language. Plugin scorers by import path are accepted
from CLI config files only, and the API accepts installed entry points by name.
Every regex from config or case data runs on the `regex` module with a time
limit, so a catastrophic pattern cannot stall a worker.

API keys come from environment variables or `*_FILE` secret files. Bearer
tokens are stored as SHA-256 digests and compared with `hmac.compare_digest`
against every configured digest, without stopping early. Known secret values
and credential-shaped strings are scrubbed from every log line, and endpoint
secret headers are excluded from the target hash and never stored. Outputs are
redacted before they are stored, cached or sent to the judge, and a pattern
that times out fails closed.

Without tokens the server refuses to listen beyond loopback. Bodies are
size-capped before buffering, unknown JSON fields are rejected, internal errors
return a generic message with the request id, and `/metrics` is behind the same
tokens as `/v1`.

## Third-party packages

Each pinned to an exact version released at least two weeks before the lock
was made (`exclude-newer` in `pyproject.toml`). The exception is a release that
fixes a published vulnerability: `certifi` and `filelock` (both transitive) are
allowed newer versions because `pip-audit` flagged the older ones.

| Package | Why it is here |
|---|---|
| `anthropic` | The official SDK: typed responses, streaming, batches, token counting. |
| `httpx2` | The HTTP client the SDK is built on; reused for HTTP targets so there is one client stack and one set of timeouts. |
| `fastapi`, `uvicorn` | Request validation from the same pydantic models the CLI uses, and an ASGI server with graceful shutdown. |
| `pydantic` | Strict validation of every config, request and stored spec. |
| `sqlalchemy`, `alembic`, `psycopg` | One schema on PostgreSQL and SQLite, versioned migrations. |
| `numpy` | Vectorised resampling: 10,000 bootstrap resamples of 1,000 cases in about 30 ms. Pure Python is two orders of magnitude slower. |
| `regex` | Per-match time limits; the standard `re` module has none. |
| `jsonschema`, `referencing` | Draft 2020-12 validation without remote reference retrieval. |
| `prometheus-client` | Standard metrics exposition. |
| dev: `pytest`, `hypothesis`, `testcontainers`, `ruff`, `mypy`, `pip-audit` | Tests, property tests, real PostgreSQL in tests, lint and format, types, vulnerability audit. |

## Bigger runs

Several things would change for many teams and much larger datasets.

Summaries are computed from the result rows on request, which is fine at tens of
thousands of cases. Beyond that, the writer would maintain per-run aggregates as
results are committed.

The response cache grows without bound. A last-used timestamp and a sweep by
age, or partitioning by month, would bound it.

Long outputs would move to object storage, keyed by content hash, with only the
reference left in the results table.
