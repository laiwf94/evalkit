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
                queued_at, priority, max_workers, cases_total,
                latency p50 and p95 (set when the run is compacted),
                matrix_id and variant (for the child runs of a matrix),
                prompt (the registry reference and the version it resolved to),
                prices (per provider: the price table version and the price of
                each model the run uses)
matrices        a group of child runs from one [matrix] config: label, dataset,
                the axes as given and the base config
case_leases     (run_id, case_id) key; position, owner, expires_at, fence, done
run_workers     (run_id, worker_id) key; joined_at, expires_at: a worker daemon's slot
results         (run_id, case_id) unique; redacted output, scores (JSON), tokens,
                cost, judge tokens and cost, attempts, latency, error
result_turns    (run_id, case_id, turn) key; one generated turn of a conversation
                case: role, redacted content, stop reason or the simulated user's
                verdict, cached, latency, tokens and cost
run_aggregates  (run_id, shard) key; counts, token and cost sums of the results
                written by one process (the shard)
run_metric_aggregates  (run_id, shard, metric) key; n, sum, sum of squares, unscorable
run_latency_buckets  (run_id, shard, bucket) key; count of fresh successful results
                whose latency falls in one logarithmic bucket
run_scores      (run_id, metric) key; a completed run's scores as one float64 array
                in case-id order, n, unscorable, mean and interval
response_cache  key = sha256(target hash, input, redaction fingerprint); output (or
                its blob digest), created_at, last_used_at, size_bytes
judge_calibrations  judge hash, kappa, weighted kappa, agreement, confusion matrix
prompts         name unique; description
prompt_versions (prompt_id, version) unique, (prompt_id, content_hash) unique;
                system, template, note; inserted, never updated
prompt_labels   (prompt_id, label) key; the version it points at, updated_at
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
output: provider, model, system prompt, template, `max_tokens`, effort, fallback
setting, and for HTTP targets the resolved URL and header names (not the secret
values). A prompt target on a provider other than Claude also includes that
provider's kind and base URL, so pointing `local` at another server is a new
target. A Claude prompt target leaves the provider field out of the hash, which
keeps the hash every Claude target had before targets named a provider, and
with it every cache entry. It keys the response cache, so a cache hit is only
possible when nothing that could change the answer changed. A process refuses
to execute a run whose prompt target, on a provider other than Claude, would
hash differently with its own provider configuration than the recorded one, for
the same reason it refuses a changed judge. HTTP targets are not checked this
way, since workers may legitimately send their own non-secret headers (the
container fleet test tags each worker's calls).

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

`tests/e2e/test_containers.py` checks all of this with the image as shipped:
an API container that only queues, eight worker containers and an
S3-compatible blob store, all on one PostgreSQL database, calling a provider
that records every request. Eight workers share a rate-limited run and all
pause on one 429; four share the next run and one is killed with SIGKILL,
after which every case has exactly one result and only cases claimed again
reached the provider twice; a third run is cancelled while three workers
execute it; and every container exits 0 on SIGTERM.

### Run aggregates and score arrays

A summary must not read every result: a million-case run would take seconds
per request. The transaction that writes a batch of results also adds the
batch's counts (completed, ok, errors, cached, fresh), token and cost sums, and
each metric's count, sum, sum of squares and unscorable count to the run's
aggregates, and counts its fresh successful results per latency bucket, with one
`INSERT ... ON CONFLICT DO UPDATE` per table. The rows are
sharded by the writing process's owner id, so eight workers on one run never
wait on one hot row; the totals are a `SUM` over a handful of shard rows. A
refused result (a lease taken over) is never added, and a process killed before
its commit adds nothing, so the totals always equal the results.

When the process that completes the run wins the compare-and-set, it writes
`run_scores`: one row per metric holding every case's score as a float64 array
in case-id order (NaN for a case not selected or unscorable), the summary
interval computed once, and the exact latency p50 and p95. Case-id order is a
Python sort, independent of database collation, and it is the order the paired
comparison has always resampled in, so verdicts are unchanged. Two runs over the
same dataset content line up element by element, so a comparison reads one row
per metric per run. If that write fails, or the run completed before migration
0005, the first summary or comparison writes it instead.

A summary of a completed run reads the run row, its shard rows and the
`run_scores` metadata, never the results. A run that is not completed is
summarised exactly from its results up to 1,000 of them; above that its
intervals are normal intervals from the aggregates, and its latency percentiles
come from the latency histogram, so it reads no results either.

The histogram's buckets are logarithmic: bucket *i* holds latencies in
(γ^(i-1), γ^i] milliseconds with γ = 1.02, and stands for 2γ^i/(γ+1), which is
within 0.99% of every latency in it (the bucketing of DDSketch). Latencies under
1 µs share one bucket. The percentiles interpolate between order statistics as
`numpy.percentile` does, with each order statistic replaced by its bucket's
value, so each is within 0.99% of the exact percentile over all of the run's
fresh results; a Hypothesis test checks that bound. Production latencies spread
over a few hundred buckets at most, so a summary sums a few hundred rows
whatever the run's size, and a batch of results writes one row per distinct
bucket in it. On a one-million-case run on PostgreSQL 16, `GET /v1/runs/{id}`
answers at p99 in 7.6 ms while the run is running and 7.8 ms once it completed
(`benchmarks/summary.py`, which fails above 50 ms or above 0.99% from the exact
percentiles). Before the histogram, the running case read the 5,000 most recent
results and took 70 to 80 ms at p99. A completed run keeps the exact
percentiles that compaction computes from the results.

`evalkit run verify RUN_ID` recomputes all of this from the results in one
repeatable-read snapshot and lists every difference (exit 3); `--repair`
replaces the aggregates with the recomputation and rewrites the arrays, and is
refused while the run is queued, running or cancelling. Migration 0005 fills in
aggregates for runs that already had results, and migration 0007 their latency
histograms (and drops the `(run_id, id)` index on results that only the old
recent-results read used).

### The response cache

Outputs are cached after redaction, keyed by (target hash,
canonical input, redaction fingerprint). Cache hits cost nothing in the new run;
the original usage stays on the cache row. Errors are never cached, and neither
is an output whose redaction timed out, so a rerun retries both. Judge verdicts
go in the same table, keyed by the judge configuration and the exact prompt, so
a rerun of an unchanged target with an unchanged judge makes no API calls at
all. `use_cache = false` turns both off.

The cache is bounded by a sweep. Each entry records its output's size in UTF-8
bytes and when it was last used; a lookup refreshes that at most once an hour,
so a hot entry costs one write an hour rather than one per run, and the
up-front lookup of a run stays one read per 500 cases. `cache.sweep` first
deletes entries unused for longer than the maximum age, then walks the rest in
least-recently-used order (index `(last_used_at, key)`) and deletes until the
total size is at most the maximum. It deletes 1,000 keys per transaction, and
each delete only removes entries whose last use is no newer than when the sweep
read them, so an entry a run uses in the meantime survives. `--dry-run`
reports exactly what a real sweep would remove. `evalkit cache sweep` runs it
on demand; when `EVALKIT_CACHE_MAX_AGE` or `EVALKIT_CACHE_MAX_BYTES` is set,
every worker daemon runs it each `EVALKIT_CACHE_SWEEP_SECONDS`, and on
PostgreSQL a session advisory lock lets only one of them do it at a time (the
others skip that round). Deleting an entry never changes a run that already
used it: the run's own results hold the output.

### Blobs

Cached outputs of `EVALKIT_BLOB_MIN_BYTES` (64 KiB) or more go to a
content-addressed blob store when `EVALKIT_BLOB_URL` names one; the row keeps
an empty output, the SHA-256 digest and the size. A blob's name is the hash of
its bytes, so writes are idempotent and two writers of the same output agree on
its name, and every read is checked against it. Blobs are laid out as
`sha256/ab/cd/<digest>`.

- **Filesystem** (`file:///path`): written to `tmp/`, fsynced, renamed into
  place and the directory fsynced, so a reader never sees a partial blob.
- **S3-compatible** (`s3+https://host[:port]/bucket[/prefix]`): MinIO, Ceph,
  SeaweedFS, versitygw and the like, path-style. Requests are signed in
  `blobs.py` with signature version 4 (about 60 lines of `hmac` and `hashlib`
  instead of a vendor SDK and its dependencies). The signed payload hash of a
  PUT is the blob's own digest, so the server rejects a body altered in
  transit. 5xx answers and connection failures are retried twice.

The blob is written before the row that refers to it is committed. If the
store cannot be written, the output is cached inline instead. A row whose blob
is missing or corrupt is deleted on lookup and counts as a miss, so the next
fresh output replaces it; a store that is unreachable is a miss and keeps the
row.

Garbage collection (after each sweep, and in `evalkit cache sweep`) lists the
store a page of 1,000 at a time, asks the database which of the page's digests
some entry still refers to (index on `output_blob`), and deletes the others
that were last written before the grace period (`EVALKIT_BLOB_GRACE`, 1 hour).
A repeated put refreshes a blob's modification time, so a blob that a run is
about to refer to again is never old enough to collect, and on the filesystem
the time is checked again just before the delete. Temporary files left by a
crash are removed after the same grace period. `evalkit cache scrub` reads
every blob and lists the ones whose bytes no longer match their name.

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
- With 8 worker processes of 4 callers each on one key at 200 requests a
  second, the end-to-end test measures 199.6 to 200.1 a second. The slots the
  workers are given never put more than burst + w × rate in any window, with
  no allowance for jitter, and no call starts before its slot. Calls do start
  up to 20-40 ms after their slots when a worker's one connection is busy with
  a statement that waited for the row lock, so a 100 ms window can briefly see
  a few calls more than the limit; the test bounds that by the measured
  lateness rather than by a fixed allowance. With the shipped image as eight worker containers calling an HTTP
  provider, the provider itself counts 199.8 to 201.1 calls a second, and
  after a 429 with Retry-After: 1 the next call from any worker comes 1.01 to
  1.04 s later. Failures without
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

## Providers

Prompt targets, judges and estimates reach a model through the `Provider`
protocol in `providers/base.py`. It has five operations: `complete` (one prompt,
one completion with usage and cost), `count_tokens`, `embed`, `price` (USD for a
set of token counts on one model) and `classify_error`. A provider is a class
with those methods and a short `name`; rate limits and cool-downs are keyed by
that name, so two targets on the same provider share them.

Every failure a provider raises is a `ProviderError`, which is a `TargetError`
carrying an `ErrorKind`, the provider's name and the HTTP status where there was
one. The kind, not the status code, decides whether a retry can help:
`rate_limited`, `overloaded`, `server`, `conflict`, `timeout` and `connection`
are retried; `auth`, `not_found`, `invalid_request`, `safety_block`,
`malformed_response`, `unsupported` and `unknown` fail the case at once.
`kind_for_status` gives the usual HTTP mapping (429 rate limited, 503 and 529
overloaded, 408 and 504 timeout, 500 and 502 server, 409 conflict, 401 and 403
auth, 404 not found, 400, 413 and 422 invalid). Unusual statuses such as 501
are `unknown` and not retried, which keeps exactly the set of statuses the
Claude client retried before kinds existed. Cancellation is never classified:
it passes through so a drain or a cancelled run stops at once.

### OpenAI-compatible servers and Gemini

`providers/openai_compat.py` and `providers/gemini.py` call their APIs as plain
JSON over httpx2, with no vendor SDK: two SDKs would add more dependency weight
than both clients put together, and raw HTTP keeps every request byte under
test. They share `providers/_http.py`, which sends one JSON body, follows no
redirects (a wrong base URL fails instead of posting the prompt elsewhere),
reads at most 64 MiB of the answer, and turns timeouts, connection failures,
error statuses and unparseable bodies into error kinds. Neither client retries:
retries stay with the run's retry policy, and `Retry-After`, OpenAI's
`retry-after-ms` and Gemini's `RetryInfo.retryDelay` become the retry delay.

The OpenAI-compatible client covers OpenAI and the servers that copy its chat
completions format (vLLM, llama.cpp, Ollama, LiteLLM). Its `name` is set per
server, so each server has its own rate limits and cool-downs. `max_tokens` is
sent as `max_completion_tokens`, or `max_tokens` for older servers. A JSON
schema goes in `response_format` and is marked `strict` only when strict mode
can accept it (every object closed and every property required), so a looser
schema still works as best-effort JSON. `reasoning_effort` is sent only when
the client is configured for reasoning models. A `refusal` gives an empty answer
with stop reason `refusal`, as on Claude; a `content_filter` finish is a
`safety_block` error. Cached prompt tokens are priced as cache reads. The API
has no token counting endpoint, so `count_tokens` is `unsupported`.

The Gemini client posts to `generateContent`, with the schema as
`responseJsonSchema` and, when configured, effort as a thinking level. Safety
filtering is never an empty answer: a blocked prompt (`promptFeedback.blockReason`)
or an answer stopped by a filter (`SAFETY`, `PROHIBITED_CONTENT`, `BLOCKLIST`,
`SPII`, `RECITATION`, `IMAGE_SAFETY`) is a `safety_block` error naming the reason
and the blocked categories, so a report tells a provider's filter from a model
that refused. Thought parts are not part of the answer, thought tokens are
billed as output, and cached content tokens are priced as cache reads.
`count_tokens` uses `countTokens`; `embed` uses `batchEmbedContents`, whose
response has no token counts, so embedding usage is recorded as zero tokens.

### Prompt targets on any provider

A prompt target names its provider like a judge does (`provider = "openai"`,
default `anthropic`) and reaches it through the same registry, so the CLI and
the API only build a Claude client when a target or judge uses Claude. Its rate
limits and cool-downs are keyed by the provider name. The model must have a
price on that provider, checked when the run is created rather than as a
missing cost at the end. Batch mode stays Claude only: it uses the Message
Batches API. `evalkit estimate` counts input tokens on the target's provider;
an OpenAI-compatible server has no counting endpoint, so its input tokens are
estimated from the text's length and labelled as an estimate (see Prices).

### Prices

`pricing.py` holds one `PriceList` per provider, `anthropic`, `openai` and
`gemini`, each with the date it was last checked against the provider's
published page as its version and that page as its source. The `PriceBook`
gives each provider name its table: `anthropic`, `openai` and `gemini` start
from their own list; a server from `EVALKIT_PROVIDERS_FILE` starts from the
list it names with `price_list`, or from none, because a self-hosted model has
no published price and guessing one would put a wrong number on every run.
`EVALKIT_PRICES_FILE` adds `[prices.<model>]` tables, which apply to every
provider (as before per-provider tables existed), and `[prices.<provider>.<model>]`
tables, which apply to one provider and win over the first kind.

A table's version is its list's (`openai@2026-11-09`) when the overrides change
nothing, and the list's plus the first eight hex digits of a SHA-256 over the
overrides that do change something (`openai@2026-11-09+local.1a2b3c4d`), so
restating a published price does not make a new version and two processes with
the same file agree on it. Long-context rates (Gemini over 200,000 prompt
tokens, Claude Haiku 5.5 over 100,000) cover input, output and cache reads.
OpenAI and Gemini cache reads have explicit rates, since their discounts are not
Claude's tenth of the input price.

Costs are computed from usage in the process that executes each case, so a run
records the prices it is costed at when it is created: for the target and each
judge, the provider's table version and the price of the model it uses
(`runs.prices`). A process whose tables price those models differently refuses
to execute the run with `failed_precondition`, naming each model with its old
and new price and version, in the same way a changed judge or a moved provider
is refused. A table version that only changes other models is not a change.
Runs created before prices were recorded execute at whatever prices the process
has.

Providers without a counting endpoint get `estimate_tokens`: one token per three
UTF-8 bytes, rounded up, plus eight for each message's chat framing. English
prose runs about four bytes a token on current tokenizers and most non-Latin
scripts about one character, two or three bytes, a token, so the estimate leans
high; a Hypothesis test checks it never falls below a four-bytes-a-token count.
`evalkit estimate` sets `target_input_estimated` or `judge_input_estimated`, adds
a note naming the provider, prints `~` and `(estimated)` on those lines and on
the total, and lists the price table version for each provider. An estimate is
never a bill: each case is costed from the usage the provider reports.

## Model matrices and the frontier

A `[matrix]` table turns one run config into a group of child runs, one per
combination of its axes: `model` (a model name, or `{provider, model, name}`),
`effort` and `max_tokens`. Expansion (`matrix.py`) is pure: the same config
gives the same children in the same order, model first, each axis in the order
written, and an axis left out keeps the base target's value. Each child is an
ordinary run with its own target hash, cache entries, leases and aggregates.
That is the point of the design: queueing, worker daemons, resume, `run verify`,
`show` and `gate` need nothing new, and a matrix of a hundred thousand cases
per variant scales the way one run does. The group row (`matrices`) records the
axes and the base config; each child records its variant (index, name and the
values it resolved to) and is labelled `<label>[<variant>]`, so
`evalkit gate label:support[gemini] label:support[claude-opus-5-5]` works.

Every variant is validated before any is recorded (its provider, the model's
price, batch mode, its judges), and the group and its children are inserted in
one transaction, so a typo in the fifth variant does not leave four orphan
runs. A matrix expands into at most 64 runs, and naming the same target twice
is refused: each child is a full paid run. The CLI executes the children one
after another (each already runs its cases concurrently, and different
providers do not share a rate limit, but sequential runs keep the progress and
the interrupt point simple); `--queue` hands them to worker daemons, which run
them in parallel. `evalkit run --resume MATRIX_ID` resumes the children that
are not completed.

`frontier.py` reads the children's summaries. Each completed variant is a
point: the mean of the quality metric (the first gated one unless
`--metric` says otherwise), the target's cost per case it actually called, and
the p50 (or `--latency p95`) latency. A variant is dominated when another is at
least as good everywhere and strictly better somewhere, and the frontier is the
undominated set; ties are all kept. Cost per *called* case leaves cached cases
out, because a variant answered from the cache is not free in general, only in
this run. Judge cost is shown but is not a dimension, since one judge grades
every variant. A dimension that some variant cannot be measured on (latency
when all its cases were cached or failed) is dropped for all of them, with a
note, instead of comparing a number against a guess.

Point estimates decide dominance, so the report adds what they cannot say: the
frontier is sorted from cheapest to most expensive and each variant is compared
with the one before it, case by case, with the paired bootstrap and permutation
test of `evalkit compare` and Holm's correction across the steps. Adjacent
steps are the questions a frontier poses ("does the next price buy a
difference this dataset can detect?"); all pairs would add comparisons nobody
asks for and dilute the correction. With 120 cases and a 10-point gap the test
finds the difference (p 0.012 in `tests/test_matrix.py`); with 30 cases and a
3-point gap it will say "no detectable change", which is the honest answer.

## The prompt registry

A prompt version is its system prompt and template, identified by SHA-256 over
both in canonical JSON (so `"ab" + "c"` and `"a" + "bc"` differ). Version rows
are only inserted: an edit is a new version, and pushing content the prompt
already has returns the version that holds it, the way a dataset import does.
Two pushes race on the unique constraints. The loser of a race for the prompt
row uses the winner's, the loser of a race for a version number reads again
and takes the next number, and the loser of a race for the same content
returns the winner's version, so concurrent pushes never mint duplicates or
skip a number. A push that loses five attempts in a row gets a conflict and
retries, rather than looping under contention.

Labels are where mutation lives, and they change only by compare-and-set. A
move is `UPDATE prompt_labels SET version = :new WHERE prompt_id = :p AND
label = :l AND version = :expected`; if no row matched, the label is somewhere
else and the caller gets a 409 (exit 3) naming the version it is at now.
Creating a label is an INSERT that the primary key refuses if another writer
created it first. There is no "move regardless" mode: the API requires
`expected_version` (possibly `null`), and the CLI fills it from the value it
read, so a promotion based on a stale view of the label fails instead of
silently undoing someone else's. The check is on the version, not a revision
counter, so a label moved away and back still accepts a move that expected the
version it points at, which is the outcome that writer asked for. `latest` is
reserved for the newest version, and labels start with a letter so they can
never be mistaken for version numbers in `NAME@REF`.

A run config can name `prompt = "support@production"` in place of `system` and
`template`. The runner resolves the reference once, before validating the
config, and from then on the run is an ordinary prompt run: the resolved
content goes into the stored target spec and the target hash, and
`runs.prompt` records the reference, the version, the label and the content
hash. Moving the label later does not touch existing runs, and resume, workers
and `run verify` never consult the registry. A matrix resolves its reference
once for all its variants, so a promotion halfway through creating the group
cannot split it across versions. Because the hash covers the content and not
the reference, `support@4` and the same text written inline are one target and
share cache entries. A target spec that still holds an unresolved reference
cannot be built into a target, so nothing executes a prompt that was not
frozen first.

## Conversations

A conversation case is one dialogue. Its input is a `Conversation`
(`datasets.py`): the opening messages, which alternate from the user and may
be empty, a persona, a goal, `max_turns`, and for the scripted simulator the
user's lines and the phrases that mean success. A conversation run checks every
selected case against that shape when it is created, so a malformed case is
named before anything is paid for.

`targets/conversation.py` loops: whenever the dialogue is empty or ends with an
assistant reply, the simulated user (`simuser.py`) gives a verdict and, to go
on, a message; then the assistant answers the whole transcript. It stops on
`goal_reached` or `gave_up`, after `max_turns` assistant replies (the user's
verdict on the last reply is still asked for and kept), or on an empty reply.
The providers take a conversation where they took a prompt: Claude's
`messages`, chat completions after the system message, and Gemini `contents`
with the assistant as `model`. The LLM simulator answers in JSON under an
output schema; an answer that does not parse fails the case without a retry,
since asking again costs as much and usually fails the same way.

### Turn caching and resume

A case is executed as a whole, so a crash or a retryable error halfway through
a dialogue would otherwise pay for its first turns again. Each turn is therefore
cached the moment it is paid for, in its own short transaction, in the response
cache table (bounded and swept like any entry). The key is a hash of who speaks,
their identity and everything they have seen: for the assistant, its own spec
and endpoint (not the simulator's) and the transcript; for the simulated user,
its spec, the case's persona, goal and script, and the transcript; both with the
redaction fingerprint. A retried or resumed dialogue replays the turns it has for
nothing and pays only for the rest, a rerun of an unchanged target pays for
nothing, and a run with another simulator keeps every assistant reply to a
dialogue it already saw. A case is not cached as a whole, since the turns are.

Turns are redacted before they are cached and before the dialogue continues, so
the assistant sees `[EMAIL]` where the user typed an address. That costs a little
realism, but a resumed dialogue then continues from exactly the transcript an
uninterrupted one would have had, and no personal data reaches the store. With
`use_cache = false` nothing is cached, and an interrupted dialogue starts over.

Each turn takes its own provider's limiter: the assistant's under the run's
rate limit, the simulated user's under `<provider>:simuser`. A turn that gets a
Retry-After cools that provider down for every caller and fails the attempt; the
run's retry policy retries the dialogue, which replays its cached turns. The
result's tokens and cost are the sum of the turns paid for in this run, it is
`cached` when none was, and its `result_turns` rows are written in the same
transaction as the result, under the same lease check.

### The tool sandbox

An agent's tools are part of what is evaluated, but calling the real ones would
make runs neither safe nor repeatable, so `tools/sandbox.py` answers them. A
`ToolSandbox` is made per session (one case) and is never shared, so limits and
scripted `times` count within a case. Each call goes through, in order: the
session's `max_calls`, a known tool, the tool's `max_calls`, then its input
schema (draft 2020-12; schemas are checked when the file is loaded, and so are
scripted results against an output schema). Failing any of these is answered
as an error the agent sees, with the reason, because how an agent recovers from
a bad call is part of what is scored; a call that names a real tool counts
towards its limit even when its arguments are invalid, so an agent cannot loop
on a broken call forever.

A valid call is answered by the first scripted rule whose `when` is a typed
subset of the arguments (`1` matches `1.0` but not `true`), else by a recorded
fixture keyed by the tool and the arguments' canonical JSON, so key order does
not matter. Several recordings of one call replay in order and the last one
repeats, which models a tool whose answer changes after a booking. Injected
failures and jitter come from a hash of the seed, the session id, the tool and
the tool's call number, not from a shared random generator, so a case sees the
same failures however many cases run beside it, and a resumed case the same
ones as before. Only calls that pass validation are numbered, so `fail_calls =
[2]` means the second real call.

Fixtures are recorded against the real tool through the same SSRF-checked
client as HTTP targets: the endpoint comes from operator config, its address is
checked and pinned, and nothing is followed. Records whose tool, arguments or
result the current schemas reject are refused when the file is loaded, so a
schema change cannot leave fixtures that silently never match. The sandbox's
identity hashes the tools, the seed, the limits and the fixtures' content, but
not the recording endpoints or the fixture file's path, so it can be part of a
target's identity: a changed answer is a changed target, a moved file is not.

## Calling Claude

All Claude calls go through one provider (`providers/anthropic.py`):

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
- The Claude API has no embeddings endpoint, so `embed` fails with
  `unsupported` without sending a request.

The pinned SDK (1.x) sends API version `2023-06-01`. No request sends the
sampling parameters that the newer API version removes, so moving to the 2.x SDK
is a version bump.

`evalkit estimate` counts input tokens with the token counting endpoint for
every uncached target prompt and every judge call, and prices output at the
worst case of `max_tokens` per call. Never a tokenizer from another vendor:
where a provider cannot count, the estimate is from text length and labelled as
one (see Prices).

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
model, effort, rubric, and for judges not on Claude the provider and its
endpoint), so a rubric edit is visibly uncalibrated.

### Judges on any provider

A judge names its provider: `anthropic` (the default), the built-in `openai`
and `gemini`, or a server the operator lists in `EVALKIT_PROVIDERS_FILE` as
`[providers.<name>]` with a `kind` (`openai_compat` or `gemini`), a `base_url`
and optionally `api_key_env`, the variable that holds its key. Only configured
names can be used, so a run config sent to the API cannot point a judge at an
arbitrary URL. `providers/registry.py` builds one client per name on first use,
so a CLI run judged on Gemini needs no Claude key. The judge reads stop reasons
through `Completion.finished` and `Completion.truncated`, which know each
provider's spelling (`end_turn`, `stop`, `STOP`; `max_tokens`, `length`,
`MAX_TOKENS`), so a verdict is cached only when the model ended it itself and a
cut-off verdict is reported, not scored. Rate limits are keyed `<provider>:judge`.

The fingerprint that keys cached verdicts and calibrations is unchanged for
Claude judges, so existing caches and calibrations stay valid. Other providers
add their name and endpoint to it, so an OpenAI model and a model on a
self-hosted server that share a name never share verdicts.

### Judge identity and the gate

A score is a property of the system under test and of the judge that graded it.
If a candidate run is graded by a different judge than its baseline, a
"regression" may be the new judge being stricter. So each run stores, in
`runs.judges`, the identity of every judge scorer when it is created: provider,
endpoint, model, mode, effort, a SHA-256 of the rubric, and the fingerprint.
Runs from before this column existed only had Claude judges, and their
identity is derived from their scorer specs, which gives exactly the identity a
new run with the same judge records.

`compare_runs` checks every compared metric: if its judge differs between the
two runs (or only one run has a judge behind that name) the comparison is
refused, naming each field that changed, unless `--allow-judge-change` (API:
`allow_judge_change`) is given; then the report lists the changes and the CLI
prints them as warnings. A refusal is exit 3, "cannot be compared", not exit 1,
"regression", because nothing is known about the system under test. Metrics
without a judge, such as an exact match, can still be compared with `--metric`.
`max_tokens` is not part of the identity: it only decides whether a verdict
fits, and one that does not fit is reported as unscorable.

A process also refuses to execute a run whose judges, as built from its own
provider configuration, differ from the recorded ones. Without this, a worker
whose `local` provider points at another server would grade part of a run with
a different judge than the rest.

### A judge from another family

A judge that shares a family with one of the variants it grades tends to prefer
that variant's answers, so a matrix over Claude, OpenAI and Gemini is best
graded by a fourth model. `examples/providers` does this with a model on a
self-hosted OpenAI-compatible server, priced by the operator in
`EVALKIT_PRICES_FILE` (`[prices.grader.<model>]`; a server from the providers
file has no list of its own). Every child run of the matrix records the same
judge identity, so the variants compare with each other and with later runs
under the same judge without `--allow-judge-change`. Because the verdict cache
is keyed by the judge's fingerprint, the case and the answer, not by the run,
two variants that give the same answer to a case share one verdict: in the
end-to-end test 177 judge calls grade 360 answers, and each child run is
charged for the verdicts it was first to need.

## Deciding whether a change is real

Evaluation datasets are small (tens to a few hundred cases) and scores are
often binary, so a raw difference of means is mostly noise. The comparison is
paired: the same case is scored in both runs, and the test works on per-case
differences. Case difficulty is shared between the two runs and cancels out,
which is why a paired test detects a regression with a fraction of the cases an
unpaired test would need (Berg-Kirkpatrick et al., *An Empirical Investigation
of Statistical Significance in NLP*).

Above 20,000 scored cases the interval for one run is the normal interval
(mean plus or minus 1.96 standard errors) instead of the bootstrap described
next: at that size the bootstrap distribution of a mean of bounded scores is
normal to within its own Monte Carlo noise (the test suite checks they agree to
5% of the width), and 10,000 resamples of a million cases would take minutes.

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
| Blob store unavailable | Long outputs are cached inline; a cache entry whose blob cannot be read is a miss and the case runs again. A missing or corrupt blob drops its entry. |
| Two processes sweep the cache | On PostgreSQL the second one skips the round (advisory lock); on SQLite the deletes are idempotent. |
| Judge refuses, runs out of tokens or returns bad JSON | That score is `unscorable` with the reason. |
| Database unavailable | `/healthz` returns 503; a running run fails and can be resumed. |
| Runs compared over different dataset contents | `failed_precondition` (CLI exit 3), never a misleading verdict. |
| A matrix variant is invalid (unknown provider, unpriced model) | Nothing is recorded: the matrix and its runs are created in one transaction after every variant is checked. |
| A process prices a run's models differently from when it was created | The run is refused with `failed_precondition`, naming each model's old and new price and table version, so one run is never costed at two prices. |
| A provider moved after a run was recorded | The run is refused (it fails with `failed_precondition`) rather than answered by another server under the old identity. |

## Limits

| Limit | Value | Why |
|---|---|---|
| Cases per dataset | 100,000 | Bounds import memory. Summaries no longer grow with run size. |
| JSONL line | 1 MiB | Checked before parsing. |
| API request body | 16 MiB (configurable, max 256 MiB) | Checked by Content-Length and while reading. |
| Concurrent cases per process (`concurrency`) | 1-256 | |
| Concurrent runs per server or worker | 2 (configurable, 0-64) | Later runs wait as `queued` for a free slot on any worker. |
| Workers per run | `max_workers`, unlimited by default | Batch-mode runs: 1. |
| Run priority | -1000 to 1000 | |
| Worker poll interval | 5 s default | Fallback when no notification arrives. |
| Drain on SIGTERM | 25 s default | Inside Kubernetes' default 30 s grace period. |
| Scorers per run | 32 | |
| Template, system prompt | 200,000 characters | The same for registry prompt versions, so a pushed version can always run. |
| Prompt version note, description | 2,000 characters | |
| Regex patterns | 1,000 characters, 250 ms per match | Patterns can come from case data. |
| Redaction patterns | 64, 1,000 characters each, 250 ms each | |
| HTTP target response | 1 MiB default, 64 MiB max | Checked before buffering. |
| HTTP target timeout | 30 s default | |
| Tools per sandbox | 64 | |
| Tool calls per session | 50 default, 10,000 max | Answered or not. |
| Tool arguments, tool result | 64 KiB, 1 MiB | Canonical JSON. |
| Fixture file | 64 MiB, 100,000 records | |
| Case timeout | 600 s default, retries included | |
| Retry attempts | 5 default, 20 max | |
| Retry-After honoured | up to 120 s by default | |
| Judge `max_tokens` | 256-16,000 | |
| Calibration items | 5,000 | |
| Result page | 500 | |
| Case lease | 30 s default | Time before a dead process's cases are taken over. |
| Cases per claim | 4 per worker, 64-500 | Bounds what a dead process strands for one lease length. |
| Response cache | unbounded by default; `EVALKIT_CACHE_MAX_AGE`, `EVALKIT_CACHE_MAX_BYTES` | Swept hourly by workers when set. |
| Cached output kept inline | under 64 KiB when a blob store is configured | `EVALKIT_BLOB_MIN_BYTES`. |
| Unreferenced blob kept | 1 hour | Covers the gap between writing a blob and committing its row. |

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
secret headers are excluded from the target hash and never stored. Blob store
credentials have their own variables (a URL with credentials in it is refused)
and are scrubbed like the rest; blob names are checked to be SHA-256 digests
before they become a path or a key. Outputs are
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
| dev: `pytest`, `hypothesis`, `testcontainers`, `ruff`, `mypy`, `pip-audit` | Tests, property tests, real PostgreSQL, S3 and the image itself in tests, lint and format, types, vulnerability audit. |

## Bigger runs

Several things would change for many teams and much larger datasets.

Long outputs in the results table would move to the blob store as cached
outputs already can, with only the digest left in the row.
