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
annotation tool (calibration labels are imported, not collected). Generated
cases do wait for a person: they are drafts until approved. Several
processes, on several machines, can share one run; what else changes at scale
is discussed under [Bigger runs](#bigger-runs).

## What is stored

```text
datasets        (name, version) unique, (name, content_hash) unique; parent (the
                version it was made from), origin (import or edit), note, and
                cases added, changed and removed relative to the parent
dataset_cases   (dataset_id, case_id) unique; input, expected, tags, split, metadata
                (JSON), case_hash (SHA-256 of the case in canonical JSON)
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
result_steps    (run_id, case_id, step) key; one step of an agent case's trajectory:
                a model reply (redacted text, stop reason) or a tool call (tool,
                call id, arguments, sandbox status, result or error, source),
                cached, latency, tokens and cost
result_stages   (run_id, case_id, position) key; one stage of a pipeline case:
                the retrieval (its ranking as JSON) or the generation (the redacted
                answer, stop reason), cached, latency, tokens and cost
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
generations     one synthetic generation: dataset name, settings, generator identity,
                dedupe settings, counts (asked, generated, invalid, duplicates),
                failed calls, tokens, cost
case_drafts     (generation_id, case_id) unique; the case (JSON), source document,
                status (pending, duplicate, approved, rejected), duplicate_of and
                similarity, note, reviewed_at, the dataset version it went into
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

### Edits, lineage and diffs

A version is never changed after it is written; an edit
(`src/evalkit/dataset_edits.py`) writes the next one. It names its base by
content hash, applies `upsert` (whole cases), `update` (some fields of a case)
and `delete` to the base's cases, keeps their positions and appends new ones,
and records the base as the new version's parent. Each case row carries its
own hash, so the change counts, the log and a diff between two versions come
from comparing (case id, hash) pairs, without reading the cases; only the
changed cases are loaded, and their fields are compared key by key down into
objects (16 levels), so a diff says `input.question` changed rather than
`input`. An import is recorded as a child of the newest version.

Concurrency is optimistic, on the version hash. An edit whose base is not the
newest version is refused with a conflict naming the newest, unless the newest
is exactly what the edit gives (the reply to an applied edit was lost and the
client sent it again). Two edits of the newest version at once both try to
insert the same version number; the unique constraint on (name, version) lets
one in, and the other rolls back, reads again, and is refused because its base
is no longer the newest. No lock is held while the cases are rewritten. The
PostgreSQL test releases twelve writers at once and checks that one wins, that
identical edits mint one version, and that writers who retry on conflict lose
none of their edits.

An edit whose content is already a version (an edit that undoes the last
one) returns that version rather than a new one, the same rule an import
follows, because (name, content hash) is unique. `parent_id` has no foreign key:
versions are never deleted, and on SQLite a column with one cannot be dropped
by a downgrade without rebuilding a table other tables reference.

### Comparing runs on two versions

Score arrays are in case-id order over each run's own version, so two runs on
different versions of one dataset are paired by position maps: the case ids in
both versions with the same case hash, looked up in each version's sorted ids.
The comparison runs on those cases only, and the report's `dataset_change`
says how many were shared, changed, or in one version only. A changed case is
left out because its score change would measure the edit, not the system under
test. Runs on different datasets, or on versions with no unchanged case in
common, are still refused.

### Generated cases and drafts

A dataset grows faster when a model drafts cases from the documents the system
answers from (`src/evalkit/generate.py`), but a model's cases are not test
cases until someone has read them. So a generation never writes to a dataset:
it stores drafts, and approval turns them into one ordinary edit of the newest
version, with the lineage, diff and comparison rules every other edit has.

A request shows the instructions in the system prompt and, in the user
message, up to 20 examples spread evenly over the example file or the
dataset's newest version (an even spread covers more of a dataset ordered by
topic than its first rows), then one document cut at 20,000 characters, and
asks for `per_document` cases in JSON constrained by a schema. Documents and
examples are marked as data, since a support article can contain
instructions. When `allowed_tags` is set the schema's tag field is an enum of
them, so a model cannot invent slice names; any it chose are added to the
configured tags. Without documents, `count` cases are asked for `per_call` at a
time; each request carries its number, so the requests (and their cache keys)
differ, and their repeats are caught as duplicates.

Calls are cached in the response cache table by the generator's identity
(provider, endpoint, model, effort, `max_tokens`, system prompt, schema) and
the exact request, under the provider's `generate` limiter and the retry
policy, so an unchanged config is free to run again and gives the same drafts;
a generation is cheap to repeat after a crash rather than resumable. One
request failing does not fail the generation: it is recorded with its
document, as are replies that were refused, cut off or off the schema, and a
case that is not a valid case (an empty input, more than one JSONL line's 1
MiB) is counted as invalid. A reply with more cases than asked is cut.

A draft's id is the prefix and the first 12 hex digits of the SHA-256 of its
input in canonical JSON, so the same question gets the same id from every
generation, and a draft that repeats a dataset case word for word has that
case's id only if that case was itself generated. Two drafts with the same
input in one generation keep one id; the later ones get `~2`, `~3` and are
duplicates of the first.

Approval upserts the approved cases into the newest version as one edit whose
base is that version's hash (or imports them as version 1). A writer that
moves the dataset on in between makes the edit a conflict; approval reads the
newest version again and retries, up to five times. If the process dies after
the edit but before the drafts are marked, approving again finds the cases
already there: the edit changes nothing, returns the version, and the drafts
are marked. Approving every pending draft first compares them with the newest
version, because cases may have been added since the generation ran; drafts
named one by one are a reviewer's decision and are added as asked, including
duplicates and rejected drafts. A draft that is in a version cannot be
rejected, since rejecting it would not take it out; that is a dataset edit.

### Near-duplicates

Generated cases repeat each other and the dataset: the same question from two
documents, or a paraphrase of an example. Exact hashing misses "How do I reset
my password?" against "How can I reset my password?", and comparing embeddings
of every pair is quadratic and costs a provider call per text. Near-duplicates
are found with MinHash and locality-sensitive hashing in numpy
(`src/evalkit/dedupe.py`), the method web-scale corpora are deduplicated with.

Texts are normalised (NFKC, case-folded, runs of non-word characters made one
space) and cut into character 5-grams, which work for short questions and for
scripts without spaces; word n-grams are available. Each window's polynomial
hash is computed for all windows at once, modulo the largest prime below
2^32. A signature is 128 minima of the multiply-add-shift hashes
(a·x + b) mod 2^64 with random odd a: the modulus is uint64 wraparound, so
there is no division, and every set is hashed in blocks of 8,192 shingles into
one reused 8 MiB buffer, its minima taken with `np.minimum.reduceat`. Two sets
share a minimum with probability equal to their Jaccard similarity, so the
fraction of equal positions estimates it with standard error
sqrt(J(1 - J) / 128), 0.035 at 0.8.

LSH splits signatures into bands; two texts are compared only when a band is
identical. The layout minimises the false positive area below the threshold
plus four times the false negative area above it, because a candidate is only
checked on its signature while a missed pair is a duplicate let into the
dataset: at 0.8 that is 12 bands of 10 rows, which makes 99.4% of pairs at 0.9
candidates and 0.1% at 0.4. Each text is matched with the most similar kept
text or dataset case at or above the threshold, so a chain of edits points at
the first text, not the previous edit.

Decisions are made on the estimate, so pairs within about two standard errors
of the threshold go either way; `num_perm` (up to 1,024) narrows that.
Measured on a 4-core Linux container with planted one-word edits: among
10,100 sixty-word texts every one of the 100 edits was found and nothing else
was flagged, in 1.7 s; among 101,000 twelve-word texts, where one word is a
large part of the text and many edits fall near 0.8, 248 of the 313 edits
whose exact similarity is at least 0.8 were found and 162 edits just below it
were flagged, and nothing else, in 7.2 s (3.2 s of it shingling, 1.3 s
signatures). Memory is the signatures, 1 KiB per text, plus one 8 MiB block.

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
- Writes to the row commit asynchronously: each statement turns
  `synchronous_commit` off for its own transaction, with `set_config(...,
  true)` in the same round trip, so pooled connections keep the server's
  setting. The row lock is held until commit, and with a WAL flush inside it
  every slow fsync stalls every process at once. With a background job
  fsyncing the same disk, the eight-worker test below fell to 160-168 requests
  a second, with calls 150-165 ms after their slots. With asynchronous commit
  it held 199.7-200.0 a second, 11-15 ms after. The row is coordination state:
  a database crash can lose its last fraction of a second of updates, which at
  worst admits that much again once.
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

### One tool loop on every provider

`providers/tools.py` describes tool use once: a transcript of `AgentMessage`s
(the user's task, assistant replies carrying the `ToolUse`s they ask for, and
tool messages answering each of those in order), tool definitions from the
sandbox, and a `ToolCompletion` back. A `ToolProvider` maps that onto its own
API: Claude `tool_use` and `tool_result` blocks, chat completions `tool_calls`
and `tool` messages, Gemini `functionCall` and `functionResponse` parts. The
transcript is checked before anything is sent. Results go back as
`{"result": ...}` or `{"error": message}` everywhere, because Gemini needs an
object there and one form keeps the transcript, and so the step cache, the same
across providers. Two provider details must come back on the next request and
ride through untouched: Claude's thinking blocks, sent first in the assistant's
message, and Gemini's thought signature on the call it came with. Chat
completions arguments that are not JSON are passed to the sandbox as text, which
answers `invalid_arguments`, rather than failing the case: recovering from a bad
call is the agent's job. Calls with no id are named by their position, so a
replayed transcript has the same ids.

`tools/loop.py` is the only loop. It stops when a reply asks for no tool, when a
reply was cut off at the output limit or refused (a tool call in it may be
incomplete, so none is run), or after `max_steps` replies. The agent target
(`targets/agent.py`) renders the case into the task and makes a fresh
`ToolSandbox` per attempt, seeded by the case id, so a retried case is answered
exactly as before. A run carries its sandbox inline: `evalkit run` reads
`sandbox = "FILE"` and its fixture file into the stored target, so any worker
can execute the run and the sandbox's content is part of the target's identity.

Each model reply is cached like a conversation turn, the moment it is paid
for, keyed by the model's settings and endpoint, the tools it was offered and
the whole transcript (with the redaction fingerprint), but not by how the
sandbox answers: the answers are in the transcript. A resumed or retried case
replays its replies for nothing, and a changed tool answer reuses every reply
before the model saw it. Reply text is redacted before it is cached and before
the loop goes on; tool arguments are not, because the sandbox matches on them.
The steps are written as `result_steps` rows in the result's transaction, under
the same lease check; a tool name or call id the model made up is clipped to the
column's size.

### Agents behind HTTP

`targets/http_agent.py` speaks the session protocol (`docs/agent-protocol.md`)
to an agent service. The protocol is stateless on purpose:

- evalkit owns the session, and every request carries the whole transcript.
- The sandbox, not the agent, answers tool calls.

Holding the session lets evalkit treat an external agent exactly like a model
in the tool loop. The trajectory it stores is the one the agent saw, every case
gets the same tool answers and faults, and each step can be cached by a hash of
the request. A stateful protocol, where the agent keeps the session and
evalkit sends only the new message, would make a resumed run depend on the
agent having kept state across the crash. It would also make the agent's own
tool calls invisible.

**Idempotent requests.** The request id is a hash of the request body. The
session id is derived from the agent's configuration and the case id, so it is
stable across attempts and processes. A retry after a timeout therefore resends
the same `Idempotency-Key`, and an agent that stores answers by key never does
the same work twice. Caching each answer the moment it arrives means a resumed
run never resends a step that was answered.

**Budgets.** Both budgets are per session: replies (`max_steps`) and reported
tokens (`max_tokens`). Neither is per turn, because a conversation that spends
its budget in one turn should stop there. A cached reply stores the tokens it
was answered with, so a replayed session spends its budget the same way. Its
cost to this run is still zero. Token counts and cost are the agent's own
report, because evalkit cannot see the models behind the endpoint. Only a
simulated user played by a model is priced from evalkit's price lists.

**Limits.** Each request waits on the endpoint's limiter (`http:NAME`, shared
with `http` targets on the same endpoint), and sessions are not paced. A
Retry-After cools down the endpoint for every session. Responses are checked
strictly and unknown fields are ignored, so later versions can add fields:

- types are strict, so a number sent as a string is refused;
- at most 64 tool calls, with ids unique within a reply;
- usage must be non-negative and finite.

A violation fails the case without a retry and names the field. An agent that
answers malformed once usually does again.

**Conformance.** `conformance.py` builds its requests with the same
functions as the target (`wire_messages`, `request_id`, `parse_reply`) and
sends them through the same `HttpTarget.post`. So what it checks is what a run
sends, and the two cannot drift apart. It probes the properties a run relies
on and that a single session cannot show:

- Answering by `request_id` is what makes retries safe.
- Statelessness is what makes a resume and concurrent cases safe.
- Recovery from tool errors is what keeps fault injection from turning into
  `max_steps`.

The probes compare answers as well as validate them. A difference only warns:
an agent backed by a sampled model may answer a recomputed step differently,
and the protocol allows that. A refused or malformed answer fails, because a
run would fail on it. The checker never retries, so a flaky agent shows as
flaky rather than as passing.

Simulated user turns go through the same function as in a `conversation`
target (`simulated_turn`). Both cache and redact a user turn the same way, so
the two targets cannot drift apart.

**What the end-to-end test shows.** `tests/test_booking_end_to_end.py` checks
the properties above together, on the reference agent over real sockets. The
run is killed with SIGKILL, not cancelled, because a cancelled run gets to mark
itself interrupted and release its leases, and a crash does not. The test
waits until a dialogue is known to be part way through a turn before the kill.
The agent's last answer in that session asked for a tool after an earlier
answer, so at least one reply is stored and one more is needed. That makes
"resumes mid-dialogue" a fact the test checks, not a likely outcome of the
timing. The resumed run is compared with an uninterrupted run of the same
config on a fresh database, case by case: outcome, scores, every step and
every turn. A cached reply costs this run nothing (see Budgets), so a case
resumed part way is charged only for the replies asked after the resume. #83
tracks counting what the run paid before the crash.

The gate compares the reference agent with its `--skip-lookup` twin on
`tool_calls`, not on whether a booking was confirmed. On this dataset both
agents confirm 42 bookings, so an answer-only metric shows no change. The blind
agent's 42 are a different set: it books the full night and fails the late one.
That is the failure trajectory scoring exists to catch.

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
| `tool_calls` | Matched fraction of the expected tool calls, in order or not; arguments exact, subset, schema or ignored. |
| `step_efficiency` | min(1, budget / used), counting tool calls or model replies. |
| `forbidden_tools` | 0 if any tool call matches a forbidden pattern, else 1. |
| `trajectory_judge` | Rubric 1-5 over the whole trajectory, mapped to (s-1)/4. |
| `ranking` | `recall@k`, `precision@k`, `hit_rate@k`, `mrr`, `map@k` or `ndcg@k` of a retrieval target's ranking against graded labels. |
| `faithfulness` | Supported claims / claims: the answer split into claims, each checked against the case's sources. |
| `citation_accuracy` | Citations whose cited source supports the cited sentence / citations. |
| `context_relevance` | Retrieved sources that help answer the question / sources. |
| `context_recall` | Reference-answer claims the sources support / reference claims. |
| `answer_completeness` | Reference-answer claims the answer states / reference claims. |
| `noise_sensitivity` | 1 - claims that are wrong and taken from a source / claims. |
| `semantic_similarity` | Cosine of the answer's and the reference's embeddings, clipped to [0, 1]; or 1 at or above `threshold`, else 0. |
| `plugin` | Any `Scorer` subclass from an `evalkit.scorers` entry point. |

A failed call scores 0 and an unscorable case is left out. If the target
errored after all retries, every metric gets 0 for that case: the system did
not produce an answer, and the gate must see that. Unscorable cases are counted
separately and excluded from that metric.

### Trajectory scorers

The runner calls `Scorer.score_steps(case, output, steps)` with the agent's
trajectory, the same `Step`s it writes as `result_steps` rows; the default
implementation ignores the steps and calls `score`, so output scorers and
plugins are unchanged. A result without steps (a prompt or HTTP target) leaves
every trajectory metric unscorable rather than zero: "made no calls" and "was
not an agent" are different facts.

Ordered tool call matching is a longest common subsequence under a match
predicate (tool name, then the argument mode), not a greedy scan: with expected
`[B, C, D]` and actual `[C, D, B]` a greedy scan takes `B` and then finds
nothing, while the longest ordered match is `C, D`. Ties keep the earlier
expectation, so the call reported missing is the later one. Unordered matching
is a maximum bipartite matching (augmenting paths), so a loosely specified call
cannot take the only actual call a stricter one could match. Both are O(n·m)
predicate checks, with at most 200 expected calls and the sandbox's call limit
bounding m. `exact` and `subset` compare JSON strictly (`true` is not `1`,
unlike Python's `==`); `subset` lists must have the same length and match by
position; `schema` treats each expected `arguments` as a Draft 2020-12 schema,
checked when the config is built if it is in the config and per case
(unscorable when invalid) otherwise, with no remote references fetched.

Deterministic trajectory scorers read raw tool arguments, which are never
redacted because the sandbox matches on them. The trajectory judge renders
every step as one numbered line (a step is cut at 2,000 characters; a
trajectory over 60,000 keeps its first and last steps and says how many were
left out between), then redacts the whole prompt before it is sent, so a tool
result carrying an email address does not leave the process. It records its
identity with mode `trajectory`, so its cached verdicts, calibrations and gate
comparisons never mix with a rubric judge's on the same rubric.

Scorers see redacted text, because redaction happens before scoring. The judge
therefore never receives personal data, and a cached rerun (which only has
the redacted text) scores exactly like the original. The cost is that an
expected answer which itself contains, say, an email address will not match;
disable that pattern for such datasets.

### Ranking metrics

A retrieval target (`evalkit.targets.retrieval`) writes its output as canonical
JSON, `{"documents": [{"id", "score", "text"}]}`, best first, each id at its
first rank and cut to `k`: a retriever that repeats a document cannot count it
twice. The ranking scorers parse that and the case's labels,
`expected.relevance` as `{id: grade}` with integer grades 0 to 10 (a list of
ids is grade 1 each). Unlabelled documents are not relevant, which is the usual
pooled-judgement assumption. Definitions follow trec_eval: precision@k divides
by k even when fewer documents came back, so returning less is not rewarded;
MAP@k divides by min(|R|, k), so a perfect top k scores 1 when there are more
relevant documents than k; nDCG uses log2(i + 1) discounts with linear gain by
default, and the ideal ranking is built from every labelled grade, not only
from the returned documents, so a retriever cannot score well by returning
little. A case
with no document at `min_grade` or above is unscorable for every metric: its
recall is 0/0, and calling it 0 or 1 would move the mean for no reason.

The ranking scorer's config fills in its cut-off and its name (`ndcg_at_5`)
when it is validated, so the stored run, the report and the gate all use the
name the metric is computed under.

Rankings have their own cache entries in the response cache table, keyed by a
hash of the retriever (endpoint fingerprint, `k` and the fields read) and the
request, so anything that wraps the same retriever reuses them. Texts are
redacted before they are stored; a text whose redaction times out is replaced
by the placeholder and the ranking is not cached. Identical queries in flight
at once share one request: the first caller sends it and the others await it.
A waiter whose own case is cancelled leaves the request running; if the
sender's case is cancelled, the waiters send their own.

### Retrieval then generation

A pipeline target (`evalkit.targets.pipeline`) is a retrieval target and a
prompt joined in one case, so a RAG application can be evaluated end to end
without standing up its own service. The retrieval stage is a
`RetrievalTarget` built from the pipeline's `retrieval` table, which is the
same `RetrieverSpec` a retrieval target uses: the same protections, the same
in-flight sharing and the same cache entries, so a retrieval run and a
pipeline over the same retriever pay for each query once between them. The
top `contexts` documents (all `k` by default) fill the template's
`{{ contexts }}` as numbered blocks (`[1] kb-2` and the text), the numbering
`citation_accuracy` reads; the template may also name `{{ query }}` and
`{{ filters }}`, and must name `{{ contexts }}`, since a generator that never
sees the documents is not a RAG pipeline.

The generation stage is cached on its own, under a key of the generator
(provider, model, system, template and the other settings, and where the
provider is) and the prompt it was sent. The prompt holds the documents, so a
retriever change re-pays only for the queries whose top documents moved; in
`tests/test_pipeline.py` a regressed retriever that changes the top two
documents of 20 queries in 30 costs 20 answers, not 30. The case-level response
cache is not used (`caches_cases = False`): a cached case would come back
without its stages, and the stages are what the scorers read. Each stage waits
on its own limiter (the retriever's endpoint, the model's provider) and only
when the cache cannot answer it, so a fully cached rerun is not paced at all.

Both stages are written as `result_stages` rows in the result's transaction,
and the trajectory endpoint returns them as `stages`, the ranking as JSON. A
case whose generation fails is an error with no stages, like an agent case
whose loop fails. Scoring reads both: a scorer with `reads_ranking` set (the
ranking metrics, and RAG judges with `contexts_from = "output"`) is given the
retrieval stage, and every other scorer is given the answer with the case's
input replaced by the query and the documents the model was given, under
`contexts`. The grounding judges therefore check the answer against exactly
the documents it was written from, not the whole ranking, and a gate between
two pipelines reads retrieval and answer metrics from the same run.

Unlike a prompt target on another provider, a pipeline is not re-identified
when a worker executes it: its identity mixes the retriever's endpoint, which
workers may tag with their own headers (and which is not checked for HTTP
targets either), with the model's provider, so a mismatch could not be
blamed on either. The model's price is checked as for every target.

### Faithfulness and citations

Faithfulness is checked claim by claim rather than graded as a whole (the
approach of Ragas and of FActScore, Min et al., 2023): a single 1-5 grade
hides which sentence is wrong and drifts with answer length. The judge first
splits the answer into self-contained claims (at most 50 are checked; the
detail says when more were made), then checks them against the sources in
batches, one call per `batch_size` claims, all batches at once. Batching
keeps a long answer to a few calls instead of one per claim while keeping each
prompt short enough that claims late in the list are read as carefully as the
first. Verdicts are three-way, because a contradicted claim and an
unsupported one call for different fixes; both count against the score. The
verdict list must give exactly one verdict per claim number. Anything else
(a skipped claim, a repeat) leaves the case unscorable rather than letting a
missing verdict count either way. An answer with no claims scores 1, as in
DeepEval: it asserts nothing the sources could fail to support.

Citations are parsed without a model. A bracket counts as a citation only when
every comma- or semicolon-separated part is a source id, a number from 1 to
999 or a range of fewer than 50; `[sic]`, `[2020]` and Markdown links are
text. A number names the source with that id if there is one, else the source
at that position, and otherwise nothing: a citation of a source that does not
exist is kept and counted as wrong without a call. Markers at the start of a
sentence belong to the sentence before (`... capital. [1] Its tower ...`).
Each cited sentence is sent with only the sources it cites, and the judge
answers per source (precision: does it support any of the sentence) and per
sentence (do the cited sources together support all of it). The score is the
precision, accurate citations / citations; the detail adds the sentence-level
count and how many sentences cite nothing.

Sources are clipped at 20,000 characters each and 200,000 together. Both
judges are `LLMJudge`s, so the runner gives them the verdict cache, the
provider's judge limiter and the retry policy, and records their identity
under modes `faithfulness` and `citation_accuracy`. Their prompts and the
settings that change verdicts (`batch_size`, `require_citations`) are the
identity's rubric, so a prompt change in a new evalkit version, or a new batch
size, is a different judge to the gate and misses the cache. Each call's cache
key includes its system prompt, since one judge asks two kinds of question.

### RAG metrics and the RAG report

The four RAG judges in `evalkit.scorers.rag` reuse faithfulness's two calls
rather than grading with a rubric, so each verdict names a claim or a source.
`context_recall` and `answer_completeness` both split the *reference* answer
into claims (with the same extraction prompt as an answer) and check them, the
first against the sources and the second against the answer as the only
source; together they separate "the retriever did not find it" from "the
answer did not say it". `context_relevance` asks per source, in batches, with
each source's rank in the prompt so the detail can say where the noise sits.
`noise_sensitivity` follows Ragas: the answer's claims are checked against the
reference and against the sources in two concurrent sets of batches, and a
claim the reference does not support but a source does counts as misled. Ragas
reports that share with lower better; every metric the gate reads is higher
better, so the stored score is 1 - sensitivity and the detail gives the
sensitivity itself. Ragas's split of misleading sources into relevant and
irrelevant ones would need the reference's claims checked as well, a third set
of calls per case, and is left out. `context_relevance` and `context_recall`
take `contexts_from = "output"` to read a retrieval target's ranking, since
they judge retrieval alone; `contexts_from` is part of their rubric so the two
settings never share verdicts.

`evalkit report rag` (`evalkit.rag_report`) gives each metric a role from its
scorer type (retrieval, precision, grounding, answer, or other for trajectory
scorers) and attributes a failing case to the earliest stage that failed, in
pipeline order: retrieval, then grounding, then the answer. Order matters
because failures cascade: an answer written from sources that lack the fact
is usually unfaithful and incomplete too, and the fix is in the retriever.
Precision metrics only decide a cause when nothing else failed, since noise the
answer survived is not a failure the user sees. "Ignored context" needs
evidence that retrieval succeeded and, unless the failing metric is
`answer_completeness` (which measures omission directly), that the answer was
grounded; without it the case is `unattributed` and a note names the missing
metric, rather than a guess that would send someone to fix the wrong stage.
Only gated metrics make a case fail, so a run can gate on the answer and keep
retrieval and grounding metrics as ungated evidence. The report reads the
result rows once with `yield_per` and holds the counts and the first `--limit`
failing cases by id, so its memory does not grow with the run.

The RAG demo (`examples/rag/demo.sh`) runs against `examples/rag/server.py`, a
fake that stands in for the retrievers, the generator and the grader at once,
with the standard library only so it starts in milliseconds and needs nothing
installed. It is reached the way real ones are: the retrievers as HTTP
endpoints in `EVALKIT_HTTP_TARGETS_FILE`, the model and the grader as an
`openai_compat` provider in `EVALKIT_PROVIDERS_FILE`, so the demo exercises the
SSRF-safe client, the provider client, the retrieval and stage caches and the
judges' real prompts and schemas rather than in-process doubles. The grader
recognises evalkit's judge prompts by their first sentence (a test pins those)
and decides support by word overlap with all numbers required, which is enough
for an extractive reader whose answers are copies of its sources; it is not a
substitute for a calibrated judge, and `examples/rag/faithfulness.toml` is what
the human labels calibrate.

### Embeddings and semantic similarity

`evalkit.embeddings.Embedder` is one embedding model on one provider, shared
by every case of a run. A case's texts first claim their keys in an in-flight
table, then are looked up in the cache; texts that miss join a queue that is
sent when it holds `batch_size` texts or 5 ms after its first text arrived, at
most four requests at once, under the provider's `embed` limiter and the
run's retry policy. Claiming before the lookup matters: cases that end
together often share a reference answer, and a case that arrives while
another is still reading the cache must wait for that one rather than pay for
the same text. A waiter whose request was cancelled with another case sends
its own, as retrieval does. A batch's tokens are split among its texts in
proportion to their length by largest remainder, so the shares are whole and
add up to what the provider billed; a text served from the cache or from
another case's request costs nothing.

Vectors are stored in `embedding_cache` as little-endian float32 bytes with
their dimension count, keyed by SHA-256 over the embedder's identity
(provider, endpoint, model) and the text, cut at 20,000 characters. Float32 is
what the providers compute in; it is 6 KiB for a 1,536-dimension vector,
against about 30 KiB as JSON, and reads back without parsing. A fresh vector
is rounded to float32 before it is used, so a rerun from the cache scores bit
for bit like the run that paid for it; cosines are then computed in float64.
A row whose bytes do not match its dimensions is deleted and fetched again,
and a cache that cannot be read or written only costs a request. Vectors are
small and shared across runs, so the sweep bounds them by age alone
(`--max-age`), not by the response cache's size budget.

`semantic_similarity` records an identity like a judge's: mode
`semantic_similarity`, the embedder's provider, endpoint and model, and as its
rubric the threshold and the reference key. The gate therefore refuses to pair
runs scored by different models or thresholds, and the run's price record
includes the embedding model. `evalkit estimate` does not count embedding
tokens; at $0.02 per million they are a rounding error next to a judge.

A raw cosine is a poor gate: most models put unrelated texts at 0.6 or more,
so a mean of cosines barely moves when answers go wrong, and what counts as
close enough differs by model. `evalkit calibrate --scorer` therefore fits the
threshold to 0/1 labels (`calibrate_threshold`). Only thresholds between two
neighbouring distinct cosines change a verdict, so it sorts the cosines once
and walks them with running true and false positive counts: one candidate
above the highest cosine and one per distinct value, each placed at the
midpoint down to the next value so that new texts near the boundary are not
decided by a hair. That is O(n log n), 8 ms for the 5,000-item limit, and a
property test checks it against brute force over every threshold. The best
candidate has the highest Cohen's kappa (not accuracy: with 90% positives,
passing everything is 90% accurate and has kappa 0), then the highest
agreement, then the strictest threshold. Fitting a threshold to the labels it
is measured on flatters it, so the report adds a k-fold cross-validated kappa
(5 folds by default; folds from a hash of each id, so the file's order does
not matter): every fold is judged at the threshold chosen on the others, and
the pooled verdicts give the kappa. The stored calibration is keyed by the
embedder's fingerprint and keeps the threshold, the confusion matrix and the
cross-validated kappa.

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

A faithfulness judge is calibrated on whole answers, labelled `faithful` or
`unfaithful`, because that is the decision a gate on it makes: a case is
faithful when every claim is supported, the score of 1. Its claim-level score
would need people to agree on the same claim split first, which the judge's
extraction step does not guarantee; the labelled sets in `examples/rag` record
the sentence each labeller marked (`expected.unsupported`) so a disagreement can
be read back to the claim.

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

### Slices

A regression confined to one category is diluted by every other case: in the
test suite 40 billing cases lose 25 answers while the other 360 win and lose
about evenly, so the whole-dataset difference is -1.0% with p = 0.80. Slicing
runs the same paired test inside each group of cases (`evalkit.slices`).

What a slice is. A tag, a split, or a value of one metadata key. Tags overlap
(a case with two tags is in two slices) and that is kept: a slice is a question
about the cases that carry a label, not a partition. A metadata list behaves
like tags; a number or boolean is named by its canonical JSON so `3` and `"3"`
do not silently merge; objects and nulls put a case in no slice, and the report
counts those cases. More than 500 slices is refused: by then the key is an id,
not a category, and each slice costs a bootstrap and a permutation test.

Multiple testing. Every (slice, metric) test with enough cases joins one Holm
family. Without it, ten tags and two metrics would give twenty chances at a
false regression, about a 64% chance of at least one at alpha 0.05. The
family is kept apart from the whole-dataset metrics so that adding `--by` never
changes the verdicts a gate gave before; the slice verdicts answer a separate
question, and a team that gates on slices opts into it with rules. One family
over every slice, even when the rules require only some, keeps the verdict
printed next to a slice the same whether or not it is required; the price is
some power when there are many slices, and `--min-slice-size` (default 20) and
fewer metrics are the levers.

Minimum size. A slice with fewer paired cases than the minimum is listed as too
small and left out of the family. On 0/1 scores a 10-case slice can only
reach p < 0.05 when almost every case flips one way, and each such test would
raise every other slice's adjusted p-value. The count is of cases scored in
both runs, so unscorable cases do not make a slice look testable.

Rules. Required slices that regress fail the gate. A required slice that is too
small also fails it: a gate that says "billing must not regress" and passes
because billing had eight cases would be the averaging problem again. A
required name that is not among the compared cases, or a metric that was not
compared, is an input error rather than a failure, because it is a mistake in
the rules (usually a typo) and should be fixed rather than retried. Rules come
from a `[gate]` table (`evalkit.config.GateRules`) or the API request; flags
go over the file.

Versions. Slice membership is read from the candidate's dataset version for
the cases being compared. On two versions those are the cases both have
unchanged, and an unchanged case has the same tags, split and metadata in both
(they are in its hash), so either version gives the same answer.

Cost. Each slice test is the same bootstrap and permutation test as a
whole-dataset metric, on fewer cases, so disjoint slices cost about one more
whole-dataset comparison. Measured on a 4-core Linux container with 10,000
resamples: 100,000 cases in 50 disjoint slices took 10.0 s against 9.5 s for
the whole-dataset test, and in 500 slices 12.0 s; 10,000 cases in 20 slices,
1.4 s against 1.3 s. Overlapping tags multiply the work by the tags per case.

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
| Seed documents per generation | 10,000; 1 MiB per file, 20,000 characters per request | |
| Cases per generation request | 50 (`per_document`, `per_call`); 100,000 per generation | |
| Generation errors kept | 100 | |
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
