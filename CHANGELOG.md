# Changelog

All notable changes to evalkit. Versions follow [Semantic Versioning](https://semver.org/);
until 1.0 a minor version may change the stored schema (migrations run on start-up)
and the API.

## 0.1.0 (unreleased)

The first version: an evaluation and regression-testing harness, the
scale-out work of week 1 of the roadmap (#22), providers and prompts from
week 2 (#23), and agents and conversations from week 3 (#24).

### Evaluating and gating

- Versioned JSONL datasets, identified by a content hash that ignores key order,
  whitespace and line order; personal data is redacted before anything is stored.
- Targets: prompt templates on Claude, the deployed app over HTTP (endpoints named
  in operator config only, with SSRF protection and DNS-rebinding defence), and a
  deterministic fake for demos and tests. Prompt targets can run through the
  Message Batches API.
- Deterministic scorers, an LLM judge with position swap, verdict caching and
  calibration against labels, and plugin scorers.
- Bootstrap intervals and a paired sign-flip permutation test with Holm
  correction; `evalkit gate` exits 1 only for a regression.
- Cost estimates from token counts, per-provider rate limits and retries that
  honour Retry-After.
- The CLI, an HTTP API with bearer tokens, Prometheus metrics, JSON logs with
  secret scrubbing, and a non-root container image.

### Scale-out runs (#22)

- **Case leases.** Several processes execute one run: cases are claimed in
  batches with `FOR UPDATE SKIP LOCKED`, results are written only under the
  fence they were claimed with, and a dead process's cases are taken over when
  their leases expire. Exactly one process completes the run.
- **A shared rate limiter.** On PostgreSQL every process admits calls through one
  GCRA row per provider key (one `UPDATE ... RETURNING` per admission), so a
  fleet keeps to the configured rate, and a 429 on any worker pauses all of them.
- **A run queue and `evalkit worker`.** Runs submitted through the API or with
  `evalkit run --queue` are executed by worker daemons that wake on
  `LISTEN/NOTIFY`, honour run priority and per-run worker caps, drain on
  SIGTERM, and stop on `POST /v1/runs/{id}/cancel` or `evalkit cancel`.
- **Aggregates and score arrays.** Every result write adds to per-writer
  aggregate shards, and a completed run keeps one compact score array per metric,
  so `GET /v1/runs/{id}` on a one-million-case run answers in under 8 ms at p99
  both while it is running and after it completed, and a comparison reads one
  row per metric. A running run's latency percentiles come from a logarithmic
  latency histogram kept with the aggregates, within 1% of the exact values.
  `evalkit run verify [--repair]` recomputes and checks all of them.
- **A bounded cache and blobs.** The response cache records last use and is swept
  by age and size (`evalkit cache sweep`, or by workers). Long cached outputs go to
  a content-addressed blob store on a filesystem or an S3-compatible server, with
  request signing in our own code, hash checks on every read, garbage collection
  and `evalkit cache scrub`.
- **Tests.** Claim races forced into chosen orders on PostgreSQL, a Hypothesis
  state machine over random lease interleavings, limiter properties over any
  window, and an end-to-end test of the image itself: an API container and eight
  worker containers on one key keep to 200 requests per second (measured within
  1%; the test allows 5%) and all pause on one 429; four share a run, one is
  killed with SIGKILL, and every case is recorded exactly once; a cancelled run
  stops on every worker within 5 seconds.

### Providers and prompts (#23)

- **A provider protocol.** `complete`, `count_tokens`, `embed`, `price` and
  `classify_error` behind one interface, with errors sorted into kinds (rate
  limit, overload, timeout, safety block and others) that decide retries. The
  Claude client moved to `providers/anthropic.py`.
- **OpenAI-compatible servers and Gemini** over raw HTTP with httpx2, no vendor
  SDKs: chat completions with JSON schema output, `generateContent` with a
  response schema and thinking levels, embeddings, and safety blocks and
  refusals reported as such. The built-in `openai` and `gemini` providers and
  any server listed in `EVALKIT_PROVIDERS_FILE` can be named by prompt targets
  and judges; only configured names can be called.
- **Judges on any provider**, with each run recording its judges' identity
  (provider, endpoint, model, mode, effort, rubric hash). `evalkit gate`
  refuses to compare runs graded by different judges (exit 3) unless
  `--allow-judge-change` is given, and a process whose configuration would
  build a different judge will not execute the run.
- **Model matrices.** A `[matrix]` table expands model, effort and
  `max_tokens` axes into up to 64 grouped child runs. `evalkit report matrix`
  marks the Pareto frontier over quality, cost per case and p50 or p95 latency,
  and compares neighbours on the frontier case by case with Holm correction.
  `POST /v1/matrices` and `GET /v1/matrices/{id}` do the same over the API.
- **A prompt registry.** Immutable versions identified by a content hash,
  labels moved only by compare-and-set, `evalkit prompt push|show|list|diff|label`
  and `/v1/prompts`. A target may name `prompt = "support@production"`; the
  version is resolved when the run is created and frozen into it.
- **Prices per provider.** Versioned price lists for Anthropic, OpenAI and
  Gemini (long-context rates included), overrides per provider in
  `EVALKIT_PRICES_FILE`, the prices a run used recorded with it (a process that
  prices them differently refuses to execute it), and input estimates for
  providers without a counting endpoint labelled as estimates.
- **Tests.** Strict fake OpenAI-compatible and Gemini servers that reject any
  request a real server would, and an end-to-end test of `examples/providers`:
  one config on Claude, OpenAI and Gemini over local sockets, judged by a
  fourth provider, with every provider's tokens and cost checked against its
  price list, then a prompt version promoted to production, run and gated
  against the matrix run it replaces.
### Fixes

- The shared rate limiter's writes no longer wait for the WAL flush while
  holding the row lock, so a slow disk no longer pushes a fleet below its rate
  (#81).
- Cancelling a run while a batch of results was being written no longer writes
  that batch a second time and reports its cases as refused (#81).
- A worker draining on SIGTERM no longer starts new cases when its event loop
  ran late. Before, cases that finished in the same passes as the stop could
  each take the next case off the queue before the drain was requested (#84).
- A sandbox with an error response can be stored and read back. Before, it was
  refused by `evalkit run` with `sandbox = FILE`, and a run submitted with one
  could not execute (#82).
- The reference agent closes the connection when it refuses a request without
  reading its body (too large, or without a length), instead of reading that
  body as the next request (#85).

### Agents and conversations (#24)

- **Conversation cases and simulated users.** A `conversation` target holds a
  dialogue with a simulated user, scripted or played by a model on any provider,
  until the goal is reached, the user gives up or `max_turns` is hit. Every
  provider takes a whole conversation. Each generated turn is a `result_turns`
  row (migration 0012) and is cached by a hash of the dialogue so far as soon as
  it is paid for, so a dialogue killed or retried halfway replays its turns for
  nothing.

- **A tool sandbox.** `evalkit.tools` answers an agent's tool calls without
  real systems: tool schemas checked with jsonschema, scripted responses matched
  on a subset of the arguments, fixtures recorded from the real tools with
  `evalkit tools record`, per-session and per-tool call limits, and latency and
  failures injected deterministically per case. Every call, answered or not, is
  kept as the case's trajectory. `evalkit tools check` validates a sandbox.

- **Agent targets and stored trajectories.** An `agent` target runs a model on
  Claude, an OpenAI-compatible server or Gemini against a tool sandbox through
  one normalised tool loop, until it answers without a tool call or reaches
  `max_steps`. Every model reply and tool call is a `result_steps` row
  (migration 0013), served at `GET /v1/runs/{id}/results/{case}/trajectory`.
  Replies are cached by a hash of the transcript so far, so a case killed or
  retried mid-trajectory pays only for the replies it lacks.

- **Trajectory scorers.** `tool_calls` matches expected tool calls in order
  (longest ordered subsequence) or out of order (maximum matching), with
  arguments compared exactly, as a subset or against a JSON Schema, and can
  count unexpected calls against the score; `step_efficiency` scores tool calls
  or model replies against a budget; `forbidden_tools` fails any call matching a
  pattern; `trajectory_judge` grades the whole trajectory with a rubric, its
  prompt redacted, and is calibrated by `evalkit calibrate` on labelled
  trajectories. Scorers receive the trajectory through `Scorer.score_steps`;
  without one, trajectory metrics are unscorable.

- **A session protocol and HTTP agent targets.** `docs/agent-protocol.md`
  defines `evalkit-agent/1`: evalkit posts the whole session (transcript, tools,
  budget left) to an agent service one step at a time, and the agent answers
  with its next message and the tools it asks for. An `http_agent` target
  speaks it on the SSRF-safe client:
  - the sandbox answers tool calls;
  - a simulated user is optional for conversation cases;
  - every request carries an idempotency key derived from its content;
  - each answer is cached on arrival;
  - responses are validated strictly;
  - the session has budgets for agent replies (`max_steps`) and reported tokens
    (`max_tokens`).

- **Agent conformance and a reference agent.** `evalkit agent conformance
  ENDPOINT --sandbox FILE` checks that an agent speaks the protocol before a
  run: protocol version, schemas, idempotent retries (also concurrent ones),
  ordering of resumed and interleaved sessions, recovery from tool errors and
  timeouts. It exits 1 on a failing check, or on a warning with `--strict`.
  `examples/agent_server.py` is a table-booking agent with no dependencies that
  passes every check.

- **Tests.** An end-to-end test of `examples/agents`: 50 booking conversations
  between a scripted simulated user and the reference agent over local sockets,
  ending on goal reached, give-up and the turn limit. The run is killed with
  SIGKILL mid-dialogue and resumed, and it matches an uninterrupted run case by
  case, with every agent step answered once. The gate then fails an agent that
  books without looking up free tables on `tool_calls` (-54%), although it
  confirms as many bookings. `examples/demo.sh` ends with that gate.

### RAG evaluation (#25, in progress)

- **Retrieval targets and ranking metrics.** A `retrieval` target posts each
  case's query (a string, or a query with filters) to a retriever named in
  operator config, on the SSRF-safe client, and keeps the top `k` documents it
  returns: ids, scores and texts, read from configurable fields, each id at its
  first rank. Retrieval cases label documents with grades from 0 to 10 in
  `expected.relevance`. The `ranking` scorer computes `recall@k`,
  `precision@k`, `hit_rate@k`, `mrr`, `map@k` and `ndcg@k` (linear or
  exponential gain) against those labels, and a case with no relevant label is
  unscorable rather than zero. Rankings are kept in a retrieval cache keyed by
  the retriever and the request, with texts redacted, and identical queries in
  flight at the same time make one request.
- **Claim-level faithfulness and citation accuracy.** The `faithfulness`
  scorer has a judge split the answer into self-contained claims, then checks
  them against the case's sources in concurrent batches (`batch_size`, 10 by
  default), each claim supported, unsupported or contradicted with the ids of
  the sources that decide it. It scores supported / claims and its detail lists
  every unsupported claim. The `citation_accuracy` scorer parses `[1]`,
  `[1, 3]`, `[2-4]`, `[^1]` and `[source-id]` citations, attaches each to the
  sentence it closes, and checks every cited sentence against each source it
  cites: accurate citations / citations, with citations of sources that do not
  exist counted as wrong. Sources are texts, `{id, text}` objects or a
  retrieval ranking under `contexts` in the case. Both record their identity
  like any judge and share its cache, limits and retries; a verdict list that
  does not cover every claim or citation is unscorable, not guessed at.
- **Retrieval and answer metrics for RAG.** `context_relevance` judges each
  retrieved source for whether it helps answer the question (relevant /
  sources, the irrelevant ones named with their ranks). `context_recall` splits
  the reference answer into claims and checks them against the sources, so a
  low score is retrieval's fault whatever the generator did.
  `answer_completeness` checks the same reference claims against the answer and
  names what it leaves out or contradicts. `noise_sensitivity` checks the
  answer's claims against the reference and the sources together and counts the
  wrong claims a source led to; it stores 1 - sensitivity so that higher is
  better, and its detail gives the sensitivity and the misleading sources. The
  first two read their sources from the case or, with `contexts_from =
  "output"`, from a retrieval target's ranking. All four reuse the faithfulness
  judge's extraction and batched checks and record their own judge identity.
- **`evalkit report rag RUN`.** Attributes each failing case of a run to a
  retrieval miss, an unsupported answer or ignored context (or noisy
  retrieval, a target error, or unattributed when the run does not measure
  enough to tell), from the roles of the run's metrics: ranking recall-type
  metrics and `context_recall` for retrieval, `faithfulness`,
  `citation_accuracy` and `noise_sensitivity` for grounding, and every other
  output metric for the answer. A case fails below `--threshold` (0.5, or per
  metric with `NAME=VALUE`) on a gated metric; ungated metrics are evidence.
  The report streams the results once and keeps only counts and the first
  `--limit` failing cases.
- **Embeddings and semantic similarity.** `evalkit.embeddings` embeds texts on
  any provider with an embeddings endpoint (OpenAI-compatible servers and
  Gemini): texts from every case of a run share requests of up to `batch_size`
  (64 by default), a text already on its way is awaited rather than sent
  again, and vectors are kept as little-endian float32 in a new
  `embedding_cache` table keyed by the provider, endpoint, model and text, so
  a rerun pays nothing and scores bit for bit the same. Each case is charged
  its share of a batch's tokens and cost, by text length. `evalkit cache sweep
  --max-age` (and workers with `EVALKIT_CACHE_MAX_AGE`) also removes vectors
  unused for longer. The `semantic_similarity` scorer compares the answer
  with the reference text: the cosine clipped to [0, 1], or 1/0 at a
  `threshold`. Its provider, endpoint, model and threshold are recorded with
  the run like a judge's identity, so the gate refuses to compare runs scored
  differently. `evalkit calibrate --scorer FILE LABELS` sweeps every threshold
  that changes a verdict against 0/1 labels in one pass over the sorted
  cosines (8 ms for 5,000 labelled items), reports the one with the highest
  Cohen's kappa with its agreement, precision and recall, a 5-fold
  cross-validated kappa (each fold judged at the threshold chosen on the
  others) and a table of thresholds, and stores the result as a calibration.
- **Labelled RAG sets.** `examples/rag` holds an 18-document help-centre
  corpus, 12 retrieval cases with graded relevance labels and reference
  answers, the rankings of a baseline and a regressed retriever, and 16
  answers labelled faithful or unfaithful. `evalkit calibrate --judge` takes a
  faithfulness judge and measures its kappa against those labels.
- **The RAG demo.** `examples/rag/demo.sh` runs a two-stage pipeline
  (`examples/rag/pipeline.toml`) on the current retriever and on a change
  under review (`pipeline-next.toml`), prints `evalkit report rag` for both
  and gates the change: recall@3, nDCG@5 and answer completeness fail it, and
  the report puts all eight failing questions down to retrieval misses.
  `examples/rag/server.py` (standard library only) stands in for both
  retrievers, the generator and the grader over HTTP, reached through
  ordinary operator config, so no key or network is needed. CI runs the demo
  on every push, and `tests/test_rag_demo.py` checks every case's stages,
  scores and cause.

### Datasets as a product (#26, in progress)

- **Case-level edits.** `PATCH /v1/datasets/{name}` and `evalkit dataset edit`
  apply an edit (`upsert` whole cases, `update` some fields of a case,
  `delete` cases, and a `note`) to the version named by its content hash, as
  the next version with that version as its parent. If the dataset has moved
  on, the edit is refused with a conflict (409, exit 3) that names the newest
  version; resending an edit that was applied returns the version it made.
  Concurrent edits race on the version number's unique constraint, so exactly
  one wins.
- **Log, lineage and diffs.** Every version records its parent, origin
  (import or edit), note and how many cases it added, changed and removed.
  `evalkit dataset log` and `GET /v1/datasets/{name}/log` list them;
  `--version` and `.../versions/{v}/lineage` give one version's ancestors.
  `evalkit dataset diff` and `GET /v1/datasets/{name}/diff` list added,
  removed and changed cases, each change down to the field and object key
  (`input.question`, `tags: +billing`). Imports are recorded as children of
  the newest version.
- **Comparison across versions.** Runs on two versions of one dataset are
  compared on the cases both versions have unchanged, and the gate prints how
  many were left out (`dataset_change` in JSON and the API). Before, any
  change to a dataset made its runs incomparable.
- Migration 0016 adds the lineage columns and a per-case hash, and fills both
  in for existing datasets, chaining each name's versions in order.
- **Slices.** `evalkit compare` and `evalkit gate` take `--by tag`, `--by split`
  or `--by metadata.KEY` and compare every metric inside each slice, with
  Holm's correction across all slice tests and a minimum slice size
  (`--min-slice-size`, default 20) below which a slice is listed but not
  tested. `--require-slice NAME`, `--any-slice` and a `[gate]` table given with
  `--rules` make the gate fail when a named slice regresses or is too small to
  test. `POST /v1/compare` takes the same rules as `slices` and returns
  `failures`.
- **Synthetic generation with drafts.** `evalkit generate CONFIG` asks a model
  on any configured provider for cases from seed documents (files or a JSONL
  of `{id, text}`) and examples (a file or the dataset's newest version), in
  JSON constrained by a schema, under a `generate` limiter with retries and
  caching. Each case gets the configured tags (plus any of `allowed_tags` the
  model chose), split, its source document and an id from its input. They are
  stored as drafts (migration 0017: `generations`, `case_drafts`) and reach the
  dataset only through `evalkit drafts approve` (or `POST
  /v1/generations/{id}/approve`), as one edit of the newest version; `drafts
  reject`, `drafts show` and `drafts list` (and the matching API routes) cover
  the rest of the review.
- **Near-duplicate removal.** `src/evalkit/dedupe.py` estimates Jaccard
  similarity of character 5-grams with 128-permutation MinHash signatures and
  finds candidate pairs with LSH banding, all in numpy. Drafts that repeat a
  dataset case, an example or an earlier draft are marked `duplicate`, with the
  case they repeat and the similarity; approving all pending drafts checks them
  again against the newest version. 101,000 short texts take 7.2 s on four
  cores.
- **Annotation queues.** `evalkit annotate create` (or `POST
  /v1/annotation-queues`) builds a queue from a dataset version or a run's
  outputs; annotators `next` (claim), `submit`, `skip`, and an adjudicator
  settles disputed items. Each item takes `overlap` labels from different
  annotators, enforced by unique slots, with claims that lapse and can be taken
  over. Items become `agreed` or `disputed` when their last label arrives, and
  `export` writes the final labels in the form `evalkit calibrate` reads.
  Migration 0018 adds `annotation_queues`, `annotation_tasks` and
  `annotation_assignments`.
- **Krippendorff's alpha.** `src/evalkit/agreement.py` computes alpha for any
  number of annotators with missing labels, on nominal, ordinal, interval and
  ratio levels, from the coincidence matrix in numpy, and matches the
  published worked examples. `evalkit annotate agreement` and `GET
  /v1/annotation-queues/{queue}` report it with each annotator's agreement
  with the final labels.
- **Failure clusters.** `evalkit report clusters RUN [--baseline RUN]`
  embeds each failing case (input, output and the detail of each failed
  metric, through the embedding cache), groups them with spherical k-means in
  numpy (k-means++ restarts, k chosen by silhouette on a sample, or `--k`),
  and names each cluster with a model shown its most typical cases, falling
  back to its most distinctive words. With a baseline both runs' failures are
  clustered together and each cluster is new, grew, same, shrank or resolved,
  with the cases that newly fail or were fixed. `--save-tags` writes the
  clusters into the newest dataset version as `cluster:NAME` tags in one edit,
  replacing an earlier clustering's, so `--by tag` slices and
  `--require-slice` follow them. 20,000 vectors of 1536 dimensions cluster in
  1.2 s on four cores (`benchmarks/clusters.py`).
