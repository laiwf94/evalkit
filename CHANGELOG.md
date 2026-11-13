# Changelog

All notable changes to evalkit. Versions follow [Semantic Versioning](https://semver.org/);
until 1.0 a minor version may change the stored schema (migrations run on start-up)
and the API.

## 0.1.0 (unreleased)

The first version: an evaluation and regression-testing harness, the
scale-out work of week 1 of the roadmap (#22), and providers and prompts from
week 2 (#23).

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
