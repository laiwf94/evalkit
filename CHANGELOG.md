# Changelog

All notable changes to evalkit. Versions follow [Semantic Versioning](https://semver.org/);
until 1.0 a minor version may change the stored schema (migrations run on start-up)
and the API.

## 0.1.0 (unreleased)

The first version: an evaluation and regression-testing harness, and the
scale-out work of week 1 of the roadmap (#22).

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
