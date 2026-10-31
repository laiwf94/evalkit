# evalkit roadmap, November 2026 to October 2027

By October 2027 evalkit is an enterprise-grade, self-hosted LLM evaluation and observability platform at feature parity with LangSmith, Braintrust, Confident AI (DeepEval), promptfoo, Arize Phoenix, OpenAI Evals and Ragas: distributed runs across many workers, every major provider behind one gateway, agent, conversation, RAG, multimodal and safety scoring, production tracing with online evaluation, dashboards and automations, a web console with playground and review queues, and SSO, SCIM, custom roles, audit, encryption, HA, Kubernetes and hybrid deployment. It keeps what it already does better than all of them (paired significance tests in the gate, crash-safe resumable runs) and wins on weight: a published, CI-gated benchmark against promptfoo, DeepEval and Phoenix holds it to an image under 100 MB, a CLI that starts in under 300 ms, under 1 ms of harness overhead per case at p99 and 20,000 ingested spans per second per instance, on the same Python, FastAPI and PostgreSQL stack with a short, justified dependency list.

Benchmarked against LangSmith, Braintrust, Confident AI (DeepEval), promptfoo, Arize Phoenix, OpenAI Evals and Ragas. Every week's why cites its matrix row. Weeks 1 to 12 are the twelve core themes, one per week; weeks 13 to 52 close the enterprise gaps and hold the footprint and speed targets (image under 100 MB, CLI cold start under 300 ms, harness overhead under 1 ms per case at p99, 20,000 spans per second ingested per instance), measured against promptfoo, DeepEval and Phoenix in weeks 13, 26, 39 and 49 and enforced as a CI regression gate.

## 2026-11 · Scale-out runs, providers, agents, RAG and datasets

The first five core themes, one per week: distributed execution, multi-provider targets with a prompt registry, agent and conversation evaluation, RAG metrics, and datasets with generation and annotation.

| Week | Dates | Deliverable | Why |
|---|---|---|---|
| 1 | 02 Nov to 08 Nov 2026 | Scale-out execution: case leases, a fleet-wide limiter and a worker daemon | One process executes a run today, so a 50,000-case judge run is bounded by one machine, rate limits are per process and the response cache grows forever. Matrix rows A1 and A2: the hosted leaders run evaluations across a fleet. |
| 2 | 09 Nov to 15 Nov 2026 | Providers and prompts: OpenAI-compatible and Gemini targets, model matrices and a prompt registry | evalkit only calls Claude, so teams cannot compare models across providers or avoid a judge grading its own family, and prompts have no history. Matrix rows B1 and F1: every leader supports several providers and a versioned prompt store. |
| 3 | 16 Nov to 22 Nov 2026 | Agents and conversations: simulated users, a tool sandbox, trajectory scorers and a session protocol | Assistants are judged on conversations and agents on the tools they call, and evalkit only scores single answers. Matrix rows C5 and C6: DeepEval, Ragas and promptfoo all score trajectories and multi-turn dialogues. |
| 4 | 23 Nov to 29 Nov 2026 | RAG evaluation: retrieval metrics, claim-level faithfulness and embeddings | Most RAG failures start at retrieval or come from answers that ignore their sources, and evalkit has no retrieval or grounding metrics. Matrix row C4: Ragas, DeepEval and promptfoo ship these as standard. |
| 5 | 30 Nov to 06 Dec 2026 | Datasets as a product: case edits, slices, synthetic generation, annotation and clustering | Datasets are imported files today, labels come from elsewhere, and averages hide regressions in one category. Matrix rows D1, D2 and D3: the leaders version datasets in place, generate cases and run annotation queues. |

## 2026-12 · Decision statistics, production traces, safety suites and teams

Power analysis and sequential gates, OTLP tracing with online scoring, red-team suites, and projects with roles, OIDC and budgets.

| Week | Dates | Deliverable | Why |
|---|---|---|---|
| 6 | 07 Dec to 13 Dec 2026 | Decision statistics: power analysis, sequential gates, repeated sampling and tournaments | Teams do not know whether their dataset can detect the regression they care about, pay for every case after the answer is clear, and ignore sampling noise. Matrix rows E2 and E3: no leader offers these, which keeps evalkit ahead on E1. |
| 7 | 14 Dec to 20 Dec 2026 | Production traces: OTLP ingestion, online scoring, drift alerts and a tracing SDK | evalkit only scores offline runs, while production behaviour is the real test set. Matrix rows G1, G2 and D4: LangSmith, Braintrust, Confident AI and Phoenix all ingest traces and score them online. |
| 8 | 21 Dec to 27 Dec 2026 | Safety: red-team suites, an adaptive attacker, indirect injection and over-refusal | Teams ship assistants without testing attacks, and a model that refuses everything passes every attack test. Matrix rows H1 and H2: promptfoo leads the category here and the other platforms have little. |
| 9 | 28 Dec to 03 Jan 2027 | Teams: projects, users and roles, OIDC sign-in and budgets | Everything shares one space and one set of bearer tokens, so evalkit cannot be offered to more than one team. Matrix rows J2, J3, J5 and J12: every commercial leader has projects, roles and sign-in. |

## 2027-01 · Console, developer workflow, operations and the benchmark

The web console, SDK and CI integrations, release and portability, then the first published benchmark against promptfoo, DeepEval and Phoenix with a CI gate.

| Week | Dates | Deliverable | Why |
|---|---|---|---|
| 10 | 04 Jan to 10 Jan 2027 | Web console: runs, comparisons, datasets, traces and a playground | Results are read through the CLI and JSON, and labelling, trace search and prompt iteration need a UI. Matrix rows J1 and F2: every commercial leader is used through its web app. |
| 11 | 11 Jan to 17 Jan 2027 | Developer workflow: TypeScript SDK, pytest plugin, pull request checks and code scoring | evalkit is reached only through its own CLI and API from Python, and coding assistants cannot be scored by whether their code runs. Matrix rows I1, I3, I4 and C8. |
| 12 | 18 Jan to 24 Jan 2027 | Operating evalkit: self-telemetry, signed releases and project portability | Operators cannot see queue depth or worker health, users install from main, and there is no way to move or back up a project. Matrix rows J11 and J13, with observability and release engineering each taking their one day of the year. |
| 13 | 25 Jan to 31 Jan 2027 | Benchmark against promptfoo, DeepEval and Phoenix with a CI regression gate | None of the leaders publishes footprint or speed numbers, and that is where a small project can win outright. Matrix row A4: this week sets the baseline and the targets that weeks 14, 26, 39 and 49 are held to. |

## 2027-02 · Small image, fast ingest, trace analytics and HA

A minimal image with fast cold start, 20,000 spans per second ingestion, rollup analytics over 100M spans, and a highly available control plane.

| Week | Dates | Deliverable | Why |
|---|---|---|---|
| 14 | 01 Feb to 07 Feb 2027 | Small image, fast cold start and a bounded memory budget | Python platforms are heavy to deploy and slow to start, and the week 13 numbers set the targets. Matrix row A4. |
| 15 | 08 Feb to 14 Feb 2027 | 20,000 spans per second ingestion with OTLP protobuf | Production tracing at scale is what the leaders sell, and JSON-only ingestion at 2,000 spans per second is a toy. Matrix rows G1 and G7. |
| 16 | 15 Feb to 21 Feb 2027 | Rollup analytics over 100M spans | Dashboards and online scores over months of traffic need aggregate queries that stay fast. Matrix row G7: LangSmith and Braintrust run dedicated analytics stores. |
| 17 | 22 Feb to 28 Feb 2027 | A highly available control plane | One API process and singleton schedulers are a single point of failure. Matrix row J8. |

## 2027-03 · Kubernetes, encryption, SAML and SCIM, roles and compliance

Helm deployment with autoscaling, envelope encryption with rotation, SAML SSO and SCIM, custom roles with attribute deny rules, and a hash-chained audit log with retention.

| Week | Dates | Deliverable | Why |
|---|---|---|---|
| 18 | 01 Mar to 07 Mar 2027 | Kubernetes deployment with autoscaling on queue depth | Enterprises deploy on Kubernetes, and the leaders ship Helm charts. Matrix row J9. |
| 19 | 08 Mar to 14 Mar 2027 | Encryption at rest with key rotation and mTLS between components | Stored prompts, outputs and traces contain customer data, and enterprises require it encrypted under their keys. Matrix row J7. |
| 20 | 15 Mar to 21 Mar 2027 | SAML SSO and SCIM 2.0 provisioning | Enterprises provision users from their identity provider and many still require SAML. Matrix rows J3 and J4: LangSmith and Braintrust offer both. |
| 21 | 22 Mar to 28 Mar 2027 | Custom roles and attribute-based deny rules | Fixed roles do not fit real organisations, and datasets with personal data need tighter rules. Matrix row J5: LangSmith ships custom roles and ABAC deny policies. |
| 22 | 29 Mar to 04 Apr 2027 | A hash-chained audit log, retention policies and erasure | Compliance reviews ask for tamper-evident audit trails, retention limits and erasure on request. Matrix row J6. |

## 2027-04 · Disaster recovery, organisations, hybrid plane and gateway

Tested backups and restore, organisations with fair-share quotas, a hybrid data plane in the customer's network, and an LLM gateway, plus the second benchmark.

| Week | Dates | Deliverable | Why |
|---|---|---|---|
| 23 | 05 Apr to 11 Apr 2027 | Tested backups and disaster recovery | Self-hosted customers need a restore they can trust. Matrix row J11. |
| 24 | 12 Apr to 18 Apr 2027 | Organisations, fair-share quotas and usage metering | A shared instance needs organisations above projects and protection from one noisy tenant. Matrix rows J2 and J12. |
| 25 | 19 Apr to 25 Apr 2027 | A hybrid data plane in the customer's network | Some customers keep data in their network while using a central control plane. Matrix row J10: LangSmith and Braintrust offer hybrid deployment. |
| 26 | 26 Apr to 02 May 2027 | An LLM gateway that routes, caches and traces, with the second benchmark | Braintrust's AI proxy and LangSmith's LLM Gateway capture traffic with no SDK and add caching and budgets. Matrix rows B2 and A4: the gateway must add under 2 ms at p99. |

## 2027-05 · Scoring breadth and AI assistance

Assertion breadth to match promptfoo, sandboxed custom scorers, judge alignment, automatic prompt optimisation, and an assistant over evaluation data.

| Week | Dates | Deliverable | Why |
|---|---|---|---|
| 27 | 03 May to 09 May 2027 | Assertion breadth to match promptfoo | promptfoo documents about 46 deterministic assertion types with negation on any of them, and evalkit has six built-in scorers. Matrix row C1. |
| 28 | 10 May to 16 May 2027 | Sandboxed custom scorers from the API and console | Braintrust, promptfoo and Phoenix let teams write scorer code without redeploying the server; evalkit only loads plugins from CLI config. Matrix row C3. |
| 29 | 17 May to 23 May 2027 | Judge alignment and automatic judge tuning | Judges disagree with human labels and drift when models change; the leaders let teams align judges with feedback. Matrix row C2: evalkit already calibrates with kappa and now improves the judge, not just measures it. |
| 30 | 24 May to 30 May 2027 | Automatic prompt optimisation | Braintrust's Loop proposes prompt changes from evaluation results; evalkit can measure a prompt but not improve one. Matrix row F4. |
| 31 | 31 May to 06 Jun 2027 | An assistant over evaluation data | LangSmith Engine and Braintrust Loop answer questions over traces and runs and summarise root causes. Matrix row G6. |

## 2027-06 · Multimodal and security testing

Image, audio and document evaluation, red-team plugins mapped to compliance frameworks, agent and MCP security tests, and a runtime guardrails API.

| Week | Dates | Deliverable | Why |
|---|---|---|---|
| 32 | 07 Jun to 13 Jun 2027 | Multimodal evaluation: images, audio and documents | Assistants read screenshots, PDFs and speech, and Ragas already scores multimodal faithfulness. Matrix row B3. |
| 33 | 14 Jun to 20 Jun 2027 | Red-team plugins mapped to OWASP LLM Top 10, NIST AI RMF and the EU AI Act | promptfoo leads the category with plugins, strategies such as Tree of Attacks with Pruning, and compliance-mapped reports; week 8 gave evalkit a first suite. Matrix rows H1 and H2. |
| 34 | 21 Jun to 27 Jun 2027 | Agent and MCP security tests | Agents with tools can be tricked into acting for the wrong user or calling tools they should not, and MCP servers can carry poisoned tool descriptions. Matrix row H3. |
| 35 | 28 Jun to 04 Jul 2027 | A runtime guardrails API | promptfoo sells guardrails that check prompts and outputs in production; evalkit's safety scorers can serve that role online. Matrix row H4, held to a p99 under 20 ms for rule-based checks. |

## 2027-07 · Structured outputs, integrations, SDKs and experiments

SQL and structured output metrics, framework auto-instrumentation, Go and Java SDKs, and experiment tracking over time, plus the third benchmark.

| Week | Dates | Deliverable | Why |
|---|---|---|---|
| 36 | 05 Jul to 11 Jul 2027 | SQL and structured output metrics | Text-to-SQL and extraction are common tasks, and Ragas scores SQL by execution equivalence. Matrix row C7. |
| 37 | 12 Jul to 18 Jul 2027 | Framework auto-instrumentation | Teams use agent frameworks and expect traces with no code changes. Matrix row I2. |
| 38 | 19 Jul to 25 Jul 2027 | Go and Java SDKs | Production services that call models are often written in Go and Java, and LangSmith and Braintrust reach them. Matrix row I1: evalkit has Python and TypeScript after week 11. |
| 39 | 26 Jul to 01 Aug 2027 | Experiment tracking over time with the third benchmark | Braintrust and LangSmith show how quality moves across many experiments, tied to commits. Matrix rows F3 and A4. |

## 2027-08 · Monitoring, automations, review at scale and 10M-case runs

Dashboards and alerts, a rules engine on traces, human review at scale, feedback and topic analytics, and streaming runs over ten million cases.

| Week | Dates | Deliverable | Why |
|---|---|---|---|
| 40 | 02 Aug to 08 Aug 2027 | Dashboards and alerts | LangSmith, Braintrust, Confident AI and Phoenix monitor quality, cost and latency on dashboards with alerts; evalkit has drift alerts but no dashboards. Matrix row G3. |
| 41 | 09 Aug to 15 Aug 2027 | Rules and automations on traces | LangSmith's automations route traces to datasets, queues, online evaluators and webhooks by rule; evalkit needs each step done by hand. Matrix row G4. |
| 42 | 16 Aug to 22 Aug 2027 | Human review at scale | Leaders run review queues over production traces with configurable forms; week 5 queues work on dataset cases only. Matrix row D3. |
| 43 | 23 Aug to 29 Aug 2027 | User feedback and topic analytics | Braintrust's pattern discovery and LangSmith's feedback views show what users complain about; evalkit ignores end-user feedback. Matrix row G5. |
| 44 | 30 Aug to 05 Sep 2027 | Streaming runs over ten million cases | Large teams evaluate on millions of cases, and the week 1 design is proven to one million. Matrix row A3, held to 5,000 cases per second of harness throughput across a worker fleet. |

## 2027-09 · Agents in the IDE, CI everywhere, benchmarks and exports

An MCP server, GitLab, JUnit and SARIF integrations, a standard benchmark registry, and exports to warehouses and fine-tuning formats.

| Week | Dates | Deliverable | Why |
|---|---|---|---|
| 45 | 06 Sep to 12 Sep 2027 | An MCP server for agents and IDEs | Braintrust exposes its data to AI tools over MCP, and developers increasingly start and read evaluations from coding agents. Matrix row I5. |
| 46 | 13 Sep to 19 Sep 2027 | GitLab, JUnit and SARIF integrations | Not every team uses GitHub, generic CI systems read JUnit XML, and code scanning dashboards read SARIF; promptfoo publishes findings into code scanning. Matrix row I3. |
| 47 | 20 Sep to 26 Sep 2027 | A standard benchmark registry | OpenAI Evals ships a registry of evals and DeepEval ships standard benchmarks, so teams can place their model choice against public numbers. Matrix row D5. |
| 48 | 27 Sep to 03 Oct 2027 | Exports to warehouses and fine-tuning formats | Teams analyse evaluation data alongside their other data, and curated cases become training data. Matrix row I6. |

## 2027-10 · Performance, security hardening, upgrades and 1.0

The fourth benchmark with tightened gates, fuzzing and a threat model, zero-downtime upgrades, and a 1.0 with a compatibility promise.

| Week | Dates | Deliverable | Why |
|---|---|---|---|
| 49 | 04 Oct to 10 Oct 2027 | Performance hardening and the fourth benchmark | The year added many features to the hot paths, and the week 13 targets must still hold. Matrix row A4: the last quarterly re-measure, with the CI gate tightened. |
| 50 | 11 Oct to 17 Oct 2027 | Security hardening: threat model, fuzzing and dependency review | The attack surface grew a year's worth: OTLP and protobuf parsing, SAML, SCIM, MCP, sandboxes, a gateway and a filter language. Matrix rows J3 to J7 and H4 depend on these being sound. |
| 51 | 18 Oct to 24 Oct 2027 | Zero-downtime upgrades | Enterprises upgrade without maintenance windows, and a year of migrations must apply safely to live data. Matrix row J13. |
| 52 | 25 Oct to 31 Oct 2027 | evalkit 1.0 | A 1.0 tells adopters what will not break and closes the year at parity with the category on the matrix. Matrix row J13. |

## Next year

Gaps from the comparison matrix still open after October 2027:

- A hosted multi-region service with data residency choices (rows J8, J10)
- Ruby, .NET and Rust SDKs (row I1)
- Span replay in the playground, re-running a recorded call with edited inputs (Phoenix has it; row F2)
- Model file security scanning (promptfoo Model Security)
- IDE plugins and static scanning of prompts in source code (promptfoo Code Scanning; row I3)
- Spreadsheet-style collaborative dataset editing with comments (row D1)
- Active-active multi-region control plane (row J8)
