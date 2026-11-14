# The evalkit agent session protocol

Version `evalkit-agent/1`. This is how evalkit evaluates an agent that runs as your own HTTP service, with the `http_agent` target (`src/evalkit/targets/http_agent.py`).

evalkit holds the session. Each request carries the whole session so far, and the agent answers with its next message. When that message asks for tools, evalkit answers the calls from the run's tool sandbox and sends the answers in the next request. The agent never calls a real tool during a run. So:

- every case sees the same tool answers and the same injected faults;
- the whole trajectory is stored with the result (`result_steps`, and `result_turns` when a simulated user is involved);
- a retried case or a resumed run asks only for the steps it is missing.

An agent can serve production traffic and evaluation from the same code. It only needs a mode where tool calls are returned instead of executed.

## Sessions

A session is one case. There are two kinds:

- **Task.** The target's `template`, filled from the case input, is the user's only message. The session ends when the agent replies without asking for a tool.
- **Conversation.** The target names a `simulator` and the case input is a conversation (`goal`, `persona`, `max_turns`, optional opening `messages`). Each agent reply without tool calls ends the agent's turn. The simulated user then answers, or ends the dialogue with `goal_reached` or `gave_up`. The dialogue also ends after `max_turns` finished agent turns.

Both kinds also end when a budget runs out (see [Budgets](#budgets)).

The session id is stable for a case and an agent configuration: `s-` followed by 32 hex characters. A retry of the case and a resumed run use the same id. The id changes when the endpoint, the tools or the budgets change.

## Request

`POST` to the endpoint URL with a JSON body:

```json
{
  "protocol": "evalkit-agent/1",
  "session_id": "s-3f0c9a…",
  "step": 2,
  "request_id": "9b1e…",
  "messages": [
    {"role": "user", "content": "Book a table on 2026-11-20"},
    {"role": "assistant", "content": "",
     "tool_calls": [{"id": "c1", "name": "find_slots", "arguments": {"date": "2026-11-20"}}]},
    {"role": "tool", "results": [{"id": "c1", "name": "find_slots", "result": ["19:00", "20:30"]}]},
    {"role": "assistant", "content": "Booking 19:00",
     "tool_calls": [{"id": "c2", "name": "book", "arguments": {"slot": "19:00"}}]},
    {"role": "tool", "results": [{"id": "c2", "name": "book", "result": {"confirmation": "BK-19"}}]}
  ],
  "tools": [
    {"name": "find_slots", "description": "Free tables on a date",
     "input_schema": {"type": "object", "properties": {"date": {"type": "string"}}, "required": ["date"]}},
    {"name": "book", "description": "Book a slot",
     "input_schema": {"type": "object", "properties": {"slot": {"type": "string"}}, "required": ["slot"]}}
  ],
  "budget": {"steps_remaining": 18, "tokens_remaining": 9640}
}
```

| Field | Meaning |
|---|---|
| `protocol` | Always `evalkit-agent/1` in this version. |
| `session_id` | The session (one case). |
| `step` | How many replies the agent has already given in this session; this request asks for reply number `step`, counting from 0. |
| `request_id` | The idempotency key of this step: 32 hex characters derived from the rest of the body. The same step of the same session always has the same id. |
| `messages` | The transcript so far, oldest first. It starts with the user, and every `assistant` message with tool calls is followed by one `tool` message answering all of them, in order. It never ends with an assistant message. |
| `tools` | The tools on offer. `input_schema` is a JSON Schema (draft 2020-12) for the arguments object. |
| `budget` | What is left of the session's budgets. `tokens_remaining` is `null` when the session has no token budget. |

Message shapes:

- `{"role": "user", "content": TEXT}`: the task, or the simulated user's turn.
- `{"role": "assistant", "content": TEXT, "tool_calls": [{"id", "name", "arguments"}]}`: one of the agent's earlier replies, as evalkit stored it. Text is redacted by the run's redaction rules; tool arguments are not.
- `{"role": "tool", "results": [{"id", "name", "result": VALUE} | {"id", "name", "error": MESSAGE}]}`: the sandbox's answers. An unknown tool, invalid arguments, an exceeded call limit or an injected fault comes back as `error`. It is never a failed request.

Headers sent with every request:

| Header | Value |
|---|---|
| `Content-Type` | `application/json` |
| `Idempotency-Key` | the `request_id` |
| `X-Evalkit-Protocol` | `evalkit-agent/1` |
| `X-Evalkit-Session` | the `session_id` |

The endpoint's configured `headers` and `secret_headers` (for example `Authorization`) come from operator config, as for `http` targets.

## Response

Status 200 with a JSON object:

```json
{
  "message": {
    "content": "Booked, confirmation BK-19",
    "tool_calls": []
  },
  "usage": {"input_tokens": 812, "output_tokens": 46, "cost_usd": 0.0049},
  "stop_reason": "end_turn"
}
```

| Field | Required | Meaning |
|---|---|---|
| `message.content` | no, default `""` | The reply text. |
| `message.tool_calls` | no, default `[]` | Up to 64 calls, each `{"id": STRING, "name": STRING, "arguments": JSON}`. `id` and `name` are 1 to 128 characters, and ids are unique within the message. `arguments` should be an object matching the tool's `input_schema`; anything else is answered with an `invalid_arguments` error. |
| `usage.input_tokens`, `usage.output_tokens` | no, default 0 | Non-negative integers: what this reply cost the agent in tokens, across every model call it made. They count against the token budget. |
| `usage.cost_usd` | no, default 0 | A non-negative finite number: what this reply cost. evalkit stores it as the step's cost and does not price agent tokens itself. |
| `stop_reason` | no | Up to 64 characters. It becomes the case's stop reason when the session ends on this reply. Without one, a finished reply ends with `end_turn`. |
| `protocol` | no | If present, must be `evalkit-agent/1`. |

A reply with tool calls continues the agent's turn: evalkit answers the calls and sends the next request. A reply without tool calls ends the turn. Unknown fields are ignored, so later versions can add fields without breaking agents.

A response is a protocol violation if it is not a JSON object, has no `message` object, has fields of the wrong JSON type (strings for numbers included), has negative or non-finite usage, has duplicate tool call ids, or names another protocol version. A violation fails the case without a retry, with an error that names the offending field, for example `agent 'booker' broke the session protocol: message.tool_calls.0.id: Field required`.

## Errors, timeouts and retries

| Answer | What evalkit does |
|---|---|
| 408, 429, 5xx, a timeout, a dropped connection | Retryable: the run's retry policy applies. `Retry-After` on any of these holds back every request to the endpoint (capped by `max_retry_after`). |
| other 4xx | Fails the case. |
| 3xx | Fails the case: redirects are not followed. |
| a body over `max_response_bytes` or not JSON | Fails the case. |

The endpoint's `timeout_seconds` applies to each request, and the run's `case_timeout_seconds` to the whole session.

**Idempotency.** A retry resends the same body with the same `request_id`. A request that timed out may have reached the agent, so an agent with side effects should treat a repeated `request_id` as the same request: return the stored answer, or compute it again without repeating the side effect. A step that already has an answer is never sent again, because evalkit caches every answer the moment it arrives, keyed by the request. That holds across retries, crashes and reruns with the cache on. A rerun with `use_cache = false` asks again, with the same ids.

**Statelessness.** Every request carries the full transcript, so an agent need not keep session state. If it does, it must key that state by `session_id` and `step`, not by arrival order: concurrent cases are separate sessions, and a resumed run may start a session at the step it reached before the crash.

## Budgets

Two budgets cover the whole session, across every agent turn of a conversation:

- `max_steps` (default 20, at most 500): agent replies. When the session has used them all, it ends with stop reason `max_steps`.
- `max_tokens` (default none): the input plus output tokens the agent reports. Before each request evalkit checks whether the reported total has reached the budget. If it has, the session ends with stop reason `token_budget`.

The tool calls in the last reply are still answered and stored. The request tells the agent what is left (`budget`), so an agent can plan within it. Replies replayed from the cache count the tokens they were first answered with, so a resumed session spends its budgets exactly as an uninterrupted one would.

## Configuration

The URL lives in operator config (`EVALKIT_HTTP_TARGETS_FILE`), never in a run. A run names the endpoint:

```toml
# endpoints.toml
[endpoints.booker]
url = "https://agents.internal.example/v1/step"
timeout_seconds = 60
allow_private = true                 # the agent runs on a private network
secret_headers = { Authorization = "BOOKER_TOKEN" }
```

```toml
# run.toml
[target]
type = "http_agent"
endpoint = "booker"
template = "Book a table on {{ date }}"
sandbox = "booking.toml"             # tools, scripted answers, fixtures, faults
max_steps = 12
max_tokens = 20000

# For conversation cases, drop the template and add a simulated user:
# [target.simulator]
# type = "llm"
# provider = "anthropic"
# model = "claude-sonnet-5-5"
```

Requests go through the same SSRF-safe client as `http` targets. evalkit resolves the host and refuses private, loopback, link-local and metadata addresses unless the endpoint sets `allow_private`. It connects to the address it checked, so DNS rebinding cannot redirect a request. It ignores proxy variables, follows no redirects and caps the response size.

## Versioning

Additive changes (new optional request or response fields) keep the version. A change that an agent written for `evalkit-agent/1` could not handle gets a new version string. Every request names the version it speaks in `protocol`, and an agent that does not recognise it should answer 400.
