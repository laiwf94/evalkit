#!/usr/bin/env bash
# Shows a regression being caught. Needs no API key: the targets are fakes with
# known accuracy. Uses a throwaway SQLite database.
set -euo pipefail
cd "$(dirname "$0")/.."

EVALKIT=${EVALKIT:-"uv run evalkit"}
DB_DIR=$(mktemp -d)
AGENT_PID=
BLIND_PID=
cleanup() {
  if [ -n "$AGENT_PID" ]; then kill "$AGENT_PID" 2>/dev/null || true; fi
  if [ -n "$BLIND_PID" ]; then kill "$BLIND_PID" 2>/dev/null || true; fi
  rm -rf "$DB_DIR"
}
trap cleanup EXIT
export EVALKIT_DATABASE_URL="sqlite:///$DB_DIR/demo.db"

step() { printf '\n== %s\n' "$*"; }

step "Baseline: the version on main"
$EVALKIT run examples/baseline.toml

step "Rerun of the baseline: every answer comes from the cache, at no cost"
$EVALKIT run examples/baseline.toml --label baseline-rerun | grep -E "^cases|^cost"

step "Candidate A: a different change of similar quality"
$EVALKIT run examples/candidate-good.toml > /dev/null
$EVALKIT gate label:baseline label:candidate-good

step "Candidate B: a change that broke something"
$EVALKIT run examples/candidate-regressed.toml > /dev/null
set +e
$EVALKIT gate label:baseline label:candidate-regressed
code=$?
set -e

if [ "$code" -ne 1 ]; then
  echo "expected the gate to fail with exit code 1, got $code" >&2
  exit 1
fi
step "The gate exited 1 for candidate B, which is what fails the CI job"

step "An agent service: the reference agent passes evalkit agent conformance"
uv run python examples/agent_server.py --port 0 --port-file "$DB_DIR/agent.port" 2>/dev/null &
AGENT_PID=$!
for _ in $(seq 1 100); do [ -s "$DB_DIR/agent.port" ] && break; sleep 0.1; done
if [ ! -s "$DB_DIR/agent.port" ]; then
  echo "the reference agent did not start" >&2
  exit 1
fi
cat > "$DB_DIR/endpoints.toml" <<EOF
[endpoints.booker]
url = "http://127.0.0.1:$(cat "$DB_DIR/agent.port")/v1/step"
allow_private = true
timeout_seconds = 10
EOF
EVALKIT_HTTP_TARGETS_FILE="$DB_DIR/endpoints.toml" \
  $EVALKIT agent conformance booker --sandbox examples/agents/booking.toml --strict

step "50 booking conversations with a simulated user, then an agent that books without looking"
uv run python examples/agent_server.py --skip-lookup --port 0 --port-file "$DB_DIR/blind.port" 2>/dev/null &
BLIND_PID=$!
for _ in $(seq 1 100); do [ -s "$DB_DIR/blind.port" ] && break; sleep 0.1; done
if [ ! -s "$DB_DIR/blind.port" ]; then
  echo "the reference agent with --skip-lookup did not start" >&2
  exit 1
fi
cat >> "$DB_DIR/endpoints.toml" <<EOF
[endpoints.booker-blind]
url = "http://127.0.0.1:$(cat "$DB_DIR/blind.port")/v1/step"
allow_private = true
timeout_seconds = 10
EOF
export EVALKIT_HTTP_TARGETS_FILE="$DB_DIR/endpoints.toml"
$EVALKIT run examples/agents/booking-run.toml | grep -E "^cases|^tool_calls|^confirmed"
$EVALKIT run examples/agents/booking-blind.toml | grep -E "^cases|^tool_calls|^confirmed"
set +e
$EVALKIT gate label:booker label:booker-blind
code=$?
set -e
if [ "$code" -ne 1 ]; then
  echo "expected the gate to fail the blind agent with exit code 1, got $code" >&2
  exit 1
fi
step "The gate exited 1: the blind agent confirms as many bookings, but skips find_slots"
