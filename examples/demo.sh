#!/usr/bin/env bash
# Shows a regression being caught. Needs no API key: the targets are fakes with
# known accuracy. Uses a throwaway SQLite database.
set -euo pipefail
cd "$(dirname "$0")/.."

EVALKIT=${EVALKIT:-"uv run evalkit"}
DB_DIR=$(mktemp -d)
trap 'rm -rf "$DB_DIR"' EXIT
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
