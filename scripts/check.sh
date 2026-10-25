#!/usr/bin/env bash
# The full local gate, the same steps CI runs.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest "$@"
