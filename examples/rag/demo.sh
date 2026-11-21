#!/usr/bin/env bash
# A RAG pipeline end to end: retrieval then generation as one target, ranking
# metrics against graded labels, claim-level grounding and completeness judged
# per case, the RAG report saying why each failing case failed, and a gate that
# fails a retriever change. Needs no API key: the retrievers, the generator and
# the grader are the fakes in examples/rag/server.py. Uses a throwaway SQLite
# database.
set -euo pipefail
cd "$(dirname "$0")/../.."

EVALKIT=${EVALKIT:-"uv run evalkit"}
WORK=$(mktemp -d)
SERVER_PID=
cleanup() {
  if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null || true; fi
  rm -rf "$WORK"
}
trap cleanup EXIT
export EVALKIT_DATABASE_URL="sqlite:///$WORK/rag.db"

step() { printf '\n== %s\n' "$*"; }

uv run python examples/rag/server.py --port 0 --port-file "$WORK/port" 2>/dev/null &
SERVER_PID=$!
for _ in $(seq 1 100); do [ -s "$WORK/port" ] && break; sleep 0.1; done
if [ ! -s "$WORK/port" ]; then
  echo "the RAG fakes did not start" >&2
  exit 1
fi
PORT=$(cat "$WORK/port")
# Operator config: the checked-in files on the port the server picked.
sed "s/:8766/:$PORT/" examples/rag/endpoints.toml > "$WORK/endpoints.toml"
sed "s/:8766/:$PORT/" examples/rag/providers.toml > "$WORK/providers.toml"
export EVALKIT_HTTP_TARGETS_FILE="$WORK/endpoints.toml"
export EVALKIT_PROVIDERS_FILE="$WORK/providers.toml"
export EVALKIT_PRICES_FILE=examples/rag/prices.toml

step "The pipeline on the current retriever: 12 questions, retrieval and answer stored per case"
$EVALKIT run examples/rag/pipeline.toml

step "Why its failing cases failed"
$EVALKIT report rag label:rag

step "The same pipeline on the retriever change under review"
$EVALKIT run examples/rag/pipeline-next.toml > /dev/null
$EVALKIT report rag label:rag-next

step "Gate the change against the current retriever"
set +e
$EVALKIT gate label:rag label:rag-next
code=$?
set -e
if [ "$code" -ne 1 ]; then
  echo "expected the gate to fail the retriever change with exit code 1, got $code" >&2
  exit 1
fi
step "The gate exited 1: the new retriever loses the answering documents, and the answers with them"
