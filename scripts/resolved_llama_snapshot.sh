#!/usr/bin/env bash
# Print local HF hub snapshot path for offline Stage 1/2 (same as backend pipeline).
# Usage:
#   ./scripts/resolved_llama_snapshot.sh         -> MODEL_BASE (or MAMMOFM_MODEL_BASE)
#   ./scripts/resolved_llama_snapshot.sh --id   -> MODEL_ID (or MAMMOFM_MODEL_ID)
#   ./scripts/resolved_llama_snapshot.sh org/repo
# Requires HF_HOME (export before call, or source local_hf_env.inc.sh like start_server).
# Exits 2 if no snapshot with config.json (matches python backend/config.py).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${MAMMOFM_PYTHON:-/restricted/projectnb/batmanlab/shawn24/llava_breast/bin/python3.10}"
exec "$PY" "$ROOT/backend/config.py" "$@"
