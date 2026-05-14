#!/usr/bin/env bash
# Interactive uvicorn on a compute node with the same local HF_HOME seeding as qsub start_server.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_HOME=/share/pkg.8/cuda/12.2/install
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TRITON_CACHE_DIR=/restricted/projectnb/batmanlab/atang4/data/.triton
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MAMMOFM_EMBED_SCRIPT=/restricted/projectnb/batmanlab/atang4/MammoFM/backend/save_img_embedding_mammofm.py

# On a GPU compute node, enable 8-bit Stage 1 (same as qsub). Leave unset for CPU-only hosts.
if [[ -z "${MAMMOFM_STAGE1_LOAD_8BIT:-}" ]] && command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
  export MAMMOFM_STAGE1_LOAD_8BIT=1
fi

# shellcheck source=/dev/null
source "$ROOT/scripts/local_hf_env.inc.sh"

echo "HF_HOME=$HF_HOME"
echo "Stage1 8bit: MAMMOFM_STAGE1_LOAD_8BIT=${MAMMOFM_STAGE1_LOAD_8BIT:-0}"

PY=/restricted/projectnb/batmanlab/shawn24/llava_breast/bin/python3.10
cd "$ROOT/backend"
export PYTHONPATH=.
"$PY" verify_backend.py
exec "$PY" -m uvicorn app:app --host 0.0.0.0 --port 8000
