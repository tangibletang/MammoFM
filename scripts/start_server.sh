#!/bin/bash -l
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=8:00:00
#$ -N mammofm_server
#$ -o /restricted/projectnb/batmanlab/atang4/data/server.log
#$ -j y

export CUDA_HOME=/share/pkg.8/cuda/12.2/install
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TOKENIZERS_PARALLELISM=false
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TRITON_CACHE_DIR=/restricted/projectnb/batmanlab/atang4/data/.triton
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MAMMOFM_EMBED_SCRIPT=/restricted/projectnb/batmanlab/atang4/MammoFM/backend/save_img_embedding_mammofm.py

# GPU: default SCC request (no queue pinned — same pattern as scripts/smoke_test_stage1*.sh).
# Optional: if your project uses a named queue, add e.g. #$ -q ece  (or #$ -q a100 + gpu_c=8.0 when available).
# Optional: MAMMOFM_NO_KV_CACHE=1 to force use_cache=False during generate (debug only).
# Max new tokens: tune if reports truncate or you need to save VRAM.
# CPU fallback if GPU still OOM: MAMMOFM_STAGE1_CPU_GENERATE=1 or ./scripts/mammofm submit-cpu.
export MAMMOFM_STAGE1_CPU_GENERATE="${MAMMOFM_STAGE1_CPU_GENERATE:-0}"
export MAMMOFM_STAGE1_MAX_NEW_TOKENS="${MAMMOFM_STAGE1_MAX_NEW_TOKENS:-128}"
# Default GPU Stage 1: 8-bit LLM (projector/vision stay fp16 via sitecustomize patch). Fidelity: UI "Generate on CPU", submit-cpu, or MAMMOFM_STAGE1_LOAD_8BIT=0.
# 8-bit layout: sitecustomize pins device_map to GPU0 ({'':0}) to avoid CPU/CUDA RoPE errors; opt into accelerate auto-map with MAMMOFM_STAGE1_BNB_DEVICE_MAP=auto
export MAMMOFM_STAGE1_LOAD_8BIT="${MAMMOFM_STAGE1_LOAD_8BIT:-1}"

# qsub copies this script under /var/spool/sge/... — do not use BASH_SOURCE for repo paths.
SCRIPT_DIR="/restricted/projectnb/batmanlab/atang4/MammoFM/scripts"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/local_hf_env.inc.sh"

NODE=$(hostname)
PY=/restricted/projectnb/batmanlab/shawn24/llava_breast/bin/python3.10

echo "=== MammoFM  node=$NODE  $(date) ==="
echo "Laptop tunnel:  scripts/mammofm tunnel"
echo "HF_HOME=$HF_HOME  STAGE1_MAXTOK=${MAMMOFM_STAGE1_MAX_NEW_TOKENS:-}  STAGE1_8BIT=${MAMMOFM_STAGE1_LOAD_8BIT:-}  STAGE1_CPU=${MAMMOFM_STAGE1_CPU_GENERATE:-}"
echo "========================================================="

cat > /restricted/projectnb/batmanlab/atang4/data/server_info.txt << INFO
Node: $NODE
Started: $(date)
HF_HOME: $HF_HOME
SSH tunnel: ssh -F /dev/null -S none -o ControlMaster=no -N -J atang4@scc1.bu.edu -L 25001:${NODE}:8000 atang4@scc4.bu.edu
URL: http://127.0.0.1:25001/   (pick another LOCAL_PORT if 25001 is taken)
Stack: FastAPI (app.py) + frontend/
INFO

cd /restricted/projectnb/batmanlab/atang4/MammoFM/backend
"$PY" verify_backend.py \
    || { echo "verify_backend.py failed — fix backend/config.py or paths before starting."; exit 1; }
exec "$PY" -m uvicorn app:app --host 0.0.0.0 --port 8000
