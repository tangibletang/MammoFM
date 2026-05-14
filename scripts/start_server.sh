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

# Stage 1: fp16 load decodes on GPU by default (set MAMMOFM_STAGE1_CPU_GENERATE=1 if you OOM — slow).
# 8-bit/4-bit remain experimental (load-only); keep MAMMOFM_STAGE1_LOAD_8BIT=0 for working e2e.
# Optional: MAMMOFM_STAGE1_MAX_NEW_TOKENS=128 to reduce decode VRAM.
export MAMMOFM_STAGE1_LOAD_8BIT="${MAMMOFM_STAGE1_LOAD_8BIT:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/local_hf_env.inc.sh"

NODE=$(hostname)
PY=/restricted/projectnb/batmanlab/shawn24/llava_breast/bin/python3.10

echo "=== MammoFM  node=$NODE  $(date) ==="
echo "Laptop tunnel:  scripts/mammofm tunnel"
echo "HF_HOME=$HF_HOME  STAGE1_8BIT=${MAMMOFM_STAGE1_LOAD_8BIT:-}  STAGE1_CPU=${MAMMOFM_STAGE1_CPU_GENERATE:-}"
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
