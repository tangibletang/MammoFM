#!/bin/bash -l
#$ -l gpus=1
#$ -l gpu_c=8.0
# TODO: tighten to A100-only if needed — BU SCC may support -l gpu_type=A100 in some queues.
#$ -l h_rt=4:00:00
#$ -N mammofm_verify_8bit
#$ -o /restricted/projectnb/batmanlab/atang4/data/validation/smoke_e2e_a100.$JOB_ID.log
#$ -j y
#
# 8-bit GPU pathway verification. Requires A100/A40-class GPU (compute capability >= 8.0).
# Writes verify_8bit.OK or verify_8bit.FAIL under /restricted/projectnb/batmanlab/atang4/data/validation/
#
#   qsub scripts/smoke_e2e_a100.sh
#   qsub -sync y scripts/smoke_e2e_a100.sh   # blocks until done
set -eo pipefail

SENTINEL_DIR=/restricted/projectnb/batmanlab/atang4/data/validation
mkdir -p "$SENTINEL_DIR"

echo "=== MammoFM 8-bit GPU pathway verification ==="
echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null || true
date

export CUDA_HOME=/share/pkg.8/cuda/12.2/install
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

export TOKENIZERS_PARALLELISM=false
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TRITON_CACHE_DIR=/restricted/projectnb/batmanlab/atang4/data/.triton
export HF_HOME=/restricted/projectnb/batmanlab/atang4/data/.hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export MAMMOFM_STAGE1_LOAD_8BIT=1
export MAMMOFM_SKIP_CPU_BEFORE_LORA_MERGE=1
export MAMMOFM_STAGE1_CUDA_ALLOC_CONF="${MAMMOFM_STAGE1_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export PYTORCH_CUDA_ALLOC_CONF="$MAMMOFM_STAGE1_CUDA_ALLOC_CONF"
export MAMMOFM_ENFORCE_IDLE_CUDA_DEVICE=1
export PYTORCH_ENABLE_MEM_EFFICIENT_SDPA=0

PYTHON=/restricted/projectnb/batmanlab/shawn24/llava_breast/bin/python3.10
BACKEND=/restricted/projectnb/batmanlab/atang4/MammoFM/backend
LOG_TAIL="$SENTINEL_DIR/smoke_e2e_a100.$JOB_ID.log"

cd "$BACKEND"

set +e
OUTPUT=$("$PYTHON" -u smoke_e2e_pipeline.py 2>&1)
EXIT_CODE=$?
set -e

echo "$OUTPUT"

# Validate: must exit 0 AND print E2E_OK AND non-empty report lengths
PRELIM_CHARS=$(echo "$OUTPUT" | grep "^preliminary chars:" | awk '{print $NF}')
FINAL_CHARS=$(echo "$OUTPUT" | grep "^final chars:" | awk '{print $NF}')
JOB_ID_SMOKE=$(echo "$OUTPUT" | grep "^E2E_OK" | awk '{print $2}')

if [[ $EXIT_CODE -eq 0 && -n "$JOB_ID_SMOKE" && "${PRELIM_CHARS:-0}" -gt 0 && "${FINAL_CHARS:-0}" -gt 0 ]]; then
    cat > "$SENTINEL_DIR/verify_8bit.OK" <<EOF
job_id=$JOB_ID_SMOKE
qsub_job_id=${JOB_ID:-unknown}
node=$(hostname)
timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
preliminary_chars=$PRELIM_CHARS
final_chars=$FINAL_CHARS
log=$LOG_TAIL
EOF
    rm -f "$SENTINEL_DIR/verify_8bit.FAIL"
    echo "=== verify_8bit.OK written ==="
else
    TAIL=$(echo "$OUTPUT" | tail -30)
    cat > "$SENTINEL_DIR/verify_8bit.FAIL" <<EOF
exit_code=$EXIT_CODE
qsub_job_id=${JOB_ID:-unknown}
node=$(hostname)
timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
preliminary_chars=${PRELIM_CHARS:-0}
final_chars=${FINAL_CHARS:-0}
log=$LOG_TAIL
tail_output<<ENDTAIL
$TAIL
ENDTAIL
EOF
    rm -f "$SENTINEL_DIR/verify_8bit.OK"
    echo "=== verify_8bit.FAIL written ==="
    exit 1
fi

echo "=== 8-bit GPU verification finished ==="
date
