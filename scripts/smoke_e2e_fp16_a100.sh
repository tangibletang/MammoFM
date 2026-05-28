#!/bin/bash -l
#$ -l gpus=1
#$ -l gpu_c=8.0
# TODO: tighten to A100-only if needed — BU SCC may support -l gpu_type=A100 in some queues.
#$ -l h_rt=4:00:00
#$ -N mammofm_verify_fp16
#$ -o /restricted/projectnb/batmanlab/atang4/data/validation/smoke_e2e_fp16_a100.$JOB_ID.log
#$ -j y
#
# 16-bit (fp16) GPU pathway verification on A100/A40-class (compute capability >= 8.0).
# This is the ORIGINAL unquantized Stage 1 pathway, kept on GPU (no CPU decode).
# Differs from smoke_e2e_a100.sh only by MAMMOFM_STAGE1_LOAD_8BIT=0.
# Writes verify_fp16.OK or verify_fp16.FAIL under /restricted/projectnb/batmanlab/atang4/data/validation/
#
#   qsub scripts/smoke_e2e_fp16_a100.sh
#   qsub -sync y scripts/smoke_e2e_fp16_a100.sh   # blocks until done
set -eo pipefail

SENTINEL_DIR=/restricted/projectnb/batmanlab/atang4/data/validation
mkdir -p "$SENTINEL_DIR"

echo "=== MammoFM 16-bit GPU (A100) pathway verification ==="
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

# fp16 on GPU: no 8-bit quantization, Stage 1 decode stays on GPU (NOT CPU)
export MAMMOFM_STAGE1_LOAD_8BIT=0
# Explicitly DO NOT set MAMMOFM_STAGE1_CPU_GENERATE — keep Stage 1 on GPU
unset MAMMOFM_STAGE1_CPU_GENERATE || true
export MAMMOFM_STAGE1_CUDA_ALLOC_CONF="${MAMMOFM_STAGE1_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export PYTORCH_CUDA_ALLOC_CONF="$MAMMOFM_STAGE1_CUDA_ALLOC_CONF"
export MAMMOFM_ENFORCE_IDLE_CUDA_DEVICE=1
export PYTORCH_ENABLE_MEM_EFFICIENT_SDPA=0

PYTHON=/restricted/projectnb/batmanlab/shawn24/llava_breast/bin/python3.10
BACKEND=/restricted/projectnb/batmanlab/atang4/MammoFM/backend
LOG_TAIL="$SENTINEL_DIR/smoke_e2e_fp16_a100.$JOB_ID.log"

cd "$BACKEND"

set +e
OUTPUT=$("$PYTHON" -u smoke_e2e_pipeline.py 2>&1)
EXIT_CODE=$?
set -e

echo "$OUTPUT"

PRELIM_CHARS=$(echo "$OUTPUT" | grep "^preliminary chars:" | awk '{print $NF}')
FINAL_CHARS=$(echo "$OUTPUT" | grep "^final chars:" | awk '{print $NF}')
JOB_ID_SMOKE=$(echo "$OUTPUT" | grep "^E2E_OK" | awk '{print $2}')

if [[ $EXIT_CODE -eq 0 && -n "$JOB_ID_SMOKE" && "${PRELIM_CHARS:-0}" -gt 0 && "${FINAL_CHARS:-0}" -gt 0 ]]; then
    cat > "$SENTINEL_DIR/verify_fp16.OK" <<EOF
job_id=$JOB_ID_SMOKE
qsub_job_id=${JOB_ID:-unknown}
node=$(hostname)
timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
preliminary_chars=$PRELIM_CHARS
final_chars=$FINAL_CHARS
log=$LOG_TAIL
EOF
    rm -f "$SENTINEL_DIR/verify_fp16.FAIL"
    echo "=== verify_fp16.OK written ==="
else
    TAIL=$(echo "$OUTPUT" | tail -30)
    cat > "$SENTINEL_DIR/verify_fp16.FAIL" <<EOF
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
    rm -f "$SENTINEL_DIR/verify_fp16.OK"
    echo "=== verify_fp16.FAIL written ==="
    exit 1
fi

echo "=== fp16 GPU (A100) verification finished ==="
date
