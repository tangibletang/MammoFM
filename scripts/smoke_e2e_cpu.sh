#!/bin/bash -l
# No GPU request — CPU-only Stage 1.
#$ -l h_rt=4:00:00
#$ -N mammofm_verify_cpu
#$ -o /restricted/projectnb/batmanlab/atang4/data/validation/smoke_e2e_cpu.$JOB_ID.log
#$ -j y
#
# CPU fallback pathway verification. Stage 1 runs on CPU (fp16, capped tokens for speed).
# Mammo-CLIP encoding and Stage 2 still use GPU when available on the assigned node.
# Writes verify_cpu.OK or verify_cpu.FAIL under /restricted/projectnb/batmanlab/atang4/data/validation/
#
#   qsub scripts/smoke_e2e_cpu.sh
#   qsub -sync y scripts/smoke_e2e_cpu.sh   # blocks until done
set -eo pipefail

SENTINEL_DIR=/restricted/projectnb/batmanlab/atang4/data/validation
mkdir -p "$SENTINEL_DIR"

echo "=== MammoFM CPU pathway verification ==="
echo "=== Node: $(hostname) ==="
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

export MAMMOFM_STAGE1_CPU_GENERATE=1
export MAMMOFM_STAGE1_LOAD_8BIT=0
export MAMMOFM_STAGE1_MAX_NEW_TOKENS="${MAMMOFM_STAGE1_MAX_NEW_TOKENS:-64}"  # capped: CPU is slow

PYTHON=/restricted/projectnb/batmanlab/shawn24/llava_breast/bin/python3.10
BACKEND=/restricted/projectnb/batmanlab/atang4/MammoFM/backend
LOG_TAIL="$SENTINEL_DIR/smoke_e2e_cpu.$JOB_ID.log"

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
    cat > "$SENTINEL_DIR/verify_cpu.OK" <<EOF
job_id=$JOB_ID_SMOKE
qsub_job_id=${JOB_ID:-unknown}
node=$(hostname)
timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
preliminary_chars=$PRELIM_CHARS
final_chars=$FINAL_CHARS
log=$LOG_TAIL
EOF
    rm -f "$SENTINEL_DIR/verify_cpu.FAIL"
    echo "=== verify_cpu.OK written ==="
else
    TAIL=$(echo "$OUTPUT" | tail -30)
    cat > "$SENTINEL_DIR/verify_cpu.FAIL" <<EOF
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
    rm -f "$SENTINEL_DIR/verify_cpu.OK"
    echo "=== verify_cpu.FAIL written ==="
    exit 1
fi

echo "=== CPU pathway verification finished ==="
date
