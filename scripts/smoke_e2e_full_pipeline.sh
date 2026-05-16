#!/bin/bash -l
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=2:00:00
#$ -N mammofm_e2e_full
#$ -o /restricted/projectnb/batmanlab/atang4/data/validation/smoke_e2e_full_pipeline.$JOB_ID.log
#$ -j y
#
# Encode + Stage 1 (LLaVA) + Stage 2 (LLaMA). Default 8-bit Stage 1 matches production.
#   qsub -sync y /path/to/smoke_e2e_full_pipeline.sh
#   qsub -v SMOKE_LOAD_8BIT=0 -sync y ...   # fp16 Stage 1
set -eo pipefail
mkdir -p /restricted/projectnb/batmanlab/atang4/data/validation

echo "=== MammoFM E2E smoke (encode + stage1 + stage2) ==="
echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
date

export CUDA_HOME=/share/pkg.8/cuda/12.2/install
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

export TOKENIZERS_PARALLELISM=false
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TRITON_CACHE_DIR=/restricted/projectnb/batmanlab/atang4/data/.triton

# Prefer shared cache; node-local HF can be empty before seed — pipeline uses fallback in config.
export HF_HOME=/restricted/projectnb/batmanlab/atang4/data/.hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

SMOKE_LOAD_8BIT="${SMOKE_LOAD_8BIT:-${MAMMOFM_STAGE1_LOAD_8BIT:-1}}"
if [[ "$SMOKE_LOAD_8BIT" == "1" || "$SMOKE_LOAD_8BIT" == "true" || "$SMOKE_LOAD_8BIT" == "yes" ]]; then
  export MAMMOFM_STAGE1_LOAD_8BIT=1
  export MAMMOFM_SKIP_CPU_BEFORE_LORA_MERGE=1
else
  export MAMMOFM_STAGE1_LOAD_8BIT=0
fi

# Avoid expandable_segments: under VRAM pressure it can trip PyTorch 2.1 CUDACachingAllocator asserts
# (!block->expandable_segment_) on shared nodes. Pipeline Stage 1 reads MAMMOFM_STAGE1_CUDA_ALLOC_CONF.
export MAMMOFM_STAGE1_CUDA_ALLOC_CONF="${MAMMOFM_STAGE1_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export PYTORCH_CUDA_ALLOC_CONF="$MAMMOFM_STAGE1_CUDA_ALLOC_CONF"
export MAMMOFM_ENFORCE_IDLE_CUDA_DEVICE=1
export PYTORCH_ENABLE_MEM_EFFICIENT_SDPA=0

PYTHON=/restricted/projectnb/batmanlab/shawn24/llava_breast/bin/python3.10
BACKEND=/restricted/projectnb/batmanlab/atang4/MammoFM/backend

echo "SMOKE_LOAD_8BIT=$SMOKE_LOAD_8BIT  MAMMOFM_STAGE1_LOAD_8BIT=$MAMMOFM_STAGE1_LOAD_8BIT"

cd "$BACKEND"
"$PYTHON" -u smoke_e2e_pipeline.py

echo "=== E2E smoke finished ==="
date
