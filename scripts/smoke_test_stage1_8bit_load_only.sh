#!/bin/bash -l
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=0:45:00
#$ -N mammofm_smoke_s1_8bit_load
#$ -o /restricted/projectnb/batmanlab/atang4/data/validation/smoke_stage1_8bit_load_only.log
#$ -j y
#
# Canary: 8-bit checkpoint load through LoRA/merge hook (no token generation).
# Bitsandbytes + this LLaVA mm_projector still fails at generate — use smoke_test_stage1_patch.sh for e2e.
#
# Submit: qsub /restricted/projectnb/batmanlab/atang4/MammoFM/scripts/smoke_test_stage1_8bit_load_only.sh
#
set -eo pipefail

ROOT=/restricted/projectnb/batmanlab/atang4/MammoFM

echo "=== smoke Stage1 8-bit load-only ==="
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MAMMOFM_SKIP_CPU_BEFORE_LORA_MERGE=1
export MAMMOFM_NO_KV_CACHE=1
export MAMMOFM_CPU_GENERATE=0

PYTHON=/restricted/projectnb/batmanlab/shawn24/llava_breast/bin/python3.10
PATCH_SITE="$ROOT/backend/patch_site"
LLAVA_SRC=/restricted/projectnb/batmanlab/shawn24/PhD/LLaVa-Breast-scc/LLaVa-Breast/src_pos_emb4views_new_loss
CHECKPOINT=/restricted/projectnb/batmanlab/atang4/data/checkpoints_v1_bu_ve_old_loss/llava-llama3.1_8B_breast_clip-finetune_512-lora/checkpoint-5500
export PYTHONPATH="${PATCH_SITE}:${LLAVA_SRC}"

"$PYTHON" "$ROOT/backend/smoke_stage1_load_merge_only.py" \
  --model-path "$CHECKPOINT" \
  --model-base meta-llama/Meta-Llama-3.1-8B-Instruct

echo "=== smoke 8-bit load-only complete ==="
date
