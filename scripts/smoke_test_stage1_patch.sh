#!/bin/bash -l
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=1:00:00
#$ -N mammofm_smoke_s1
#$ -o /restricted/projectnb/batmanlab/atang4/data/validation/smoke_stage1_patch.log
#$ -j y
#
# Default: GPU Stage 1 with 8-bit LLM (matches start_server.sh / pipeline when MAMMOFM_STAGE1_LOAD_8BIT=1).
#   qsub -v SMOKE_LOAD_8BIT=0            # fp16 path
#   qsub -v SMOKE_CPU_STAGE1=1           # CPU decode (forces fp16 load, no 8-bit)
#   qsub -v SMOKE_MAX_NEW_TOKENS=256
SMOKE_CPU_STAGE1="${SMOKE_CPU_STAGE1:-0}"
SMOKE_MAX_NEW_TOKENS="${SMOKE_MAX_NEW_TOKENS:-128}"
if [[ "$SMOKE_CPU_STAGE1" == "1" ]]; then
  SMOKE_LOAD_8BIT=0
else
  SMOKE_LOAD_8BIT="${SMOKE_LOAD_8BIT:-${MAMMOFM_STAGE1_LOAD_8BIT:-1}}"
fi
echo "SMOKE_CPU_STAGE1=$SMOKE_CPU_STAGE1  SMOKE_LOAD_8BIT=$SMOKE_LOAD_8BIT  SMOKE_MAX_NEW_TOKENS=$SMOKE_MAX_NEW_TOKENS"

set -eo pipefail
mkdir -p /restricted/projectnb/batmanlab/atang4/data/validation/smoke_stage1_patch

echo "=== smoke Stage1 + sitecustomize patch ==="
echo "=== Node: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null) ==="
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
export MAMMOFM_CPU_GENERATE="$SMOKE_CPU_STAGE1"
if [[ "$SMOKE_LOAD_8BIT" == "1" || "$SMOKE_LOAD_8BIT" == "true" || "$SMOKE_LOAD_8BIT" == "yes" ]]; then
  export MAMMOFM_SKIP_CPU_BEFORE_LORA_MERGE=1
fi
# Match production: leave MAMMOFM_NO_KV_CACHE unset (HF incremental decode). Set =1 only to debug.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Avoid CUTLASS SDPA backend issues on V100-class GPUs (see pipeline Stage 1/2 env).
export PYTORCH_ENABLE_MEM_EFFICIENT_SDPA=0

PYTHON=/restricted/projectnb/batmanlab/shawn24/llava_breast/bin/python3.10
LLAVA_SRC=/restricted/projectnb/batmanlab/shawn24/PhD/LLaVa-Breast-scc/LLaVa-Breast/src_pos_emb4views_new_loss
PATCH_SITE=/restricted/projectnb/batmanlab/atang4/MammoFM/backend/patch_site
MAMMOFM_ROOT=/restricted/projectnb/batmanlab/atang4/MammoFM
CHECKPOINT=/restricted/projectnb/batmanlab/atang4/data/checkpoints_v1_bu_ve_old_loss/llava-llama3.1_8B_breast_clip-finetune_512-lora/checkpoint-5500

MODEL_BASE_LOCAL=$("$PYTHON" "$MAMMOFM_ROOT/backend/config.py") || {
  echo "ERROR: Llama snapshot not under HF_HOME=$HF_HOME (see README / local_hf_env.inc.sh)"
  exit 1
}

# Reuse a completed job with embeddings + source.json (no UI).
JOB=/restricted/projectnb/batmanlab/atang4/data/jobs/500f8458-82ba-455f-a6c7-43036a287d3b
OUT=/restricted/projectnb/batmanlab/atang4/data/validation/smoke_stage1_patch/val_results.json

export PYTHONPATH="${PATCH_SITE}:${LLAVA_SRC}"

cd "$LLAVA_SRC"

# Sanity: sitecustomize + our mean-init patch
/restricted/projectnb/batmanlab/shawn24/llava_breast/bin/python3.10 -c "
import sys
assert 'sitecustomize' in sys.modules
import transformers.modeling_utils as m
assert m.PreTrainedModel._init_added_embeddings_weights_with_mean.__name__ == '_low_gpu_mean_init'
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
assert LlavaLlamaForCausalLM.from_pretrained.__name__ == '_llava_from_pretrained_full_materialize'
print('patch_ok')
"

echo "=== Running Stage 1 validation (plain torch — ctchat does not use DeepSpeed config) ==="
CTCHAT_EXTRA=()
if [[ "$SMOKE_LOAD_8BIT" == "1" || "$SMOKE_LOAD_8BIT" == "true" || "$SMOKE_LOAD_8BIT" == "yes" ]]; then
  CTCHAT_EXTRA+=(--load-8bit)
fi
"$PYTHON" -u llava/serve/ctchat_validation_llama.py \
    --model-path "$CHECKPOINT" \
    --model-base "$MODEL_BASE_LOCAL" \
    --source_json "$JOB/source.json" \
    --bu_path "$JOB/embed_bu" \
    --out-path "$OUT" \
    --max-new-tokens "$SMOKE_MAX_NEW_TOKENS" \
    "${CTCHAT_EXTRA[@]}"

/restricted/projectnb/batmanlab/shawn24/llava_breast/bin/python3.10 -c "
import json
with open('$OUT') as f:
    d = json.load(f)
assert d and 'conversations_out' in d[0]
ans = d[0]['conversations_out'][0]['answer']
assert len(ans) > 20
print('STAGE1_SMOKE_OK', len(ans), 'chars')
print(ans[:400], '...')
"

echo "=== smoke complete ==="
date
