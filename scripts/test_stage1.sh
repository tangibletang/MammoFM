#!/bin/bash -l
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=2:00:00
#$ -N mammofm_test_stage1
#$ -o /restricted/projectnb/batmanlab/atang4/data/validation/stage1_test/comparison_log.txt
#$ -j y

set -e
mkdir -p /restricted/projectnb/batmanlab/atang4/data/validation/stage1_test

echo "=== Node: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) ==="
date

export CUDA_HOME=/share/pkg.8/cuda/12.2/install
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TOKENIZERS_PARALLELISM=false
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TRITON_CACHE_DIR=/restricted/projectnb/batmanlab/atang4/data/.triton
export HF_HOME=/restricted/projectnb/batmanlab/atang4/data/.hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

PYTHON=/restricted/projectnb/batmanlab/shawn24/llava_breast/bin/python3.10
DEEPSPEED=/restricted/projectnb/batmanlab/shawn24/llava_breast/bin/deepspeed
LLAVA_SRC=/restricted/projectnb/batmanlab/shawn24/PhD/LLaVa-Breast-scc/LLaVa-Breast/src_pos_emb4views_new_loss
CHECKPOINT=/restricted/projectnb/batmanlab/atang4/data/checkpoints_v1_bu_ve_old_loss/llava-llama3.1_8B_breast_clip-finetune_512-lora/checkpoint-5500
JOBS=/restricted/projectnb/batmanlab/atang4/data/jobs
VAL_BASELINE=/restricted/projectnb/batmanlab/atang4/data/checkpoints_v1_bu_ve_old_loss/llava-llama3.1_8B_breast_clip-finetune_512-lora/checkpoint-5500/val_results.json

cd $LLAVA_SRC
export PYTHONPATH=$(pwd)

for EXAM in E172868 E172906 E172907; do
    JOB_DIR=$JOBS/stage1_test_${EXAM}
    echo ""
    echo "=== Stage 1 test: exam=$EXAM ==="

    $DEEPSPEED --master_port 12438 llava/serve/ctchat_validation_llama.py \
        --deepspeed ./zero3.json \
        --model-path $CHECKPOINT \
        --model-base meta-llama/Meta-Llama-3.1-8B-Instruct \
        --source_json $JOB_DIR/source.json \
        --bu_path $JOB_DIR/embed_bu \
        --out-path $JOB_DIR/val_results.json

    echo "--- Our Stage 1 output ---"
    $PYTHON -c "
import json
with open('$JOB_DIR/val_results.json') as f:
    data = json.load(f)
print(data[0]['conversations_out'][0]['answer'])
"

    echo "--- Baseline Stage 1 output ---"
    $PYTHON -c "
import json
with open('$VAL_BASELINE') as f:
    data = json.load(f)
match = [r for r in data if r['exam_id'] == '$EXAM']
print(match[0]['conversations_out'][0]['answer'] if match else 'NOT FOUND')
"

    # Save output
    cp $JOB_DIR/val_results.json \
       /restricted/projectnb/batmanlab/atang4/data/validation/stage1_test/${EXAM}_val_results.json
done

echo ""
echo "=== Stage 1 test complete ==="
date
