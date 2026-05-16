#!/bin/bash -l
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=1:00:00
#$ -N mammofm_test_stage2
#$ -o /restricted/projectnb/batmanlab/atang4/data/jobs/test_stage2_v2.out
#$ -j y

set -e
echo "=== Node: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi') ==="
date

export CUDA_HOME=/share/pkg.8/cuda/12.2/install
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export HF_HOME=/restricted/projectnb/batmanlab/atang4/data/.hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRITON_CACHE_DIR=/restricted/projectnb/batmanlab/atang4/data/.triton

PYTHON=/restricted/projectnb/batmanlab/shawn24/llava_breast/bin/python3.10
LLAVA_SRC=/restricted/projectnb/batmanlab/shawn24/PhD/LLaVa-Breast-scc/LLaVa-Breast/src_pos_emb4views_new_loss
JOBS=/restricted/projectnb/batmanlab/atang4/data/jobs
BACKEND=/restricted/projectnb/batmanlab/atang4/MammoFM/backend
BASELINE=/restricted/projectnb/batmanlab/shawn24/PhD/LLaVa-Breast-scc/LLaVa-Breast/analysis/out/upmc_bu_embed_mayo_src_pos_emb4views_new_checkpoints_v1_bu_ve_old_loss_llava-llama3.1_8B_breast_clip-finetune_512-lora_checkpoint-5500_Merged_final_report_LP_with_findings.csv

MODEL_ID_LOCAL=$("$PYTHON" "$BACKEND/config.py" --id) || {
  echo "ERROR: Llama snapshot not under HF_HOME=$HF_HOME"
  exit 1
}

for PATIENT_EXAM in "P810759:E172868" "P810763:E172906" "P810763:E172907"; do
    PATIENT=${PATIENT_EXAM%%:*}
    EXAM=${PATIENT_EXAM##*:}
    JOB_DIR=$JOBS/test_${EXAM}
    echo ""
    echo "=== Testing patient=$PATIENT exam=$EXAM ==="

    # bridge — pass baseline CSV to get real zero_shot_per_image values
    $PYTHON $BACKEND/json_to_csv.py \
        --val-results-json $JOB_DIR/val_results.json \
        --output-csv $JOB_DIR/intermediate_zs.csv \
        --patient-id $PATIENT --exam-id $EXAM \
        --classifier-csv $BASELINE

    # Stage 2
    $PYTHON $LLAVA_SRC/final_stage.py \
        --input-csv  $JOB_DIR/intermediate_zs.csv \
        --output-csv $JOB_DIR/final_zs.csv \
        --model-id $MODEL_ID_LOCAL \
        --max-new-tokens 320 --temperature 0.3 --top-p 0.9 \
        --patient-id $PATIENT --exam-id $EXAM

    echo "--- Our Stage 2 output (with zero_shot_per_image) ---"
    $PYTHON -c "
import pandas as pd
df = pd.read_csv('$JOB_DIR/final_zs.csv')
print(df['final_generated_report_zs'].iloc[0])
"

    echo "--- Baseline output ---"
    $PYTHON -c "
import pandas as pd
df = pd.read_csv('$BASELINE')
row = df[(df['patient_id']=='$PATIENT') & (df['exam_id']=='$EXAM')]
print(row['final_generated_report_zs'].iloc[0] if not row.empty else 'NOT FOUND')
"
done

echo ""
echo "=== Test complete ==="
date
