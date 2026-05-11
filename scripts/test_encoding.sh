#!/bin/bash -l
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=1:00:00
#$ -N mammofm_test_encode
#$ -o /restricted/projectnb/batmanlab/atang4/data/validation/encode_test.log
#$ -j y

set -e
echo "=== Node: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
date

export CUDA_HOME=/share/pkg.8/cuda/12.2/install
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/restricted/projectnb/batmanlab/shawn24/PhD/LLaVa-Breast-scc/LLaVa-Breast/src_pos_emb4views_new_loss
export HF_HOME=/restricted/projectnb/batmanlab/atang4/data/.hf_cache
export TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=/restricted/projectnb/batmanlab/atang4/data/.triton

PYTHON=/restricted/projectnb/batmanlab/shawn24/llava_breast/bin/python3.10
LLAVA_SRC=/restricted/projectnb/batmanlab/shawn24/PhD/LLaVa-Breast-scc/LLaVa-Breast/src_pos_emb4views_new_loss
JOB=/restricted/projectnb/batmanlab/atang4/data/jobs/test_encode
EXAM=test_exam

mkdir -p $JOB/embed_bu/controls/test_images_png/$EXAM

# Copy test images
for VIEW in LMLO LCC RMLO RCC; do
    cp /restricted/projectnb/batmanlab/atang4/Breast-CLIP-downstream/Breast-CLIP-downstream/${VIEW}.png \
       $JOB/embed_bu/controls/test_images_png/$EXAM/${VIEW}.png
done
echo "Images copied"

# Build CSV
cat > $JOB/images.csv << 'EOF'
dataset,file_path
BU,/restricted/projectnb/pixel/hariri/MGdata_for_mirai/png/controls/test_exam/LMLO.png
BU,/restricted/projectnb/pixel/hariri/MGdata_for_mirai/png/controls/test_exam/LCC.png
BU,/restricted/projectnb/pixel/hariri/MGdata_for_mirai/png/controls/test_exam/RMLO.png
BU,/restricted/projectnb/pixel/hariri/MGdata_for_mirai/png/controls/test_exam/RCC.png
EOF
echo "CSV built"

# Run encoding
echo "=== Running encoding ==="
$PYTHON $LLAVA_SRC/llava/model/multimodal_encoder/save_img_embedding.py \
    --mammo-clip-chkpt /restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP/src/codebase/outputs/mayo/MammoCLIP-MayoClinic-epoch4.tar \
    --data-csv $JOB/images.csv \
    --bu_path $JOB/embed_bu \
    --inference-mode y

echo "=== Encoding done. .pt files: ==="
ls -lh $JOB/embed_bu/controls/test_images_png/$EXAM/

echo "=== Test complete ==="
date
