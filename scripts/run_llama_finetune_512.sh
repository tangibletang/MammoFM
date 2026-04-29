#!/bin/sh
#SBATCH --output=/restricted/projectnb/batmanlab/shawn24/PhD/LLaVa-Breast-scc/LLaVa-Breast/src/logs/scc_logs/512_finetune_%j.out

pwd
hostname
date

slurm_output_train1=/restricted/projectnb/batmanlab/shawn24/PhD/LLaVa-Breast-scc/LLaVa-Breast/src/logs/scc_logs/512_finetune_$CURRENT.out

export CUDA_HOME=/share/pkg.8/cuda/12.2/install
export CUDA_PATH=/share/pkg.8/cuda/12.2/install
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/include:$CPATH
export TOKENIZERS_PARALLELISM=false
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

module load miniconda
conda activate /restricted/projectnb/batmanlab/shawn24/llava_breast

echo -e '\nPretrain llama3.1_8B 512 attention!\n'
cd /restricted/projectnb/batmanlab/shawn24/PhD/LLaVa-Breast-scc/LLaVa-Breast/src_pos_emb_4_views_wo_short_answers
export PYTHONPATH=$(pwd)


 python extract_gt_findings.py --report_col "conversations_out" \
 --json_file "/restricted/projectnb/batmanlab/shawn24/PhD/LLaVa-Breast-scc/LLaVa-Breast/src_pos_emb_4_views_wo_short_answers/checkpoints_mayo_pretrain_wo_short_answers/llava-llama3.1_8B_breast_clip-finetune_512-lora/checkpoint-9500/val_results.json" \
 --save_file_name "./out/upmc_bu_embed_mayo_src_pos_emb_4_views_llava_wo_short_answers_llava-llama3.1_8B_breast_clip-finetune_512-lora_checkpoint-9500.csv"



deepspeed --master_port 12438 llava/serve/ctchat_validation_llama.py \
    --deepspeed ./zero3.json \
    --model-path /restricted/projectnb/batmanlab/shawn24/PhD/LLaVa-Breast-scc/LLaVa-Breast/src_pos_emb_4_views_wo_short_answers/checkpoints_mayo_pretrain_wo_short_answers/llava-llama3.1_8B_breast_clip-finetune_512-lora/checkpoint-9500 \
    --model-base meta-llama/Meta-Llama-3.1-8B-Instruct \
    --out-path /restricted/projectnb/batmanlab/shawn24/PhD/LLaVa-Breast-scc/LLaVa-Breast/src_pos_emb_4_views_wo_short_answers/checkpoints_mayo_pretrain_wo_short_answers/llava-llama3.1_8B_breast_clip-finetune_512-lora/checkpoint-9500/val_results.json


deepspeed --master_port 12540  llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./zero3.json \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --version llama3_1 \
    --data_path /restricted/projectnb/batmanlab/shawn24/PhD/LLaVa-Breast-scc/LLaVa-Breast/dataset_json/train_vqa.json \
    --image_folder path_to_train_encodings/encodings/ \
    --vision_tower /work/nvme/beaq/shg121/LLaVa-Breast/src/Mammo-clip-chk_pt/model-best.tar \
    --mm_projector_type "attn_pool+mlp2x_gelu" \
    --mm_n_queries 512 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --pretrain_mm_mlp_adapter ./checkpoints_mayo_pretrain_wo_short_answers/llava-llama3.1_8B_breast_clip-pretrain_512attention/mm_projector.bin \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints_mayo_pretrain_wo_short_answers/llava-llama3.1_8B_breast_clip-finetune_512-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 10000 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 128000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 10 \
    --lazy_preprocess True \
    --report_to wandb
