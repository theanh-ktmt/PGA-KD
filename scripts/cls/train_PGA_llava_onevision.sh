#!/bin/bash

# GPU per node
NUM_GPUS_PER_NODE=8
LORA_R=16
LORA_A=32
BATCH_SIZE=8
MSE_W=1.0
PGA_W=2.0
SCL_W=0.01
VAR_T=0.85

# Configs
TRAIN_SCRIPT="main.py"
EXP_NAME="PGA_Onevision_full_cls_r${LORA_R}_bs${BATCH_SIZE}"
USE_FULLSET=true

if [ "$USE_FULLSET" = true ]; then
    SUBSETS=("ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397")
    echo "Running with FULL dataset set."
else
    SUBSETS=("ImageNet_1K")
    echo "Running with SINGLE dataset (ImageNet_1K)."
fi

# Run with torchrun
torchrun --nproc_per_node=$NUM_GPUS_PER_NODE $TRAIN_SCRIPT \
    --model_name "llava-hf/llava-onevision-qwen2-0.5b-ov-hf" \
    --teacher_model_name "raghavlite/B3_Qwen2_2B" \
    --lora True \
    --teacher_lora True \
    --lora_r $LORA_R \
    --lora_alpha $LORA_A \
    --teacher_lora_r 8 \
    --teacher_pooling "eos" \
    --teacher_backbone "qwen2_vl" \
    --model_backbone "llava_onevision" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split "original" \
    --image_dir "vlm2vec_train/MMEB-train" \
    --output_dir "training/$EXP_NAME" \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --bf16 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --seed 42 \
    --weight_decay 0.01 \
    --normalize True \
    --teacher_normalize True \
    --lr_scheduler_type "constant" \
    --warmup_ratio 0.03 \
    --kd_weight 0.3 \
    --kd_loss_type "pga" \
    --image_resolution "low" \
    --projector_config_path "./config/projector_config.json" \
    --projector_lr 5e-4 \
    --report_to "wandb" \
    --run_name "$EXP_NAME" \
    --pga_mse_loss_weight $MSE_W \
    --pga_loss_weight $PGA_W \
    --pga_scl_loss_weight $SCL_W \
    --pga_spectral_variance_threshold $VAR_T