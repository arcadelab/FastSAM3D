#!/bin/bash

TASK_NAME="union_train"
CLICK_TYPE="random"
MULTI_CLICK=false
MODEL_TYPE="vit_b_ori"
CHECKPOINT="./work_dir/SAM/sam_vit_b.pth"
DEVICE="cuda"
WORK_DIR="./work_dir1"
NUM_WORKERS=0
GPU_IDS="0 1"  
MULTI_GPU=false
RESUME=false
LR_SCHEDULER="multisteplr"
STEP_SIZE="120 180" 
GAMMA=0.1
NUM_EPOCHS=42
IMG_SIZE=128
BATCH_SIZE=4
ACCUMULATION_STEPS=8
LR=8e-4
WEIGHT_DECAY=0.1
PORT=12361
IMAGE_PATH="./data/ttrain/images"
LABEL_PATH="./data/ttrain/label"
TEACHER_WEIGHTS_PATH="weights/teacher_model_weights.pth"

python distillation1.py \
    --task_name $TASK_NAME \
    --click_type $CLICK_TYPE \
    --multi_click $MULTI_CLICK \
    --model_type $MODEL_TYPE \
    --checkpoint $CHECKPOINT \
    --device $DEVICE \
    --work_dir $WORK_DIR \
    --num_workers $NUM_WORKERS \
    --gpu_ids $GPU_IDS \
    --multi_gpu $MULTI_GPU \
    --resume $RESUME \
    --lr_scheduler $LR_SCHEDULER \
    --gamma $GAMMA \
    --num_epochs $NUM_EPOCHS \
    --img_size $IMG_SIZE \
    --batch_size $BATCH_SIZE \
    --accumulation_steps $ACCUMULATION_STEPS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --port $PORT \
    --image_path $IMAGE_PATH \
    --label_path $LABEL_PATH \
    --teacher_weights_path $TEACHER_WEIGHTS_PATH
