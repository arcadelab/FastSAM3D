#!/bin/bash

TEST_DATA_PATH="./data/validation"
VIS_PATH="./visualization"
CHECKPOINT_PATH="./ckpt/sam_med3d_turbo.pth"
SAVE_NAME="union_out_dice.py"

IMAGE_SIZE=256
CROP_SIZE=128
DEVICE="cuda"
MODEL_TYPE="vit_b_ori"
NUM_CLICKS=5
POINT_METHOD="default"
DATA_TYPE="Ts"

THRESHOLD=0
DIM=3
SPLIT_IDX=0
SPLIT_NUM=1
FT2D=false
SEED=2023
CLICK_METHOD="default"

python validatioin1.py \
  --test_data_path $TEST_DATA_PATH \
  --vis_path $VIS_PATH \
  --checkpoint_path $CHECKPOINT_PATH \
  --save_name $SAVE_NAME \
  --image_size $IMAGE_SIZE \
  --crop_size $CROP_SIZE \
  --device $DEVICE \
  --model_type $MODEL_TYPE \
  --num_clicks $NUM_CLICKS \
  --point_method $POINT_METHOD \
  --data_type $DATA_TYPE \
  --threshold $THRESHOLD \
  --dim $DIM \
  --split_idx $SPLIT_IDX \
  --split_num $SPLIT_NUM \
  --ft2d $FT2D \
  --seed $SEED \
  #--click_method $CLICK_METHOD



# parser.add_argument('-tdp', '--test_data_path', type=str, default='./data/validation')
# parser.add_argument('-vp', '--vis_path', type=str, default='./visualization')
# parser.add_argument('-cp', '--checkpoint_path', type=str, default='./ckpt/sam_med3d_turbo.pth')
# parser.add_argument('--save_name', type=str, default='union_out_dice.py')

# parser.add_argument('--image_size', type=int, default=256)
# parser.add_argument('--crop_size', type=int, default=128)
# parser.add_argument('--device', type=str, default='cuda')
# parser.add_argument('-mt', '--model_type', type=str, default='vit_b_ori')
# parser.add_argument('-nc', '--num_clicks', type=int, default=5)
# parser.add_argument('-pm', '--point_method', type=str, default='default')
# parser.add_argument('-dt', '--data_type', type=str, default='Ts')

# parser.add_argument('--threshold', type=int, default=0)
# parser.add_argument('--dim', type=int, default=3)
# parser.add_argument('--split_idx', type=int, default=0)
# parser.add_argument('--split_num', type=int, default=1)
# parser.add_argument('--ft2d', action='store_true', default=False)
# parser.add_argument('--seed', type=int, default=2023)
# parser.add_argument('-cm', '--click_method', type=str, default='default', help='Method for generating click points.')
