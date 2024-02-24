
DATA_VALIDATION_PATH="data/initial_train_dataset"
LABEL_PATH_BASE="data/augumentation_train/label"
TRAIN_PATH_BASE="data/augumentation_train/images"
CHECKPOINT_PATH="ckpt/sam_med3d_turbo.pth"
CROP_SIZE=[128,128,128]
BATCH_SIZE=1
DEVICE="cuda"
# be noted that crop size 128 should modify in preparelabel directly if we want!
python preparelabel.py \
    --data_validation_path $DATA_VALIDATION_PATH \
    --label_path_base $LABEL_PATH_BASE \
    --train_path_base $TRAIN_PATH_BASE \
    --checkpoint_path $CHECKPOINT_PATH \
    --batch_size $BATCH_SIZE \
    --device $DEVICE