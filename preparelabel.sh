python ./preparelabel.py \
  --data_train_path "./data/train" \  # Used for teacher model's encoder to generate logits, 
  --label_path_base "./data/augmentation/label" \
  --train_path_base "./data/augmentation/images" \
  --checkpoint_path "./ckpt/sam_med3d_turbo.pth" \
  --crop_size 128 128 128 \
  --batch_size 1 \
  --shuffle \
  --device "cuda"
