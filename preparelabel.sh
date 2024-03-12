python ./preparelabel.py.py \
  --data_validation_path "./data/validation" \
  --label_path_base "./data/augmentation/label" \
  --train_path_base "./data/augmentation/images" \
  --checkpoint_path "./ckpt/sam_med3d_turbo.pth" \
  --crop_size 128 128 128 \
  --batch_size 1 \
  --shuffle \
  --device "cuda"