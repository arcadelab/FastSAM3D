import torch
from tqdm import tqdm
import os
import argparse
join = os.path.join
from glob import glob
from segment_anything.build_sam3D import sam_model_registry3D
from torch.utils.data import DataLoader
import torchio as tio
from utils.dataloader_augument import Dataset_Union_ALL_Val
import numpy as np
from segment_anything.modeling.imageencoder1 import ImageEncoderViT3D
from segment_anything.modeling import Sam3D, MaskDecoder3D, PromptEncoder3D
from functools import partial
from scipy.ndimage import rotate

parser = argparse.ArgumentParser(description='Store labels for augmented data.')
parser.add_argument('--data_validation_path', type=str, default='./data/validation', help='Path to the validation data')
parser.add_argument('--label_path_base', type=str, default='./data/augumentation/label', help='Base path for saving labels')
parser.add_argument('--train_path_base', type=str, default='./data/augumentation/images', help='Base path for saving training images')
parser.add_argument('--checkpoint_path', type=str, default='./ckpt/sam_med3d_turbo.pth', help='Path to the model checkpoint')
parser.add_argument('--crop_size', nargs=3, type=int, default=[128, 128, 128], help='Crop size for the images')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loading')
parser.add_argument('--shuffle', action='store_true', help='Whether to shuffle the dataset')
parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
args = parser.parse_args()


def store_label(model, data: DataLoader, args):
    i = 0
    for batch_data in tqdm(data):
        image3D, _, _ = batch_data
        norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
        image3D = norm_transform(image3D.squeeze(dim=1))
        image3D = image3D.unsqueeze(dim=1)
        train_path = f"{args.train_path_base}/images{i}.pt"
        label_path = f"{args.label_path_base}/label{i}.pt"
        i += 1
        image3D = image3D.float().to(args.device)
        output = model(image3D)
        for j in range(len(output)):
            output[j] = output[j].cpu().squeeze(dim=0)
        image3D = image3D.cpu().squeeze(dim=0)
        torch.save(image3D, train_path)
        torch.save(output, label_path)
        print(f"Batch {i}: Saved image to {train_path} and label to {label_path}")


augmentation_transforms = tio.Compose([
    # 随机仿射变换，包括旋转和平移
    # tio.RandomAffine(
    #     degrees=(10, 15),
    #     translate=(5, 10),
    #     scale=(0.9, 1.1),
    #     isotropic=False,
    #     center='image',
    #     default_pad_value='minimum',
    # ),
    # 随机弹性形变
    tio.RandomElasticDeformation(
        num_control_points=(5, 5, 5),  # Increased the number of control points
        max_displacement=(5, 5, 5),
        locked_borders=2,
    ),
    # 随机噪声
    tio.RandomNoise(
        mean=0,
        std=(0.01, 0.05),
        p=0.5,
    ),
    # 随机运动伪影
    tio.RandomMotion(
        degrees=10,
        translation=10,
        num_transforms=2,
        p=0.2,
    ),
    # 随机偏置场模拟 MRI 中的强度不均匀性
    tio.RandomBiasField(
        coefficients=0.5,
        p=0.3,
    ),
    # 随机鬼影，模拟 MRI 中的重复模式伪影
    tio.RandomGhosting(
        intensity=(0.5, 1),
        p=0.2,
    ),
    # 随机模糊
    tio.RandomBlur(
        std=(0.5, 1.5),
        p=0.25,
    ),
    # 随机翻转
    tio.RandomFlip(
        axes=(0, 1, 2),
        flip_probability=0.5,
    ),
])

if __name__ == "__main__":  
    np.random.seed(2023)
    torch.manual_seed(2023)  
    all_dataset_paths = glob(join(args.data_validation_path, "*", "*"))
    all_dataset_paths = list(filter(os.path.isdir, all_dataset_paths))
    print("get", len(all_dataset_paths), "datasets")

    infer_transform = [
        tio.ToCanonical(),
        tio.CropOrPad(mask_name='label', target_shape=(128,128,128)),
    ]
    
    test_dataset = Dataset_Union_ALL_Val(
        paths=all_dataset_paths, 
        mode="Val", 
        data_type='Ts', 
        transform=tio.Compose(infer_transform),
        threshold=0,
        split_num=1,
        split_idx=0,
        pcc=False,
        augmentations=augmentation_transforms
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        sampler=None,
        batch_size=1, 
        shuffle=True
    )
    device = args.device
    sam_model_tune = sam_model_registry3D['vit_b_ori'](checkpoint=None).to(device)
    model_dict = torch.load(args.checkpoint_path, map_location=device)
    state_dict = model_dict['model_state_dict']
    sam_model_tune.load_state_dict(state_dict)

    store_label(sam_model_tune.image_encoder, test_dataloader, args)