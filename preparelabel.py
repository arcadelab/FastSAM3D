import torch
from tqdm import tqdm
import os

join = os.path.join
from glob import glob
from segment_anything.build_sam3D import sam_model_registry3D
from torch.utils.data import DataLoader
import torchio as tio
from utils.data_loader import Dataset_Union_ALL_Val
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Store labels for augmented data.")
parser.add_argument(
    "--data_validation_path",
    type=str,
    default="./data/validation",
    help="Path to the validation data",
)
parser.add_argument(
    "--label_path_base",
    type=str,
    default="./data/augumentation/label",
    help="Base path for saving labels",
)
parser.add_argument(
    "--train_path_base",
    type=str,
    default="./data/augumentation/images",
    help="Base path for saving training images",
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="./ckpt/sam_med3d_turbo.pth",
    help="Path to the model checkpoint",
)
parser.add_argument(
    "--crop_size",
    nargs=3,
    type=int,
    default=[128, 128, 128],
    help="Crop size for the images",
)
parser.add_argument(
    "--batch_size", type=int, default=1, help="Batch size for data loading"
)
parser.add_argument(
    "--shuffle", action="store_true", help="Whether to shuffle the dataset"
)
parser.add_argument(
    "--device", type=str, default="cuda", help="Device to run the model on"
)
args = parser.parse_args()


def save_model_weights(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")


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


if __name__ == "__main__":
    np.random.seed(2023)
    torch.manual_seed(2023)
    all_dataset_paths = glob(join(args.data_validation_path))
    all_dataset_paths = list(filter(os.path.isdir, all_dataset_paths))
    print("get", len(all_dataset_paths), "datasets")

    infer_transform = [
        tio.ToCanonical(),
        tio.CropOrPad(mask_name="label", target_shape=(128, 128, 128)),
    ]

    test_dataset = Dataset_Union_ALL_Val(
        paths=all_dataset_paths,
        mode="Val",
        data_type="Ts",
        transform=tio.Compose(infer_transform),
        threshold=0,
        split_num=1,
        split_idx=0,
        pcc=False,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, sampler=None, batch_size=1, shuffle=True
    )
    device = args.device
    sam_model_tune = sam_model_registry3D["vit_b_ori"](checkpoint=None).to(device)
    model_dict = torch.load(args.checkpoint_path, map_location=device)
    state_dict = model_dict["model_state_dict"]
    sam_model_tune.load_state_dict(state_dict)

    store_label(sam_model_tune.image_encoder, test_dataloader, args)
