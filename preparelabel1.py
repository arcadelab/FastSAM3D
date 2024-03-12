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
from segment_anything.modeling.imageencoder1 import ImageEncoderViT3D
from segment_anything.modeling import Sam3D, MaskDecoder3D, PromptEncoder3D
from functools import partial
from scipy.ndimage import rotate


def top(input: torch.tensor, size=384):
    t = input.sum(dim=0).sum(dim=1).sum(dim=-2)
    _, index = torch.sort(t)
    return input[:, :, :, index[:size]]


def rotates(input, angle=[10, 30], axis=(1, 0)):
    output = []
    for i in angle:
        output.append(rotate(input, angle=i, axes=axis, reshape=False))
    return output


def store_label(
    model,
    data: DataLoader,
    label_path_base="./data/train/label",
    train_path_base="./data/train/images",
    device="cuda",
):
    i = 0
    for batch_data in tqdm(data):
        image3D, _, _ = batch_data
        image = rotates(image3D)
        for j in range(len(image)):
            i += 1
            train_path = train_path_base + "/images" + str(i) + ".pt"
            label_path = label_path_base + "/label" + str(i) + ".pt"
            images = image[j].float()
            output = model(images.to(device))
            for j in range(len(output)):
                output[j] = output[j].cpu()
                output[j] = output[j].squeeze(dim=0)
            image3D = images.cpu()
            image3D = images.squeeze(dim=0)
            print(images.shape)
            torch.save(images, train_path)
            torch.save(output, label_path)
            print(f"Batch {i}: Saved image to {train_path} and label to {label_path}")


if __name__ == "__main__":
    np.random.seed(2023)
    torch.manual_seed(2023)
    all_dataset_paths = glob(join("./data/validation", "*", "*"))
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
    device = "cuda"
    print("device:", device)
    sam_model_tune = Sam3D(
        image_encoder=ImageEncoderViT3D(
            depth=12,
            embed_dim=768,
            img_size=128,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=384,
        ),
        prompt_encoder=PromptEncoder3D(
            embed_dim=384,
            image_embedding_size=(8, 8, 8),
            input_image_size=(128, 128, 128),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder3D(
            num_multimask_outputs=3,
            transformer_dim=384,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    ).to("cuda")
    model_dict = torch.load("./ckpt/sam_med3d_turbo.pth", map_location=device)
    state_dict = model_dict["model_state_dict"]
    sam_model_tune.load_state_dict(state_dict)
    store_label(sam_model_tune.image_encoder, test_dataloader)
