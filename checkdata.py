import os
from glob import glob
import torch

def print_shapes(data_path):
    files = glob(os.path.join(data_path, '*.pt'))[:50]  # 只考虑前50个文件
    for i, file_path in enumerate(files, 1):
        data = torch.load(file_path)
        # 检查data是否为列表
        if isinstance(data, list):
            # 如果是列表，假设列表中的每个元素都是tensor
            print(f"File {i}: {file_path.split('/')[-1]} contains list of tensors with shapes: {[t.shape for t in data]}")
        else:
            # 如果不是列表，直接打印tensor的形状
            print(f"File {i}: {file_path.split('/')[-1]} Shape: {data.shape}")

# 路径定义
image_paths_ttrain = './data/ttrain/images'
label_paths_ttrain = './data/ttrain/label'
image_paths_aug = './data/augumentation/images'
label_paths_aug = './data/augumentation/label'

# 打印形状
print("Ttrain Images:")
print_shapes(image_paths_ttrain)
print("\nTtrain Labels:")
print_shapes(label_paths_ttrain)
print("\nAugmentation Images:")
print_shapes(image_paths_aug)
print("\nAugmentation Labels:")
print_shapes(label_paths_aug)
