
import numpy as np
import nibabel as nib
import os
import torch

#path = 'ckpt/sam_med3d_turbo.pth'
#p = 'work_dir1/union_train/sam_model_latest.pth'
#weight = torch.load(path)
#w = torch.load(p)
#print(weight['model_state_dict'].get('image_encoder.patch_embed.proj.weight') - w['model_state_dict'].get('patch_embed.proj.weight'))
path = 'data/augumentation_train/images'
data = 'data/augumentation_train/label'
o = os.listdir(path)
e = os.listdir(data)
a = 0
j = 0
for i in o:
    j += 1
    p = torch.load(path + '/' + i)
    print(p.shape)
print(j)
# for i in o:
#     p = torch.load(path + '/' + i)
#     for k,j in enumerate(p):
#         if torch.isnan(j).any():
#             print('find nan' + path + '/' + i + 'output' + k)
# a = 0
# for i in e:
#     p = torch.load(data + '/' + i)
#     if torch.isnan(p).any():
#         print('find nan in ' + data + '/' + i)