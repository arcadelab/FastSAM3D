import torch
from tqdm import tqdm
import os
join = os.path.join
from glob import glob
from segment_anything.build_sam3D import sam_model_registry3D
from torch.utils.data import DataLoader
import torchio as tio
import torch.nn.functional as F
from utils.click_method import get_next_click3D_torch_ritm, get_next_click3D_torch_2
from utils.data_loader import Dataset_Union_ALL_Val
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--click_type', type=str, default='random') #
parser.add_argument('--multi_click', action='store_true', default=False) #
parser.add_argument('--img_size', type=int, default=128)#


args = parser.parse_args()

device = "cuda"

click_methods = {
    'default': get_next_click3D_torch_ritm,
    'ritm': get_next_click3D_torch_ritm,
    'random': get_next_click3D_torch_2,
}

device = 'cuda'
print("device:", device)
sam_model = sam_model_registry3D['vit_b_ori'](checkpoint=None).to(device)
model_dict = torch.load('./ckpt/sam_med3d_turbo.pth', map_location=device)
state_dict = model_dict['model_state_dict']
sam_model.load_state_dict(state_dict)


def get_points(self, prev_masks, gt3D):
    batch_points, batch_labels = click_methods[self.args.click_type](prev_masks, gt3D)

    points_co = torch.cat(batch_points, dim=0).to(device)
    points_la = torch.cat(batch_labels, dim=0).to(device)

    self.click_points.append(points_co)
    self.click_labels.append(points_la)

    points_multi = torch.cat(self.click_points, dim=1).to(device)
    labels_multi = torch.cat(self.click_labels, dim=1).to(device)

    if self.args.multi_click:
        points_input = points_multi
        labels_input = labels_multi
    else:
        points_input = points_co
        labels_input = points_la
    return points_input, labels_input

def interaction(self, sam_model, image_embedding, gt3D, num_clicks=10):
    return_loss = 0
    prev_masks = torch.zeros_like(gt3D).to(gt3D.device)
    low_res_masks = F.interpolate(prev_masks.float(), size=(args.img_size//4,args.img_size//4,args.img_size//4))
    random_insert = np.random.randint(2, 9)
    for num_click in range(num_clicks):
        points_input, labels_input = self.get_points(prev_masks, gt3D)

        if num_click == random_insert or num_click == num_clicks - 1:
            low_res_masks, prev_masks = self.batch_forward(sam_model, image_embedding, gt3D, low_res_masks, points=None)
        else:
            low_res_masks, prev_masks = self.batch_forward(sam_model, image_embedding, gt3D, low_res_masks, points=[points_input, labels_input])
        loss = self.seg_loss(prev_masks, gt3D)
        return_loss += loss
    return prev_masks, return_loss

def store_label(model, data: DataLoader, label_path_base = './data/test/label', train_path_base = './data/test/images', device='cuda'):
    i = 0
    for batch_data in tqdm(data):
        image3D, gt3D, _ = batch_data
        image3D = image3D.squeeze(dim=1)  # 去掉不必要的维度
        image3D = tio.ZNormalization()(image3D)  # 应用 ZNormalization
        image3D = image3D.unsqueeze(dim=1)  # 添加必要的维度

                
        image3D = image3D.to(device)
        gt3D = gt3D.to(device).type(torch.long)
        
        image_embedding = sam_model.image_encoder(image3D)
        prev_masks, loss = interaction(sam_model, image_embedding, gt3D, num_clicks=11) 
        train_path = train_path_base + '/images' + str(i) + '.pt'
        label_path = label_path_base + '/label' + str(i) + '.pt'
        i += 1
        image3D = image3D.float()
        # output = model(image3D.to(device))
        # output = output.cpu()
        # image3D = image3D.cpu()
        # image3D = image3D.squeeze(dim = 0)
        # output = output.squeeze(dim = 0)
        # torch.save(image3D, train_path)        
        torch.save(prev_masks, label_path)
        print(f"Batch {i}: Saved image to {train_path} and label to {label_path}")
if __name__ == "__main__":    
    all_dataset_paths = glob(join('./data/test', "*", "*"))
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
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        sampler=None,
        batch_size=1, 
        shuffle=True
    )
    device = 'cuda'
    print("device:", device)
    sam_model_tune = sam_model_registry3D['vit_b_ori'](checkpoint=None).to(device)
    model_dict = torch.load('./ckpt/sam_med3d_turbo.pth', map_location=device)
    state_dict = model_dict['model_state_dict']
    sam_model_tune.load_state_dict(state_dict)
    store_label(sam_model_tune.image_encoder, test_dataloader)
