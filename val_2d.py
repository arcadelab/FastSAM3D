import os
join = os.path.join
import numpy as np
from glob import glob
import torch
from segment_anything.build_sam3D import sam_model_registry3D
from segment_anything.utils.transforms3D import ResizeLongestSide3D
from originalsam.segment_anything.build_sam import sam_model_registry
from tqdm import tqdm
import argparse
import SimpleITK as sitk
import torch.nn.functional as F
from torch.utils.data import DataLoader
import SimpleITK as sitk
import torchio as tio
import numpy as np
from collections import OrderedDict, defaultdict
import json
import pickle
from utils.click_method import get_next_click3D_torch_ritm, get_next_click3D_torch_2
from utils.data_loader import Dataset_Union_ALL_Val
import time
from thop import profile
from torchinfo import summary
from monai.losses import DiceCELoss

parser = argparse.ArgumentParser()
parser.add_argument('-tdp', '--test_data_path', type=str, default='/content/drive/MyDrive/paper_visual_results/totalseg')
parser.add_argument('-vp', '--vis_path', type=str, default='/content/drive/MyDrive/paper_visual_results/totalseg0441/med_sam_2d')
parser.add_argument('-cp', '--checkpoint_path', type=str, default='/content/drive/MyDrive/lighting_sam_3d/ckpt/sam_vit_b_01ec64.pth')
parser.add_argument('-sn', '--save_name', type=str, default='/content/drive/MyDrive/paper_visual_results/totalseg2d.py')

parser.add_argument('--image_size', type=int, default=1024)  #
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('-mt', '--model_type', type=str, default='vit_b')
parser.add_argument('-nc', '--num_clicks', type=int, default=10)
parser.add_argument('-pm', '--point_method', type=str, default='default')
parser.add_argument('-dt', '--data_type', type=str, default='Tr')
parser.add_argument("--encoder_adapter", type=bool, default=False, help="use adapter")
parser.add_argument('--threshold', type=int, default=0)
parser.add_argument('--dim', type=int, default=2)
parser.add_argument('--split_idx', type=int, default=0)
parser.add_argument('--split_num', type=int, default=1)
parser.add_argument('--ft2d', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=2023)

args = parser.parse_args()

SEED = args.seed
print("set seed as", SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.init()

click_methods = {
    'default': get_next_click3D_torch_ritm,
    'ritm': get_next_click3D_torch_ritm,
    'random': get_next_click3D_torch_2,
}

def compute_iou(pred_mask, gt_semantic_seg):
    in_mask = np.logical_and(gt_semantic_seg, pred_mask)
    out_mask = np.logical_or(gt_semantic_seg, pred_mask)
    iou = np.sum(in_mask) / np.sum(out_mask)
    return iou

def batch_forward(sam_model, image_embedding, gt3D, low_res_masks, points=None, device='cuda'):
    #device = "cuda"
    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
        points=points,
        boxes=None,
        masks=low_res_masks,
    )
    low_res_masks, iou_predictions = sam_model.mask_decoder(
        image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
        image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )
    prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)
    return low_res_masks, prev_masks

    
def get_points(click_type, prev_masks, gt3D, click_points, click_labels, device):
    batch_points, batch_labels = click_methods[click_type](prev_masks, gt3D)

    points_co = torch.cat(batch_points, dim=0).to(device)
    points_la = torch.cat(batch_labels, dim=0).to(device)

    click_points.append(points_co)
    click_labels.append(points_la)

    points_multi = torch.cat(click_points, dim=1).to(device)
    labels_multi = torch.cat(click_labels, dim=1).to(device)
    
    points_input = points_multi
    labels_input = labels_multi

    return points_input, labels_input, click_points, click_labels

def interaction(sam_model, image_embedding, gt3D, num_clicks):
    click_type = 'random'
    seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    img_size = 128
    device = "cuda"
    return_loss = 0
    prev_masks = torch.zeros_like(gt3D).to(gt3D.device)
    low_res_masks = F.interpolate(prev_masks.float(), size=(img_size//4,img_size//4,img_size//4))

    click_points = []
    click_labels = []

    for num_click in range(num_clicks):
        random_insert = np.random.randint(2, 9)
        points_input, labels_input, click_points, click_labels = get_points(click_type, prev_masks, gt3D, click_points, click_labels, device)

        if num_click == random_insert or num_click == num_clicks - 1:
            low_res_masks, prev_masks = batch_forward(sam_model, image_embedding, gt3D, low_res_masks, points=None)
        else:
            low_res_masks, prev_masks = batch_forward(sam_model, image_embedding, gt3D, low_res_masks, points=[points_input, labels_input])
        loss = seg_loss(prev_masks, gt3D)
        return_loss += loss
    return prev_masks, return_loss


def compute_dice(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum

def postprocess_masks(low_res_masks, image_size, original_size):
    ori_h, ori_w = original_size
    masks = F.interpolate(
        low_res_masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
        )
    if args.ft2d and ori_h < image_size and ori_w < image_size:
        top = (image_size - ori_h) // 2
        left = (image_size - ori_w) // 2
        masks = masks[..., top : ori_h + top, left : ori_w + left]
        pad = (top, left)
    else:
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        pad = None 
    return masks, pad

def sam_decoder_inference(target_size, points_coords, points_labels, model, image_embeddings, mask_inputs=None, multimask = False):
    
    with torch.no_grad():
        sparse_embeddings, dense_embeddings= model.prompt_encoder(
            points=(points_coords.to(model.device), points_labels.to(model.device)),
            boxes=None,
            masks=mask_inputs,
        )
        torch.cuda.reset_max_memory_allocated(model.device)
        low_res_masks, iou_predictions, t = model.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output = multimask,
        )
        # print(low_res_masks.shape)
        memory = torch.cuda.max_memory_allocated(model.device)
        # print(memory)
        # print(profile(model.mask_decoder,(image_embeddings,model.prompt_encoder.get_dense_pe(),sparse_embeddings,dense_embeddings,True))[0])
    if multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)
    masks = F.interpolate(low_res_masks, (target_size, target_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions, t

def repixel_value(arr, is_seg=False):
    if not is_seg:
        min_val = arr.min()
        max_val = arr.max()
        new_arr = (arr - min_val) / (max_val - min_val + 1e-10) * 255.
    return new_arr

def random_point_sampling(mask, get_point = 1):
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    fg_coords = np.argwhere(mask == 1)[:,::-1]
    bg_coords = np.argwhere(mask == 0)[:,::-1]

    fg_size = len(fg_coords)
    bg_size = len(bg_coords)

    if get_point == 1:
        if fg_size > 0:
            index = np.random.randint(fg_size)
            fg_coord = fg_coords[index]
            label = 1
        else:
            index = np.random.randint(bg_size)
            fg_coord = bg_coords[index]
            label = 0
        return torch.as_tensor([fg_coord.tolist()], dtype=torch.float), torch.as_tensor([label], dtype=torch.int)
    else:
        num_fg = get_point // 2
        num_bg = get_point - num_fg
        fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
        bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
        fg_coords = fg_coords[fg_indices]
        bg_coords = bg_coords[bg_indices]
        coords = np.concatenate([fg_coords, bg_coords], axis=0)
        labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)
        indices = np.random.permutation(get_point)
        coords, labels = torch.as_tensor(coords[indices], dtype=torch.float), torch.as_tensor(labels[indices], dtype=torch.int)
        return coords, labels


def finetune_model_predict2D(img3D, gt3D, sam_model_tune, target_size=256, click_method='random', device='cuda', num_clicks=1, prev_masks=None):
    
    pred_list = []
    iou_list = []
    dice_list = []
    slice_mask_list = defaultdict(list)

    memory = []
    img3D = torch.repeat_interleave(img3D, repeats=3, dim=1) # 1 channel -> 3 channel (align to RG)
    click_points = []
    click_labels = []
    dt = 0
    t = 0
    for slice_idx in tqdm(range(img3D.size(-1)), desc="transverse slices", leave=False):
        img2D, gt2D = repixel_value(img3D[..., slice_idx]), gt3D[..., slice_idx]

        if (gt2D==0).all():
            empty_result = torch.zeros(list(gt3D.size()[:-1])+[1]).to(device)
            for iter in range(num_clicks):
                slice_mask_list[iter].append(empty_result)
            continue

        img2D = F.interpolate(img2D, (target_size, target_size), mode="bilinear", align_corners=False)
        gt2D = F.interpolate(gt2D.float(), (target_size, target_size), mode="nearest").int()
        
        img2D, gt2D = img2D.to(device), gt2D.to(device)
        img2D = (img2D - img2D.mean()) / img2D.std()
        torch.cuda.reset_max_memory_allocated(device)
        with torch.no_grad():
            image_embeddings,times = sam_model_tune.image_encoder(img2D.float())
        t += times
        # print(profile(sam_model_tune.image_encoder,(img2D.float().to(device),))[0])
        memory.append(torch.cuda.max_memory_allocated(device))
        points_co, points_la = torch.zeros(1,0,2).to(device), torch.zeros(1,0).to(device)
        low_res_masks = None
        gt_semantic_seg = gt2D[0, 0].to(device)
        true_masks = (gt_semantic_seg > 0)
        decodetimes = 0
        for iter in range(num_clicks):
            if(low_res_masks==None):
                pred_masks = torch.zeros_like(true_masks).to(device)

            else:
                pred_masks = (prev_masks[0, 0] > 0.0).to(device)
            fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
            fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)
            mask_to_sample = torch.logical_or(fn_masks, fp_masks)
            new_points_co, _ = random_point_sampling(mask_to_sample.cpu(), get_point=1)
            new_points_la = torch.Tensor([1]).to(torch.int64) if(true_masks[new_points_co[0,1].int(), new_points_co[0,0].int()]) else torch.Tensor([0]).to(torch.int64)
            new_points_co, new_points_la = new_points_co[None].to(device), new_points_la[None].to(device)
            points_co = torch.cat([points_co, new_points_co],dim=1)
            points_la = torch.cat([points_la, new_points_la],dim=1)
            prev_masks, low_res_masks, iou_predictions,decodetime = sam_decoder_inference(
                target_size, points_co, points_la, sam_model_tune, image_embeddings, 
                mask_inputs = low_res_masks,multimask = True)
            click_points.append(new_points_co)
            click_labels.append(new_points_la)
            decodetimes += decodetime
            slice_mask, _ = postprocess_masks(low_res_masks, target_size, (gt3D.size(2), gt3D.size(3)))
            slice_mask_list[iter].append(slice_mask[..., None]) # append (B, C, H, W, 1)
        dt += decodetimes / num_clicks
    for iter in range(num_clicks):
        medsam_seg = torch.cat(slice_mask_list[iter], dim=-1).cpu().numpy().squeeze()
        medsam_seg = medsam_seg > sam_model_tune.mask_threshold
        medsam_seg = medsam_seg.astype(np.uint8)

        pred_list.append(medsam_seg) 
        iou_list.append(round(compute_iou(medsam_seg, gt3D[0][0].detach().cpu().numpy()), 4))
        dice_list.append(round(compute_dice(gt3D[0][0].detach().cpu().numpy().astype(np.uint8), medsam_seg), 4))
    print(t)
    print(dt)
    print(np.average(memory))
    
    return pred_list, click_points, click_labels, iou_list, dice_list,t,dt


def finetune_model_predict3D(img3D, gt3D, sam_model_tune, device='cuda', click_method='random', num_clicks=10, prev_masks=None):
    torch.cuda.reset_max_memory_allocated(device)
    encoder_time = 0 #
    decoder_time = []
    img3D = norm_transform(img3D.squeeze(dim=1)) # (N, C, W, H, D)
    img3D = img3D.unsqueeze(dim=1)
    click_points = []
    click_labels = []
    FLOPS = np.zeros(num_clicks)

    pred_list = []
    iou_list = []
    dice_list = []
    if prev_masks is None:
        prev_masks = torch.zeros_like(gt3D).to(device)
    low_res_masks = F.interpolate(prev_masks.float(), size=(args.crop_size//4,args.crop_size//4,args.crop_size//4))
    start_time = time.time() 
    
    with torch.no_grad():
        image_embedding,times = sam_model_tune.image_encoder(img3D.to(device)) # (1, 384, 16, 16, 16)
    image_embedding = image_embedding[-1]
    memory_before = torch.cuda.max_memory_allocated(device) 
    # print(memory_before) #
    torch.cuda.reset_max_memory_allocated(device)
    for num_click in range(num_clicks):
        #
        with torch.no_grad():
            if(num_click>1):
                click_method = "random"
            batch_points, batch_labels = click_methods[click_method](prev_masks.to(device), gt3D.to(device))
            
            points_co = torch.cat(batch_points, dim=0).to(device)  
            points_la = torch.cat(batch_labels, dim=0).to(device)  

            click_points.append(points_co)
            click_labels.append(points_la)

            points_input = points_co
            labels_input = points_la

            sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
                points=[points_input, labels_input],
                boxes=None,  #
                masks=low_res_masks.to(device),
            )
            FLOPS[num_click] += profile(sam_model_tune.prompt_encoder,([points_input, labels_input],None,low_res_masks.to(device),))[0]
            start_time = time.time()
            low_res_masks, _ = sam_model_tune.mask_decoder(
                image_embeddings=image_embedding.to(device), # (B, 384, 64, 64, 64)
                image_pe=sam_model_tune.prompt_encoder.get_dense_pe(), # (1, 384, 64, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 384)
                dense_prompt_embeddings=dense_embeddings, # (B, 384, 64, 64, 64)
                multimask_output=False,
                )
            FLOPS[num_click] += profile(sam_model_tune.mask_decoder,(image_embedding,sam_model_tune.prompt_encoder.get_dense_pe(),sparse_embeddings,dense_embeddings,False,))[0]
            # end_time = time.time()
            # decoder_time.append(end_time - start_time)
            memory_decoder = torch.cuda.max_memory_allocated(device) #
            # print(memory_decoder)
            prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)

            medsam_seg_prob = torch.sigmoid(prev_masks)  # (B, 1, 64, 64, 64)
            # convert prob to mask
            medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
            medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
            pred_list.append(medsam_seg)

            iou_list.append(round(compute_iou(medsam_seg, gt3D[0][0].detach().cpu().numpy()), 4))
            dice_list.append(round(compute_dice(gt3D[0][0].detach().cpu().numpy().astype(np.uint8), medsam_seg), 4))
    # print(np.average(FLOPS))
    return pred_list, click_points, click_labels, iou_list, dice_list,encoder_time,decoder_time,memory_before, memory_decoder, FLOPS

if __name__ == "__main__":   
    st = time.time() 
    all_dataset_paths = glob(join(args.test_data_path))
    all_dataset_paths = list(filter(os.path.isdir, all_dataset_paths))
    print("get", len(all_dataset_paths), "datasets")
    

    infer_transform = [
        tio.ToCanonical(),
        tio.CropOrPad(mask_name='label', target_shape=(args.crop_size,args.crop_size,args.crop_size)),
    ]
    
    test_dataset = Dataset_Union_ALL_Val(
        paths=all_dataset_paths, 
        mode="Val", 
        data_type=args.data_type, 
        transform=tio.Compose(infer_transform),
        threshold=0,
        split_num=args.split_num,
        split_idx=args.split_idx,
        pcc=False,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        sampler=None,
        batch_size=1, 
        shuffle=True
    )

    checkpoint_path = args.checkpoint_path

    device = args.device
    print("device:", device)
    

    
    if(args.dim==3):
        sam_model_tune = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
        if checkpoint_path is not None:
            model_dict = torch.load(checkpoint_path, map_location=device)
            state_dict = model_dict['model_state_dict']
            sam_model_tune.load_state_dict(state_dict)
    elif(args.dim==2):
        args.sam_checkpoint = args.checkpoint_path
        sam_model_tune = sam_model_registry[args.model_type](args.checkpoint_path).to(device)


    sam_trans = ResizeLongestSide3D(sam_model_tune.image_encoder.img_size)
    all_iou_list = []
    all_dice_list = []  
    

    out_dice = dict()
    out_dice_all = OrderedDict()
    encoder_times = []
    decoder_times = []
    average_decoder_times = []
    memory_befores = []
    memory_decoders=[]
    FLOPSS = []
    w = []
    for batch_data in tqdm(test_dataloader):
        for i in range(0,1):
            image3D, gt3D, img_name = batch_data
            image3D = image3D.float()
            image3D=image3D.to(device)
            sz = image3D.size()
            if(sz[2]<args.crop_size or sz[3]<args.crop_size or sz[4]<args.crop_size):
                print("[ERROR] wrong size", sz, "for", img_name)
            modality = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(img_name[0]))))
            dataset = os.path.basename(os.path.dirname(os.path.dirname(img_name[0])))
            vis_root = os.path.join(os.path.dirname(__file__), args.vis_path, modality, dataset)
            click_suffix = f"_pred{args.num_clicks - 1}.nii.gz"
            pred_path = os.path.join(vis_root, os.path.basename(img_name[0]).replace(".nii.gz", click_suffix))
            
            
            sam_model = sam_model_registry3D['vit_b_ori'](checkpoint=None).to(device)
            model_dict = torch.load('./ckpt/sam_med3d_turbo.pth', map_location=device)
            state_dict = model_dict['model_state_dict']
            sam_model.load_state_dict(state_dict)
            image_embedding = sam_model.image_encoder(image3D)
            start_time = time.time()
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"self.interaction excution time:{elapsed_time} seconds")
            if(1 == 0):
                iou_list, dice_list = [], []
                for iter in range(args.num_clicks):
                    curr_pred_path = os.path.join(vis_root, os.path.basename(img_name[0]).replace(".nii.gz", f"_pred{iter}.nii.gz"))
                    medsam_seg = sitk.GetArrayFromImage(sitk.ReadImage(curr_pred_path))
                    iou_list.append(round(compute_iou(medsam_seg, gt3D[0][0].detach().cpu().numpy()), 4))
                    dice_list.append(round(compute_dice(gt3D[0][0].detach().cpu().numpy().astype(np.uint8), medsam_seg), 4))
            else:
                norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
                if(args.dim==3):
                    seg_mask_list, points, labels, iou_list, dice_list,t,decoder_time,memory_before, memory_decoder, FLOPS = finetune_model_predict3D(
                        image3D, gt3D, sam_model_tune, device=device, 
                        click_method=args.point_method, num_clicks=args.num_clicks, 
                        prev_masks=None)
                elif(args.dim==2):
                    seg_mask_list, points, labels, iou_list, dice_list,t,decoder_time = finetune_model_predict2D(
                        image3D, gt3D, sam_model_tune, device=device, target_size=args.image_size,
                        click_method=args.point_method, num_clicks=args.num_clicks, 
                        prev_masks=None)
                os.makedirs(vis_root, exist_ok=True)
                points = [p.cpu().numpy() for p in points]
                labels = [l.cpu().numpy() for l in labels]
                pt_info = dict(points=points, labels=labels)
                print("save to", os.path.join(vis_root, os.path.basename(img_name[0]).replace(".nii.gz", "_pred.nii.gz")))
                pt_path=os.path.join(vis_root, os.path.basename(img_name[0]).replace(".nii.gz", "_pt.pkl"))
                pickle.dump(pt_info, open(pt_path, "wb"))
                for idx, pred3D in enumerate(seg_mask_list):
                    out = sitk.GetImageFromArray(pred3D)
                    sitk.WriteImage(out, os.path.join(vis_root, os.path.basename(img_name[0]).replace(".nii.gz", f"_pred{idx}.nii.gz")))
            per_iou = max(iou_list)
            all_iou_list.append(per_iou)
            all_dice_list.append(max(dice_list))
            print(dice_list)
            out_dice[img_name] = max(dice_list)
            cur_dice_dict = OrderedDict()
            encoder_times.append(t)
            decoder_times.append(decoder_time)
            for i, dice in enumerate(dice_list):
                cur_dice_dict[f'{i}'] = dice
            out_dice_all[img_name[0]] = cur_dice_dict
    print('Mean IoU : ', sum(all_iou_list)/len(all_iou_list))
    print('Mean Dice: ', sum(all_dice_list)/len(all_dice_list))

    final_dice_dict = OrderedDict()
    for k, v in out_dice_all.items():
        organ = k.split('/')[-4]
        final_dice_dict[organ] = OrderedDict()
    for k, v in out_dice_all.items():
        organ = k.split('/')[-4]
        final_dice_dict[organ][k] = v

    if(args.split_num>1):
        args.save_name = args.save_name.replace('.py', f'_s{args.split_num}i{args.split_idx}.py')

    print("Save to", args.save_name)
    with open(args.save_name, 'w') as f:
        f.writelines(f'# mean dice: \t{np.mean(all_dice_list)}\n')
        f.writelines('dice_Ts = {')
        for k, v in out_dice.items():
            f.writelines(f'\'{str(k[0])}\': {v},\n')
    with open(args.save_name.replace('.py', '.json'), 'w') as f:
        json.dump(final_dice_dict, f, indent=4)

    print(np.mean(encoder_times))
    print(np.mean(decoder_times))
    print("Done")
    eo=time.time()-st
    print(eo)