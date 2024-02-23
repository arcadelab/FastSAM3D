# -*- encoding: utf-8 -*-
'''
@File    :   prepare_data_from_nnUNet.py
@Time    :   2023/12/10 23:07:39
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   pre-process nnUNet-style dataset into SAM-Med3D-style
'''

import os.path as osp
import os
import json
import shutil
import nibabel as nib
from tqdm import tqdm
import torchio as tio

def resample_nii(input_path: str, output_path: str, target_spacing: tuple = (1.5, 1.5, 1.5), n=None, reference_image=None):
    """
    Resample a nii.gz file to a specified spacing using torchio.

    Parameters:
    - input_path: Path to the input .nii.gz file.
    - output_path: Path to save the resampled .nii.gz file.
    - target_spacing: Desired spacing for resampling. Default is (1.5, 1.5, 1.5).
    """
    
    # Load the nii.gz file using torchio
    subject = tio.Subject(
        img=tio.ScalarImage(input_path)
    )

    # Resample the image to the target spacing
    resampler = tio.Resample(target=target_spacing)
    resampled_subject = resampler(subject)
    if(n!=None):
        image = resampled_subject.img
        tensor_data = image.data
        if(isinstance(n, int)):
            n = [n]
        for ni in n:
            tensor_data[tensor_data == ni] = -1
        tensor_data[tensor_data != -1] = 0
        tensor_data[tensor_data != 0] = 1
        save_image = tio.ScalarImage(tensor=tensor_data, affine=image.affine)
        reference_size = reference_image.shape[1:]  # omitting the channel dimension
        # padding_amount = [ref_dim - in_dim for ref_dim, in_dim in zip(reference_size, save_image.shape[1:])] + [0,0,0]
        # padding_transform = tio.Pad(padding_amount)
        cropper_or_padder = tio.CropOrPad(reference_size)
        save_image = cropper_or_padder(save_image)
    else:
        save_image = resampled_subject.img
    
    # Save the resampled image to the specified output path
    save_image.save(output_path)

# train_images_path = 'data/initial_test_dataset/amos/imagesTr'
# train_label_path = 'data/initial_test_dataset/amos/labelsTr'
# target_train_images_path = 'data/prepross_test_dataset/amos/imagesTr'
# target_train_label_path = 'data/prepross_test_dataset/amos/labelsTr'
# idx = 0
# for i in os.listdir(train_images_path):
#     print(i)
#     target_gt_path = target_train_label_path + '/' + i
#     target_img_path = target_train_images_path + '/' + i
#     path_images = train_images_path + '/' + i
#     path_labels = train_label_path + '/' + i
#     img = path_images
#     gt_img = nib.load(path_labels)    
#     spacing = tuple(gt_img.header['pixdim'][1:4])
#     # Calculate the product of the spacing values
#     spacing_voxel = spacing[0] * spacing[1] * spacing[2]
#     gt_arr = gt_img.get_fdata()
#     gt_arr[gt_arr != idx] = 0
#     gt_arr[gt_arr != 0] = 1
#     volume = gt_arr.sum()*spacing_voxel
#     # print("volume:", volume)
#     if(volume<10): 
#         continue
#     # if(not osp.exists(target_gt_path)):
#     reference_image = tio.ScalarImage(img)
#     resample_nii(path_labels, target_gt_path, n=idx, reference_image=reference_image)
#     shutil.copy(img, target_img_path)
#     new_gt_img = nib.Nifti1Image(gt_arr, gt_img.affine, gt_img.header)
#     new_gt_img.to_filename(target_gt_path)
#     idx += 1
dataset_list = ['amos']
dataset_root = 'data/initial_test_dataset'
target_dir = ''
for dataset in dataset_list:
    dataset_dir = osp.join(dataset_root, dataset)
    meta_info = json.load(open(osp.join(dataset_dir, "dataset.json")))

    #print(meta_info['name'], meta_info['modality'])
    num_classes = len(meta_info["labels"])-1
    #print("num_classes:", num_classes, meta_info["labels"])
    resample_dir = osp.join(dataset_dir, "imagesTr_1.5") 
    os.makedirs(resample_dir, exist_ok=True)
    print(len(meta_info["training"]), "is found")
    for idx, cls_name in meta_info["labels"].items():
        cls_name = cls_name.replace(" ", "_")
        idx = int(idx)
        dataset_name = dataset
        target_cls_dir = osp.join(target_dir, cls_name, dataset_name)
        target_img_dir = osp.join(target_cls_dir, "imagesTr")
        target_gt_dir = osp.join(target_cls_dir, "labelsTr")
        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_gt_dir, exist_ok=True)
        for item in tqdm(meta_info["training"], desc=f"{dataset_name}-{cls_name}"):
            img, gt = item["image"], item["label"]
            img = osp.join(dataset_dir, img.replace(".nii.gz", "_0000.nii.gz"))
            gt = osp.join(dataset_dir, gt)
            resample_img = osp.join(resample_dir, osp.basename(img))
            if(not osp.exists(resample_img)):
                resample_nii(img, resample_img)
            img = resample_img

            target_img_path = osp.join(target_img_dir, osp.basename(img).replace("_0000.nii.gz", ".nii.gz"))
            # print(f"{img}->{target_img}")
            target_gt_path = osp.join(target_gt_dir, osp.basename(gt).replace("_0000.nii.gz", ".nii.gz"))
            # print(f"{gt}->{target_gt_path}")

            gt_img = nib.load(gt)    
            spacing = tuple(gt_img.header['pixdim'][1:4])
            # Calculate the product of the spacing values
            spacing_voxel = spacing[0] * spacing[1] * spacing[2]
            gt_arr = gt_img.get_fdata()
            gt_arr[gt_arr != idx] = 0
            gt_arr[gt_arr != 0] = 1
            volume = gt_arr.sum()*spacing_voxel
            # print("volume:", volume)
            if(volume<10): 
                print("skip", target_img_path)
                continue
            # if(not osp.exists(target_gt_path)):
            reference_image = tio.ScalarImage(img)
            if(meta_info['name']=="kits23" and idx==1):
                resample_nii(gt, target_gt_path, n=[1,2,3], reference_image=reference_image)
            else:
                resample_nii(gt, target_gt_path, n=idx, reference_image=reference_image)
            shutil.copy(img, target_img_path)
            new_gt_img = nib.Nifti1Image(gt_arr, gt_img.affine, gt_img.header)
            new_gt_img.to_filename(target_gt_path)



