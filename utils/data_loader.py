from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchio as tio
import torch
import numpy as np
import os
import torch
import SimpleITK as sitk
from prefetch_generator import BackgroundGenerator


class Dataset_Union_ALL(Dataset): 
    def __init__(self, paths, mode='train', data_type='Tr', image_size=128, 
                 transform=None, threshold=500,
                 split_num=1, split_idx=0, pcc=False):
        self.paths = paths
        self.data_type = data_type
        self.split_num=split_num
        self.split_idx=split_idx

        self._set_file_paths(self.paths)
        self.image_size = image_size
        self.transform = transform
        self.threshold = threshold
        self.mode = mode
        self.pcc = pcc
    
    def __len__(self):
        print(f"Dataset size: {len(self.label_paths)}")
        return len(self.label_paths)

    def __getitem__(self, index):
        sitk_image = sitk.ReadImage(self.image_paths[index])
        sitk_label = sitk.ReadImage(self.label_paths[index])

        if sitk_image.GetOrigin() != sitk_label.GetOrigin():
            sitk_image.SetOrigin(sitk_label.GetOrigin())
        if sitk_image.GetDirection() != sitk_label.GetDirection():
            sitk_image.SetDirection(sitk_label.GetDirection())

        subject = tio.Subject(
            image = tio.ScalarImage.from_sitk(sitk_image),
            label = tio.LabelMap.from_sitk(sitk_label),
        )
        image_shape = subject.image.data.shape
        print("Image shape:", image_shape)
        # if '/ct_' in self.image_paths[index]:
        # subject = tio.Clamp(-1000,1000)(subject)
        if self.transform:
            try:
                subject = self.transform(subject)
            except:
                print(self.image_paths[index])
        
              
        # save_dir = "/content/drive/MyDrive/paper_visual_results/brats00412"  # 指定保存目录
        # os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

        # # 转换PyTorch张量为NumPy数组，并保存图像
        # image_np = subject.image.data.squeeze().numpy().astype(np.float32)  # 假设图像是单通道的
        # image_sitk = sitk.GetImageFromArray(image_np)
        # image_filename = os.path.basename(self.image_paths[index]).replace('.nii.gz', '_processed.nii.gz')
        # image_filepath = os.path.join(save_dir, image_filename)
        # sitk.WriteImage(image_sitk, image_filepath)

        if(self.pcc):
            print("using pcc setting")
            # crop from random click point
            random_index = torch.argwhere(subject.label.data == 1)
            if(len(random_index)>=1):
                random_index = random_index[np.random.randint(0, len(random_index))]
                # print(random_index)
                crop_mask = torch.zeros_like(subject.label.data)
                # print(crop_mask.shape)
                crop_mask[random_index[0]][random_index[1]][random_index[2]][random_index[3]] = 1
                subject.add_image(tio.LabelMap(tensor=crop_mask,
                                                affine=subject.label.affine),
                                    image_name="crop_mask")
                subject = tio.CropOrPad(mask_name='crop_mask', 
                                        target_shape=(self.image_size,self.image_size,self.image_size))(subject)

        if subject.label.data.sum() <= self.threshold:
            return self.__getitem__(np.random.randint(self.__len__()))
        if self.mode == "train" and self.data_type == 'Tr':
            return subject.image.data.clone().detach(), subject.label.data.clone().detach()
        else:
            return subject.image.data.clone().detach(), subject.label.data.clone().detach(), self.image_paths[index]   
 
    def _set_file_paths(self, paths):
        print(f"Given paths: {paths}")
        self.image_paths = []
        self.label_paths = []

        # if ${path}/labelsTr exists, search all .nii.gz
        for path in paths:
            d = os.path.join(path, f'images{self.data_type}')
            if os.path.exists(d):
                for name in os.listdir(d):
                    base = os.path.basename(name).split('.nii.gz')[0]
                    label_path = os.path.join(path, f'labels{self.data_type}', f'{base}.nii.gz')
                    self.image_paths.append(os.path.join(path, f'images{self.data_type}', f'{base}.nii.gz'))
                    self.label_paths.append(label_path)
                    print(f"Found {len(self.image_paths)} image(s) and {len(self.label_paths)} label(s)") 

class Dataset_Union_ALL_Val(Dataset_Union_ALL):
    def _set_file_paths(self, paths):
        self.image_paths = []
        self.label_paths = []

        # if ${path}/labelsTr exists, search all .nii.gz
        for path in paths:
            for dt in ["Tr", "Val", "Ts"]:
                d = os.path.join(path, f'images{dt}')
                # print(path)
                # print(d)
                if os.path.exists(d):
                    for name in os.listdir(d):
                        
                        base = os.path.basename(name).split('.nii.gz')[0]
                        label_path = os.path.join(path, f'labels{dt}', f'{base}.nii.gz') 
                        self.image_paths.append(os.path.join(path, f'images{dt}', f'{base}.nii.gz'))
                        self.label_paths.append(label_path)
        self.image_paths = self.image_paths[self.split_idx::self.split_num]
        self.label_paths = self.label_paths[self.split_idx::self.split_num]

class distillation_data(Dataset):
    def __init__(self,image_path,label_path):
        self.image_path = image_path
        self.label_path = label_path
        self._set_file_paths()
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        image = torch.load(self.image_paths[index])
        label = torch.load(self.label_paths[index])
        for i in range(0,len(label)):
            label[i] = label[i].detach()
        return image.detach(), label

    def _set_file_paths(self):
        self.image_paths = []
        self.label_paths = []
        ima = os.listdir(self.image_path)
        self.length = len(ima)
        for i in ima:
            self.image_paths.append(self.image_path + '/' + i)
            self.label_paths.append(self.label_path + '/' + i.replace('images', 'label'))
        

class Union_Dataloader(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Test_Single(Dataset): 
    def __init__(self, paths, image_size=128, transform=None, threshold=500):
        self.paths = paths

        self._set_file_paths(self.paths)
        self.image_size = image_size
        self.transform = transform
        self.threshold = threshold
    
    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):

        sitk_image = sitk.ReadImage(self.image_paths[index])
        sitk_label = sitk.ReadImage(self.label_paths[index])

        if sitk_image.GetOrigin() != sitk_label.GetOrigin():
            sitk_image.SetOrigin(sitk_label.GetOrigin())
        if sitk_image.GetDirection() != sitk_label.GetDirection():
            sitk_image.SetDirection(sitk_label.GetDirection())

        subject = tio.Subject(
            image = tio.ScalarImage.from_sitk(sitk_image),
            label = tio.LabelMap.from_sitk(sitk_label),
        )

        if '/ct_' in self.image_paths[index]:
            subject = tio.Clamp(-1000,1000)(subject)

        if self.transform:
            try:
                subject = self.transform(subject)
            except:
                print(self.image_paths[index])


        if subject.label.data.sum() <= self.threshold:
            return self.__getitem__(np.random.randint(self.__len__()))
        

        return subject.image.data.clone().detach(), subject.label.data.clone().detach(), self.image_paths[index]
    
    def _set_file_paths(self, paths):
        self.image_paths = []
        self.label_paths = []

        self.image_paths.append(paths)
        self.label_paths.append(paths.replace('images', 'labels'))



if __name__ == "__main__":
    test_dataset = Dataset_Union_ALL(
        paths=['/cpfs01/shared/gmai/medical_preprocessed/3d/iseg/ori_totalseg_two_class/liver/Totalsegmentator_dataset_ct/',], 
        data_type='Ts', 
        transform=tio.Compose([
            tio.ToCanonical(),
            tio.CropOrPad(mask_name='label', target_shape=(128,128,128)),
        ]), 
        threshold=0)

    test_dataloader = Union_Dataloader(
        dataset=test_dataset,
        sampler=None,
        batch_size=1, 
        shuffle=True
    )
    for i,j,n in test_dataloader:
        # print(i.shape)
        # print(j.shape)
        # print(n)
        continue
