import torch
from torch.cuda import amp
import torchio as tio
import random 
import logging
from torch.backends import cudnn
import argparse
import numpy as np
import os
join = os.path.join
import torch.distributed as dist
from utils.data_paths import img_datas
from utils.data_loader import distillation_data, Union_Dataloader
from torchinfo import summary 
import matplotlib.pyplot as plt
from contextlib import nullcontext
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import numpy as np
import random 
import datetime
import logging
import matplotlib.pyplot as plt
import os
join = os.path.join
from tqdm import tqdm
from torch.backends import cudnn
import torch.distributed as dist
import torch.nn.functional as F
import torchio as tio
from torch.utils.data.distributed import DistributedSampler
import argparse
from torch.cuda import amp
import torch.multiprocessing as mp
from contextlib import nullcontext
from utils.data_paths import img_datas
from segment_anything.modeling.image_encoder3D import ImageEncoderViT3D
#from segment_anything.modeling.swin_flash import SwinTransformer
from monai.utils import ensure_tuple_rep
from functools import partial
from contextlib import nullcontext
from segment_anything.build_sam3D import sam_model_registry3D

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='union_train')
parser.add_argument('--click_type', type=str, default='random')
parser.add_argument('--multi_click', action='store_true', default=False)
parser.add_argument('--model_type', type=str, default='vit_b_ori')
parser.add_argument('--checkpoint', type=str, default='./work_dir/SAM/sam_vit_b.pth')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--work_dir', type=str, default='./work_dir1')  #导出权重路径

# train
parser.add_argument('--num_workers', type=int, default=0) #
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0,1])
parser.add_argument('--multi_gpu', action='store_true', default=False)
parser.add_argument('--resume', action='store_true', default=False)

# lr_scheduler
parser.add_argument('--lr_scheduler', type=str, default='multisteplr')
parser.add_argument('--step_size', type=list, default=[120, 180])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=42)
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default = 4) #
parser.add_argument('--accumulation_steps', type=int, default=8) #
parser.add_argument('--lr', type=float, default=5e-3)   #
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--port', type=int, default=12361)
parser.add_argument('--image_path', type=str, default='data/augumentation_train/images', help='Path to the directory containing images')
parser.add_argument('--label_path', type=str, default='data/augumentation_train/label', help='Path to the directory containing labels')
parser.add_argument('--teacher_weights_path', type=str, default='weights/teacher_model_weights.pth', help='Path to the teacher model weights') # only difference with turbo is this just the encoder weight

args = parser.parse_args()

device = args.device
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])
logger = logging.getLogger(__name__)
LOG_OUT_DIR = join(args.work_dir, args.task_name)
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def init_seeds(seed=2023, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def build_model(args):
    patch_size = ensure_tuple_rep(2, 3)
    window_size = ensure_tuple_rep(7, 3)
    if (args.multi_gpu):
        model = torch.nn.DataParallel(ImageEncoderViT3D(
            depth=6,
            embed_dim=768,
            img_size=128,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=8,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2,5],
            window_size=14,
            out_chans=384,
        ))
    else:
        model = ImageEncoderViT3D(
            depth=6,
            embed_dim=768,
            img_size=128,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=8,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2,5,8,11],
            window_size=0,
            out_chans=384,
            layeroutput=1,
        )
    model = model.to(args.device)
    return model
def get_dataloaders(args):
    train_dataset = distillation_data(image_path=args.image_path, label_path=args.label_path)

    if args.multi_gpu:
        train_sampler = DistributedSampler(train_dataset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_dataloader = Union_Dataloader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size, 
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=False,  #
    )
    return train_dataloader

def device_config(args):
    try:
        if not args.multi_gpu:
            # Single GPU
            if args.device == 'mps':
                args.device = torch.device('mps')
            else:
                args.device = torch.device(f"cuda:{args.gpu_ids[0]}")
        else:
            args.nodes = 1
            args.ngpus_per_node = len(args.gpu_ids)
            args.world_size = args.nodes * args.ngpus_per_node

    except RuntimeError as e:
        print(e)
        
def print_weight_stats(model, epoch):
    print(f"Epoch: {epoch} - Weight Statistics")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'{name}: Max={param.data.max()}, Min={param.data.min()}, Mean={param.data.mean()}, Std={param.data.std()}')

def check_weight_change(old_weights, model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            old_data = old_weights[name]
            change = (param.data - old_data).abs().sum()
            print(f"Change in {name}: {change}")


class BaseTrainer:
    def __init__(self, model, dataloaders, args):
        self.dataloaders = dataloaders
        self.model = model
        self.args = args
        self.best_loss = np.inf
        self.step_best_loss = np.inf
        self.layerepoch = 6
        self.curlayer = 1
        self.losses = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        self.init_checkpoint(join(self.args.work_dir, self.args.task_name, 'sam_model_latest.pth'))
        #self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
        #self.initial_weights = {name: param.data.clone() for name, param in model.named_parameters()} 
        self.initialmodel()
    
    def initialmodel(self, path=args.teacher_weights_path, freezelayer = ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'neck.0.weight', 'neck.1.weight', 'neck.1.bias', 'neck.2.weight', 'neck.3.weight', 'neck.3.bias']):
        # 加载权重
        p = torch.load(path)
        for name, parameter in self.model.named_parameters():
            if name in freezelayer: 
                parameter.data = p.get(name).clone()
                parameter.requires_grad = False
            # if 'mlp' in name:
            #     parameter.data = p.get(name).clone()
            #     parameter.requires_grad = False
            # if 'norm' in name:
            #     parameter.data = p.get(name).clone()
            #     parameter.requires_grad = False
    
    def set_loss_fn(self):
        self.seg_loss = torch.nn.MSELoss()
    
    def set_optimizer(self):
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model

        self.optimizer = torch.optim.AdamW([
            {'params': sam_model.parameters()}, 
        ], lr=self.args.lr, betas=(0.9,0.999), weight_decay=self.args.weight_decay)

    def set_lr_scheduler(self):
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                self.args.step_size,
                                                                self.args.gamma)
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                self.args.step_size[0],
                                                                self.args.gamma)
        elif self.args.lr_scheduler == 'coswarm':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 0.1)

    def init_checkpoint(self, ckp_path):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.multi_gpu:
                dist.barrier()
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)
            else:
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)
        
        if last_ckpt:
            if self.args.multi_gpu:
                self.model.module.load_state_dict(last_ckpt['model_state_dict'])
            else:
                self.model.load_state_dict(last_ckpt['model_state_dict'])
            if not self.args.resume:
                self.start_epoch = 0 
            else:
                self.start_epoch = last_ckpt['epoch']
                self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
                self.losses = last_ckpt['losses']
                self.best_loss = last_ckpt['best_loss']
            print(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch})")
        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")

    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "losses": self.losses,
            "best_loss": self.best_loss,
            "args": self.args,
            "used_datas": img_datas,
        }, join(MODEL_SAVE_PATH, f"sam_model_{describe}.pth"))

    def train_epoch(self, epoch):
        #print_weight_stats(self.model, epoch)
        
        epoch_loss = 0
        self.model.train()
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model
            self.args.rank = -1
        
        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            tbar = tqdm(self.dataloaders)
        else:
            tbar = self.dataloaders
        self.optimizer.zero_grad()
        step_loss = 0
        for step, (image3D, label) in enumerate(tbar):
            my_context = self.model.no_sync if self.args.rank != -1 and step % self.args.accumulation_steps != 0 else nullcontext
            with my_context():
                #image3D = self.norm_transform(image3D.squeeze(dim=1)) # (N, C, W, H, D)
                #image3D = image3D.unsqueeze(dim=1)
                #image3D = image3D.squeeze(1)  # 移除多余的维度
                
                image3D = image3D.to(device)
                for i in range(len(label)):
                    label[i] = label[i].to(device)
                # Debug: Print input data stats
                #print(f"Preprocessed input Max: {image3D.max()}")
                #print(f"Preprocessed input Min: {image3D.min()}")
                #print(f"Preprocessed input Mean: {image3D.mean()}")                    
                with amp.autocast():
                        #print("Shape of image3D before model:", image3D.shape)
                        
                        output = self.model(image3D)
                        
                        #print(f"output[-1] shape: {output[-1].shape}")
                        #print(f"label[-1] shape: {label[-1].shape}")
                        loss = self.seg_loss(output[self.curlayer], label[self.curlayer])
                    # Debug: Print output stats and current loss
                    #print(f"Output Max: {output[self.curlayer].max()}")
                    #print(f"Output Min: {output[self.curlayer].min()}")
                    #print(f"Output Mean: {output[self.curlayer].mean()}")
                epoch_loss += loss.item()

                cur_loss = loss.item()

                loss /= self.args.accumulation_steps
                
                self.scaler.scale(loss).backward()
                #torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1000)
                
                # Debug: Check gradients
                # for name, param in sam_model.named_parameters():
                #     if param.grad is not None:
                #         print(f"Gradient of {name}, Max: {param.grad.data.max()}, Min: {param.grad.data.min()}")

            if step % self.args.accumulation_steps == 0 and step != 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                print_loss = step_loss / self.args.accumulation_steps
                step_loss = 0
            else:
                step_loss += cur_loss

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                if step % self.args.accumulation_steps == 0 and step != 0:
                    print(f'Epoch: {epoch}, Step: {step}, Loss: {print_loss}')
                    if print_loss < self.step_best_loss:
                        self.step_best_loss = print_loss    
                        if print_loss < 0.001:
                            self.save_checkpoint(
                                epoch,
                                self.model.state_dict(),
                                describe=f'{epoch}_step_dice:{print_loss}_best'
                            )
        epoch_loss /= step
        #print_weight_stats(self.model, epoch)
        return epoch_loss
    
    def plot_result(self, plot_data, description, save_name):
        plt.plot(plot_data)
        plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        plt.savefig(join(MODEL_SAVE_PATH, f'{save_name}.png'))
        plt.close()


    def train(self):
        self.scaler = amp.GradScaler()
        i = 0
        for epoch in range(self.start_epoch, self.args.num_epochs):
            if i % self.layerepoch == 0 and i > 0 :
                self.curlayer += 1
            i += 1
            print(f'Epoch: {epoch}/{self.args.num_epochs - 1}')

            if self.args.multi_gpu:
                dist.barrier()
                self.dataloaders.sampler.set_epoch(epoch)
            epoch_loss = self.train_epoch(epoch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.args.multi_gpu:
                dist.barrier()
        
            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                self.losses.append(epoch_loss)
                print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
                logger.info(f'Epoch\t {epoch}\t : loss: {epoch_loss}')

                if self.args.multi_gpu:
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()
                
                # save latest checkpoint
                self.save_checkpoint(
                    epoch, 
                    state_dict, 
                    describe='latest'
                )

                # save train loss best checkpoint
                if epoch_loss < self.best_loss: 
                    self.best_loss = epoch_loss
                    self.save_checkpoint(
                        epoch,
                        state_dict,
                        describe='loss_best'
                    )
                
                # save train dice best checkpoint
                self.plot_result(self.losses, 'Dice + Cross Entropy Loss', 'Loss')
                # self.plot_result(self.dices, 'Dice', 'Dice')
        logger.info('=====================================================================')
        logger.info(f'Best loss: {self.best_loss}')
        logger.info(f'Total loss: {self.losses}')
        logger.info('=====================================================================')
        logger.info(f'args : {self.args}')
        logger.info(f'Used datasets : {img_datas}')
        logger.info('=====================================================================')

def main():
    device_config(args)
    if args.multi_gpu:
        mp.set_sharing_strategy('file_system')
        mp.spawn(
            main_worker,
            nprocs=args.world_size,
            args=(args, )
        )
    else:
        random.seed(2023)
        np.random.seed(2023)
        torch.manual_seed(2023)
        # Load datasets
        dataloaders = get_dataloaders(args)
        # Build model
        model = build_model(args)
        # Create trainer
        trainer = BaseTrainer(model, dataloaders, args)
        # Train
        trainer.train()

def main_worker(rank, args):
    setup(rank, args.world_size)

    torch.cuda.set_device(rank)
    args.num_workers = int(args.num_workers / args.ngpus_per_node)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank

    init_seeds(2023 + rank)

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO if rank in [-1, 0] else logging.WARN,
        filemode='w',
        filename=os.path.join(LOG_OUT_DIR, f'output_{cur_time}.log'))
    
    dataloaders = get_dataloaders(args)
    model = build_model(args)
    trainer = BaseTrainer(model, dataloaders, args)
    trainer.train()
    cleanup()


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=world_size,
        rank=rank
    )

def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    main()