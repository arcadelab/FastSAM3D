# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from typing import Optional, Tuple, Type
from segment_anything.modeling.dilated_utils import XPOS, RelativePositionBias
#from zeta.nn.attention.flash_attention import FlashAttention
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))
    
class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT3D(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int, int]] = None,
        global_attn_indexes: Tuple[int, ...] = (),
        layeroutput = 2,
        
        # New parameter
        dilation: int = 1,  
        segment_size: int = 64,
        dropout: float = 0.0,
        causal: bool = False,
        use_xpos: bool = False,
        use_rel_pos_bias: bool = False,
        qk_norm: bool = False,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda:0",
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size
        self.layeroutput = layeroutput
        self.patch_embed = PatchEmbed3D(
            kernel_size=(patch_size, patch_size, patch_size),
            stride=(patch_size, patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(2):
            self.blocks.append(Block3D_woatt(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size, img_size // patch_size),
        ))
        for i in range(depth -2):
            block = Block3D(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv3d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            # nn.LayerNorm(out_chans),
            LayerNorm3d(out_chans),
            nn.Conv3d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm3d(out_chans),
            # nn.LayerNorm(out_chans),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input_size = [1,1,256,256,256]
        # import IPython; IPython.embed()
        t = time.time()
        listx = []
        x = self.patch_embed(x)
        
        # x = [1,16,16,16,768]
        # import pdb; pdb.set_trace()
        if self.pos_embed is not None:
            x = x + self.pos_embed
        listx.append(x)
        i = 0
        for blk in self.blocks:
            i += 1
            x,x1 = blk(x)
            if i % self.layeroutput == 0:
                listx.append(x1)
            
        # x = [1,16,16,16,768]
        x = self.neck(x.permute(0, 4, 1, 2, 3))
        listx.append(x)
        # output_size = [1,256,16,16,16]
        return listx,time.time()-t



class Block3D(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int, int]] = None,
        # New parameter
        dilation: int = 1,  
        segment_size: int = 64,
        dropout: float = 0.0,
        causal: bool = False,
        use_xpos: bool = False,
        use_rel_pos_bias: bool = False,
        qk_norm: bool = False,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda:0",
        
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = DilatedAttention(
            dim=dim,
            heads=num_heads,
            dilation_rate=dilation,
            segment_size=segment_size,
            dropout=dropout,
            causal=causal,
            use_xpos=use_xpos,
            use_rel_pos_bias=use_rel_pos_bias,
            qk_norm=qk_norm,
            dtype=dtype,
            device=device,

        )
       
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x1 = self.attn(x)
        #print("x shape",x.shape)
        #print("x1 shape",x1.shape)
        #x1 = x1.view(B, D, H, W, C)
        x = shortcut + x1
        x = self.norm2(x)
        x = x + self.mlp(x)

        return x,x1


class Block3D_woatt(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        # self.norm1 = norm_layer(dim)
        # self.attn = Attention(
        #     dim,
        #     num_heads=num_heads,
        #     qkv_bias=qkv_bias,
        #     use_rel_pos=use_rel_pos,
        #     rel_pos_zero_init=rel_pos_zero_init,
        #     input_size=input_size if window_size == 0 else (window_size, window_size, window_size),
        # )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm2(x)
        
        x = x + self.mlp(x)

        return x,None

# add alibi, qk layer norm, one write head, multihway,
class DilatedAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dilation_rate: int,
        segment_size: int,
        dropout: float = 0.0,
        causal: bool = False,
        use_xpos: bool = False,
        use_rel_pos_bias: bool = False,
        qk_norm: bool = False,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda:0",
    ) -> None:
        super(DilatedAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dilation_rate = dilation_rate
        self.segment_size = segment_size
        self.dropout = nn.Dropout(dropout)
        self.causal = causal
        self.use_xpos = use_xpos
        self.use_rel_pos_bias = use_rel_pos_bias
        self.qk_norm = qk_norm
        self.dtype = dtype
        self.device = device
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        # self.seqlen=512 #(8*8*8)
        
        # self.attention = FlashAttention(causal=self.causal, dropout=dropout).to(
        #     device
        # )

        if use_xpos:
            self.xpos = XPOS(head_dim=dim // heads)
        if use_rel_pos_bias:
            self.relative_bias = RelativePositionBias(
                num_buckets=32, max_distance=128, n_heads=heads
            )

        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)

        # head offsets
        #self.head_offsets = nn.Parameter(torch.randn(heads, dim))

        
    def get_mask(self, i, j):
        """i = row, j=column"""
        return torch.ones((i, j), device=self.device, dtype=torch.bool).triu(
            j - i + 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DilatedAttention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        batch_size, depth, height, width, _ = x.shape
        x_dilated = x[:, ::self.dilation_rate, ::self.dilation_rate, ::self.dilation_rate, :]
        _, d_dilated, h_dilated, w_dilated, _ = x_dilated.shape
        padding_depth = depth - d_dilated
        padding_height = height - h_dilated
        padding_width = width - w_dilated
        x = F.pad(x_dilated, (0, 0, 0, padding_width, 0, padding_height, 0, padding_depth))
        qkv = self.qkv(x)
        new_seq_length = depth * height * width
        qkv = qkv.reshape(batch_size, new_seq_length, 3, self.heads, -1)
        qkv = qkv.half()
        attn = flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False)
        attn_output = attn.float()
        attn_output = attn_output.view(batch_size, depth, height, width, -1)
        x=attn_output
        return x




class PatchEmbed3D(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16, 16),
        stride: Tuple[int, int] = (16, 16, 16),
        padding: Tuple[int, int] = (0, 0, 0),
        in_chans: int = 1,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C X Y Z -> B X Y Z C
        x = x.permute(0, 2, 3, 4, 1)
        return x


