from functools import partial
from typing import Optional, Union, cast

import torch
from einops import rearrange
from torch import Tensor, nn

from soft_mixture_of_experts.transformer import SoftMoEEncoder, SoftMoEEncoderLayer


class ViTWrapper(nn.Module):
    def __init__(
        self,
        num_classes: Optional[int],
        encoder: Union[SoftMoEEncoder, nn.TransformerEncoder],
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        dropout: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if not image_size % patch_size == 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by "
                f"patch_size ({patch_size})"
            )
        self.num_classes = num_classes
        self.encoder = encoder
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels

        # Extract model dimension from the first layer of the encoder.
        # TODO: Find a cleaner way to do this?  Unfortunately, TransformerEncoder
        # and TransformerEncoderLayer don't have a 'd_model' property.
        encoder_layer = cast(
            Union[SoftMoEEncoderLayer, nn.TransformerEncoderLayer],
            encoder.layers[0],
        )
        norm_layer = cast(nn.LayerNorm, encoder_layer.norm1)
        d_model = norm_layer.normalized_shape[0]
        num_patches = (image_size // patch_size) ** 2
        patch_dim = num_channels * patch_size**2

        self.patch_to_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim, device=device, dtype=dtype),
            nn.Linear(patch_dim, d_model, device=device, dtype=dtype),
            nn.LayerNorm(d_model, device=device, dtype=dtype),
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches, d_model, device=device, dtype=dtype)
        )
        self.dropout = nn.Dropout(dropout)

        self.out: nn.Module
        if num_classes is not None:
            self.out = nn.Linear(d_model, num_classes, device=device, dtype=dtype)
        else:
            self.out = nn.Identity()

    def forward(self, x: Tensor, return_features: bool = False) -> Tensor:
        if not x.size(1) == self.num_channels:
            raise ValueError(
                f"Expected num_channels={self.num_channels} but found {x.size(1)}"
            )
        elif not x.size(2) == x.size(3) == self.image_size:
            raise ValueError(
                f"Expected image_size={self.image_size} but found {x.size(2)}x{x.size(3)}"
            )

        x = rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        x = self.patch_to_embedding(x)
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.encoder(x)

        if return_features:
            return x

        x = x.mean(dim=-2)
        return self.out(x)


class ViT(ViTWrapper):
    def __init__(
        self,
        num_classes: Optional[int],
        image_size: int = 224,
        patch_size: int = 16,
        d_model: int = 512,
        dim_feedforward: int = 2048,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_channels: int = 3,
        dropout: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            device=device,
            dtype=dtype,
        )
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        super().__init__(
            num_classes=num_classes,
            encoder=encoder,
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )


class SoftMoEViT(ViTWrapper):
    def __init__(
        self,
        num_classes: Optional[int],
        image_size: int = 224,
        patch_size: int = 16,
        d_model: int = 512,
        dim_feedforward: int = 2048,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_experts: int = 128,
        slots_per_expert: int = 1,
        num_channels: int = 3,
        dropout: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        encoder_layer = SoftMoEEncoderLayer(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            num_experts=num_experts,
            slots_per_expert=slots_per_expert,
            device=device,
            dtype=dtype,
        )
        encoder = SoftMoEEncoder(encoder_layer, num_layers=num_encoder_layers)
        super().__init__(
            num_classes=num_classes,
            encoder=encoder,
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )


def _build_vit(
    num_classes: Optional[int],
    image_size: int,
    patch_size: int,
    d_model: int,
    nhead: int,
    num_encoder_layers: int,
    mlp_ratio: float = 4.0,
    num_channels: int = 3,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> ViT:
    return ViT(
        num_classes=num_classes,
        image_size=image_size,
        patch_size=patch_size,
        d_model=d_model,
        dim_feedforward=int(d_model * mlp_ratio),
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_channels=num_channels,
        device=device,
        dtype=dtype,
    )


vit_small = partial(_build_vit, d_model=384, nhead=6, num_encoder_layers=12)
vit_base = partial(_build_vit, d_model=768, nhead=12, num_encoder_layers=12)
vit_large = partial(_build_vit, d_model=1024, nhead=16, num_encoder_layers=24)
vit_huge = partial(_build_vit, d_model=1280, nhead=16, num_encoder_layers=32)


def _build_soft_moe_vit(
    num_classes: Optional[int],
    image_size: int,
    patch_size: int,
    d_model: int,
    nhead: int,
    num_encoder_layers: int,
    num_experts: int,
    slots_per_expert: int = 1,
    mlp_ratio: float = 4.0,
    num_channels: int = 3,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> SoftMoEViT:
    return SoftMoEViT(
        num_classes=num_classes,
        image_size=image_size,
        patch_size=patch_size,
        d_model=d_model,
        dim_feedforward=int(d_model * mlp_ratio),
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_experts=num_experts,
        slots_per_expert=slots_per_expert,
        num_channels=num_channels,
        device=device,
        dtype=dtype,
    )


soft_moe_vit_small = partial(
    _build_soft_moe_vit, d_model=384, nhead=6, num_encoder_layers=12
)
soft_moe_vit_base = partial(
    _build_soft_moe_vit, d_model=768, nhead=12, num_encoder_layers=12
)
soft_moe_vit_large = partial(
    _build_soft_moe_vit, d_model=1024, nhead=16, num_encoder_layers=24
)
soft_moe_vit_huge = partial(
    _build_soft_moe_vit, d_model=1280, nhead=16, num_encoder_layers=32
)
