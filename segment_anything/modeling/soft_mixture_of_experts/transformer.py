from copy import deepcopy
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.transformer import _get_activation_fn

from segment_anything.modeling.soft_mixture_of_experts.soft_moe import SoftMoE


class SoftMoEEncoderLayer(nn.Module):
    """PyTorch module for Soft-MoE Transformer Encoder Layer, as described in:
        https://arxiv.org/pdf/2308.00951.pdf

    NOTE: Nearly identical to a standard Transformer encoder layer, except that we
    replace the second feedforward layer with 'SoftMoE'.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        num_experts: int = 128,
        slots_per_expert: int = 1,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        self.d_model = d_model
        self.norm_first = norm_first
        self.activation = activation

        self.dropout = nn.Dropout(dropout)

        # self-attention block
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
            device=device,
            dtype=dtype,
        )

        # feedforward / soft-moe block
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.linear = nn.Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.moe = SoftMoE(
            in_features=dim_feedforward,
            out_features=d_model,
            num_experts=num_experts,
            slots_per_expert=slots_per_expert,
            device=device,
            dtype=dtype,
        )

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )
        return self.dropout(x)

    # feedforward / soft-moe block
    def _ff_block(self, x: Tensor) -> Tensor:
        """Forward pass for the FeedForward block, which now includes a SoftMoE layer.
        Mostly copy-pasta from 'nn.TransformerEncoderLayer'.  The only difference
        is swapping 'self.linear2' for 'self.moe'.
        """
        x = self.moe(self.dropout(self.activation(self.linear(x))))
        return self.dropout(x)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal)
            )
            x = self.norm2(x + self._ff_block(x))

        return x


class SoftMoEEncoder(nn.Module):
    """PyTorch module for Soft-MoE Transformer Encoder, as described in:
        https://arxiv.org/pdf/2308.00951.pdf

    NOTE: Nearly identical to a standard Transformer encoder, except that we
    replace the second feedforward (nn.Linear) in each layer with 'SoftMoE'.
    """

    def __init__(self, encoder_layer: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [deepcopy(encoder_layer) for _ in range(num_layers)]
        )

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        x = src
        for layer in self.layers:
            x = layer(
                x,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )
        return x


class SoftMoEDecoderLayer(nn.Module):
    """PyTorch module for Soft-MoE Transformer Decoder Layer, as described in:
        https://arxiv.org/pdf/2308.00951.pdf

    NOTE: Nearly identical to a standard Transformer decoder layer, except that we
    replace the second feedforward layer with 'SoftMoE'.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_experts: int = 128,
        slots_per_expert: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        self.d_model = d_model
        self.norm_first = norm_first
        self.activation = activation

        self.dropout = nn.Dropout(dropout)

        # self-attention block
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )

        # cross-attention block
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )

        # feedforward / soft-moe block
        self.norm3 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.linear = nn.Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.moe = SoftMoE(
            in_features=dim_feedforward,
            out_features=d_model,
            num_experts=num_experts,
            slots_per_expert=slots_per_expert,
            device=device,
            dtype=dtype,
        )

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )
        return self.dropout(x)

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x, _ = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )
        return self.dropout(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.moe(self.dropout(self.activation(self.linear(x))))
        return self.dropout(x)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            )
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            )
            x = self.norm2(
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
                )
            )
            x = self.norm3(x + self._ff_block(x))

        return x


class SoftMoEDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [deepcopy(decoder_layer) for _ in range(num_layers)]
        )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        x = tgt
        for layer in self.layers:
            x = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )
        return x
