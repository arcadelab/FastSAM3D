from __future__ import annotations

import math
from typing import Optional, Union

import torch
from einops import einsum, rearrange
from torch import Tensor, nn


class MultiExpertLayer(nn.Module):
    """A more efficient alternative to creating 'n' separate expert layers (likely
    from 'nn.Linear' modules).  Instead, we create a single set of batched weights
    and biases, and apply all 'experts' in parallel.

    Args:
        embed_dim (int): embedding dimension (d)
        num_experts (int): number of experts (n)
        bias (bool): whether to include a bias term. Default: True
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        bias: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts

        self.weight = nn.Parameter(
            torch.empty(
                (num_experts, in_features, out_features), device=device, dtype=dtype
            )
        )
        bias_param: Optional[nn.Parameter] = None
        if bias:
            bias_param = nn.Parameter(
                torch.empty((num_experts, out_features), device=device, dtype=dtype)
            )
        # Include type annotation for mypy :D
        self.bias: Optional[nn.Parameter]
        self.register_parameter("bias", bias_param)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # NOTE: Mostly copy-pasta from 'nn.Linear.reset_parameters'
        #
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Expected input with embed_dim={self.in_features} (dim=-1), but "
                f"found {x.size(-1)}"
            )
        elif x.size(1) != self.num_experts:
            raise ValueError(
                f"Expected input with num_experts={self.num_experts} (dim=1), but "
                f"found {x.size(1)}"
            )

        # NOTE: 'd1' and 'd2' are both equal to 'embed_dim'. But for 'einsum' to
        # work correctly, we have to give them different names.
        x = einsum(x, self.weight, "b n ... d1, n d1 d2 -> b n ... d2")

        if self.bias is not None:
            # NOTE: When used with 'SoftMoE' the inputs to 'MultiExpertLayer' will
            # always be 4-dimensional.  But it's easy enough to generalize for 3D
            # inputs as well, so I decided to include that here.
            if x.ndim == 3:
                bias = rearrange(self.bias, "n d -> () n d")
            elif x.ndim == 4:
                bias = rearrange(self.bias, "n d -> () n () d")
            else:
                raise ValueError(
                    f"Expected input to have 3 or 4 dimensions, but got {x.ndim}"
                )
            x = x + bias

        return x

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_experts={self.num_experts}, bias={self.bias is not None}"
        )
