from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.modules.utils import _pair


@torch.no_grad()
def constant_init_(tensor, constant_value=0):
    nn.init.constant_(tensor, constant_value)


@torch.no_grad()
def kaiming_init_(
    tensor, a=0, mode="fan_out", nonlinearity="relu", distribution="normal"
):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.kaiming_uniform_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)


class PatchEmbed(nn.Module):
    """Images to Patch Embedding.

    Args:
            img_size: Size of input image.
            patch_size: Size of one patch.
            tube_size: Size of temporal field of one 3D patch.
            in_chans: Channel num of input features.
            embed_dim: Dimensions of embedding.
            conv_type: Type for convolution layer.
    """

    def __init__(
        self,
        img_size: int | Tuple[int, int] = 224,
        patch_size: int | Tuple[int, int] = 16,
        tube_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 768,
        bias: bool = True,
        conv_type: str = "Conv2d",
        norm_layer: nn.Module | None = None,
    ):
        super().__init__()

        if type(img_size) is int:
            img_size = _pair(img_size)
        if type(patch_size) is int:
            patch_size = _pair(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        num_patches = (self.img_size[1] // self.patch_size[1]) * (
            self.img_size[0] // self.patch_size[0]
        )
        assert (
            num_patches * self.patch_size[0] * self.patch_size[1]
            == self.img_size[0] * self.img_size[1],
            "The image size H*W must be divisible by patch size",
        )
        self.num_patches = num_patches

        # Use conv layer to embed
        if conv_type == "Conv2d":
            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias=bias,
            )
        elif conv_type == "Conv3d":
            self.proj = nn.Conv3d(
                in_chans,
                embed_dim,
                kernel_size=(tube_size, patch_size[0], patch_size[1]),
                stride=(tube_size, patch_size[0], patch_size[1]),
                bias=bias,
            )
        else:
            raise TypeError(f"Unsupported conv layer type {conv_type}")

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def init_weights(self):
        if hasattr(self.proj, "weight") and self.proj.weight is not None:
            kaiming_init_(self.proj.weight, mode="fan_in", nonlinearity="relu")
        if hasattr(self.proj, "bias") and self.proj.bias is not None:
            constant_init_(self.proj.bias, constant_value=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer_type = type(self.proj)
        if layer_type == nn.Conv3d:
            x = self.proj(x)
            x = rearrange(x, "b c t h w -> (b t) (h w) c")
        elif layer_type == nn.Conv2d:
            if x.ndim == 5:
                x = rearrange(x, "b c t h w -> (b t) c h w")
                x = self.proj(x)
                x = rearrange(x, "b c h w -> b (h w) c")
            else:
                x = self.proj(x)
                x = rearrange(x, "b c h w -> b (h w) c")
        else:
            raise TypeError(f"Unsupported conv layer type {layer_type}")

        x = self.norm(x)
        return x


def get_patch_embed(**kwargs) -> nn.Module:
    if kwargs["conv_type"] == "identity":
        return nn.Identity()
    return PatchEmbed(**kwargs)
