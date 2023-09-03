from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
from timm.models.helpers import named_apply
from timm.models.layers import to_ntuple, trunc_normal_
from timm.models.swin_transformer import BasicLayer, PatchMerging
from timm.models.vision_transformer import get_init_weights_vit
from torch.nn.modules.utils import _pair

from eztorch.models.trunks.transformer.patch_embed import get_patch_embed


class Swin(nn.Module):
    r"""Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        head_dim (int, tuple(int)):
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
    """

    def __init__(
        self,
        img_size: int | Tuple[int, int] = 224,
        patch_size: int | Tuple[int, int] = 4,
        in_chans: int = 3,
        global_pool: str = "avg",
        embed_dim: int = 96,
        depths: Tuple[int] = (2, 2, 6, 2),
        num_heads: Tuple[int] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module | None = None,
        ape: bool = False,
        patch_norm: bool = True,
        weight_init: str = "",
        conv_type: str = "Conv2d",
        tube_size: int = 2,
        **kwargs,
    ):

        super().__init__()
        assert global_pool in ("", "avg")
        self.global_pool = global_pool
        self.num_swin_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_swin_layers - 1))
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        if type(img_size) is int:
            img_size = _pair(img_size)
        if type(patch_size) is int:
            patch_size = _pair(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.depths = depths

        self.patch_grid = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        patch_embed = get_patch_embed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
            conv_type=conv_type,
            tube_size=tube_size,
        )

        self.patch_embed = patch_embed if type(patch_embed) is not nn.Identity else None

        if self.patch_embed is not None:
            num_patches = self.patch_embed.num_patches

        # absolute position embedding
        self.absolute_pos_embed = (
            nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            if ape and self.patch_embed is not None
            else None
        )

        self.pos_drop = (
            nn.Dropout(p=drop_rate) if self.patch_embed is not None else None
        )

        # build layers
        if not isinstance(embed_dim, (tuple, list)):
            embed_dim = [int(embed_dim * 2**i) for i in range(self.num_swin_layers)]
        embed_out_dim = embed_dim[1:] + [None]
        head_dim = to_ntuple(self.num_swin_layers)(None)
        window_size = to_ntuple(self.num_swin_layers)(window_size)
        mlp_ratio = to_ntuple(self.num_swin_layers)(mlp_ratio)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        layers = []
        for i in range(self.num_swin_layers):
            layers += [
                BasicLayer(
                    dim=embed_dim[i],
                    out_dim=embed_out_dim[i],
                    input_resolution=(
                        self.patch_grid[0] // (2**i),
                        self.patch_grid[1] // (2**i),
                    ),
                    depth=depths[i],
                    num_heads=num_heads[i],
                    head_dim=head_dim[i],
                    window_size=window_size[i],
                    mlp_ratio=mlp_ratio[i],
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    norm_layer=norm_layer,
                    downsample=PatchMerging if (i < self.num_swin_layers - 1) else None,
                )
            ]
        self.layers = nn.Sequential(*layers)

        self.norm = norm_layer(self.num_features)

        if weight_init != "skip":
            self.init_weights(weight_init)

    @torch.jit.ignore
    def init_weights(self, mode=""):
        assert mode in ("jax", "jax_nlhb", "moco", "")
        if self.absolute_pos_embed is not None:
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        named_apply(get_init_weights_vit(mode, head_bias=0), self)

    @property
    def num_layers(self) -> int:
        """Number of layers of the model."""
        sum_depths = sum(self.depths)
        return sum_depths + (1 if self.patch_embed is not None else 0)

    def get_param_layer_id(self, name: str) -> int:
        """Get the layer id of the named parameter.

        Args:
            name: The name of the parameter.
        """
        if name.startswith("patch_embed."):
            return 0
        elif name in ("absolute_pos_embed"):
            return 0
        elif name.startswith("layers."):
            add = 1 if self.patch_embed is not None else 0
            splitted_name = name.split(".")
            if splitted_name[3] in ["norm", "reduction"]:
                return add + sum(self.depths[: int(splitted_name[1]) + 1]) - 1
            layer_id, block_id = int(splitted_name[1]), int(splitted_name[3])
            return add + sum(self.depths[:layer_id]) + block_id
        else:
            return self.num_layers - 1

    def forward(self, x):
        if self.patch_embed is not None:
            x = self.patch_embed(x)
            if self.absolute_pos_embed is not None:
                x = x + self.absolute_pos_embed
            x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)
        if self.global_pool == "avg":
            x = x.mean(dim=1)
        return x


def create_swin(
    img_size: int | Tuple[int, int] = 224,
    patch_size: int | Tuple[int, int] = 4,
    in_chans: int = 3,
    global_pool: str = "avg",
    embed_dim: int = 96,
    depths: Tuple[int] = (2, 2, 6, 2),
    num_heads: Tuple[int] = (3, 6, 12, 24),
    window_size: int = 7,
    mlp_ratio: float = 4.0,
    qkv_bias: bool = True,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    norm_layer: nn.Module = None,
    ape: bool = False,
    patch_norm: bool = True,
    weight_init: str = "",
    conv_type: str = "Conv2d",
    tube_size: int = 2,
    pretrain_pth: str | None = None,
    **kwargs,
) -> Swin:
    """Instantiate `Swin`

    Args:
        img_size: input image size.
        patch_size: patch size.
        in_chans: number of input channels.
        global_pool: type of global pooling for final sequence.
        embed_dim: embedding dimension.
        depth: depth of transformer.
        num_heads: number of attention heads.
        mlp_ratio: ratio of mlp hidden dim to embedding dim.
        qkv_bias: enable bias for qkv if True.
        init_values: layer-scale init values.
        class_token: use class token.
        drop_rate: dropout rate
        attn_drop_rate: attention dropout rate
        drop_path_rate: stochastic depth rate
        weight_init: weight init scheme
        embed_layer: patch embedding layer
        norm_layer: normalization layer
        act_layer: MLP activation layer
        block_fn: Block layer
        has_multi_res: Whether the model will receive crops at different resolution
        pretrain_pth: Checkpoint to load pretrained ViT.
        conv_type: Type of convolution used for the patch embedder.
        tube_size: Tube size used in case the patch embedder uses a convolution 3D.

    Returns:
            The ViT model
    """

    model = Swin(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        global_pool=global_pool,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        norm_layer=norm_layer,
        ape=ape,
        patch_norm=patch_norm,
        weight_init=weight_init,
        conv_type=conv_type,
        tube_size=tube_size,
        **kwargs,
    )

    if pretrain_pth is not None:
        raise NotImplementedError("TODO")

    return model


def create_swin_tiny(*args, **kwargs) -> Swin:
    return create_swin(
        *args,
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        **kwargs,
    )


def create_swin_small(*args, **kwargs) -> Swin:
    return create_swin(
        *args,
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        **kwargs,
    )


def create_swin_base(*args, **kwargs) -> Swin:
    return create_swin(
        *args,
        patch_size=4,
        window_size=7,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        **kwargs,
    )
