import math
from functools import partial
from typing import Tuple

import torch
from timm.models.vision_transformer import (Block, VisionTransformer,
                                            _load_weights, checkpoint_seq)
from torch import nn

from eztorch.models.trunks.transformer.patch_embed import get_patch_embed


class AttentionViT(nn.Module):
    """Only the attention part of the Vision Transformer.

    Should be wrapped in a Model that contains patch embedding and/or handling of output tokens.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4.0,
        qkv_bias: bool = True,
        init_values: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module | None = None,
        act_layer: nn.Module = nn.GELU,
        block_fn: nn.Module = Block,
    ):
        """
        Args:
            patch_size: patch size.
            in_chans: number of input channels.
            embed_dim: embedding dimension.
            depth: depth of transformer.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: enable bias for qkv if True.
            init_values: layer-scale init values.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer: normalization layer.
            act_layer: MLP activation layer.
            block_fn: Block function.
        """
        super().__init__()
        use_norm = norm_layer is not None

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    init_values=init_values,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim, eps=1e-6) if use_norm else nn.Identity()

    def forward(self, x):
        x = self.blocks(x)
        x = self.norm(x)
        return x

    @property
    def num_layers(self) -> int:
        """Number of layers of the model."""
        return len(self.blocks)

    def get_param_layer_id(self, name: str) -> int:
        """Get the layer id of the named parameter.

        Args:
            name: The name of the parameter.
        """
        if name.startswith("blocks."):
            layer_id = int(name.split(".")[1])
            return layer_id
        else:
            return self.num_layers - 1


class ViT(VisionTransformer):
    def __init__(
        self,
        img_size: int | Tuple[int, int] = 224,
        patch_size: int | Tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        init_values: float | None = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        pre_norm: bool = False,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
        weight_init: str = "",
        embed_layer: nn.Module | None = None,
        norm_layer: nn.Module | None = None,
        act_layer: nn.Module | None = None,
        block_fn: nn.Module | None = None,
        has_multi_res: bool = False,
        conv_type: str = "Conv2d",
        tube_size: int = 2,
    ):
        embed_layer = embed_layer or partial(
            get_patch_embed, conv_type=conv_type, tube_size=tube_size
        )
        block_fn = block_fn or Block

        assert not no_embed_class
        assert class_token

        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,
            global_pool="token",
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            class_token=class_token,
            no_embed_class=no_embed_class,
            pre_norm=pre_norm,
            fc_norm=None,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            weight_init=weight_init,
            embed_layer=embed_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_fn=block_fn,
        )
        self.has_multi_res = has_multi_res
        self.class_token = class_token

    @property
    def num_layers(self) -> int:
        """Number of layers of the model."""
        return len(self.blocks) + 1

    def get_param_layer_id(self, name: str) -> int:
        """Get the layer id of the named parameter.

        Args:
            name: The name of the parameter.
        """
        if name in ("cls_token", "pos_embed"):
            return 0
        elif name.startswith("patch_embed"):
            return 0
        elif name.startswith("blocks."):
            layer_id = int(name.split(".")[1])
            return layer_id + 1
        else:
            return self.num_layers - 1

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        if npatch == N and w == h:
            return self.pos_embed

        if self.class_token:
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
        else:
            patch_pos_embed = self.pos_embed
        dim = x.shape[-1]
        h0 = h // self.patch_embed.patch_size[0]
        w0 = w // self.patch_embed.patch_size[1]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        if self.class_token:
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        else:
            return patch_pos_embed

    def prepare_multi_res_tokens(self, x: torch.Tensor) -> torch.Tensor:
        B, *dims, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        if self.class_token:
            # add the [CLS] token to the embed patch tokens
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_multi_res:
            x = self.prepare_multi_res_tokens(x)
            x = self.norm_pre(x)
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint_seq(self.blocks, x)
            else:
                x = self.blocks(x)
            x = self.norm(x)
        else:
            x = self.forward_features(x)

        x = (
            x[:, self.num_prefix_tokens :].mean(dim=1)
            if self.global_pool == "avg"
            else x[:, 0]
        )
        return x


def create_vit(
    img_size: int | Tuple[int, int] = 224,
    patch_size: int | Tuple[int, int] = 16,
    in_chans: int = 3,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    mlp_ratio: int = 4,
    qkv_bias: bool = True,
    init_values: float | None = None,
    class_token: bool = True,
    no_embed_class: bool = False,
    pre_norm: bool = False,
    drop_rate: float = 0,
    attn_drop_rate: float = 0,
    drop_path_rate: float = 0,
    weight_init: str = "",
    embed_layer: nn.Module | None = None,
    norm_layer: nn.Module | None = None,
    act_layer: nn.Module | None = None,
    block_fn: nn.Module | None = None,
    pretrain_pth: str | None = None,
    has_multi_res: bool = False,
    conv_type: str = "Conv2d",
    tube_size: int = 2,
    **kwargs,
) -> ViT:
    """Instantiate `ViT`

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

    model = ViT(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        init_values=init_values,
        class_token=class_token,
        no_embed_class=no_embed_class,
        pre_norm=pre_norm,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        weight_init=weight_init,
        embed_layer=embed_layer,
        norm_layer=norm_layer,
        act_layer=act_layer,
        block_fn=block_fn,
        has_multi_res=has_multi_res,
        conv_type=conv_type,
        tube_size=tube_size,
        **kwargs,
    )

    if pretrain_pth is not None:
        _load_weights(model, pretrain_pth)

    return model


def create_vit_tiny(*args, **kwargs) -> ViT:
    return create_vit(*args, embed_dim=192, num_heads=3, depth=12, **kwargs)


def create_vit_small(*args, **kwargs) -> ViT:
    return create_vit(*args, embed_dim=384, num_heads=6, depth=12, **kwargs)


def create_vit_base(*args, **kwargs) -> ViT:
    return create_vit(*args, embed_dim=768, num_heads=12, depth=12, **kwargs)
