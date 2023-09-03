# --------------------------------------------------------------------------------------------
# Modified from VideoTransformer-pytorch (https://github.com/mx-mark/VideoTransformer-pytorch)
# Not licensed
# --------------------------------------------------------------------------------------------

from functools import partial
from typing import Optional, Tuple

import torch
from einops import rearrange, reduce, repeat
from timm.models.helpers import named_apply
from timm.models.vision_transformer import get_init_weights_vit
from torch import Tensor, nn
from torch.nn import LayerNorm, Module, Parameter

from eztorch.models.trunks.transformer.patch_embed import PatchEmbed
from eztorch.models.trunks.transformer.vivit_utils import (
    get_sine_cosine_pos_emb, init_from_vit_pretrain)
from eztorch.models.trunks.vit import AttentionViT
from eztorch.utils.mask import batch_mask_tube_in_sequence


class ViViT(Module):
    """A PyTorch implementation of `ViViT: A Video Vision Transformer`. Only supports the model 2 version.

            <https://arxiv.org/abs/2103.15691>
    Args:
        num_frames: Number of frames in the video.
        img_size: Size of input image.
        patch_size: Size of one patch.
        spatial_class_token: Whether to use class token for spatial encoder.
        temporal_class_token: Whether to use class token for temporal encoder.
        embed_dim: Dimensions of embedding.
        spatial_num_heads: Number of parallel spatial attention heads.
        temporal_num_heads: Number of parallel temporal attention heads.
        spatial_depth: Depth of the spatial transformer layers.
        temporal_depth: Number of temporal transformer layers.
        in_chans: Channel num of input features.
        dropout_p: Probability of dropout spatial and temporal layer paths.
        dropout_rate: Probability of dropout layer.
        time_dropout_rate: Probability of time dropout.
        attention_dropout_rate: Probability of attention dropout layer.
        tube_size: Dimension of the kernel size in Conv3d.
        conv_type: Type of the convolution in PatchEmbed layer.
        norm_layer: Config for norm layers.
        use_learnable_pos_emb: Whether to use learnable position embeddings.
        use_learnable_time_emb: Whether to use learnable temporal embeddings.
        freeze_spatial: Whether to freeze the spatial encoder.
        temporal_mask_token: Whether to use temporal mask tokens.
        temporal_mask_ratio: Ratio of temporal masking from ViViT.
        temporal_mask_tube: Tube size of temporal masking from ViViT
    """

    def __init__(
        self,
        num_frames: int,
        img_size: int | Tuple[int, int] = 224,
        patch_size: int | Tuple[int, int] = 16,
        spatial_class_token: bool = True,
        temporal_class_token: bool = True,
        embed_dim: int = 768,
        spatial_num_heads: int = 12,
        temporal_num_heads: int = 12,
        spatial_depth: int = 12,
        temporal_depth: int = 4,
        in_chans: int = 3,
        dropout_p: int = 0.0,
        dropout_rate: float = 0.0,
        time_dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        tube_size: int = 2,
        conv_type: str = "Conv3d",
        norm_layer: Module | None = None,
        use_learnable_pos_emb: bool = True,
        use_learnable_time_emb: bool = True,
        freeze_spatial: bool = False,
        temporal_mask_token: bool = False,
        temporal_mask_ratio: float = 0.0,
        temporal_mask_tube: int = 2,
        **kwargs,
    ):
        super().__init__()

        norm_layer = norm_layer or partial(LayerNorm, eps=1e-6)

        num_frames = num_frames // tube_size
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.spatial_depth = spatial_depth
        self.temporal_depth = temporal_depth
        self.conv_type = conv_type
        self.tube_size = tube_size
        self.use_learnable_pos_emb = use_learnable_pos_emb
        self.spatial_class_token = spatial_class_token
        self.temporal_class_token = temporal_class_token
        self.freeze_spatial = freeze_spatial
        self.temporal_mask_ratio = temporal_mask_ratio
        self.temporal_mask_tube = temporal_mask_tube

        # tokenize & position embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            tube_size=tube_size,
            conv_type=conv_type,
        )

        self.pos_drop = nn.Dropout(p=dropout_rate)

        num_patches = self.patch_embed.num_patches

        self.spatial_transformer = AttentionViT(
            embed_dim=embed_dim,
            depth=spatial_depth,
            num_heads=spatial_num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            init_values=None,
            drop_rate=dropout_rate,
            attn_drop_rate=attention_dropout_rate,
            drop_path_rate=dropout_p,
            norm_layer=norm_layer,
            act_layer=nn.GELU,
        )

        self.time_drop = nn.Dropout(p=time_dropout_rate)

        self.temporal_transformer = AttentionViT(
            embed_dim=embed_dim,
            depth=temporal_depth,
            num_heads=temporal_num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            init_values=None,
            drop_rate=dropout_rate,
            attn_drop_rate=attention_dropout_rate,
            drop_path_rate=dropout_p,
            norm_layer=norm_layer,
            act_layer=nn.GELU,
        )

        if spatial_class_token:
            self.spatial_cls_token = Parameter(torch.zeros(1, 1, embed_dim))
        if temporal_class_token:
            self.temporal_cls_token = Parameter(torch.zeros(1, 1, embed_dim))

        if temporal_mask_token:
            self.temporal_mask_token = Parameter(torch.zeros(1, 1, embed_dim))

        num_prefix_spatial_tokens = 1 if spatial_class_token else 0
        num_prefix_temporal_tokens = 1 if temporal_class_token else 0

        num_patches = num_patches + num_prefix_spatial_tokens
        num_frames = num_frames + num_prefix_temporal_tokens

        if use_learnable_pos_emb:
            self.pos_embed = Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            self.register_buffer(
                "pos_embed", get_sine_cosine_pos_emb(num_patches, embed_dim)
            )
        if use_learnable_time_emb:
            self.time_embed = Parameter(torch.zeros(1, num_frames, embed_dim))
        else:
            self.register_buffer(
                "time_embed", get_sine_cosine_pos_emb(num_frames, embed_dim)
            )

        self.init_weights()

        if freeze_spatial:
            for param in self.spatial_transformer.parameters():
                param.requires_grad = False
            if use_learnable_pos_emb:
                self.pos_embed.requires_grad = False
            if spatial_class_token:
                self.spatial_cls_token.requires_grad = False
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            self.freeze_spatial_transformer()

    def init_weights(self):
        named_apply(get_init_weights_vit("", 0), self)
        if self.use_learnable_pos_emb:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.time_embed, std=0.02)
        if self.spatial_class_token:
            nn.init.trunc_normal_(self.spatial_cls_token, std=0.02)
        if self.temporal_class_token:
            nn.init.trunc_normal_(self.temporal_cls_token, std=0.02)

    def prepare_tokens(self, x: Tensor):
        b = x.shape[0]
        x = self.patch_embed(x)

        if self.spatial_class_token:
            # Add Position Embedding
            spatial_cls_tokens = repeat(
                self.spatial_cls_token, "b ... -> (repeat b) ...", repeat=x.shape[0]
            )
            x = torch.cat((spatial_cls_tokens, x), dim=1)

        if self.use_learnable_pos_emb:
            x = x + self.pos_embed
        else:
            x = x + self.pos_embed.type_as(x)

        return x, b

    def forward_spatial(
        self, x: Tensor
    ) -> Tuple[Tensor, int, Tensor | None, Tensor | None]:
        (x, b) = self.prepare_tokens(x)

        x = self.pos_drop(x)
        x = self.spatial_transformer(x)

        if self.spatial_class_token:
            x = rearrange(x[:, 0], "(b t) d -> b t d", b=b)
        else:
            x = rearrange(x, "(b t) p d -> b t p d", b=b)
            x = reduce(x, "b t p d -> b t d", "mean")

        return x, b

    def forward(
        self,
        x: Tensor,
        return_spatial: bool = False,
        detach_spatial: bool = False,
        inversed_temporal_masked_indices: Tensor | None = None,
    ) -> Tensor:
        if self.temporal_mask_ratio > 0.0 and self.training:
            b, c, t, h, w = x.shape

            (
                _,
                indices_kept,
                inversed_temporal_masked_indices,
                num_masked_tokens,
            ) = batch_mask_tube_in_sequence(
                self.temporal_mask_ratio,
                self.temporal_mask_tube,
                t,
                b,
                x.device,
            )

            x = x.permute([0, 2, 1, 3, 4])
            old_x = x
            x = torch.empty(
                (b, t - num_masked_tokens // b, c, h, w), device=x.device, dtype=x.dtype
            )
            for i in range(b):
                x[i] = old_x[i][indices_kept[i]]
            x = x.permute([0, 2, 1, 3, 4])

            inversed_temporal_masked_indices = torch.div(
                inversed_temporal_masked_indices[:, :: self.tube_size],
                self.tube_size,
                rounding_mode="floor",
            )

            num_masked_tokens //= self.tube_size

            in_model_mask = True
        else:
            in_model_mask = False

        if self.freeze_spatial:
            with torch.no_grad():
                (
                    x_spatial,
                    b,
                ) = self.forward_spatial(x)
        else:
            (
                x_spatial,
                b,
            ) = self.forward_spatial(x)

        if detach_spatial:
            x = x_spatial.detach()
        else:
            x = x_spatial

        if inversed_temporal_masked_indices is not None and self.training:
            if not in_model_mask:
                inversed_temporal_masked_indices = torch.div(
                    inversed_temporal_masked_indices[:, :: self.tube_size],
                    self.tube_size,
                    rounding_mode="floor",
                )

                num_masked_tokens = inversed_temporal_masked_indices.numel() - (
                    x_spatial.shape[0] * x_spatial.shape[1]
                )

            temporal_mask_tokens = self.temporal_mask_token.repeat(
                b, num_masked_tokens // b, 1
            )

            x = torch.cat((temporal_mask_tokens, x), 1)
            old_x = x
            x = torch.empty_like(old_x)
            for i in range(x.shape[0]):
                x[i] = old_x[i][inversed_temporal_masked_indices[i]]

        if self.temporal_class_token:
            cls_temporal_tokens = repeat(
                self.temporal_cls_token, "() n d -> b n d", b=b
            )
            x = torch.cat((cls_temporal_tokens, x), dim=1)

        if self.use_learnable_pos_emb:
            x = x + self.time_embed
        else:
            x = x + self.time_embed.type_as(x)

        x = self.time_drop(x)

        x = self.temporal_transformer(x)

        if return_spatial:
            return x_spatial, x

        return x

    @property
    def num_layers(self) -> int:
        """Number of layers of the model."""
        return (
            1 + self.spatial_transformer.num_layers
            if not self.freeze_spatial
            else 0 + self.temporal_transformer.num_layers
        )

    def get_param_layer_id(self, name: str) -> int:
        """Get the layer id of the named parameter.

        Args:
            name: The name of the parameter.
        """
        if name in ("spatial_cls_token", "pos_embed"):
            return 0
        elif name.startswith("patch_embed"):
            return 0
        elif name in ("time_embed", "temporal_cls_token"):
            return self.spatial_transformer.num_layers if not self.freeze_spatial else 0
        elif name.startswith("spatial_transformer."):
            return (
                1
                + self.spatial_transformer.get_param_layer_id(
                    name[len("spatial_transformer.") :]
                )
                if not self.freeze_spatial
                else 0
            )
        elif name.startswith("temporal_transformer."):
            return (
                1 + self.spatial_transformer.num_layers
                if not self.freeze_spatial
                else 0
                + self.temporal_transformer.get_param_layer_id(
                    name[len("temporal_transformer.") :]
                )
            )
        else:
            return self.num_layers - 1

    def freeze_spatial_transformer(self):
        self.spatial_transformer.eval()
        self.patch_embed.eval()
        return

    def train(self, mode: bool = True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        if self.freeze_spatial:
            self.freeze_spatial_transformer()

        return self


def create_vivit(
    num_frames: int,
    img_size: int | Tuple[int, int] = 224,
    patch_size: int | Tuple[int, int] = 16,
    embed_dim: int = 768,
    spatial_num_heads: int = 12,
    temporal_num_heads: int = 12,
    spatial_depth: int = 12,
    temporal_depth: int = 4,
    in_chans: int = 3,
    dropout_p: float = 0.0,
    dropout_rate: float = 0.0,
    time_dropout_rate: float = 0.0,
    attention_dropout_rate: float = 0.0,
    tube_size: int = 2,
    conv_type: str = "Conv3d",
    norm_layer: Module | None = None,
    use_learnable_pos_emb: bool = True,
    use_learnable_time_emb: bool = True,
    freeze_spatial: bool = False,
    temporal_mask_token: bool = False,
    temporal_mask_ratio: float = 0.0,
    temporal_mask_tube: int = 2,
    pretrain_pth: Optional[str] = None,
    weights_from: str = "spatial",
    vit_prefix: str = "trunk.",
    copy_strategy: str = "repeat",
    extend_strategy: str = "temporal_avg",
    **kwargs,
) -> ViViT:
    """Instantiate `ViViT: A Video Vision Transformer` model 2.

        <https://arxiv.org/abs/2103.15691>

    Args:
        num_frames: Number of frames in the video.
        img_size: Size of input image.
        patch_size: Size of one patch.
        spatial_class_token: Whether to use class token for spatial encoder.
        temporal_class_token: Whether to use class token for temporal encoder.
        embed_dim: Dimensions of embedding.
        spatial_num_heads: Number of parallel spatial attention heads.
        temporal_num_heads: Number of parallel temporal attention heads.
        spatial_depth: Depth of the spatial transformer layers.
        temporal_depth: Number of temporal transformer layers.
        in_chans: Channel num of input features.
        dropout_p: Probability of dropout spatial and temporal layer paths.
        dropout_rate: Probability of dropout layer.
        time_dropout_rate: Probability of time dropout.
        attention_dropout_rate: Probability of attention dropout layer.
        tube_size: Dimension of the kernel size in Conv3d.
        conv_type: Type of the convolution in PatchEmbed layer.
        norm_layer: Config for norm layers.
        use_learnable_pos_emb: Whether to use learnable position embeddings.
        use_learnable_time_emb: Whether to use learnable temporal embeddings.
        freeze_spatial: Whether to freeze the spatial encoder.
        temporal_mask_token: Whether to use temporal mask tokens.
        temporal_mask_ratio: Ratio of temporal masking from ViViT.
        temporal_mask_tube: Tube size of temporal masking from ViViT
        pretrain_path: Path of the pretrain checkpoint.
        weight_from: From what part of the backbone the pretrained checkpoint refer.
        vit_prefix: Prefix to remove from the checkpoint to retrieve the spatial vit pretrained in Eztorch.
        copy_strategy: Copy or Initial to zero towards the new additional layer.
        extend_strategy: How to initialize the weights of Conv3d from pre-trained Conv2d.

    Returns:
            The ViViT model
    """

    model = ViViT(
        num_frames=num_frames,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        spatial_num_heads=spatial_num_heads,
        temporal_num_heads=temporal_num_heads,
        spatial_depth=spatial_depth,
        temporal_depth=temporal_depth,
        in_chans=in_chans,
        dropout_p=dropout_p,
        dropout_rate=dropout_rate,
        temporal_dropout_rate=time_dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        tube_size=tube_size,
        conv_type=conv_type,
        norm_layer=norm_layer,
        use_learnable_pos_emb=use_learnable_pos_emb,
        use_learnable_time_emb=use_learnable_time_emb,
        freeze_spatial=freeze_spatial,
        temporal_mask_token=temporal_mask_token,
        temporal_mask_ratio=temporal_mask_ratio,
        temporal_mask_tube=temporal_mask_tube,
        **kwargs,
    )

    if pretrain_pth is not None:
        if weights_from == "spatial":
            init_from_vit_pretrain(
                model=model,
                pretrained=pretrain_pth,
                conv_type=conv_type,
                copy_strategy=copy_strategy,
                extend_strategy=extend_strategy,
                tube_size=tube_size,
                trunk_prefix=vit_prefix,
            )
        else:
            raise TypeError(f"Do not support the weights_from: {weights_from}")

    return model


def create_vivit_tiny(*args, **kwargs) -> ViViT:
    temporal_num_heads = kwargs.pop("temporal_num_heads", 3)
    temporal_depth = kwargs.pop("temporal_depth", 4)
    return create_vivit(
        *args,
        embed_dim=192,
        spatial_num_heads=3,
        temporal_num_heads=temporal_num_heads,
        spatial_depth=12,
        temporal_depth=temporal_depth,
        **kwargs,
    )


def create_vivit_small(*args, **kwargs) -> ViViT:
    temporal_num_heads = kwargs.pop("temporal_num_heads", 6)
    temporal_depth = kwargs.pop("temporal_depth", 4)
    return create_vivit(
        *args,
        embed_dim=384,
        spatial_num_heads=6,
        temporal_num_heads=temporal_num_heads,
        spatial_depth=12,
        temporal_depth=temporal_depth,
        **kwargs,
    )


def create_vivit_base(*args, **kwargs) -> ViViT:
    temporal_num_heads = kwargs.pop("temporal_num_heads", 12)
    temporal_depth = kwargs.pop("temporal_depth", 4)
    return create_vivit(
        *args,
        embed_dim=768,
        spatial_num_heads=12,
        temporal_num_heads=temporal_num_heads,
        spatial_depth=12,
        temporal_depth=temporal_depth,
        **kwargs,
    )
