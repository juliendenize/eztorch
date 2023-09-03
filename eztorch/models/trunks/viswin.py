from functools import partial
from typing import Tuple

import torch
from einops import rearrange, reduce, repeat
from timm.models.helpers import named_apply
from timm.models.vision_transformer import get_init_weights_vit
from torch import Tensor, nn
from torch.nn import LayerNorm, Module, Parameter

from eztorch.models.trunks.swin import Swin
from eztorch.models.trunks.transformer.patch_embed import PatchEmbed
from eztorch.models.trunks.transformer.viswin_utils import \
    init_from_swin_pretrain
from eztorch.models.trunks.transformer.vivit_utils import \
    get_sine_cosine_pos_emb
from eztorch.models.trunks.vit import AttentionViT
from eztorch.utils.mask import batch_mask_tube_in_sequence


class ViSwin(Module):
    """ViSwin.

    Args:
        num_frames: Number of frames in the video.
        img_size: Size of input image.
        patch_size: Size of one patch.
        in_chans: Channel num of input features.
        embed_dim: Dimensions of embedding.
        norm_layer: Config for norm layers.
        spatial_depths: Depths of the spatial transformer layers.
        spatial_num_heads: Number of spatial parallel attention heads.
        window_size: Size of the window for Swin attention.
        temporal_class_token: Whether to use class token for temporal encoder.
        use_learnable_time_emb: Whether to use learnable temporal embeddings.
        temporal_num_heads: Number of temporal parallel attention heads.
        temporal_depth: Number of temporal transformer layers.
        temporal_mask_token: Whether to use temporal mask tokens.
        temporal_mask_ratio: Ratio of temporal masking from ViViT.
        temporal_mask_tube: Tube size of temporal masking from ViViT
        dropout_p: Probability of dropout spatial and temporal layer paths.
        dropout_rate: Probability of dropout layer.
        time_dropout_rate: Probability of time dropout.
        attention_dropout_rate: Probability of attention dropout layer.
        tube_size: Dimension of the kernel size in Conv3d.
        conv_type: Type of the convolution in PatchEmbed layer.
        patch_norm: Whether to normalize the patches.
        freeze_spatial: Whether to freeze the spatial encoder.
    """

    def __init__(
        self,
        num_frames: int,
        img_size: int | Tuple[int, int] = 224,
        patch_size: int | Tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: Module | None = None,
        spatial_depths: Tuple[int] = (2, 2, 6, 2),
        spatial_num_heads: Tuple[int] = (3, 6, 12, 24),
        window_size: int = 7,
        temporal_class_token: bool = False,
        use_learnable_time_emb: bool = False,
        temporal_num_heads: int = 12,
        temporal_depth: int = 4,
        temporal_mask_token: bool = False,
        temporal_mask_ratio: float = 0.0,
        temporal_mask_tube: int = 2,
        dropout_p: int = 0.0,
        dropout_rate: float = 0.0,
        time_dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        tube_size: int = 2,
        conv_type: str = "Conv3d",
        patch_norm: bool = True,
        freeze_spatial: bool = False,
        **kwargs,
    ):
        super().__init__()

        norm_layer = norm_layer or partial(LayerNorm, eps=1e-6)

        num_frames = num_frames // tube_size
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.spatial_depths = spatial_depths
        self.temporal_depth = temporal_depth
        self.conv_type = conv_type
        self.tube_size = tube_size
        self.use_learnable_time_emb = use_learnable_time_emb
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
            norm_layer=norm_layer if patch_norm else None,
        )

        temp_dim = int(embed_dim * 2 ** (len(spatial_depths) - 1))

        self.pos_drop = nn.Dropout(p=dropout_rate)

        self.spatial_transformer = Swin(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            global_pool="",
            embed_dim=embed_dim,
            depths=spatial_depths,
            num_heads=spatial_num_heads,
            window_size=window_size,
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=dropout_rate,
            attn_drop_rate=attention_dropout_rate,
            drop_path_rate=dropout_p,
            norm_layer=norm_layer,
            ape=False,
            patch_norm=False,
            weight_init="skip",
            conv_type="identity",
            tube_size=tube_size,
        )

        self.time_drop = nn.Dropout(p=time_dropout_rate)

        self.temporal_transformer = AttentionViT(
            embed_dim=temp_dim,
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

        if temporal_class_token:
            self.temporal_cls_token = Parameter(torch.zeros(1, 1, temp_dim))

        if temporal_mask_token:
            self.temporal_mask_token = Parameter(torch.zeros(1, 1, temp_dim))

        num_prefix_temporal_tokens = 1 if temporal_class_token else 0
        num_frames = num_frames + num_prefix_temporal_tokens

        if use_learnable_time_emb:
            self.time_embed = Parameter(torch.zeros(1, num_frames, temp_dim))
        else:
            self.register_buffer(
                "time_embed", get_sine_cosine_pos_emb(num_frames, temp_dim)
            )

        self.init_weights()

        if freeze_spatial:
            for param in self.spatial_transformer.parameters():
                param.requires_grad = False
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            self.freeze_spatial_transformer()

    def init_weights(self):
        named_apply(get_init_weights_vit("", 0), self)
        if self.use_learnable_time_emb:
            nn.init.trunc_normal_(self.time_embed, std=0.02)
        if self.temporal_class_token:
            nn.init.trunc_normal_(self.temporal_cls_token, std=0.02)

    def prepare_tokens(self, x):
        # Tokenize
        b = x.shape[0]
        x = self.patch_embed(x)

        return x, b

    def forward_spatial(self, x: Tensor) -> Tuple[Tensor, int]:
        x, b = self.prepare_tokens(x)
        x = self.pos_drop(x)
        x = self.spatial_transformer(x)
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
                x_spatial, b = self.forward_spatial(x)
        else:
            x_spatial, b = self.forward_spatial(x)

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

        if self.use_learnable_time_emb:
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
            1 + self.spatial_transformer.num_layers if not self.freeze_spatial else 0
        ) + self.temporal_transformer.num_layers

    def get_param_layer_id(self, name: str) -> int:
        """Get the layer id of the named parameter.

        Args:
            name: The name of the parameter.
        """
        if name.startswith("patch_embed"):
            return 0
        elif name in ("time_embed", "temporal_cls_token"):
            return self.spatial_transformer.num_layers if not self.freeze_spatial else 0
        elif name.startswith("spatial_transformer."):
            return 1 + (
                self.spatial_transformer.get_param_layer_id(
                    name[len("spatial_transformer.") :]
                )
                if not self.freeze_spatial
                else 0
            )
        elif name.startswith("temporal_transformer."):
            return (
                1
                + (
                    self.spatial_transformer.num_layers
                    if not self.freeze_spatial
                    else 0
                )
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


def create_viswin(
    num_frames: int,
    img_size: int | Tuple[int, int] = 224,
    patch_size: int | Tuple[int, int] = 16,
    in_chans: int = 3,
    embed_dim: int = 96,
    norm_layer: Module | None = None,
    spatial_depths: Tuple[int] = (2, 2, 6, 2),
    spatial_num_heads: Tuple[int] = (3, 6, 12, 24),
    window_size: int = 7,
    temporal_class_token: bool = False,
    use_learnable_time_emb: bool = False,
    temporal_num_heads: int = 12,
    temporal_depth: int = 4,
    temporal_mask_token: bool = False,
    temporal_mask_ratio: float = 0.0,
    temporal_mask_tube: int = 2,
    dropout_p: int = 0.0,
    dropout_rate: float = 0.0,
    time_dropout_rate: float = 0.0,
    attention_dropout_rate: float = 0.0,
    tube_size: int = 2,
    conv_type: str = "Conv3d",
    patch_norm: bool = True,
    freeze_spatial: bool = False,
    pretrain_pth: str | None = None,
    weights_from: str = "spatial",
    copy_strategy: str = "repeat",
    extend_strategy: str = "temporal_avg",
    **kwargs,
) -> ViSwin:
    """Instantiate ViSwin.

    Args:
        num_frames: Number of frames in the video.
        img_size: Size of input image.
        patch_size: Size of one patch.
        in_chans: Channel num of input features.
        embed_dim: Dimensions of embedding.
        norm_layer: Config for norm layers.
        spatial_depths: Depths of the spatial transformer layers.
        spatial_num_heads: Number of spatial parallel attention heads.
        window_size: Size of the window for Swin attention.
        temporal_class_token: Whether to use class token for temporal encoder.
        use_learnable_time_emb: Whether to use learnable temporal embeddings.
        temporal_num_heads: Number of temporal parallel attention heads.
        temporal_depth: Number of temporal transformer layers.
        temporal_mask_token: Whether to use temporal mask tokens.
        temporal_mask_ratio: Ratio of temporal masking from ViViT.
        temporal_mask_tube: Tube size of temporal masking from ViViT
        dropout_p: Probability of dropout spatial and temporal layer paths.
        dropout_rate: Probability of dropout layer.
        time_dropout_rate: Probability of time dropout.
        attention_dropout_rate: Probability of attention dropout layer.
        tube_size: Dimension of the kernel size in Conv3d.
        conv_type: Type of the convolution in PatchEmbed layer.
        patch_norm: Whether to normalize the patches.
        freeze_spatial: Whether to freeze the spatial encoder.

    Returns:
            The ViSwin model
    """

    model = ViSwin(
        num_frames=num_frames,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        norm_layer=norm_layer,
        spatial_depths=spatial_depths,
        spatial_num_heads=spatial_num_heads,
        window_size=window_size,
        temporal_class_token=temporal_class_token,
        use_learnable_time_emb=use_learnable_time_emb,
        temporal_num_heads=temporal_num_heads,
        temporal_depth=temporal_depth,
        temporal_mask_token=temporal_mask_token,
        temporal_mask_ratio=temporal_mask_ratio,
        temporal_mask_tube=temporal_mask_tube,
        dropout_p=dropout_p,
        dropout_rate=dropout_rate,
        time_dropout_rate=time_dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        tube_size=tube_size,
        conv_type=conv_type,
        patch_norm=patch_norm,
        freeze_spatial=freeze_spatial,
        pretrain_pth=pretrain_pth,
        weights_from=weights_from,
        copy_strategy=copy_strategy,
        extend_strategy=extend_strategy,
        **kwargs,
    )

    if pretrain_pth is not None:
        if weights_from == "spatial":
            init_from_swin_pretrain(
                model=model,
                pretrained=pretrain_pth,
                conv_type=conv_type,
                copy_strategy=copy_strategy,
                extend_strategy=extend_strategy,
                tube_size=tube_size,
            )
        else:
            raise TypeError(f"Do not support the weights_from: {weights_from}")

    return model


def create_viswin_tiny(*args, **kwargs) -> ViSwin:
    temporal_num_heads = kwargs.pop("temporal_num_heads", 3)
    return create_viswin(
        *args,
        patch_size=4,
        window_size=7,
        embed_dim=96,
        spatial_depths=(2, 2, 6, 2),
        spatial_num_heads=(3, 6, 12, 24),
        temporal_num_heads=temporal_num_heads,
        **kwargs,
    )


def create_viswin_small(*args, **kwargs) -> ViSwin:
    temporal_num_heads = kwargs.pop("temporal_num_heads", 6)
    return create_viswin(
        *args,
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 18, 2),
        spatial_num_heads=(3, 6, 12, 24),
        temporal_num_heads=temporal_num_heads,
        **kwargs,
    )


def create_viswin_base(*args, **kwargs) -> ViSwin:
    temporal_num_heads = kwargs.pop("temporal_num_heads", 16)
    return create_viswin(
        *args,
        patch_size=4,
        window_size=7,
        embed_dim=128,
        spatial_depths=(2, 2, 18, 2),
        spatial_num_heads=(4, 8, 16, 32),
        temporal_num_heads=temporal_num_heads,
        **kwargs,
    )
