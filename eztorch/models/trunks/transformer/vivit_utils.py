import re

import numpy as np
import torch
import torch.nn as nn
from einops import repeat
from lightning.pytorch.utilities import rank_zero_info
from timm.models.helpers import adapt_input_conv
from timm.models.vision_transformer import resize_pos_embed


def get_sine_cosine_pos_emb(n_position: int, d_hid: int) -> torch.Tensor:
    """Sinusoid position encoding table."""
    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


@torch.no_grad()
def init_from_npz_vit(
    model: nn.Module,
    checkpoint_path: str,
    conv_type: str,
    copy_strategy: str,
    extend_strategy: str,
    tube_size: int,
    prefix: str = "",
):
    """Init the ViViT model from a ViT pretrained model.

    Args:
        model: The ViViT model.
        checkpoint_path: Path to the pretrained model.
        conv_type: Convolution type of the ViViT patch embedder.
        copy_strategy: Strategy to copy the weights for the temporal transformer.
        extend_strategy: Extend strategy for the convolution patch embedder in case it is 3D.
        tube_size: Size of the tube for making tubelets.
        prefix: Prefix for the checkpoint.
    """

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and "opt/target/embedding/kernel" in w:
        prefix = "opt/target/"

    embed_conv_w = adapt_input_conv(
        model.patch_embed.proj.weight.shape[1], _n2p(w[f"{prefix}embedding/kernel"])
    )

    new_embed_conv_w = embed_conv_w
    if conv_type == "Conv3d":
        new_embed_conv_w = repeat(embed_conv_w, "d c h w -> d c t h w", t=tube_size)
        if extend_strategy == "temporal_avg":
            new_embed_conv_w = new_embed_conv_w / tube_size
        elif extend_strategy == "center_frame":
            new_embed_conv_w.zero_()
            new_embed_conv_w[:, :, tube_size // 2, :, :] = embed_conv_w

    model.patch_embed.proj.weight.copy_(new_embed_conv_w)

    model.patch_embed.proj.bias.copy_(_n2p(w[f"{prefix}embedding/bias"]))

    if model.spatial_class_token:
        model.spatial_cls_token.copy_(_n2p(w[f"{prefix}cls"], t=False))
    if model.temporal_class_token:
        if copy_strategy == "set_zero":
            model.temporal_cls_token.copy_(_n2p(w[f"{prefix}cls"], t=False)).zero_()
        elif copy_strategy == "repeat":
            model.temporal_cls_token.copy_(_n2p(w[f"{prefix}cls"], t=False))

    pos_embed_w = _n2p(w[f"{prefix}Transformer/posembed_input/pos_embedding"], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w,
            model.pos_embed,
            getattr(model, "num_prefix_tokens", 1),
            model.patch_embed.grid_size,
        )
    model.pos_embed.copy_(pos_embed_w)
    model.spatial_transformer.norm.weight.copy_(
        _n2p(w[f"{prefix}Transformer/encoder_norm/scale"])
    )
    model.spatial_transformer.norm.bias.copy_(
        _n2p(w[f"{prefix}Transformer/encoder_norm/bias"])
    )
    if copy_strategy == "set_zero":
        model.temporal_transformer.norm.weight.copy_(
            model.spatial_transformer.norm.weight
        ).zero_()
        model.temporal_transformer.norm.bias.copy_(
            model.spatial_transformer.norm.bias
        ).zero_()
    elif copy_strategy == "repeat":
        model.temporal_transformer.norm.weight.copy_(
            model.spatial_transformer.norm.weight
        )
        model.temporal_transformer.norm.bias.copy_(model.spatial_transformer.norm.bias)

    # NOTE representation layer has been removed, not used in latest 21k/1k pretrained weights
    # if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
    #     model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
    #     model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.spatial_transformer.blocks.children()):
        block_prefix = f"{prefix}Transformer/encoderblock_{i}/"
        mha_prefix = block_prefix + "MultiHeadDotProductAttention_1/"
        block.norm1.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/scale"]))
        block.norm1.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/bias"]))
        block.attn.qkv.weight.copy_(
            torch.cat(
                [
                    _n2p(w[f"{mha_prefix}{n}/kernel"], t=False).flatten(1).T
                    for n in ("query", "key", "value")
                ]
            )
        )
        block.attn.qkv.bias.copy_(
            torch.cat(
                [
                    _n2p(w[f"{mha_prefix}{n}/bias"], t=False).reshape(-1)
                    for n in ("query", "key", "value")
                ]
            )
        )
        block.attn.proj.weight.copy_(_n2p(w[f"{mha_prefix}out/kernel"]).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f"{mha_prefix}out/bias"]))
        for r in range(2):
            getattr(block.mlp, f"fc{r + 1}").weight.copy_(
                _n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/kernel"])
            )
            getattr(block.mlp, f"fc{r + 1}").bias.copy_(
                _n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/bias"])
            )
        block.norm2.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_2/scale"]))
        block.norm2.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_2/bias"]))
    for i, block in enumerate(model.temporal_transformer.blocks.children()):
        if copy_strategy == "set_zero":
            block_prefix = f"{prefix}Transformer/encoderblock_{i}/"
            mha_prefix = block_prefix + "MultiHeadDotProductAttention_1/"
            block.norm1.weight.copy_(
                _n2p(w[f"{block_prefix}LayerNorm_0/scale"])
            ).zero_()
            block.norm1.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/bias"])).zero_()
            block.attn.qkv.weight.copy_(
                torch.cat(
                    [
                        _n2p(w[f"{mha_prefix}{n}/kernel"], t=False).flatten(1).T
                        for n in ("query", "key", "value")
                    ]
                )
            ).zero_()
            block.attn.qkv.bias.copy_(
                torch.cat(
                    [
                        _n2p(w[f"{mha_prefix}{n}/bias"], t=False).reshape(-1)
                        for n in ("query", "key", "value")
                    ]
                )
            ).zero_()
            block.attn.proj.weight.copy_(
                _n2p(w[f"{mha_prefix}out/kernel"]).flatten(1)
            ).zero_()
            block.attn.proj.bias.copy_(_n2p(w[f"{mha_prefix}out/bias"])).zero_()
            for r in range(2):
                getattr(block.mlp, f"fc{r + 1}").weight.copy_(
                    _n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/kernel"])
                ).zero_()
                getattr(block.mlp, f"fc{r + 1}").bias.copy_(
                    _n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/bias"])
                ).zero_()
            block.norm2.weight.copy_(
                _n2p(w[f"{block_prefix}LayerNorm_2/scale"])
            ).zero_()
            block.norm2.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_2/bias"])).zero_()
        elif copy_strategy == "repeat":
            block_prefix = f"{prefix}Transformer/encoderblock_{i}/"
            mha_prefix = block_prefix + "MultiHeadDotProductAttention_1/"
            block.norm1.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/scale"]))
            block.norm1.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/bias"]))
            block.attn.qkv.weight.copy_(
                torch.cat(
                    [
                        _n2p(w[f"{mha_prefix}{n}/kernel"], t=False).flatten(1).T
                        for n in ("query", "key", "value")
                    ]
                )
            )
            block.attn.qkv.bias.copy_(
                torch.cat(
                    [
                        _n2p(w[f"{mha_prefix}{n}/bias"], t=False).reshape(-1)
                        for n in ("query", "key", "value")
                    ]
                )
            )
            block.attn.proj.weight.copy_(_n2p(w[f"{mha_prefix}out/kernel"]).flatten(1))
            block.attn.proj.bias.copy_(_n2p(w[f"{mha_prefix}out/bias"]))
            for r in range(2):
                getattr(block.mlp, f"fc{r + 1}").weight.copy_(
                    _n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/kernel"])
                )
                getattr(block.mlp, f"fc{r + 1}").bias.copy_(
                    _n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/bias"])
                )
            block.norm2.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_2/scale"]))
            block.norm2.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_2/bias"]))


@torch.no_grad()
def init_from_torch_vit(
    model: nn.Module,
    checkpoint_path: str,
    conv_type: str,
    copy_strategy: str,
    extend_strategy: str,
    tube_size: int,
    prefix: str = "trunk.",
):
    """Init the ViViT model from a ViT pretrained model.

    Args:
        model: The ViViT model.
        checkpoint_path: Path to the pretrained model.
        conv_type: Convolution type of the ViViT patch embedder.
        copy_strategy: Strategy to copy the weights for the temporal transformer.
        extend_strategy: Extend strategy for the convolution patch embedder in case it is 3D.
        tube_size: Size of the tube for making tubelets.
        prefix: Prefix for the trunk keys in the checkpoint.
    """

    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    old_state_dict_keys = list(state_dict.keys())

    if prefix != "":
        for old_key in old_state_dict_keys:
            weight = state_dict.pop(old_key)
            if old_key.startswith(prefix):
                new_key = old_key[len(prefix) :]
                state_dict[new_key] = weight

    old_state_dict_keys = list(state_dict.keys())
    for old_key in old_state_dict_keys:
        if old_key == "patch_embed.proj.weight":
            weight = state_dict[old_key]
            if conv_type == "Conv3d" and weight.ndim == 4:
                new_weight = repeat(weight, "d c h w -> d c t h w", t=tube_size)
                if extend_strategy == "temporal_avg":
                    new_weight = new_weight / tube_size
                elif extend_strategy == "center_frame":
                    new_weight.zero_()
                    new_weight[:, :, tube_size // 2, :, :] = weight
                state_dict[old_key] = new_weight
        elif old_key.startswith("blocks."):
            weight = state_dict.pop(old_key)
            state_dict["spatial_transformer." + old_key] = weight
            pattern = re.compile(r"blocks.(\d+)")
            matchObj = pattern.findall(old_key)
            if len(matchObj) >= 1 and int(matchObj[0]) < model.temporal_depth:
                if copy_strategy == "repeat":
                    state_dict["temporal_transformer." + old_key] = weight.clone()
                elif copy_strategy == "set_zero":
                    state_dict[
                        "temporal_transformer." + old_key
                    ] = weight.clone().zero_()
        elif old_key.startswith("cls_token"):
            weight = state_dict.pop(old_key)
            if model.spatial_class_token:
                state_dict["spatial_" + old_key] = weight
            if model.temporal_class_token:
                if copy_strategy == "repeat":
                    state_dict["temporal_" + old_key] = weight.clone()
                elif copy_strategy == "set_zero":
                    state_dict["temporal_" + old_key] = weight.clone().zero_()
        elif old_key.startswith("norm."):
            weight = state_dict.pop(old_key)
            state_dict["spatial_transformer." + old_key] = weight
            if copy_strategy == "repeat":
                state_dict["temporal_transformer." + old_key] = weight.clone()
            elif copy_strategy == "set_zero":
                state_dict["temporal_transformer." + old_key] = weight.clone().zero_()

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    rank_zero_info(
        "Loaded vivit from pretrained torch model.\n"
        f"missing_keys:{missing_keys}\n "
        f"unexpected_keys:{unexpected_keys}"
    )


@torch.no_grad()
def init_from_vit_pretrain(
    model: nn.Module,
    pretrained: str,
    conv_type: str,
    copy_strategy: str,
    extend_strategy: str = "temporal_avg",
    tube_size: int = 2,
    trunk_prefix: str = "trunk.",
) -> None:
    """Init the ViViT model from a ViT pretrained model.

    Args:
        model: The ViViT model.
        pretrained: Path to the pretrained model.
        conv_type: Convolution type of the ViViT patch embedder.
        copy_strategy: Strategy to copy the weights for the temporal transformer.
        extend_strategy: Extend strategy for the convolution patch embedder in case it is 3D.
        tube_size: Size of the tube for making tubelets.
        trunk_prefix: Prefix used to store the checkpoint only for torch for the keys.
    """

    if isinstance(pretrained, str):
        if pretrained[-4:] == ".npz":
            init_from_npz_vit(
                model=model,
                checkpoint_path=pretrained,
                conv_type=conv_type,
                copy_strategy=copy_strategy,
                extend_strategy=extend_strategy,
                tube_size=tube_size,
                prefix="",
            )

        else:
            init_from_torch_vit(
                model=model,
                checkpoint_path=pretrained,
                conv_type=conv_type,
                copy_strategy=copy_strategy,
                extend_strategy=extend_strategy,
                tube_size=tube_size,
                prefix=trunk_prefix,
            )

        rank_zero_info(
            f"Model loaded with {pretrained} checkpoint using the '{copy_strategy}' copy strategy and the '{extend_strategy}' extend strategy."
        )
        return
