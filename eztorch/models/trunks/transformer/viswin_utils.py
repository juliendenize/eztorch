import torch
import torch.nn as nn
from einops import repeat
from lightning.pytorch.utilities import rank_zero_info


@torch.no_grad()
def init_from_npz_swin(
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
    raise NotImplementedError("Numpy checkpoint for swin not supported.")


@torch.no_grad()
def init_from_torch_swin(
    model: nn.Module,
    checkpoint_path: str,
    conv_type: str,
    copy_strategy: str,
    extend_strategy: str,
    tube_size: int,
    prefix: str = "trunk.",
):
    """Init the ViSwin model from a Swin pretrained model.

    Args:
        model: The ViSwin model.
        checkpoint_path: Path to the pretrained model.
        conv_type: Convolution type of the ViSwin patch embedder.
        copy_strategy: Strategy to copy the weights for the temporal transformer.
        extend_strategy: Extend strategy for the convolution patch embedder in case it is 3D.
        tube_size: Size of the tube for making tubelets.
        prefix: Prefix for the trunk keys in the checkpoint.
    """

    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if copy_strategy not in [""]:
        raise NotImplementedError(f"{copy_strategy} not supported.")
    if extend_strategy not in ["temporal_avg", "center_frame", ""]:
        raise NotImplementedError(f"{copy_strategy} not supported.")

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
        elif old_key.startswith("layers."):
            weight = state_dict.pop(old_key)
            state_dict["spatial_transformer." + old_key] = weight
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
def init_from_swin_pretrain(
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
            init_from_npz_swin(
                model=model,
                checkpoint_path=pretrained,
                conv_type=conv_type,
                copy_strategy=copy_strategy,
                extend_strategy=extend_strategy,
                tube_size=tube_size,
                prefix="",
            )

        else:
            init_from_torch_swin(
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
