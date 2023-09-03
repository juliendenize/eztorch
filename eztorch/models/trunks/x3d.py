import math
from typing import Callable, Tuple

import numpy as np
import torch
from pytorchvideo.layers.swish import Swish
from pytorchvideo.layers.utils import round_repeats, round_width
from pytorchvideo.models.net import Net
from pytorchvideo.models.x3d import (create_x3d_bottleneck_block,
                                     create_x3d_head, create_x3d_res_stage,
                                     create_x3d_stem)
from torch.nn import BatchNorm3d, Module, ModuleList, ReLU, Softmax


def create_x3d(
    *,
    # Input clip configs.
    input_channel: int = 3,
    input_clip_length: int = 13,
    input_crop_size: int = 160,
    # Model configs.
    model_num_class: int = 400,
    dropout_rate: float = 0.5,
    width_factor: float = 2.0,
    depth_factor: float = 2.2,
    # Normalization configs.
    norm: Callable = BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    # Activation configs.
    activation: Callable = ReLU,
    # Stem configs.
    stem_dim_in: int = 12,
    stem_conv_kernel_size: Tuple[int] = (5, 3, 3),
    stem_conv_stride: Tuple[int] = (1, 2, 2),
    # Stage configs.
    stage_conv_kernel_size: Tuple[Tuple[int]] = (
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
    ),
    stage_spatial_stride: Tuple[int] = (2, 2, 2, 2),
    stage_temporal_stride: Tuple[int] = (1, 1, 1, 1),
    bottleneck: Callable = create_x3d_bottleneck_block,
    bottleneck_factor: float = 2.25,
    se_ratio: float = 0.0625,
    inner_act: Callable = Swish,
    # Head configs.
    head: Callable = create_x3d_head,
    head_dim_out: int = 2048,
    head_pool_act: Callable = ReLU,
    head_bn_lin5_on: bool = False,
    head_activation: Callable = Softmax,
    head_output_with_global_average: bool = True,
) -> Module:
    """X3D model builder. It builds a X3D network backbone, which is a ResNet.

    Christoph Feichtenhofer.
    "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730

    ::

                                         Input
                                           ↓
                                         Stem
                                           ↓
                                         Stage 1
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                         Stage N
                                           ↓
                                         Head

    Args:
        input_channel: Number of channels for the input video clip.
        input_clip_length: Length of the input video clip. Value for
            different models: X3D-XS: 4; X3D-S: 13; X3D-M: 16; X3D-L: 16.
        input_crop_size: Spatial resolution of the input video clip.
            Value for different models: X3D-XS: 160; X3D-S: 160; X3D-M: 224;
            X3D-L: 312.

        model_num_class: The number of classes for the video dataset.
        dropout_rate: Dropout rate.
        width_factor: Width expansion factor.
        depth_factor: Depth expansion factor. Value for different
            models: X3D-XS: 2.2; X3D-S: 2.2; X3D-M: 2.2; X3D-L: 5.0.

        norm: A callable that constructs normalization layer.
        norm_eps: Normalization epsilon.
        norm_momentum: Normalization momentum.

        activation: A callable that constructs activation layer.

        stem_dim_in: Input channel size for stem before expansion.
        stem_conv_kernel_size: Convolutional kernel size(s) of stem.
        stem_conv_stride: Convolutional stride size(s) of stem.

        stage_conv_kernel_size: Convolutional kernel size(s) for ``conv_b``.
        stage_spatial_stride: The spatial stride for each stage.
        stage_temporal_stride: The temporal stride for each stage.
        bottleneck_factor: Bottleneck expansion factor for the 3x3x3 conv.
        se_ratio: if > 0, apply SE to the 3x3x3 conv, with the SE
            channel dimensionality being se_ratio times the 3x3x3 conv dim.
        inner_act: Whether use Swish activation for ``act_b`` or not.

        head_dim_out: Output channel size of the X3D head.
        head_pool_act: A callable that constructs resnet pool activation
            layer such as ``ReLU``.
        head_bn_lin5_on: If ``True``, perform normalization on the features
            before the classifier.
        head_activation: A callable that constructs activation layer.
        head_output_with_global_average: If ``True``, perform global averaging on
            the head output.

    Returns:
        The X3D network.
    """

    torch._C._log_api_usage_once("PYTORCHVIDEO.model.create_x3d")

    blocks = []
    # Create stem for X3D.
    stem_dim_out = round_width(stem_dim_in, width_factor)
    stem = create_x3d_stem(
        in_channels=input_channel,
        out_channels=stem_dim_out,
        conv_kernel_size=stem_conv_kernel_size,
        conv_stride=stem_conv_stride,
        conv_padding=[size // 2 for size in stem_conv_kernel_size],
        norm=norm,
        norm_eps=norm_eps,
        norm_momentum=norm_momentum,
        activation=activation,
    )
    blocks.append(stem)

    # Compute the depth and dimension for each stage
    stage_depths = [1, 2, 5, 3]
    exp_stage = 2.0
    stage_dim1 = stem_dim_in
    stage_dim2 = round_width(stage_dim1, exp_stage, divisor=8)
    stage_dim3 = round_width(stage_dim2, exp_stage, divisor=8)
    stage_dim4 = round_width(stage_dim3, exp_stage, divisor=8)
    stage_dims = [stage_dim1, stage_dim2, stage_dim3, stage_dim4]

    dim_in = stem_dim_out
    # Create each stage for X3D.
    for idx in range(len(stage_depths)):
        dim_out = round_width(stage_dims[idx], width_factor)
        dim_inner = int(bottleneck_factor * dim_out)
        depth = round_repeats(stage_depths[idx], depth_factor)

        stage_conv_stride = (
            stage_temporal_stride[idx],
            stage_spatial_stride[idx],
            stage_spatial_stride[idx],
        )

        stage = create_x3d_res_stage(
            depth=depth,
            dim_in=dim_in,
            dim_inner=dim_inner,
            dim_out=dim_out,
            bottleneck=bottleneck,
            conv_kernel_size=stage_conv_kernel_size[idx],
            conv_stride=stage_conv_stride,
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            se_ratio=se_ratio,
            activation=activation,
            inner_act=inner_act,
        )
        blocks.append(stage)
        dim_in = dim_out

    # Create head for X3D.
    total_spatial_stride = stem_conv_stride[1] * np.prod(stage_spatial_stride)
    total_temporal_stride = stem_conv_stride[0] * np.prod(stage_temporal_stride)

    assert (
        input_clip_length >= total_temporal_stride
    ), "Clip length doesn't match temporal stride!"
    assert (
        input_crop_size >= total_spatial_stride
    ), "Crop size doesn't match spatial stride!"

    head_pool_kernel_size = (
        input_clip_length // total_temporal_stride,
        int(math.ceil(input_crop_size / total_spatial_stride)),
        int(math.ceil(input_crop_size / total_spatial_stride)),
    )

    if head is not None:
        head = head(
            dim_in=dim_out,
            dim_inner=dim_inner,
            dim_out=head_dim_out,
            num_classes=model_num_class,
            pool_act=head_pool_act,
            pool_kernel_size=head_pool_kernel_size,
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            bn_lin5_on=head_bn_lin5_on,
            dropout_rate=dropout_rate,
            activation=head_activation,
            output_with_global_average=head_output_with_global_average,
        )
        blocks.append(head)

    return Net(blocks=ModuleList(blocks))
