from typing import Callable, Tuple

import torch
from pytorchvideo.models.head import create_res_basic_head
from pytorchvideo.models.net import Net
from pytorchvideo.models.r2plus1d import create_2plus1d_bottleneck_block
from pytorchvideo.models.resnet import create_res_stage
from pytorchvideo.models.stem import create_res_basic_stem
from torch.nn import AvgPool3d, BatchNorm3d, Module, ModuleList, ReLU, Softmax


def create_r2plus1d(
    *,
    # Input clip configs.
    input_channel: int = 3,
    # Model configs.
    model_depth: int = 50,
    model_num_class: int = 400,
    dropout_rate: float = 0.0,
    # Normalization configs.
    norm: Callable = BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    # Activation configs.
    activation: Callable = ReLU,
    # Stem configs.
    stem_dim_out: int = 64,
    stem_conv_kernel_size: Tuple[int] = (1, 7, 7),
    stem_conv_stride: Tuple[int] = (1, 2, 2),
    # Stage configs.
    stage_conv_a_kernel_size: Tuple[Tuple[int]] = (
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
    ),
    stage_conv_b_kernel_size: Tuple[Tuple[int]] = (
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
    ),
    stage_conv_b_num_groups: Tuple[int] = (1, 1, 1, 1),
    stage_conv_b_dilation: Tuple[Tuple[int]] = (
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
    ),
    stage_spatial_stride: Tuple[int] = (2, 2, 2, 2),
    stage_temporal_stride: Tuple[int] = (1, 1, 2, 2),
    stage_bottleneck: Tuple[Callable] = (
        create_2plus1d_bottleneck_block,
        create_2plus1d_bottleneck_block,
        create_2plus1d_bottleneck_block,
        create_2plus1d_bottleneck_block,
    ),
    # Head configs.
    head: Callable = create_res_basic_head,
    head_pool: Callable = AvgPool3d,
    head_pool_kernel_size: Tuple[int] = (4, 7, 7),
    head_output_size: Tuple[int] = (1, 1, 1),
    head_activation: Callable = Softmax,
    head_output_with_global_average: bool = True,
) -> Module:
    """Build the R(2+1)D network from:: A closer look at spatiotemporal convolutions for action recognition. Du
    Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, Manohar Paluri. CVPR 2018.

    R(2+1)D follows the ResNet style architecture including three parts: Stem,
    Stages and Head. The three parts are assembled in the following order:

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

        model_depth: The depth of the resnet.
        model_num_class: The number of classes for the video dataset.
        dropout_rate: Dropout rate.

        norm: A callable that constructs normalization layer.
        norm_eps: Normalization epsilon.
        norm_momentum: Normalization momentum.

        activation: A callable that constructs activation layer.

        stem_dim_out: Output channel size for stem.
        stem_conv_kernel_size: Convolutional kernel size(s) of stem.
        stem_conv_stride: Convolutional stride size(s) of stem.

        stage_conv_a_kernel_size: Convolutional kernel size(s) for conv_a.
        stage_conv_b_kernel_size: Convolutional kernel size(s) for conv_b.
        stage_conv_b_num_groups: Number of groups for groupwise convolution
            for conv_b. 1 for ResNet, and larger than 1 for ResNeXt.
        stage_conv_b_dilation: Dilation for 3D convolution for conv_b.
        stage_spatial_stride: The spatial stride for each stage.
        stage_temporal_stride: The temporal stride for each stage.
        stage_bottleneck: A callable that constructs bottleneck block layer
            for each stage. Examples include: :func:`create_bottleneck_block`,
            :func:`create_2plus1d_bottleneck_block`.

        head_pool: A callable that constructs resnet head pooling layer.
        head_pool_kernel_size: The pooling kernel size.
        head_output_size: The size of output tensor for head.
        head_activation: A callable that constructs activation layer.
        head_output_with_global_average: If ``True``, perform global averaging on
            the head output.

    Returns:
        Basic resnet.
    """

    torch._C._log_api_usage_once("PYTORCHVIDEO.model.create_r2plus1d")

    # Number of blocks for different stages given the model depth.
    _MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3)}

    # Given a model depth, get the number of blocks for each stage.
    assert (
        model_depth in _MODEL_STAGE_DEPTH.keys()
    ), f"{model_depth} is not in {_MODEL_STAGE_DEPTH.keys()}"
    stage_depths = _MODEL_STAGE_DEPTH[model_depth]

    blocks = []
    # Create stem for R(2+1)D.
    stem = create_res_basic_stem(
        in_channels=input_channel,
        out_channels=stem_dim_out,
        conv_kernel_size=stem_conv_kernel_size,
        conv_stride=stem_conv_stride,
        conv_padding=[size // 2 for size in stem_conv_kernel_size],
        pool=None,
        norm=norm,
        activation=activation,
    )
    blocks.append(stem)

    stage_dim_in = stem_dim_out
    stage_dim_out = stage_dim_in * 4

    # Create each stage for R(2+1)D.
    for idx in range(len(stage_depths)):
        stage_dim_inner = stage_dim_out // 4
        depth = stage_depths[idx]

        stage_conv_b_stride = (
            stage_temporal_stride[idx],
            stage_spatial_stride[idx],
            stage_spatial_stride[idx],
        )

        stage = create_res_stage(
            depth=depth,
            dim_in=stage_dim_in,
            dim_inner=stage_dim_inner,
            dim_out=stage_dim_out,
            bottleneck=stage_bottleneck[idx],
            conv_a_kernel_size=stage_conv_a_kernel_size[idx],
            conv_a_stride=[1, 1, 1],
            conv_a_padding=[size // 2 for size in stage_conv_a_kernel_size[idx]],
            conv_b_kernel_size=stage_conv_b_kernel_size[idx],
            conv_b_stride=stage_conv_b_stride,
            conv_b_padding=[size // 2 for size in stage_conv_b_kernel_size[idx]],
            conv_b_num_groups=stage_conv_b_num_groups[idx],
            conv_b_dilation=stage_conv_b_dilation[idx],
            norm=norm,
            activation=activation,
        )

        blocks.append(stage)
        stage_dim_in = stage_dim_out
        stage_dim_out = stage_dim_out * 2

    if head is not None:
        # Create head for R(2+1)D.
        head = head(
            in_features=stage_dim_in,
            out_features=model_num_class,
            pool=head_pool,
            output_size=head_output_size,
            pool_kernel_size=head_pool_kernel_size,
            dropout_rate=dropout_rate,
            activation=head_activation,
            output_with_global_average=head_output_with_global_average,
        )
        blocks.append(head)
    return Net(blocks=ModuleList(blocks))
