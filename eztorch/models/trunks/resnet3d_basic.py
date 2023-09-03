from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from pytorchvideo.layers.utils import set_attributes
from pytorchvideo.models.head import create_res_basic_head
from pytorchvideo.models.net import Net
from pytorchvideo.models.resnet import (_MODEL_STAGE_DEPTH, ResStage,
                                        _trivial_sum)
from pytorchvideo.models.stem import create_res_basic_stem
from torch.nn import (AvgPool3d, BatchNorm3d, Conv3d, MaxPool3d, Module,
                      ModuleList, ReLU)


def create_basic_block(
    *,
    # Convolution configs.
    dim_in: int,
    dim_out: int,
    conv_a_kernel_size: Tuple[int] = (3, 3, 3),
    conv_a_stride: Tuple[int] = (2, 2, 2),
    conv_a_padding: Tuple[int] = (1, 1, 1),
    conv_a: Callable = Conv3d,
    conv_b_kernel_size: Tuple[int] = (3, 3, 3),
    conv_b_stride: Tuple[int] = (2, 2, 2),
    conv_b_padding: Tuple[int] = (1, 1, 1),
    conv_b: Callable = Conv3d,
    # Norm configs.
    norm: Callable = BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    # Activation configs.
    activation: Callable = ReLU,
) -> Module:
    """
    Basic block: a sequence of spatiotemporal Convolution, Normalization,
    and Activations repeated in the following order:

    ::

                                    Conv3d (conv_a)
                                           ↓
                                 Normalization (norm_a)
                                           ↓
                                   Activation (act_a)
                                           ↓
                                    Conv3d (conv_b)
                                           ↓
                                 Normalization (norm_b)


    Normalization examples include: ``BatchNorm3d`` and ``None`` (no normalization).
    Activation examples include: ``ReLU``, ``Softmax``, ``Sigmoid``, and ``None`` (no activation).

    Args:
        dim_in: Input channel size to the basicblock block.
        dim_inner: Intermediate channel size of the basicblock.
        dim_out: Output channel size of the basicblock.
        basicblock: A callable that constructs basicblock block layer.
            Examples include: :func:`create_basicblock_block`.
        conv_a_kernel_size: Convolutional kernel size(s) for ``conv_a``.
        conv_a_stride: Convolutional stride size(s) for ``conv_a``.
        conv_a_padding: Convolutional padding(s) for ``conv_a``.
        conv_a: A callable that constructs the ``conv_a`` conv layer, examples
            include ``Conv3d``, ``OctaveConv``, etc
        conv_b_kernel_size: Convolutional kernel size(s) for ``conv_b``.
        conv_b_stride: Convolutional stride size(s) for ``conv_b``.
        conv_b_padding: Convolutional padding(s) for ``conv_b``.
        conv_b: A callable that constructs the ``conv_b`` conv layer, examples
            include ``Conv3d``, ``OctaveConv``, etc

        norm: A callable that constructs normalization layer, examples
            include ``BatchNorm3d``, ``None`` (not performing normalization).
        norm_eps: Normalization epsilon.
        norm_momentum: Normalization momentum.

        activation_basicblock: A callable that constructs activation layer, examples
            include: ``ReLU``, ``Softmax``, ``Sigmoid``, and ``None`` (not performing
            activation).
        activation_block: A callable that constructs activation layer used
            at the end of the block. Examples include: ``ReLU``, ``Softmax``, ``Sigmoid``,
            and None (not performing activation).

    Returns:
        Resnet basicblock block.
    """
    conv_a = conv_a(
        in_channels=dim_in,
        out_channels=dim_out,
        kernel_size=conv_a_kernel_size,
        stride=conv_a_stride,
        padding=conv_a_padding,
        bias=False,
    )
    norm_a = (
        None
        if norm is None
        else norm(num_features=dim_out, eps=norm_eps, momentum=norm_momentum)
    )
    act_a = None if activation is None else activation()

    conv_b = conv_b(
        in_channels=dim_out,
        out_channels=dim_out,
        kernel_size=conv_b_kernel_size,
        stride=conv_b_stride,
        padding=conv_b_padding,
        bias=False,
    )
    norm_b = (
        None
        if norm is None
        else norm(num_features=dim_out, eps=norm_eps, momentum=norm_momentum)
    )

    return BasicBlock(
        conv_a=conv_a,
        norm_a=norm_a,
        act_a=act_a,
        conv_b=conv_b,
        norm_b=norm_b,
    )


def create_res_basic_block(
    *,
    # Bottleneck Block configs.
    dim_in: int,
    dim_out: int,
    basicblock: Callable,
    use_shortcut: bool = False,
    branch_fusion: Callable = _trivial_sum,
    # Conv configs.
    conv_a_kernel_size: Tuple[int] = (3, 3, 3),
    conv_a_stride: Tuple[int] = (2, 2, 2),
    conv_a_padding: Tuple[int] = (1, 1, 1),
    conv_a: Callable = Conv3d,
    conv_b_kernel_size: Tuple[int] = (3, 3, 3),
    conv_b_stride: Tuple[int] = (2, 2, 2),
    conv_b_padding: Tuple[int] = (1, 1, 1),
    conv_b: Callable = Conv3d,
    conv_skip: Callable = Conv3d,
    # Norm configs.
    norm: Callable = BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    # Activation configs.
    activation_basicblock: Callable = ReLU,
    activation_block: Callable = ReLU,
) -> Module:
    """
    Residual block. Performs a summation between an identity shortcut in branch1 and a
    main block in branch2. When the input and output dimensions are different, a
    convolution followed by a normalization will be performed.

    ::


                                         Input
                                           |-------+
                                           ↓       |
                                         Block     |
                                           ↓       |
                                       Summation ←-+
                                           ↓
                                       Activation

    Normalization examples include: ``BatchNorm3d`` and ``None`` (no normalization).
    Activation examples include: ``ReLU``, ``Softmax``, ``Sigmoid``, and ``None`` (no activation).
    Transform examples include: :class:`BottleneckBlock`.

    Args:
        dim_in: Input channel size to the bottleneck block.
        dim_out: Output channel size of the bottleneck.
        bottleneck: A callable that constructs bottleneck block layer.
            Examples include: create_bottleneck_block.
        use_shortcut: If ``True``, use conv and norm layers in skip connection.
        branch_fusion: A callable that constructs summation layer.
            Examples include: lambda x, y: x + y, OctaveSum.

        conv_a_kernel_size: Convolutional kernel size(s) for ``conv_a``.
        conv_a_stride: Convolutional stride size(s) for ``conv_a``.
        conv_a_padding: Convolutional padding(s) for ``conv_a``.
        conv_a: A callable that constructs the ``conv_a`` conv layer, examples
            include ``Conv3d``, ``OctaveConv``, etc
        conv_b_kernel_size: Convolutional kernel size(s) for ``conv_b``.
        conv_b_stride: Convolutional stride size(s) for ``conv_b``.
        conv_b_padding: Convolutional padding(s) for ``conv_b``.
        conv_b: A callable that constructs the ``conv_b`` conv layer, examples
            include ``Conv3d``, ``OctaveConv``, etc
        conv_skip: A callable that constructs the ``conv_skip`` conv layer,
        examples include ``Conv3d``, ``OctaveConv``, etc

        norm: A callable that constructs normalization layer. Examples
            include BatchNorm3d, None (not performing normalization).
        norm_eps: Normalization epsilon.
        norm_momentum: Normalization momentum.

        activation_basicblock: A callable that constructs activation layer in
            basicblock. Examples include: ``ReLU``, ``Softmax``, ``Sigmoid``, and ``None``
            (not performing activation).
        activation_block: A callable that constructs activation layer used
            at the end of the block. Examples include: ``ReLU``, ``Softmax``, ``Sigmoid``,
            and ``None`` (not performing activation).

    Returns:
        Resnet basic block layer.
    """
    branch1_conv_stride = tuple(map(np.prod, zip(conv_a_stride, conv_b_stride)))
    norm_model = None
    if use_shortcut or (
        norm is not None and (dim_in != dim_out or np.prod(branch1_conv_stride) != 1)
    ):
        norm_model = norm(num_features=dim_out, eps=norm_eps, momentum=norm_momentum)

    return ResBlock(
        branch1_conv=conv_skip(
            dim_in,
            dim_out,
            kernel_size=(1, 1, 1),
            stride=branch1_conv_stride,
            bias=False,
        )
        if (dim_in != dim_out or np.prod(branch1_conv_stride) != 1) or use_shortcut
        else None,
        branch1_norm=norm_model,
        branch2=basicblock(
            dim_in=dim_in,
            dim_out=dim_out,
            conv_a_kernel_size=conv_a_kernel_size,
            conv_a_stride=conv_a_stride,
            conv_a_padding=conv_a_padding,
            conv_a=conv_a,
            conv_b_kernel_size=conv_b_kernel_size,
            conv_b_stride=conv_b_stride,
            conv_b_padding=conv_b_padding,
            conv_b=conv_b,
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            activation=activation_basicblock,
        ),
        activation=None if activation_block is None else activation_block(),
        branch_fusion=branch_fusion,
    )


def create_res_basic_stage(
    *,
    # Stage configs.
    depth: int,
    # basicblock Block configs.
    dim_in: int,
    dim_out: int,
    basicblock: Callable,
    # Conv configs.
    conv_a_kernel_size: Union[Tuple[int], List[Tuple[int]]] = (3, 3, 3),
    conv_a_stride: Tuple[int] = (2, 2, 2),
    conv_a_padding: Union[Tuple[int], List[Tuple[int]]] = (1, 1, 1),
    conv_a: Callable = Conv3d,
    conv_b_kernel_size: Tuple[int] = (3, 3, 3),
    conv_b_stride: Tuple[int] = (2, 2, 2),
    conv_b_padding: Tuple[int] = (1, 1, 1),
    conv_b: Callable = Conv3d,
    # Norm configs.
    norm: Callable = BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    # Activation configs.
    activation: Callable = ReLU,
) -> Module:
    """
    Create Residual Stage, which composes sequential blocks that make up a ResNet. These
    blocks could be, for example, Residual blocks, Non-Local layers, or
    Squeeze-Excitation layers.

    ::


                                        Input
                                           ↓
                                       ResBlock
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                       ResBlock

    Normalization examples include: ``BatchNorm3d`` and ``None`` (no normalization).
    Activation examples include: ``ReLU``, ``Softmax``, ``Sigmoid``, and ``None`` (no activation).
    basicblock examples include: :func:`create_basicblock_block`.

    Args:
        depth: Number of blocks to create.

        dim_in: Input channel size to the basicblock block.
        dim_out: Output channel size of the basicblock.
        basicblock: A callable that constructs basicblock block layer.
            Examples include: :func:`create_basicblock_block`.

        conv_a_kernel_size: Convolutional kernel size(s)
            for conv_a. If ``conv_a_kernel_size`` is a tuple, use it for all blocks in
            the stage. If ``conv_a_kernel_size`` is a list of tuple, the kernel sizes
            will be repeated until having same length of depth in the stage. For
            example, for conv_a_kernel_size = [(3, 1, 1), (1, 1, 1)], the kernel
            size for the first 6 blocks would be [(3, 1, 1), (1, 1, 1), (3, 1, 1),
            (1, 1, 1), (3, 1, 1)].
        conv_a_stride: Convolutional stride size(s) for ``conv_a``.
        conv_a_padding: Convolutional padding(s) for
            ``conv_a``. If ``conv_a_padding`` is a tuple, use it for all blocks in
            the stage. If ``conv_a_padding`` is a list of tuple, the padding sizes
            will be repeated until having same length of depth in the stage.
        conv_a: A callable that constructs the conv_a conv layer, examples
            include Conv3d, OctaveConv, etc
        conv_b_kernel_size: Convolutional kernel size(s) for ``conv_b``.
        conv_b_stride: Convolutional stride size(s) for ``conv_b``.
        conv_b_padding: Convolutional padding(s) for ``conv_b``.
        conv_b: A callable that constructs the ``conv_b`` conv layer, examples
            include Conv3d, OctaveConv, etc

        norm: A callable that constructs normalization layer. Examples
            include ``BatchNorm3d``, and ``None`` (not performing normalization).
        norm_eps: Normalization epsilon.
        norm_momentum: Normalization momentum.

        activation: A callable that constructs activation layer. Examples
            include: ``ReLU``, ``Softmax``, ``Sigmoid``, and ``None`` (not performing
            activation).

    Returns:
        resnet basic stage layer.
    """
    res_blocks = []
    if isinstance(conv_a_kernel_size[0], int):
        conv_a_kernel_size = [conv_a_kernel_size]
    if isinstance(conv_a_padding[0], int):
        conv_a_padding = [conv_a_padding]
    # Repeat conv_a kernels until having same length of depth in the stage.
    conv_a_kernel_size = (conv_a_kernel_size * depth)[:depth]
    conv_a_padding = (conv_a_padding * depth)[:depth]

    for ind in range(depth):
        block = create_res_basic_block(
            dim_in=dim_in if ind == 0 else dim_out,
            dim_out=dim_out,
            basicblock=basicblock,
            conv_a_kernel_size=conv_a_kernel_size[ind],
            conv_a_stride=conv_a_stride if ind == 0 else (1, 1, 1),
            conv_a_padding=conv_a_padding[ind],
            conv_a=conv_a,
            conv_b_kernel_size=conv_b_kernel_size,
            conv_b_stride=conv_b_stride if ind == 0 else (1, 1, 1),
            conv_b_padding=conv_b_padding,
            conv_b=conv_b,
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            activation_basicblock=activation,
            activation_block=activation,
        )
        res_blocks.append(block)
    return ResStage(res_blocks=ModuleList(res_blocks))


def create_resnet3d_basic(
    *,
    # Input clip configs.
    input_channel: int = 3,
    # Model configs.
    model_depth: int = 50,
    model_num_class: int = 400,
    dropout_rate: float = 0.5,
    # Normalization configs.
    norm: Callable = BatchNorm3d,
    # Activation configs.
    activation: Callable = ReLU,
    # Stem configs.
    stem_activation: Optional[Callable] = ReLU,
    stem_dim_out: int = 64,
    stem_conv_kernel_size: Tuple[int] = (1, 7, 7),
    stem_conv_stride: Tuple[int] = (1, 2, 2),
    stem_pool: Optional[Callable] = MaxPool3d,
    stem_pool_kernel_size: Tuple[int] = (1, 3, 3),
    stem_pool_stride: Tuple[int] = (1, 2, 2),
    stem: Optional[Callable] = create_res_basic_stem,
    # Stage configs.
    stage1_pool: Callable = None,
    stage1_pool_kernel_size: Tuple[int] = (2, 1, 1),
    stage_conv_a_kernel_size: Union[Tuple[int], Tuple[Tuple[int]]] = (
        (1, 3, 3),
        (1, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
    ),
    stage_conv_b_kernel_size: Union[Tuple[int], Tuple[Tuple[int]]] = (
        (1, 3, 3),
        (1, 3, 3),
        (1, 3, 3),
        (1, 3, 3),
    ),
    stage_spatial_h_stride: Tuple[int] = (1, 2, 2, 2),
    stage_spatial_w_stride: Tuple[int] = (1, 2, 2, 2),
    stage_temporal_stride: Tuple[int] = (1, 1, 1, 1),
    basicblock: Union[Tuple[Callable], Callable] = create_basic_block,
    # Head configs.
    head: Callable = create_res_basic_head,
    head_pool: Callable = AvgPool3d,
    head_pool_kernel_size: Tuple[int] = (4, 7, 7),
    head_output_size: Tuple[int] = (1, 1, 1),
    head_activation: Callable = None,
    head_output_with_global_average: bool = True,
) -> Module:
    """
    Build ResNet style models for video recognition. ResNet has three parts:
    Stem, Stages and Head. Stem is the first Convolution layer (Conv1) with an
    optional pooling layer. Stages are grouped residual blocks. There are usually
    multiple stages and each stage may include multiple residual blocks. Head
    may include pooling, dropout, a fully-connected layer and global spatial
    temporal averaging. The three parts are assembled in the following order:

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

        model_depth: The depth of the resnet. Options include: :math:`18, 50, 101, 152`.
        model_num_class: The number of classes for the video dataset.
        dropout_rate: Dropout rate.


        norm: A callable that constructs normalization layer.

        activation: A callable that constructs activation layer.

        stem_activation: A callable that constructs activation layer of stem.
        stem_dim_out: Output channel size to stem.
        stem_conv_kernel_size: Convolutional kernel size(s) of stem.
        stem_conv_stride: Convolutional stride size(s) of stem.
        stem_pool: A callable that constructs resnet head pooling layer.
        stem_pool_kernel_size: Pooling kernel size(s).
        stem_pool_stride: Pooling stride size(s).
        stem: A callable that constructs stem layer.
            Examples include: :func:`create_res_video_stem`.

        stage_conv_a_kernel_size: Convolutional kernel size(s) for ``conv_a``.
        stage_conv_b_kernel_size: Convolutional kernel size(s) for ``conv_b``.
        stage_spatial_h_stride: The spatial height stride for each stage.
        stage_spatial_w_stride: The spatial width stride for each stage.
        stage_temporal_stride: The temporal stride for each stage.
        basicblock: A callable that constructs basicblock block layer.
            Examples include: :func:`create_basicblock_block`.

        head: A callable that constructs the resnet-style head.
            Ex: create_res_basic_head
        head_pool: A callable that constructs resnet head pooling layer.
        head_pool_kernel_size: The pooling kernel size.
        head_output_size: The size of output tensor for head.
        head_activation: A callable that constructs activation layer.
        head_output_with_global_average: if ``True``, perform global averaging on
            the head output.

    Returns:
        Basic resnet.
    """

    torch._C._log_api_usage_once("PYTORCHVIDEO.model.create_resnet3d_basic")

    # Given a model depth, get the number of blocks for each stage.
    assert (
        model_depth in _MODEL_STAGE_DEPTH.keys()
    ), f"{model_depth} is not in {_MODEL_STAGE_DEPTH.keys()}"
    stage_depths = _MODEL_STAGE_DEPTH[model_depth]

    # Broadcast single element to tuple if given.
    if isinstance(stage_conv_a_kernel_size[0], int):
        stage_conv_a_kernel_size = (stage_conv_a_kernel_size,) * len(stage_depths)

    if isinstance(stage_conv_b_kernel_size[0], int):
        stage_conv_b_kernel_size = (stage_conv_b_kernel_size,) * len(stage_depths)

    if isinstance(basicblock, Callable):
        basicblock = [
            basicblock,
        ] * len(stage_depths)

    blocks = []
    # Create stem for resnet.
    stem = stem(
        in_channels=input_channel,
        out_channels=stem_dim_out,
        conv_kernel_size=stem_conv_kernel_size,
        conv_stride=stem_conv_stride,
        conv_padding=[size // 2 for size in stem_conv_kernel_size],
        pool=stem_pool,
        pool_kernel_size=stem_pool_kernel_size,
        pool_stride=stem_pool_stride,
        pool_padding=[size // 2 for size in stem_pool_kernel_size],
        norm=norm,
        activation=stem_activation,
    )
    blocks.append(stem)

    stage_dim_in = stem_dim_out
    stage_dim_out = stage_dim_in

    # Create each stage for resnet.
    for idx in range(len(stage_depths)):
        depth = stage_depths[idx]

        stage_conv_a_kernel = stage_conv_a_kernel_size[idx]
        stage_conv_a_stride = (
            stage_temporal_stride[idx],
            stage_spatial_h_stride[idx],
            stage_spatial_w_stride[idx],
        )
        stage_conv_a_padding = [1, 1, 1] if idx > 1 else [0, 1, 1]

        stage_conv_b_kernel = stage_conv_b_kernel_size[idx]
        stage_conv_b_stride = (1, 1, 1)
        stage_conv_b_padding = [0, 1, 1]

        stage = create_res_basic_stage(
            depth=depth,
            dim_in=stage_dim_in,
            dim_out=stage_dim_out,
            basicblock=basicblock[idx],
            conv_a_kernel_size=stage_conv_a_kernel,
            conv_a_stride=stage_conv_a_stride,
            conv_a_padding=stage_conv_a_padding,
            conv_b_kernel_size=stage_conv_b_kernel,
            conv_b_stride=stage_conv_b_stride,
            conv_b_padding=stage_conv_b_padding,
            norm=norm,
            activation=activation,
        )

        blocks.append(stage)
        stage_dim_in = stage_dim_out
        stage_dim_out = stage_dim_out * 2

        if idx == 0 and stage1_pool is not None:
            blocks.append(
                stage1_pool(
                    kernel_size=stage1_pool_kernel_size,
                    stride=stage1_pool_kernel_size,
                    padding=(0, 0, 0),
                )
            )
    if head is not None:
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


class ResBlock(Module):
    """
    Residual block. Performs a summation between an identity shortcut in branch1 and a
    main block in branch2. When the input and output dimensions are different, a
    convolution followed by a normalization will be performed.

    ::


                                         Input
                                           |-------+
                                           ↓       |
                                         Block     |
                                           ↓       |
                                       Summation ←-+
                                           ↓
                                       Activation

    The builder can be found in `create_res_block`.
    """

    def __init__(
        self,
        branch1_conv: Module = None,
        branch1_norm: Module = None,
        branch2: Module = None,
        activation: Module = None,
        branch_fusion: Callable = None,
    ) -> Module:
        """
        Args:
            branch1_conv: Convolutional module in branch1.
            branch1_norm: Normalization module in branch1.
            branch2: Basicblock block module in branch2.
            activation: Activation module.
            branch_fusion: A callable or layer that combines branch1
                and branch2.
        """
        super().__init__()
        set_attributes(self, locals())
        assert self.branch2 is not None

    def forward(self, x) -> torch.Tensor:
        if self.branch1_conv is None:
            x = self.branch_fusion(x, self.branch2(x))
        else:
            shortcut = self.branch1_conv(x)
            if self.branch1_norm is not None:
                shortcut = self.branch1_norm(shortcut)
            x = self.branch_fusion(shortcut, self.branch2(x))
        if self.activation is not None:
            x = self.activation(x)
        return x


class BasicBlock(Module):
    """
    basicblock block: a sequence of spatiotemporal Convolution, Normalization,
    and Activations repeated in the following order:

    ::


                                    Conv3d (conv_a)
                                           ↓
                                 Normalization (norm_a)
                                           ↓
                                   Activation (act_a)
                                           ↓
                                    Conv3d (conv_b)
                                           ↓
                                 Normalization (norm_b)


    The builder can be found in :func:`create_basicblock_block`.
    """

    def __init__(
        self,
        *,
        conv_a: Module = None,
        norm_a: Module = None,
        act_a: Module = None,
        conv_b: Module = None,
        norm_b: Module = None,
    ) -> None:
        """
        Args:
            conv_a: Convolutional module.
            norm_a: Normalization module.
            act_a: Activation module.
            conv_b: Convolutional module.
            norm_b: Normalization module.
        """
        super().__init__()
        set_attributes(self, locals())
        assert all(op is not None for op in (self.conv_a, self.conv_b))
        if self.norm_b is not None:
            # This flag is used for weight initialization.
            self.norm_b.block_final_bn = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Explicitly forward every layer.
        # Branch2a, for example Tx1x1, BN, ReLU.
        x = self.conv_a(x)
        if self.norm_a is not None:
            x = self.norm_a(x)
        if self.act_a is not None:
            x = self.act_a(x)
        # Branch2b, for example 1xHxW, BN, ReLU.
        x = self.conv_b(x)
        if self.norm_b is not None:
            x = self.norm_b(x)
        return x
