import math
from typing import List

import torch.nn as nn
from torch.nn import Module
from torch.nn.modules.utils import _triple


class SpatioTemporalConv(nn.Module):
    """Applies a factored 3D convolution over an input signal composed of several input planes with distinct
    spatial and time axes, by performing a 2D convolution over the spatial axes to an intermediate subspace,
    followed by a 1D convolution over the time axis to produce the final output.

    Args:
        in_chans: Number of channels in the input tensor
        out_channels: Number of channels produced by the convolution
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution.
        padding: Zero-padding added to the sides of the input during their respective convolutions.
        bias: If ``True``, adds a learnable bias to the output.
    """

    def __init__(
        self,
        in_chans,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=False,
        first_conv=False,
    ):
        super().__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size = (1, kernel_size[1], kernel_size[2])
        spatial_stride = (1, stride[1], stride[2])
        spatial_padding = (0, padding[1], padding[2])

        temporal_kernel_size = (kernel_size[0], 1, 1)
        temporal_stride = (stride[0], 1, 1)
        temporal_padding = (padding[0], 0, 0)

        # compute the number of intermediary channels (M) using formula
        # from the paper section 3.5
        intermed_channels = int(
            math.floor(
                (
                    kernel_size[0]
                    * kernel_size[1]
                    * kernel_size[2]
                    * in_chans
                    * out_channels
                )
                / (
                    kernel_size[1] * kernel_size[2] * in_chans
                    + kernel_size[0] * out_channels
                )
            )
        )
        # print(intermed_channels)

        # the spatial conv is effectively a 2D conv due to the
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = nn.Conv3d(
            in_chans,
            intermed_channels,
            spatial_kernel_size,
            stride=spatial_stride,
            padding=spatial_padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()

        # the temporal conv is effectively a 1D conv, but has batch norm
        # and ReLU added inside the model constructor, not here. This is an
        # intentional design choice, to allow this module to externally act
        # identical to a standard Conv3D, so it can be reused easily in any
        # other codebase
        self.temporal_conv = nn.Conv3d(
            intermed_channels,
            out_channels,
            temporal_kernel_size,
            stride=temporal_stride,
            padding=temporal_padding,
            bias=bias,
        )

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x


class SpatioTemporalResBlock(nn.Module):
    r"""Single block for the ResNet network.

    Uses SpatioTemporalConv in
    the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)
    Args:
        in_chans: Number of channels in the input tensor.
        out_channels: Number of channels in the output produced by the block.
        kernel_size: Size of the convolving kernels.
        downsample: If ``True``, the output size is to be smaller than the input.
    """

    def __init__(self, in_chans, out_channels, kernel_size, downsample=False):
        super().__init__()

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a separate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample

        # to allow for SAME padding
        padding = kernel_size // 2

        if self.downsample:
            # downsample with stride =2 the input x
            self.downsampleconv = SpatioTemporalConv(
                in_chans, out_channels, 1, stride=2
            )
            self.downsamplebn = nn.BatchNorm3d(out_channels)

            # downsample with stride = 2 when producing the residual
            self.conv1 = SpatioTemporalConv(
                in_chans, out_channels, kernel_size, padding=padding, stride=2
            )
        else:
            self.conv1 = SpatioTemporalConv(
                in_chans, out_channels, kernel_size, padding=padding
            )

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = SpatioTemporalConv(
            out_channels, out_channels, kernel_size, padding=padding
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.outrelu(x + res)


class SpatioTemporalResLayer(nn.Module):
    r"""Forms a single layer of the ResNet network, with a number of repeating
    blocks of same output size stacked on top of each other
        Args:
            in_chans: Number of channels in the input tensor.
            out_channels: Number of channels in the output produced by the layer.
            kernel_size: Size of the convolving kernels.
            layer_size: Number of blocks to be stacked to form the layer
            block_type: Type of block that is to be used to form the layer.
            downsample: If ``True``, the first block in layer will implement downsampling.
    """

    def __init__(
        self,
        in_chans,
        out_channels,
        kernel_size,
        layer_size,
        block_type=SpatioTemporalResBlock,
        downsample=False,
    ):

        super().__init__()

        # implement the first block
        self.block1 = block_type(in_chans, out_channels, kernel_size, downsample)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x


class R2Plus1DDownSample(nn.Module):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in each layer
    set by layer_sizes, and by performing a global average pool at the end producing a 512-dimensional vector for
    each element in the batch.

    Args:
        stem: stem used or input
        layer_sizes: An iterable containing the number of blocks in each layer
        block_type: Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
        num_classes: number of classes for classification.
    """

    def __init__(
        self,
        stem: Module,
        layer_sizes: List[int] = [1, 1, 1, 1],
        block_type: Module = SpatioTemporalResBlock,
        num_classes: int = 101,
    ):
        super().__init__()
        # self.num_classes = num_classes

        # first conv, with stride 1x2x2 and kernel size 1x7x7
        self.stem = stem
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.layer1 = SpatioTemporalResLayer(
            64, 64, 3, layer_sizes[0], block_type=block_type
        )
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.layer2 = SpatioTemporalResLayer(
            64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True
        )
        self.layer3 = SpatioTemporalResLayer(
            128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True
        )
        self.layer4 = SpatioTemporalResLayer(
            256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True
        )

        # global average pooling of the output
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(-1, 512)

        x = self.fc(x)

        return x
