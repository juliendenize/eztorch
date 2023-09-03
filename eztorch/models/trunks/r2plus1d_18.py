from typing import List, Union

from torch.nn import (BatchNorm3d, Conv3d, Identity, Linear, Module, ReLU,
                      Sequential)
from torchvision.models.video.resnet import (BasicBlock, Conv2Plus1D,
                                             R2Plus1dStem, _video_resnet)

from eztorch.models.trunks.r2plus1d_18_downsample import R2Plus1DDownSample


class LargeR2Plus1dStem(Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution."""

    def __init__(self) -> None:
        super().__init__(
            Conv3d(
                3,
                83,
                kernel_size=(1, 7, 7),
                stride=(1, 2, 2),
                padding=(0, 3, 3),
                bias=False,
            ),
            BatchNorm3d(83),
            ReLU(inplace=True),
            Conv3d(
                83,
                64,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                padding=(1, 0, 0),
                bias=False,
            ),
            BatchNorm3d(64),
            ReLU(inplace=True),
        )


_STEMS = {"large": LargeR2Plus1dStem, "normal": R2Plus1dStem}


def create_r2plus1d_18(
    downsample: bool = True,
    num_classes: int = 101,
    layers: List[int] = [1, 1, 1, 1],
    progress: bool = True,
    pretrained: bool = False,
    stem: Union[str, Module] = LargeR2Plus1dStem,
    **kwargs
) -> Module:
    """Build R2+1D_18 from torchvision for video.

    Args:
        num_classes: If not :math:`0`, replace the last fully connected layer with ``num_classes`` output, if :math:`0` replace by identity.
        pretrained: If ``True``, returns a model pre-trained on ImageNet.
        progress: If ``True``, displays a progress bar of the download to stderr
        layers: Number of layers per block.
        stem: Stem to use for input.
        **kwargs: arguments specific to torchvision constructors for ResNet.

    Returns:
        Basic resnet.
    """

    if type(stem) is str:
        stem = _STEMS[stem]

    if downsample:
        model = R2Plus1DDownSample(stem(), layers)
        model.num_features = 512
    else:
        model = _video_resnet(
            "r2plus1d_18",
            pretrained,
            progress,
            block=BasicBlock,
            conv_makers=[Conv2Plus1D] * 4,
            layers=layers,
            stem=stem,
            **kwargs
        )
        model.num_features = model.inplanes

    if num_classes == 0:
        model.fc = Identity()
    else:
        model.fc = Linear(model.fc.in_features, num_classes)

    return model
