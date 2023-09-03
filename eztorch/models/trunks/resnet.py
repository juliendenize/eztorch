from torch.nn import Conv2d, Identity, Linear, Module
from torchvision.models.resnet import (resnet18, resnet34, resnet50, resnet101,
                                       resnet152, resnext50_32x4d,
                                       resnext101_32x8d, wide_resnet50_2,
                                       wide_resnet101_2)

_ResNets = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "resnext50_32x4d": resnext50_32x4d,
    "resnext101_32x8d": resnext101_32x8d,
    "wide_resnet50_2": wide_resnet50_2,
    "wide_resnet101_2": wide_resnet101_2,
}


def create_resnet(
    name: str,
    num_classes: int = 1000,
    progress: bool = True,
    pretrained: bool = False,
    small_input: bool = False,
    **kwargs,
) -> Module:
    """Build ResNet from torchvision for image.

    Args:
        name: name of the resnet model (such as resnet18).
        num_classes: If not :math:`0`, replace the last fully connected layer with num_classes output, if :math:`0` replace by identity.
        pretrained: If ``True``, returns a model pre-trained on ImageNet.
        progress: If ``True``, displays a progress bar of the download to stderr.
        small_input: If ``True``, replace the first conv2d for small images and replace first maxpool by identity.
        **kwargs: arguments specific to torchvision constructors for ResNet.

    Returns:
        Basic resnet.
    """
    assert (
        name in _ResNets
    ), f"ResNet {name} is not supported please add the corresponding entry in _ResNets directory or provide the right name."
    func = _ResNets[name]

    model = func(pretrained=pretrained, progress=progress, **kwargs)

    if num_classes == 0:
        model.fc = Identity()
    else:
        model.fc = Linear(model.fc.in_features, num_classes)

    if small_input:
        model.conv1 = Conv2d(
            3, model.conv1.out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = Identity()

    model.num_features = model.inplanes

    return model
