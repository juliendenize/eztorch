import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode

_MEANS = {
    "cifar10": [0.4914, 0.4822, 0.4465],
    "cifar100": [0.5071, 0.4865, 0.4409],
    "stl10": [0.4914, 0.4822, 0.4465],
    "imagenet": [0.485, 0.456, 0.406],
}

_STDS = {
    "cifar10": [0.2023, 0.1994, 0.2010],
    "cifar100": [0.2009, 0.1984, 0.2023],
    "stl10": [0.2471, 0.2435, 0.2616],
    "imagenet": [0.229, 0.224, 0.225],
}

_INTERPOLATION = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
}
