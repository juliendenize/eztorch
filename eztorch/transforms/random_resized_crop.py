from typing import Iterable, Union

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from eztorch.transforms.utils import _INTERPOLATION


class RandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(
        self,
        size: Union[int, Iterable[int]],
        scale: Iterable[float] = [0.08, 1.0],
        ratio: Iterable[float] = [3 / 4, 4 / 3],
        interpolation: Union[str, InterpolationMode] = "bilinear",
        antialias: bool = True,
        **kwargs,
    ) -> None:
        if type(interpolation) is str:
            interpolation = _INTERPOLATION[interpolation]
        super().__init__(
            size,
            scale=scale,
            ratio=ratio,
            interpolation=interpolation,
            antialias=antialias,
            **kwargs,
        )
