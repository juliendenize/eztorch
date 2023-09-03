from typing import Any, Iterable, List

from PIL.Image import Image
from torch import Tensor


class MultiCropTransform:
    """Define multi crop transform that apply several sets of transform to the inputs.

    Args:
        set_transforms: List of Dictionary of sets of transforms specifying transforms and number of views per set.

    Example::

        set_transforms = [
            {'transform': [...], 'num_views': ...},
            {'transform': [...], 'num_views': ...},
            ...
        ]

        transform = MultiCropTransform(
            set_transforms
        )
    """

    def __init__(self, set_transforms: List[Any]) -> None:
        super().__init__()

        self.set_transforms = set_transforms
        transforms = []
        for set_transform in self.set_transforms:
            if "num_views" not in set_transform:
                set_transform["num_views"] = 1
            transforms.extend([set_transform["transform"]] * set_transform["num_views"])
        self.transforms = transforms

    def __call__(self, img: Image | Tensor | Iterable[Image | Tensor]) -> Tensor:
        if type(img) not in [Image, Tensor]:
            transformed_images = [
                transform(image)
                for transform, image in zip(self.transforms, img, strict=True)
            ]
        else:
            transformed_images = [transform(img) for transform in self.transforms]
        return transformed_images

    def __repr__(self) -> str:
        format_string = self.__class__.__name__

        for set_transform in self.set_transforms:
            format_string += "(\n"
            format_string += "    num views={}\n".format(set_transform["num_views"])
            format_string += "    transforms={}".format(set_transform["transform"])
            format_string += "\n)"
        return format_string
