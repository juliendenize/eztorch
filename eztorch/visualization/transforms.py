import copy
import math
from typing import Any, Dict, Iterable, List, Mapping

import hydra
import torch
import torchvision
from torch import Tensor
from torch.nn import Module


def apply_several_transforms(
    images: Iterable[Tensor], transforms: Iterable[Module]
) -> List[List[Tensor]]:
    """Apply several transformations to a list of images.

    Args:
        images: The images.
        transforms: The transformations to apply to the images.

    Returns:
        List of list of transformed images.
    """

    transformed_images = [
        [transform(image) for image in images] for transform in transforms
    ]
    return transformed_images


def apply_several_video_transforms(
    videos: Iterable[Dict[str, Any]], transforms: Iterable[Module]
) -> List[List[Tensor]]:
    """Apply several transformations to a list of videos.

    Args:
        videos: The videos.
        transforms: The transformations to apply to the videos.

    Returns:
        List of list of transformed videos.
    """

    transformed_images = [
        [transform(copy.deepcopy(video)) for video in videos]
        for transform in transforms
    ]
    return transformed_images


def make_grid_from_several_transforms(
    sets_images: Iterable[Iterable[Tensor]], n_images_per_row: int = 8
) -> Tensor:
    """Make a grid of images by aligning images from several transformations vertically.

    Args:
        sets_images: Sets of transformed images aligned, base_image(sets_images[0][?]) == ... == base_images(sets_images[-1][?]).
        n_images_per_row: Number of images displayed per row.

    Returns:
        Grid of images.
    """

    n_images = len(sets_images[0])
    for set_image in sets_images:
        assert len(set_image) == n_images

    if n_images_per_row == -1:
        nrow = n_images
        all_images = torch.cat(*sets_images, dim=0)

    else:
        nrow = n_images_per_row
        all_images = []

        number_of_rows = math.ceil(n_images // n_images_per_row)

        for row_idx in range(number_of_rows):
            for set_image in sets_images:
                for i in range(
                    row_idx * n_images_per_row,
                    min(n_images, (row_idx + 1) * n_images_per_row),
                ):
                    all_images.append(set_image[i])

    grid = torchvision.utils.make_grid(all_images, nrow=nrow)

    return grid


def make_several_transforms_from_config(
    cfg_transforms: Mapping[Any, Any]
) -> List[Module]:
    """Make several transformations from a configuration dictionary.

    Args:
        cfg_transforms: Configuration of transformations in the form {'transform1': {...}, 'transform2': {...}}.

    Returns:
        List of transformations.
    """

    transforms = [
        hydra.utils.instantiate(conf_transform)
        for _, conf_transform in cfg_transforms.items()
    ]
    return transforms
