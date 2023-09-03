# --------------------------------------------------------------------------------------------
# Modified from pytorchvideo (https://github.com/facebookresearch/pytorchvideo)
# Licensed under the Apache License, Version 2.0
# --------------------------------------------------------------------------------------------


import math
from typing import Tuple

import torch


def _get_param_spatial_crop(
    scale: Tuple[float, float],
    ratio: Tuple[float, float],
    height: int,
    width: int,
    log_uniform_ratio: bool = True,
    num_tries: int = 10,
) -> Tuple[int, int, int, int]:
    """Given scale, ratio, height and width, return sampled coordinates of the videos.

    Args:
        scale (Tuple[float, float]): Scale range of Inception-style area based
            random resizing.
        ratio (Tuple[float, float]): Aspect ratio range of Inception-style
            area based random resizing.
        height (int): Height of the original image.
        width (int): Width of the original image.
        log_uniform_ratio (bool): Whether to use a log-uniform distribution to
            sample the aspect ratio. Default is True.
        num_tries (int): The number of times to attempt a randomly resized crop.
            Falls back to a central crop after all attempts are exhausted.
            Default is 10.

    Returns:
        Tuple containing i, j, h, w. (i, j) are the coordinates of the top left
        corner of the crop. (h, w) are the height and width of the crop.
    """
    assert num_tries >= 1, "num_tries must be at least 1"

    if scale[0] > scale[1]:
        scale = (scale[1], scale[0])
    if ratio is not None and ratio[0] > ratio[1]:
        ratio = (ratio[1], ratio[0])

    for _ in range(num_tries):
        area = height * width
        target_area = area * (scale[0] + torch.rand(1).item() * (scale[1] - scale[0]))
        if ratio is None:
            aspect_ratio = float(width) / float(height)
        elif log_uniform_ratio:
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(
                log_ratio[0] + torch.rand(1).item() * (log_ratio[1] - log_ratio[0])
            )
        else:
            aspect_ratio = ratio[0] + torch.rand(1).item() * (ratio[1] - ratio[0])

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if 0 < w <= width and 0 < h <= height:
            i = torch.randint(0, height - h + 1, (1,)).item()
            j = torch.randint(0, width - w + 1, (1,)).item()
            return i, j, h, w

    # Fallback to central crop.
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:  # whole image
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w


def random_resized_crop(
    frames: torch.Tensor,
    target_height: int,
    target_width: int,
    scale: Tuple[float, float],
    aspect_ratio: Tuple[float, float] | None,
    shift: bool = False,
    log_uniform_ratio: bool = True,
    interpolation: str = "bilinear",
    num_tries: int = 10,
) -> torch.Tensor:
    """Crop the given images to random size and aspect ratio. A crop of random size relative to the original size
    and a random aspect ratio is made. This crop is finally resized to given size. This is popularly used to train
    the Inception networks.

    Args:
        frames (torch.Tensor): Video tensor to be resized with shape (C, T, H, W).
        target_height (int): Desired height after cropping.
        target_width (int): Desired width after cropping.
        scale (Tuple[float, float]): Scale range of Inception-style area based
            random resizing. Should be between 0.0 and 1.0.
        aspect_ratio (Tuple[float, float]): Aspect ratio range of Inception-style
            area based random resizing. Should be between 0.0 and +infinity.
        shift (bool): Bool that determines whether or not to sample two different
            boxes (for cropping) for the first and last frame. If True, it then
            linearly interpolates the two boxes for other frames. If False, the
            same box is cropped for every frame. Default is False.
        log_uniform_ratio (bool): Whether to use a log-uniform distribution to
            sample the aspect ratio. Default is True.
        interpolation (str): Algorithm used for upsampling. Currently supports
            'nearest', 'bilinear', 'bicubic', 'area'. Default is 'bilinear'.
        num_tries (int): The number of times to attempt a randomly resized crop.
            Falls back to a central crop after all attempts are exhausted.
            Default is 10.

    Returns:
        cropped (tensor): A cropped video tensor of shape (C, T, target_height, target_width).
    """
    assert (
        scale[0] > 0 and scale[1] > 0
    ), "min and max of scale range must be greater than 0"
    assert aspect_ratio is None or (
        aspect_ratio[0] > 0 and aspect_ratio[1] > 0
    ), "min and max of aspect_ratio range must be greater than 0"

    device, dtype = frames.device, frames.dtype

    channels, t, height, width = frames.shape

    i, j, h, w = _get_param_spatial_crop(
        scale,
        aspect_ratio,
        height,
        width,
        log_uniform_ratio,
        num_tries,
    )

    if not shift:
        cropped = frames[:, :, i : i + h, j : j + w]
        return torch.nn.functional.interpolate(
            cropped,
            size=(target_height, target_width),
            mode=interpolation,
        )

    i_, j_, h_, w_ = _get_param_spatial_crop(
        scale, aspect_ratio, height, width, log_uniform_ratio, num_tries, device
    )
    i_s = [int(i) for i in torch.linspace(i, i_, steps=t).tolist()]
    j_s = [int(i) for i in torch.linspace(j, j_, steps=t).tolist()]
    h_s = [int(i) for i in torch.linspace(h, h_, steps=t).tolist()]
    w_s = [int(i) for i in torch.linspace(w, w_, steps=t).tolist()]
    cropped = torch.zeros(
        (channels, t, target_height, target_width), device=device, dtype=dtype
    )
    for ind in range(t):
        cropped[:, ind : ind + 1, :, :] = torch.nn.functional.interpolate(
            frames[
                :,
                ind : ind + 1,
                i_s[ind] : i_s[ind] + h_s[ind],
                j_s[ind] : j_s[ind] + w_s[ind],
            ],
            size=(target_height, target_width),
            mode=interpolation,
        )
    return cropped


class RandomResizedCrop(torch.nn.Module):
    """``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.random_resized_crop``."""

    def __init__(
        self,
        target_height: int,
        target_width: int,
        scale: Tuple[float, float],
        aspect_ratio: Tuple[float, float],
        shift: bool = False,
        log_uniform_ratio: bool = True,
        interpolation: str = "bilinear",
        num_tries: int = 10,
    ) -> None:

        super().__init__()
        self._target_height = target_height
        self._target_width = target_width
        self._scale = scale
        self._aspect_ratio = aspect_ratio
        self._shift = shift
        self._log_uniform_ratio = log_uniform_ratio
        self._interpolation = interpolation
        self._num_tries = num_tries

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input video tensor with shape (C, T, H, W).
        """
        return random_resized_crop(
            x,
            self._target_height,
            self._target_width,
            self._scale,
            self._aspect_ratio,
            self._shift,
            self._log_uniform_ratio,
            self._interpolation,
            self._num_tries,
        )

    def __repr__(self):
        return (
            f"{__class__.__name__}(target_height={self._target_height}, target_width={self._target_width}, "
            f"scale={self._scale}, aspect_ratio={self._aspect_ratio}, shift={self._shift}, "
            f"log_uniform_ratio={self._log_uniform_ratio}, interpolation={self._interpolation}, "
            f"num_tries={self._num_tries})"
        )
