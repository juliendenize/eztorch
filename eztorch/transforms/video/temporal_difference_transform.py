import torch
from torch import Tensor
from torch.nn import Module
from torchvision.transforms import RandomApply
from torchvision.transforms.functional import rgb_to_grayscale


def temporal_difference(
    frames: Tensor,
    use_grayscale: bool = False,
    absolute: bool = False,
) -> Tensor:
    """Compute temporal differences from consecutives frames.

    Args:
        frames: The frames.
        use_grayscale: If ``True``, apply grayscale transformation before computing the temporal difference.
        absolute: If ``True``, take the absolute value of the difference, else shift to the mean value (:math`255` for int, :math:`0.5` for float) and divide by 2.

    Returns:
        The frames after temporal difference.
    """

    if use_grayscale:
        frames = frames.permute(1, 0, 2, 3)
        frames = rgb_to_grayscale(frames, 3)
        frames = frames.permute(1, 0, 2, 3)
    else:
        frames = frames

    if frames.dtype == torch.uint8:
        convert = True
        frames = frames.to(torch.float32)
    else:
        convert = False

    out_frames = torch.zeros_like(frames)
    t = frames.shape[1]

    dt = frames[:, 0 : t - 1, :, :] - frames[:, 1:t, :, :]
    if absolute:
        dt = dt.abs()
    out_frames[:, 0 : t - 1, :, :] = dt
    if t <= 1:
        return out_frames
    out_frames[:, -1, :, :] = dt[:, -1, :, :]

    if not absolute:
        out_frames += 255.0
        out_frames /= 2.0

    if convert:
        out_frames = out_frames.round().clamp(0, 255).to(torch.uint8)

    return out_frames


class TemporalDifference(Module):
    """Compute temporal differences from consecutives frames.

    Args:
        use_grayscale: If ``True``, apply grayscale transformation before computing the temporal difference.
        absolute: If ``True``, take the absolute value of the difference, else shift to the mean value (:math:`255` for int, :math:`0.5` for float) and divide by 2.
    """

    def __init__(self, use_grayscale: bool = True, absolute: bool = False) -> None:
        super().__init__()
        self.use_grayscale = use_grayscale
        self.absolute = absolute

    def forward(self, input: Tensor) -> Tensor:
        return temporal_difference(input, self.use_grayscale, self.absolute)


class RandomTemporalDifference(RandomApply):
    """Randomly compute temporal differences from consecutives frames.

    Args:
        use_grayscale: If ``True``, apply grayscale transformation before computing the temporal difference.
        absolute: If ``True``, take the absolute value of the difference, else shift to the mean value (:math:`255` for int, :math:`0.5` for float) and divide by 2.
        p: The probability to compute temporal difference.
    """

    def __init__(
        self, use_grayscale: bool = True, absolute: bool = False, p: float = 0.2
    ) -> None:
        transform = TemporalDifference(use_grayscale, absolute)
        super().__init__(transforms=[transform], p=p)
