import random
from fractions import Fraction
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from pytorchvideo.data.clip_sampling import ClipInfo, ClipInfoList, ClipSampler
from torch import Tensor


def compute_jittered_speed(factor: float, speed: int) -> float:
    """Compute jittered speed.

    Args:
        factor: The jittering factor.
        speed: The speed to jitter.

    Returns:
        float: the jitter speed.
    """
    min_speed = speed * (1 - factor)
    max_speed = speed * (1 + factor)

    jittered_speed = np.random.uniform(min_speed, max_speed)

    return jittered_speed


class RandomClipSampler(ClipSampler):
    r"""Randomly samples clip of size clip_duration from the videos.

    Args:
        clip_duration: Duration of a clip.
        speeds: If not ``None``, the list of speeds to randomly apply on clip duration. At each call, :math:`clip\_duration *= choice(speeds)`.
        jitter_factor: The jitter factor bound to apply on clip duration. At each call, :math:`clip\_duration *= (1 + \pm rand(0, jitter\_factor))`.
    """

    def __init__(
        self,
        clip_duration: Union[float, Fraction],
        speeds: Optional[List[int]] = None,
        jitter_factor: float = 0.0,
    ) -> None:
        super().__init__(clip_duration)
        self._speeds = speeds
        if self._speeds is not None and len(self._speeds) == 1 and self._speeds[0] == 1:
            self._speeds = None
        self._jitter_factor = jitter_factor

    def __call__(
        self, last_clip_time: float, video_duration: float, annotation: Dict[str, Any]
    ) -> ClipInfo:

        if self._speeds is not None:
            speed = np.random.choice(self._speeds)
            speed_fraction = Fraction(speed)
            clip_duration = Fraction(self._clip_duration, Fraction(1, speed_fraction))
        else:
            clip_duration = self._clip_duration

        if self._jitter_factor != 0.0:
            jittered_speed = compute_jittered_speed(self._jitter_factor, 1)
            jittered_speed_fraction = Fraction(jittered_speed)
            clip_duration = Fraction(
                clip_duration, Fraction(1, jittered_speed_fraction)
            )

        max_possible_clip_start = max(video_duration - clip_duration, 0)
        clip_start_sec = Fraction(random.uniform(0, max_possible_clip_start))
        return ClipInfo(clip_start_sec, clip_start_sec + clip_duration, 0, 0, True)


class RandomMultiClipSampler(RandomClipSampler):
    r"""Randomly samples multiple clip of size clip_duration from the videos.

    Args:
        clip_duration: Duration of a clip.
        num_clips: Number of clips to sample.
        speeds: If not ``None``, the list of speeds to randomly apply on clip duration. At each call, :math:`clip\_duration *= choice(speeds)`.
        jitter_factor: The jitter factor bound to apply on clip duration. At each call, :math:`clip\_duration *= (1 + \pm rand(0, jitter\_factor))`.
    """

    def __init__(
        self,
        clip_duration: float,
        num_clips: int,
        speeds: Optional[List[int]] = None,
        jitter_factor: float = 0.0,
    ) -> None:
        super().__init__(clip_duration, speeds=speeds, jitter_factor=jitter_factor)
        self._num_clips = num_clips

    def __call__(
        self, last_clip_time: float, video_duration: float, annotation: Dict[str, Any]
    ) -> ClipInfoList:

        (
            clip_start_list,
            clip_end_list,
            clip_index_list,
            aug_index_list,
            is_last_clip_list,
        ) = (
            self._num_clips * [None],
            self._num_clips * [None],
            self._num_clips * [None],
            self._num_clips * [None],
            self._num_clips * [None],
        )
        for i in range(self._num_clips):
            (
                clip_start_list[i],
                clip_end_list[i],
                clip_index_list[i],
                aug_index_list[i],
                is_last_clip_list[i],
            ) = super().__call__(last_clip_time, video_duration, annotation)

        return ClipInfoList(
            clip_start_list,
            clip_end_list,
            clip_index_list,
            aug_index_list,
            is_last_clip_list,
        )


class RandomCVRLSampler(ClipSampler):
    r"""Randomly samples two clip of size clip_duration from the videos. The second clip is sampled after the first
    one following a Power law for the starting time.

    References:
        - https://arxiv.org/abs/2008.03800

    Args:
        clip_duration: Duration of a clip.
        speeds: If not ``None``, the list of speeds to randomly apply on clip duration. At each call, :math:`clip\_duration *= choice(speeds)`.
        jitter_factor: The jitter factor bound to apply on clip duration. At each call, :math:`clip\_duration *= (1 + \pm rand(0, jitter\_factor))`.
        shuffle: If ``True``, shuffle the clip order for the output.
        power_cdf: Power coefficient for the power law.
        decreasing_cdf: Whether the power law curve is ascending or descending.
    """

    def __init__(
        self,
        clip_duration: Union[float, Fraction],
        speeds: Optional[List[int]] = None,
        jitter_factor: float = 0.0,
        shuffle: bool = True,
        power_cdf: float = 1.0,
        decreasing_cdf: bool = True,
    ) -> None:
        super().__init__(clip_duration)
        self._speeds = speeds
        if self._speeds is not None and len(self._speeds) == 1 and self._speeds[0] == 1:
            self._speeds = None
        self._jitter_factor = jitter_factor
        self._shuffle = shuffle
        self._power_cdf = power_cdf
        self._decreasing_cdf = decreasing_cdf

    def __call__(
        self, last_clip_time: float, video_duration: float, annotation: Dict[str, Any]
    ) -> ClipInfo:

        if self._speeds is not None:
            speed = np.random.choice(self._speeds)
            speed_fraction = Fraction(speed)
            clip_duration = Fraction(self._clip_duration, Fraction(1, speed_fraction))
        else:
            clip_duration = self._clip_duration

        if self._jitter_factor != 0.0:
            jittered_speed = compute_jittered_speed(self._jitter_factor, 1)
            jittered_speed_fraction = Fraction(jittered_speed)
            clip_duration = Fraction(
                clip_duration, Fraction(1, jittered_speed_fraction)
            )

        if video_duration <= clip_duration:
            start = Fraction(0)
            end = Fraction(video_duration) + clip_duration

            return ClipInfoList(
                [start, start], [end, end], [0, 0], [0, 0], [True, True]
            )

        else:
            max_start = float(video_duration - clip_duration)
            max_start_tensor = torch.tensor(max_start)
            max_start_fraction = Fraction(max_start)

            def cdf(k: Tensor, power: float = 1.0):
                if self._decreasing_cdf:
                    p = -torch.pow(k, power + 1) / (
                        power * torch.pow(max_start_tensor, power + 1)
                    ) + k * (power + 1) / (power * max_start_tensor)
                else:
                    p = torch.pow(k, power + 1) / (
                        power * torch.pow(max_start_tensor, power + 1)
                    ) + k * (power + 1) / (power * max_start_tensor)
                return p

            u = torch.rand(1)
            k_low = Fraction(0)
            k_up = max_start_fraction
            two = Fraction(2.0)
            k = Fraction(max_start_fraction, two)

            while abs(k_up - k_low) > 1e-3:
                k = Fraction(k_up + k_low, two)
                if cdf(torch.tensor(float(k)), self._power_cdf) > u:
                    k_up = k
                else:
                    k_low = k

            max_start_1 = max_start_fraction - k
            start_1 = Fraction(float(torch.rand(1))) * max_start_1
            start_2 = start_1 + k

        keep_order = random.randint(0, 1) if self._shuffle else 1

        if keep_order:
            starts = [start_1, start_2]
            ends = [start_1 + clip_duration, start_2 + clip_duration]
        else:
            starts = [start_2, start_1]
            ends = [start_2 + clip_duration, start_1 + clip_duration]

        return ClipInfoList(starts, ends, [0, 0], [0, 0], [True, True])
