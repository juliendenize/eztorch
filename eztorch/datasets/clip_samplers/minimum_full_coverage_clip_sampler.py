from fractions import Fraction
from math import ceil
from typing import Any, Dict

from pytorchvideo.data.clip_sampling import ClipInfo, ClipSampler


class MinimumFullCoverageClipSampler(ClipSampler):
    """Find the minmimum number of clips to cover the full video.

    Args:
        clip_duration: Duration of a clip.
        augs_per_clip: Number of augmentations to be applied on each clip.
    """

    def __init__(self, clip_duration: float, augs_per_clip: int = 1) -> None:
        super().__init__(clip_duration)
        self._augs_per_clip = augs_per_clip

    def __call__(
        self, last_clip_time: float, video_duration: float, annotation: Dict[str, Any]
    ) -> ClipInfo:
        """
        Args:
            last_clip_time: Not used.
            video_duration:: The duration (in seconds) for the video that's
                being sampled.
            annotation: Not used by this sampler.
        Returns:
            includes the clip information of (clip_start_time,
                clip_end_time, clip_index, aug_index, is_last_clip). The times are in seconds.
                ``is_last_clip`` is ``True`` after the end of the video is reached.

        """
        max_possible_clip_start = Fraction(max(video_duration - self._clip_duration, 0))
        clips_per_video = ceil(Fraction(video_duration / self._clip_duration))

        uniform_clip = Fraction(max_possible_clip_start, max(clips_per_video - 1, 1))

        clip_start_sec = uniform_clip * self._current_clip_index
        clip_index = self._current_clip_index
        aug_index = self._current_aug_index

        self._current_aug_index += 1
        if self._current_aug_index >= self._augs_per_clip:
            self._current_clip_index += 1
            self._current_aug_index = 0

        # Last clip is True if sampled self._clips_per_video or if end of video is reached.
        is_last_clip = False
        if (
            self._current_clip_index >= clips_per_video
            or uniform_clip * self._current_clip_index > max_possible_clip_start
        ):
            self._current_clip_index = 0
            is_last_clip = True

        if is_last_clip:
            self.reset()

        return ClipInfo(
            clip_start_sec,
            clip_start_sec + self._clip_duration,
            clip_index,
            aug_index,
            is_last_clip,
        )

    def reset(self):
        self._current_clip_index = 0
        self._current_aug_index = 0
