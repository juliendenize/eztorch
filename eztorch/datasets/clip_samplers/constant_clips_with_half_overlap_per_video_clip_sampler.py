from fractions import Fraction
from typing import Any, Dict

from pytorchvideo.data.clip_sampling import ClipInfo, ClipSampler


class ConstantClipsWithHalfOverlapPerVideoClipSampler(ClipSampler):
    """Evenly splits the video into clips_per_video increments and samples clips of size clip_duration at these
    increments.

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
            last_clip_time: Not used for ConstantClipsPerVideoSampler.
            video_duration:: the duration (in seconds) for the video that's
                being sampled.
            annotation: Not used by this sampler.

        Returns:
            the clip information composed of (clip_start_time,
                clip_end_time, clip_index, aug_index, is_last_clip). The times are in seconds.
                is_last_clip is True after clips_per_video clips have been sampled or the end
                of the video is reached.
        """

        num_clips = max(video_duration // self._clip_duration * 2 - 1, 0)

        clip_start_sec = Fraction(self._clip_duration, 2) * self._current_clip_index

        clip_index = self._current_clip_index
        aug_index = self._current_aug_index

        self._current_aug_index += 1
        if self._current_aug_index >= self._augs_per_clip:
            self._current_clip_index += 1
            self._current_aug_index = 0

        # Last clip is True if sampled self._clips_per_video or if end of video is reached.
        is_last_clip = False
        if self._current_clip_index >= num_clips:
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
