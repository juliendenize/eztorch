from fractions import Fraction
from math import floor
from typing import Any, List

import torch

from eztorch.datasets.clip_samplers.spot.spot_clip_sampler import \
    SpotClipSampler
from eztorch.datasets.spot import Spot


class SlidingWindowSpotClipSampler(SpotClipSampler):
    """Sampler windows that slide across the whole video. Possibility to overlap windows. The last window is always between (half_duration - window_num_frames, window_num_frames).

    Args:
        data_source: SoccerNet dataset.
        window_num_frames: Duration of a window.
        overlap_window: Overlap duration between two windows.
        shuffle: Whether to shuffle indices.
    """

    def __init__(
        self,
        data_source: Spot,
        window_num_frames: int = 32,
        overlap_window: int = 1,
        shuffle: bool = False,
    ) -> None:
        super().__init__(data_source, shuffle=shuffle)

        self.window_num_frames = window_num_frames
        self.overlap_window = overlap_window
        self._shuffle = shuffle

        self.correct_window_per_video = (
            (self.data_source._annotated_videos.num_frames_per_video - overlap_window)
            // (window_num_frames - overlap_window)
        ).to(dtype=torch.int)

        self.one_more_window_per_video = (
            (self.data_source._annotated_videos.num_frames_per_video - overlap_window)
            % (window_num_frames - overlap_window)
            > 0
        ).to(dtype=torch.int)

        self.total_windows_per_video = (
            self.correct_window_per_video + self.one_more_window_per_video
        )

        self.total_windows = self.total_windows_per_video.sum(0)

        self.window_to_sample = self.total_windows

        self._precompute_indices()

    def _precompute_indices(self) -> List[Any]:
        indices = [None for _ in range(self.total_windows)]
        video_idx = 0
        global_idx = 0
        for i in range(len(self.total_windows_per_video)):
            for j in range(self.correct_window_per_video[i]):
                clip_start_frame = j * (self.window_num_frames - self.overlap_window)
                clip_end_frame = clip_start_frame + self.window_num_frames - 1

                indices[global_idx] = (
                    video_idx,
                    clip_start_frame,
                    clip_end_frame,
                )
                global_idx += 1

            if self.one_more_window_per_video[i]:
                duration = self.data_source._annotated_videos.num_frames_per_video[i]
                clip_start_frame = floor(duration) - self.window_num_frames
                clip_end_frame = clip_start_frame + self.window_num_frames - 1

                indices[global_idx] = (
                    video_idx,
                    clip_start_frame,
                    clip_end_frame,
                )
                global_idx += 1

            video_idx += 1

        self._raw_indices = indices

    def __iter__(self) -> List[Any]:
        indices = self._raw_indices

        if self._shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

        if self._shuffle:
            indices = [
                indices[idx] for idx in torch.randperm(len(indices), generator=g)
            ]

        return iter(indices)

    def __len__(self) -> int:
        return self.window_to_sample

    def __repr__(self) -> str:
        return f"{__class__.__name__}(len={self.__len__()}, window_num_frames={self.window_num_frames}, overlap_window={self.overlap_window}, shuffle={self._shuffle}, seed={self.seed})"
