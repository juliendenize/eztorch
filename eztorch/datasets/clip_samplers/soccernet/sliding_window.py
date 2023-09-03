from fractions import Fraction
from math import floor
from typing import Any, List

import torch

from eztorch.datasets.clip_samplers.soccernet.soccernet_clip_sampler import \
    SoccerNetClipSampler
from eztorch.datasets.soccernet import SoccerNet


class SlidingWindowSoccerNetClipSampler(SoccerNetClipSampler):
    """Sampler windows that slide across the whole video. Possibility to overlap windows. The last window is always between (half_duration - window_duration, window_duration).

    Args:
        data_source: SoccerNet dataset.
        window_duration: Duration of a window.
        overlap_window: Overlap duration between two windows.
        shuffle: Whether to shuffle indices.
    """

    def __init__(
        self,
        data_source: SoccerNet,
        window_duration: float = 32.0,
        overlap_window: float = 1.0,
        shuffle: bool = False,
    ) -> None:
        super().__init__(data_source, shuffle=shuffle)

        self.window_duration = window_duration
        self.overlap_window = overlap_window
        self._shuffle = shuffle

        self.correct_window_per_half = (
            (self.data_source._annotated_videos._duration_per_half - overlap_window)
            // (window_duration - overlap_window)
        ).to(dtype=torch.int)

        self.one_more_window_per_half = (
            (self.data_source._annotated_videos._duration_per_half - overlap_window)
            % (window_duration - overlap_window)
            > 0
        ).to(dtype=torch.int)

        self.total_windows_per_half = (
            self.correct_window_per_half + self.one_more_window_per_half
        )

        self.total_windows = self.total_windows_per_half.sum(0)

        self.window_to_sample = self.total_windows

        self._precompute_indices()

    def _precompute_indices(self) -> List[Any]:
        indices = [None for _ in range(self.total_windows)]
        video_idx = 0
        half_idx = 0
        global_idx = 0
        for i in range(len(self.total_windows_per_half)):
            for j in range(self.correct_window_per_half[i]):
                clip_start_sec = Fraction(
                    float(j * (self.window_duration - self.overlap_window))
                )
                clip_end_sec = clip_start_sec + self.window_duration

                indices[global_idx] = (
                    video_idx,
                    half_idx % 2,
                    clip_start_sec,
                    clip_end_sec,
                )
                global_idx += 1

            if self.one_more_window_per_half[i]:
                duration = self.data_source._annotated_videos._duration_per_half[i]
                clip_start_sec = Fraction(floor(duration) - self.window_duration)
                clip_end_sec = clip_start_sec + self.window_duration

                indices[global_idx] = (
                    video_idx,
                    half_idx % 2,
                    clip_start_sec,
                    clip_end_sec,
                )
                global_idx += 1

            half_idx += 1
            if half_idx % 2 == 0:
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
        return f"{__class__.__name__}(len={self.__len__()}, window_duration={self.window_duration}, overlap_window={self.overlap_window}, shuffle={self._shuffle}, seed={self.seed})"
