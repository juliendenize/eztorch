from math import ceil, floor
from typing import Any, List

import numpy as np
import torch

from eztorch.datasets.clip_samplers.spot.spot_clip_sampler import \
    SpotClipSampler
from eztorch.datasets.spot import Spot


def random_start_subsequences(
    clip_duration: int = 32,
    video_num_frames: int = 2700,
    num_subsequences: int = 50,
    sample_edges: bool = True,
    prevent_resample_edges: bool = True,
    generator: torch.Generator | None = None,
):
    possible_start_idx: np.ndarray = np.arange(0, video_num_frames - clip_duration)

    subsequences = [None for _ in range(num_subsequences)]

    if sample_edges:
        subsequences[-1] = 0
        subsequences[-2] = video_num_frames - clip_duration

        if prevent_resample_edges:
            possible_start_idx = possible_start_idx[ceil(clip_duration / 2) :]
            possible_start_idx = possible_start_idx[: -floor(clip_duration / 2)]

        num_subsequences -= 2

    max_possible_start = possible_start_idx[-1]
    for i in range(num_subsequences):
        if possible_start_idx.shape[0] == 0:
            raise AttributeError(
                f"Impossible to sample without overlap {num_subsequences} clips of {clip_duration} seconds in video of {video_num_frames} num frames, try changing the numbers."
            )
        idx_tensor: int = torch.randint(
            0, possible_start_idx.shape[0], size=(1,), generator=generator
        ).item()
        start_idx = possible_start_idx[idx_tensor]

        min_remove = max(start_idx - clip_duration + 1, 0)
        max_remove = min(start_idx + clip_duration + 1, max_possible_start)

        possible_start_idx = possible_start_idx[
            np.logical_or(
                possible_start_idx < min_remove, possible_start_idx > max_remove
            )
        ]

        subsequences[i] = start_idx
    return subsequences


class UniformWindowWithoutOverlapSpotClipSampler(SpotClipSampler):
    """Sampler uniformly randoml windows in Spot videos.

    Args:
        data_source: Spot dataset.
        windows_per_video: Number of windows to sampler per video.
        window_num_frames: Duration of a window.
        sample_edges: Whether to force the sample of edges in the videos. Useful for first or last second actions.
        prevent_resample_edges: Whether to prevent resample of edges. If True, prevent half of the window duration of edges to be sampled again.
        shuffle: Whether to shuffle indices.
    """

    def __init__(
        self,
        data_source: Spot,
        windows_per_video: int = 50,
        window_num_frames: int = 32,
        sample_edges: bool = False,
        prevent_resample_edges: bool = True,
        shuffle: bool = False,
    ) -> None:
        super().__init__(data_source, shuffle=shuffle)

        self.windows_per_video = windows_per_video
        self.window_num_frames = window_num_frames
        self.sample_edges = sample_edges
        self.prevent_resample_edges = prevent_resample_edges
        self._shuffle = shuffle

    def __iter__(self) -> List[Any]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = [None for _ in range(len(self.data_source) * self.windows_per_video)]
        global_idx = 0
        for idx in range(len(self.data_source)):
            video_metadata = self.data_source.get_video_metadata(idx)
            video_random_starts = random_start_subsequences(
                clip_duration=self.window_num_frames,
                video_num_frames=video_metadata["num_frames"],
                num_subsequences=self.windows_per_video,
                sample_edges=self.sample_edges,
                prevent_resample_edges=self.prevent_resample_edges,
                generator=g,
            )

            for video_random_start in video_random_starts:
                indices[global_idx] = (
                    idx,
                    video_random_start,
                    video_random_start + self.window_num_frames - 1,
                )
                global_idx += 1

        if self._shuffle:
            indices = [
                indices[idx] for idx in torch.randperm(len(indices), generator=g)
            ]

        return iter(indices)

    def __len__(self) -> int:
        return len(self.data_source) * self.windows_per_video

    def __repr__(self) -> str:
        return (
            f"{__class__.__name__}(len={self.__len__()}, windows_per_video={self.windows_per_video}, "
            f"window_num_frames={self.window_num_frames}, sample_edges={self.sample_edges}, "
            f"prevent_resample_edges={self.prevent_resample_edges} shuffle={self._shuffle}, seed={self.seed})"
        )
