from fractions import Fraction
from math import ceil, floor
from typing import Any, List

import numpy as np
import torch

from eztorch.datasets.clip_samplers.soccernet.soccernet_clip_sampler import \
    SoccerNetClipSampler
from eztorch.datasets.soccernet import SoccerNet


def random_start_subsequences(
    clip_duration: float = 32,
    video_duration: float = 2700,
    num_clips: int = 50,
    fps: int = 25,
    sample_edges: bool = True,
    prevent_resample_edges: bool = True,
    generator: torch.Generator | None = None,
):
    """Sammple starting point of clips inside a video uniformly. Prevent overlap.

    Args:
        clip_duration: Duration of a clip.
        video_duration: Duration of the video.
        num_clips: Number of clips to sample.
        fps: FPS of the clips.
        sample_edges: Whether to force the sample of edges in the videos. Useful for first or last second actions.
        prevent_resample_edges: Whether to prevent resample of edges. If True, prevent half of the window duration of edges to be sampled again.
        generator: Generator for generating random Pytorch tensors.

    Raises:
        AttributeError: _description_

    Returns:
        _type_: _description_
    """
    possible_start_idx: np.ndarray = np.arange(
        0, int((video_duration - clip_duration) * fps)
    )

    subsequences = [None for _ in range(num_clips)]

    if sample_edges:
        subsequences[-1] = Fraction(0)
        subsequences[-2] = Fraction(video_duration - clip_duration)

        if prevent_resample_edges:
            possible_start_idx = possible_start_idx[ceil(clip_duration / 2) * fps :]
            possible_start_idx = possible_start_idx[: -floor(clip_duration / 2) * fps]

        num_clips -= 2

    max_possible_start = possible_start_idx[-1]
    for i in range(num_clips):
        if possible_start_idx.shape[0] == 0:
            raise AttributeError(
                f"Impossible to sample without overlap {num_clips} clips of {clip_duration} seconds in video of {video_duration} seconds, try changing the numbers."
            )
        idx_tensor: int = torch.randint(
            0, possible_start_idx.shape[0], size=(1,), generator=generator
        ).item()
        start_idx = possible_start_idx[idx_tensor]
        start_sec = Fraction(start_idx, fps)

        min_remove = max(start_idx - int(clip_duration * fps) + 1, 0)
        max_remove = min(start_idx + int(clip_duration * fps) + 1, max_possible_start)

        possible_start_idx = possible_start_idx[
            np.logical_or(
                possible_start_idx < min_remove, possible_start_idx > max_remove
            )
        ]

        subsequences[i] = start_sec
    return subsequences


class UniformWindowWithoutOverlapSoccerNetClipSampler(SoccerNetClipSampler):
    """Sampler uniformly randoml windows in SoccerNet videos.

    Args:
        data_source: SoccerNet dataset.
        windows_per_video: Number of windows to sampler per video.
        window_duration: Duration of a window.
        sample_edges: Whether to force the sample of edges in the videos. Useful for first or last second actions.
        prevent_resample_edges: Whether to prevent resample of edges. If True, prevent half of the window duration of edges to be sampled again.
        shuffle: Whether to shuffle indices.
    """

    def __init__(
        self,
        data_source: SoccerNet,
        windows_per_video: int = 50,
        window_duration: float = 32.0,
        sample_edges: bool = False,
        prevent_resample_edges: bool = True,
        shuffle: bool = False,
    ) -> None:
        super().__init__(data_source, shuffle=shuffle)

        assert windows_per_video % 2 == 0, "Windows per video should be an even number."

        self.windows_per_video = windows_per_video
        self.windows_per_half = windows_per_video // 2
        self.window_duration = window_duration
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
            for half_idx in range(video_metadata["num_halves"]):
                half_random_starts = random_start_subsequences(
                    clip_duration=self.window_duration,
                    video_duration=video_metadata["duration"][half_idx],
                    num_clips=self.windows_per_half,
                    fps=self.data_source._annotated_videos.fps_videos,
                    sample_edges=self.sample_edges,
                    prevent_resample_edges=self.prevent_resample_edges,
                    generator=g,
                )

                for half_random_start in half_random_starts:
                    indices[global_idx] = (
                        idx,
                        half_idx,
                        half_random_start,
                        half_random_start + self.window_duration,
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
            f"window_duration={self.window_duration}, sample_edges={self.sample_edges}, "
            f"prevent_resample_edges={self.prevent_resample_edges} shuffle={self._shuffle}, seed={self.seed})"
        )
