from fractions import Fraction
from typing import Any, List

import torch

from eztorch.datasets.clip_samplers.soccernet.soccernet_clip_sampler import \
    SoccerNetClipSampler
from eztorch.datasets.soccernet import SoccerNet


class UniformWindowSoccerNetClipSampler(SoccerNetClipSampler):
    """Sampler uniformly randomly windows in SoccerNet videos.

    Args:
        data_source: SoccerNet dataset.
        windows_per_video: Number of windows to sampler per video.
        window_duration: Duration of a window.
        sample_edges: Whether to force the sample of edges in the videos. Useful for kick-offs or last second actions.
        shuffle: Whether to shuffle indices.
    """

    def __init__(
        self,
        data_source: SoccerNet,
        windows_per_video: int = 50,
        window_duration: float = 32.0,
        sample_edges: bool = False,
        shuffle: bool = False,
    ) -> None:
        super().__init__(data_source, shuffle=shuffle)

        assert windows_per_video % 2 == 0, "Windows per video should be an even number."

        self.windows_per_video = windows_per_video
        self.windows_per_half = windows_per_video // 2
        self.window_duration = window_duration
        self.sample_edges = sample_edges
        self._shuffle = shuffle

    def __iter__(self) -> List[Any]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = [None for i in range(len(self.data_source) * self.windows_per_video)]
        global_idx = 0
        for idx in range(len(self.data_source)):
            video_metadata = self.data_source.get_video_metadata(idx)
            for half_idx in range(video_metadata["num_halves"]):
                max_possible_clip_start = Fraction(
                    max(video_metadata["duration"][half_idx] - self.window_duration, 0)
                )

                if self.sample_edges:
                    indices[global_idx] = (
                        idx,
                        half_idx,
                        Fraction(0),
                        Fraction(self.window_duration),
                    )
                    global_idx += 1

                windows_per_half = (
                    self.windows_per_half - 2
                    if self.sample_edges
                    else self.windows_per_half
                )

                windows_start = (
                    float(max_possible_clip_start)
                    * torch.rand(windows_per_half).sort()[0]
                )

                for window_start in windows_start.tolist():
                    indices[global_idx] = (
                        idx,
                        half_idx,
                        Fraction(window_start),
                        Fraction(window_start + self.window_duration),
                    )
                    global_idx += 1

                if self.sample_edges:
                    indices[global_idx] = (
                        idx,
                        half_idx,
                        Fraction(
                            video_metadata["duration"][half_idx] - self.window_duration
                        ),
                        Fraction(video_metadata["duration"][half_idx]),
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
        return f"{__class__.__name__}(len={self.__len__()}, windows_per_video={self.windows_per_video}, window_duration={self.window_duration}, sample_edges={self.sample_edges}, shuffle={self._shuffle}, seed={self.seed})"
