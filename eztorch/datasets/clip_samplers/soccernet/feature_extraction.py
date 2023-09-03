from fractions import Fraction
from typing import Any, List

import torch

from eztorch.datasets.clip_samplers.soccernet.soccernet_clip_sampler import \
    SoccerNetClipSampler
from eztorch.datasets.soccernet import SoccerNet


class FeatureExtractionSoccerNetClipSampler(SoccerNetClipSampler):
    """Sampler windows that slide across the whole video to extract features at a specified fps.

    Args:
        data_source: SoccerNet dataset.
        window_duration: Duration of a window.
        fps: fps to extract features.
        shuffle: Whether to shuffle indices.
    """

    def __init__(
        self,
        data_source: SoccerNet,
        window_duration: float = 2.56,
        fps: int = 2,
        shuffle: bool = False,
    ) -> None:
        super().__init__(data_source, shuffle=shuffle)

        self.window_duration = window_duration
        self.fps = fps
        self._shuffle = shuffle
        self.indices = self._precompute_indices()

    def _precompute_indices(self) -> List[Any]:
        indices = []
        frac_fps = Fraction(self.fps)
        over_frac_fps = Fraction(1, self.fps)
        for i in range(self.data_source.num_videos):
            video_metadata = self.data_source.get_video_metadata(i)
            for j in range(video_metadata["num_halves"]):
                start_sec = 0

                end_sec = Fraction(
                    Fraction(int(video_metadata["duration"][j] * self.fps)), frac_fps
                )
                all_times = torch.arange(
                    start_sec,
                    float(end_sec),
                    float(over_frac_fps),
                )
                start_times = torch.maximum(
                    all_times - self.window_duration / 2, torch.tensor(0)
                )
                end_times = torch.minimum(
                    all_times + self.window_duration / 2, torch.tensor(float(end_sec))
                )

                indices.extend(
                    [(i, j, start, end) for start, end in zip(start_times, end_times)]
                )

        return indices

    def __iter__(self) -> List[Any]:
        indices = self.indices

        if self._shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            indices = [
                indices[idx] for idx in torch.randperm(len(indices), generator=g)
            ]

        return iter(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __repr__(self) -> str:
        return (
            f"{__class__.__name__}("
            f"len={self.__len__()},"
            f" window_duration={self.window_duration},"
            f" fps={self.fps},"
            f" shuffle={self._shuffle},"
            f" seed={self.seed})"
        )
