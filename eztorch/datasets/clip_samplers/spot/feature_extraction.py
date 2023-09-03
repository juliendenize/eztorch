from typing import Any, List

import torch

from eztorch.datasets.clip_samplers.spot.spot_clip_sampler import \
    SpotClipSampler
from eztorch.datasets.soccernet import SoccerNet


class FeatureExtractionSpotClipSampler(SpotClipSampler):
    """Sampler windows that slide across the whole video to extract features.

    Args:
        data_source: SoccerNet dataset.
        window_num_frames: Duration of a window.
        shuffle: Whether to shuffle indices.
    """

    def __init__(
        self,
        data_source: SoccerNet,
        window_num_frames: float = 16,
        shuffle: bool = False,
    ) -> None:
        super().__init__(data_source, shuffle=shuffle)

        self.window_num_frames = window_num_frames
        self._shuffle = shuffle
        self.indices = self._precompute_indices()

    def _precompute_indices(self) -> List[Any]:
        indices = []
        for i in range(self.data_source.num_videos):
            video_metadata = self.data_source.get_video_metadata(i)
            start_frame = 0
            end_frame = int(video_metadata["num_frames"]) - 1
            all_frames = torch.arange(
                start_frame,
                end_frame + 1,
            ).long()
            start_frames = torch.maximum(
                all_frames - self.window_num_frames // 2, torch.tensor(0)
            )
            end_frames = torch.minimum(
                all_frames + self.window_num_frames // 2 - 1, torch.tensor(end_frame)
            )

            indices.extend(
                [(i, start, end) for start, end in zip(start_frames, end_frames)]
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
            f" window_num_frames={self.window_num_frames},"
            f" shuffle={self._shuffle},"
            f" seed={self.seed})"
        )
