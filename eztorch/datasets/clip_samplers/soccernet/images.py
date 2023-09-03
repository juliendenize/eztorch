from typing import Any, List

import torch

from eztorch.datasets.clip_samplers.soccernet.soccernet_clip_sampler import \
    SoccerNetClipSampler
from eztorch.datasets.soccernet import ImageSoccerNet


class ImagesSoccerNetClipSampler(SoccerNetClipSampler):
    """Sampler of images in an ImageSoccerNet dataset.

    Args:
        data_source: SoccerNet dataset.
        images_per_video: Number of images per video to sample.
        shuffle: Whether to shuffle indices.
    """

    def __init__(
        self,
        data_source: ImageSoccerNet,
        images_per_video: int | None = None,
        shuffle: bool = False,
    ) -> None:
        super().__init__(data_source, shuffle=shuffle)

        self.images_per_video = images_per_video

        assert (
            self.images_per_video is None or self.images_per_video % 2 == 0
        ), "images_per_video should be divible by 2 to correctly sample in the halves."

    def __iter__(self) -> List[Any]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = [None for i in range(len(self))]
        global_idx = 0
        for idx in range(self.data_source.num_videos):
            video_metadata = self.data_source.get_video_metadata(idx)
            for half_idx in range(video_metadata["num_halves"]):
                num_frames_half = video_metadata["num_frames_fps"][half_idx]
                if self.images_per_video is None:
                    for i in range(num_frames_half):
                        indices[global_idx] = (idx, half_idx, i)
                        global_idx += 1
                else:
                    random_frames = torch.randperm(num_frames_half, generator=g)[
                        : self.images_per_video // 2
                    ]
                    random_frames = torch.sort(random_frames)[0].tolist()
                    for i in random_frames:
                        indices[global_idx] = (idx, half_idx, i)
                        global_idx += 1

        if self._shuffle:
            indices = [
                indices[idx] for idx in torch.randperm(len(indices), generator=g)
            ]

        return iter(indices)

    def __len__(self) -> int:
        return (
            len(self.data_source)
            if self.images_per_video is None
            else self.images_per_video * self.data_source.num_videos
        )

    def __repr__(self) -> str:
        return f"{__class__.__name__}(images_per_video={self.images_per_video}, shuffle={self._shuffle}, seed={self.seed})"
