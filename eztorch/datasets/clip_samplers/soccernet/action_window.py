from math import floor
from typing import Any, List

import torch

from eztorch.datasets.clip_samplers.soccernet.soccernet_clip_sampler import \
    SoccerNetClipSampler
from eztorch.datasets.soccernet import SoccerNet


class ActionWindowSoccerNetClipSampler(SoccerNetClipSampler):
    """Sampler windows randomly around actions in SoccerNet videos.

    Args:
        data_source: SoccerNet dataset.
        window_duration: Duration of a window.
        offset_action: Minimum offset before and after the action.
        shuffle: Whether to shuffle indices.
    """

    def __init__(
        self,
        data_source: SoccerNet,
        window_duration: float = 32.0,
        offset_action: float = 0.0,
        shuffle: bool = False,
    ) -> None:
        super().__init__(data_source, shuffle=shuffle)

        self.window_duration = window_duration
        self.offset_action = offset_action
        self.num_actions = self.data_source._annotated_videos.num_actions

    def __iter__(self) -> List[Any]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = [None for i in range(self.num_actions)]
        global_idx = 0
        for idx in range(len(self.data_source)):
            video_metadata = self.data_source.get_video_metadata(idx)
            for half_idx in range(video_metadata["num_halves"]):
                for position in video_metadata["annotations"][half_idx]["position"]:
                    if position <= self.offset_action:
                        min_clip_start_sec = 0
                        max_clip_start_sec = 0
                    elif position >= (
                        video_metadata["duration"][half_idx] - self.offset_action
                    ):
                        min_clip_start_sec = (
                            video_metadata["duration"][half_idx] - self.window_duration
                        )
                        max_clip_start_sec = (
                            video_metadata["duration"][half_idx] - self.window_duration
                        )
                    else:
                        min_clip_start_sec = max(
                            floor(position - self.window_duration + self.offset_action),
                            0,
                        )
                        max_clip_start_sec = max(
                            min(
                                position - self.offset_action,
                                video_metadata["duration"][half_idx]
                                - self.window_duration,
                            ),
                            0,
                        )

                    clip_start_sec = float(
                        (min_clip_start_sec - max_clip_start_sec)
                        * torch.rand(1, generator=g)
                        + max_clip_start_sec
                    )
                    clip_end_sec = clip_start_sec + self.window_duration

                    indices[global_idx] = (idx, half_idx, clip_start_sec, clip_end_sec)
                    global_idx += 1

        if self._shuffle:
            indices = [
                indices[idx] for idx in torch.randperm(len(indices), generator=g)
            ]

        return iter(indices)

    def __len__(self) -> int:
        return self.num_actions

    def __repr__(self) -> str:
        return f"{__class__.__name__}(len={self.__len__()}, window_duration={self.window_duration}, offset_action={self.offset_action}, shuffle={self._shuffle}, seed={self.seed})"
