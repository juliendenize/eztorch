from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from iopath.common.file_io import g_pathmgr

from eztorch.datasets.spot_utils.parse_utils import (LABELS_SPOT_DATASETS,
                                                     SpotDatasets)
from eztorch.utils.utils import get_global_rank, get_world_size


def process_event(events: dict[str, Any], labels_dictionary: dict[str, int]):
    """Process event from spots dictionary.

    Args:
        annotation: The annotation to process.
        labels_dictionary: Labels actions to int.

    Returns:
        The processed annotation.
    """
    new_annotation = {}
    for element, value in events.items():
        if element == "label":
            new_annotation[element] = labels_dictionary[value]
        elif element == "frame":
            new_annotation[element] = int(value)
        elif element == "comment":
            new_annotation[element] = value
        else:
            new_annotation[element] = value
    return new_annotation


class SpotPaths:
    """SpotPaths contains dictionaries describing videos from SoccerNet.

    Args:
        annotations: A list of dictionaries describing the videos.
        path_prefix: Path prefix to add to video paths.
    """

    @classmethod
    def from_path(
        cls,
        data_path: str,
        path_prefix: str = "",
        dataset: SpotDatasets = SpotDatasets.TENNIS,
    ) -> SpotPaths:
        """Factory function that creates a SpotPaths object depending on the path type.

        Only supports json for now.

        Args:
            data_path: The path to the file to be read.
            path_prefix: Path prefix to add to video paths.

        Returns:
            The SpotPaths object.
        """

        if g_pathmgr.isfile(data_path):
            if Path(data_path).suffix == ".json":
                return SpotPaths.from_json(data_path, path_prefix)
            raise NotImplementedError
        elif g_pathmgr.isdir(data_path):
            NotImplementedError
        else:
            raise FileNotFoundError(f"{data_path} not found.")

    @classmethod
    def from_json(
        cls,
        json_file: str,
        path_prefix: str = "",
        dataset: SpotDatasets = SpotDatasets.TENNIS,
    ) -> SpotPaths:
        """Factory function that creates a SpotPaths object by parsing the structure of the given json file. It
        expects the json to be created from soccernet_utils jsons.

        Args:
            json_file: Root directory to the SoccerNet json.
            path_prefix: Path prefix to add to video paths.

        Returns:
            The SpotPaths object.
        """

        assert g_pathmgr.exists(json_file), f"{json_file} not found."

        json_content = json.load(open(json_file))

        paths_and_annotations = [{} for _ in json_content]

        i = 0

        for content in json_content:
            new_content = copy.copy(content)
            new_content["events"] = [
                process_event(event, LABELS_SPOT_DATASETS[SpotDatasets(dataset)])
                for event in new_content["events"]
            ]
            new_content["num_frames"] = int(new_content["num_frames"])
            new_content["height"] = int(new_content["height"])
            new_content["width"] = int(new_content["width"])
            new_content["num_events"] = int(new_content["num_events"])

            paths_and_annotations[i] = new_content
            i += 1

        return cls(paths_and_annotations, path_prefix)

    def __init__(
        self,
        annotations: list[dict[str, Any]],
        path_prefix: str | Path = "",
    ) -> None:
        self._annotations = annotations

        self._path_prefix = Path(path_prefix)

        self._serialize_annotations()

    def _serialize_annotations(self):
        """Serialize annotations for the dataset."""
        self._video_paths = np.array(
            [video_content["video"] for video_content in self._annotations]
        ).astype(np.string_)

        self._num_frames_per_video = torch.tensor(
            [video_content["num_frames"] for video_content in self._annotations],
            dtype=torch.int32,
        )

        self._cum_num_frames_per_video = self._num_frames_per_video.cumsum(0)

        self._fps_per_video = torch.tensor(
            [int(video_content["fps"]) for video_content in self._annotations],
            dtype=torch.uint8,
        )

        self._height_per_video = torch.tensor(
            [int(video_content["height"]) for video_content in self._annotations],
            dtype=torch.int32,
        )

        self._width_per_video = torch.tensor(
            [int(video_content["width"]) for video_content in self._annotations],
            dtype=torch.int32,
        )

        self._num_events_per_video = torch.tensor(
            [int(video_content["num_events"]) for video_content in self._annotations],
            dtype=torch.int32,
        )

        self._num_events_per_video = torch.tensor(
            [int(video_content["num_events"]) for video_content in self._annotations],
            dtype=torch.int16,
        )
        self._end_event_video_idx = self._num_events_per_video.to(
            dtype=torch.int32
        ).cumsum(
            dim=0,
        )
        self._start_event_video_idx = torch.roll(self._end_event_video_idx, 1)
        self._start_event_video_idx[0] = 0

        self._events_video = torch.tensor(
            [
                video_idx
                for video_idx, video_content in enumerate(self._annotations)
                for _ in range(len(video_content["events"]))
            ],
            dtype=torch.int32,
        )

        # Annotations
        self._label_event_per_video = torch.tensor(
            [
                event_content["label"]
                for video_content in self._annotations
                for event_content in video_content["events"]
            ],
            dtype=torch.uint8,
        )
        self._frame_event_per_video = torch.tensor(
            [
                event_content["frame"]
                for video_content in self._annotations
                for event_content in video_content["events"]
            ],
        )

        self._comment_event_per_video = np.array(
            [
                event_content["comment"]
                for video_content in self._annotations
                for event_content in video_content["events"]
            ]
        ).astype(np.string_)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Args:
            index: The video index.

        Returns:
            The video annotation for the given index.
        """

        video_event_start_idx = self._start_event_video_idx[index]
        video_event_end_idx = self._end_event_video_idx[index]

        return {
            "video_path": self._path_prefix / self._video_paths[index].decode(),
            "video_name": self._video_paths[index].decode(),
            "num_frames": self._num_frames_per_video[index].item(),
            "events": {
                "label": self._label_event_per_video[
                    video_event_start_idx:video_event_end_idx
                ],
                "frame": self._frame_event_per_video[
                    video_event_start_idx:video_event_end_idx
                ],
                "comment": self._comment_event_per_video[
                    video_event_start_idx:video_event_end_idx
                ],
            },
        }

    @property
    def num_frames_per_video(self) -> torch.Tensor:
        """Number of frames per video for decode."""
        return self._num_frames_per_video

    @property
    def cum_num_frames_per_video(self) -> torch.Tensor:
        """Number of cumulative frames per video for decode."""
        return self._cum_num_frames_per_video

    @property
    def number_of_frames(self) -> int:
        """Number of total frames in the dataset."""
        return int(self._num_frames_per_video.sum())

    @property
    def num_videos(self) -> int:
        """Number of videos."""
        return len(self._video_paths)

    @property
    def path_prefix(self) -> Path:
        """The prefix to add to video paths."""
        return self._path_prefix

    @path_prefix.setter
    def path_prefix(self, value: str | Path):
        self._path_prefix = Path(value)

    @property
    def global_rank(self):
        """Global rank of the process."""
        return get_global_rank()

    @property
    def worlf_size(self):
        """World size, number of the processes."""
        return get_world_size()

    def __len__(self) -> int:
        """
        Returns:
            The number of videos.
        """
        return len(self._annotations)
