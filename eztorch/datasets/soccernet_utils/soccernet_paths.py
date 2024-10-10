from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from iopath.common.file_io import g_pathmgr

from eztorch.datasets.soccernet_utils.parse_utils import (SoccerNetTask,
                                                          process_annotation)
from eztorch.utils.utils import get_global_rank, get_world_size


class SoccerNetPaths:
    """SoccerNetPaths contains dictionaries describing videos from SoccerNet.

    Args:
        annotations: A list of dictionaries describing the videos.
        path_prefix: Path prefix to add to video paths.
        task: The SoccerNet task.
    """

    @classmethod
    def from_path(
        cls,
        data_path: str,
        path_prefix: str = "",
        task: SoccerNetTask = SoccerNetTask.ACTION,
    ) -> SoccerNetPaths:
        """Factory function that creates a SoccerNetPaths object depending on the path type.

        Only supports json for now.

        Args:
            data_path: The path to the file to be read.
            path_prefix: Path prefix to add to video paths.
            task: The SoccerNet task.

        Returns:
            The SoccerNetPaths object.
        """

        if g_pathmgr.isfile(data_path):
            if Path(data_path).suffix == ".json":
                return SoccerNetPaths.from_json(data_path, path_prefix, task)
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
        task: SoccerNetTask = SoccerNetTask.ACTION,
    ) -> SoccerNetPaths:
        """Factory function that creates a SoccerNetPaths object by parsing the structure of the given json file.
        It expects the json to be created from soccernet_utils jsons.

        Args:
            json_file: Root directory to the SoccerNet json.
            path_prefix: Path prefix to add to video paths.
            task: The SoccerNet task.

        Returns:
            The SoccerNetPaths object.
        """

        assert g_pathmgr.exists(json_file), f"{json_file} not found."

        json_content = json.load(open(json_file))

        paths_and_annotations = [{} for match in json_content]

        i = 0

        for match in json_content:
            new_content = {}
            new_content["halves"] = {}
            for key, content in json_content[match].items():
                if key == "halves":
                    for half, half_content in content.items():
                        new_half_content = copy.copy(half_content)
                        new_half_content["annotations"] = [
                            process_annotation(annotation, task)
                            for annotation in new_half_content["annotations"]
                        ]
                        new_half_content["duration"] = float(
                            new_half_content["duration"]
                        )
                        new_half_content["fps"] = int(new_half_content["fps"])
                        new_half_content["num_frames"] = int(
                            new_half_content["num_frames"]
                        )
                        new_content["halves"][half] = new_half_content
                else:
                    new_content[key] = json_content[match][key]

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
            [match_content["UrlLocal"] for match_content in self._annotations]
        ).astype(np.string_)
        self._halves_per_video = torch.tensor(
            [len(match_content["halves"]) for match_content in self._annotations],
            dtype=torch.uint8,
        )
        self._half_paths = np.array(
            [
                half_content["UrlLocal"]
                for match_content in self._annotations
                for _, half_content in match_content["halves"].items()
            ]
        ).astype(np.string_)

        self._half_ids = torch.tensor(
            [
                int(half_id)
                for match_content in self._annotations
                for half_id, _ in match_content["halves"].items()
            ],
            dtype=torch.uint8,
        )

        self._num_frames_per_half = torch.tensor(
            [
                half_content["num_frames"]
                for match_content in self._annotations
                for _, half_content in match_content["halves"].items()
            ],
            dtype=torch.int32,
        )

        self._duration_per_half = torch.tensor(
            [
                int(half_content["duration"])
                for match_content in self._annotations
                for _, half_content in match_content["halves"].items()
            ],
        )

        self._fps_per_half = torch.tensor(
            [
                int(half_content["fps"])
                for match_content in self._annotations
                for _, half_content in match_content["halves"].items()
            ],
            dtype=torch.uint8,
        )

        if not torch.all(self._fps_per_half == self._fps_per_half[0]):
            raise AttributeError("All videos should have same fps for now")

        # Flags to know where indexing for videos starts and ends
        self._end_video_idx = self._halves_per_video.to(dtype=torch.int32).cumsum(dim=0)
        self._start_video_idx = torch.roll(self._end_video_idx, 1)
        self._start_video_idx[0] = 0

        # Flags to know where indexing for halves annotations starts and ends
        self._num_annotations_per_half = torch.tensor(
            [
                len(half_content["annotations"])
                for match_content in self._annotations
                for _, half_content in match_content["halves"].items()
            ],
            dtype=torch.int16,
        )
        self._end_annotation_half_idx = self._num_annotations_per_half.to(
            dtype=torch.int32
        ).cumsum(
            dim=0,
        )
        self._start_annotation_half_idx = torch.roll(self._end_annotation_half_idx, 1)
        self._start_annotation_half_idx[0] = 0

        # Flags to know where indexing for halves annotations starts and ends
        self._annotations_half = torch.tensor(
            [
                video_idx * 2 + half_idx
                for video_idx, match_content in enumerate(self._annotations)
                for half_idx, (_, half_content) in enumerate(
                    match_content["halves"].items()
                )
                for _ in range(len(half_content["annotations"]))
            ],
            dtype=torch.int32,
        )

        # Annotations
        self._label_annotation_per_half = torch.tensor(
            [
                annotation_content["label"]
                for match_content in self._annotations
                for _, half_content in match_content["halves"].items()
                for annotation_content in half_content["annotations"]
            ],
            dtype=torch.uint8,
        )
        self._position_annotation_per_half = torch.tensor(
            [
                annotation_content["position"]
                for match_content in self._annotations
                for _, half_content in match_content["halves"].items()
                for annotation_content in half_content["annotations"]
            ],
        )
        self._team_per_half = torch.tensor(
            [
                annotation_content["team"]
                for match_content in self._annotations
                for _, half_content in match_content["halves"].items()
                for annotation_content in half_content["annotations"]
            ],
            dtype=torch.uint8,
        )
        self._visibility_per_half = torch.tensor(
            [
                annotation_content["visibility"]
                for match_content in self._annotations
                for _, half_content in match_content["halves"].items()
                for annotation_content in half_content["annotations"]
            ],
            dtype=torch.uint8,
        )

        self._video_idx_annotation_per_half = torch.tensor(
            [
                video_idx
                for video_idx, match_content in enumerate(self._annotations)
                for _, half_content in match_content["halves"].items()
                for annotation_content in half_content["annotations"]
            ],
            dtype=torch.int32,
        )

        self._half_idx_annotation_per_half = torch.tensor(
            [
                half_idx
                for match_content in self._annotations
                for half_idx, (_, half_content) in enumerate(
                    match_content["halves"].items()
                )
                for annotation_content in half_content["annotations"]
            ],
            dtype=torch.int32,
        )

        # Useful for caching
        self._cumsum_num_frames_per_half = self._num_frames_per_half.to(
            dtype=torch.int32
        ).cumsum(dim=0)
        self._prev_cumsum_num_frames_per_half = torch.roll(
            self._cumsum_num_frames_per_half, 1
        )
        self._prev_cumsum_num_frames_per_half[0] = 0

        self.set_fps(self._fps_per_half[0])

    def get_half_metadata(self, video_index: int, half_index: int):
        """Get the metadata of the specified half.

        Args:
            video_index: The video index.
            half_index: The half index.

        Returns:
            The metadata of the half.
        """
        return self.__getitem__((video_index, half_index))

    def get_video_metadata(self, video_index):
        """Get the metadata of the specified video.

        Args:
            video_index: The video index.

        Returns:
            The metadata of the video.
        """
        num_halves = self._halves_per_video[video_index]

        halves_metadata = [
            self.get_half_metadata(video_index, half_index)
            for half_index in range(num_halves)
        ]

        video_metadata = {
            "video_path": self._path_prefix / self._video_paths[video_index].decode(),
            "url_local": self._video_paths[video_index].decode(),
            "num_halves": num_halves,
        }

        for key in halves_metadata[0].keys():
            if key in video_metadata:
                continue
            video_metadata[key] = [
                half_metadata[key] for half_metadata in halves_metadata
            ]

        return video_metadata

    def __getitem__(self, index: tuple[int]) -> dict[str, Any]:
        """
        Args:
            index: The video index. Tuple containing the video index and the half index.

        Returns:
            The video annotation for the given index.
        """

        video_index, half_index = index

        start_video_idx = self._start_video_idx[video_index]

        half_annotation_start_idx = self._start_annotation_half_idx[
            start_video_idx + half_index
        ]
        half_annotation_end_idx = self._end_annotation_half_idx[
            start_video_idx + half_index
        ]

        start_video_idx = self._start_video_idx[video_index]

        return {
            "video_path": self._path_prefix / self._video_paths[video_index].decode(),
            "url_local": self._video_paths[video_index].decode(),
            "half_path": self._path_prefix
            / self._half_paths[start_video_idx + half_index].decode(),
            "half_id": self._half_ids[start_video_idx + half_index].item(),
            "duration": self._duration_per_half[start_video_idx + half_index].item(),
            "num_frames": self._num_frames_per_half[
                start_video_idx + half_index
            ].item(),
            "num_frames_fps": self._num_frames_per_half_fps[
                start_video_idx + half_index
            ].item(),
            "annotations": {
                "label": self._label_annotation_per_half[
                    half_annotation_start_idx:half_annotation_end_idx
                ],
                "position": self._position_annotation_per_half[
                    half_annotation_start_idx:half_annotation_end_idx
                ],
                "team": self._team_per_half[
                    half_annotation_start_idx:half_annotation_end_idx
                ],
                "visibility": self._visibility_per_half[
                    half_annotation_start_idx:half_annotation_end_idx
                ],
            },
            "start_video_idx": start_video_idx,
            "half_idx": start_video_idx + half_index,
        }

    def set_fps(self, fps: int) -> None:
        self._fps = fps
        if fps == self.fps_videos:
            self._num_frames_per_half_fps = self._num_frames_per_half
            self._cumsum_num_frames_per_half_fps = self._cumsum_num_frames_per_half
            self._number_of_frames_fps = self.number_of_frames
            self._prev_cumsum_num_frames_per_half_fps = (
                self._prev_cumsum_num_frames_per_half
            )
        else:
            self._num_frames_per_half_fps = self._duration_per_half * fps
            self._cumsum_num_frames_per_half_fps = self._num_frames_per_half_fps.cumsum(
                dim=0
            )
            self._number_of_frames_fps = int(self._cumsum_num_frames_per_half_fps.sum())
            self._prev_cumsum_num_frames_per_half_fps = torch.roll(
                self._cumsum_num_frames_per_half_fps, 1
            )
            self._prev_cumsum_num_frames_per_half_fps[0] = 0

    @property
    def fps_videos(self) -> int:
        return int(self._fps_per_half[0])

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def num_frames_per_half_fps(self) -> torch.Tensor:
        """Number of frames per half for decode fps."""
        return self._num_frames_per_half_fps

    @property
    def cumsum_num_frames_per_half_fps(self) -> torch.Tensor:
        """Number of cumulative frames per half for decode fps."""
        return self._cumsum_num_frames_per_half_fps

    @property
    def prev_cumsum_num_frames_per_half_fps(self):
        """Number of cumulative frames per half shifted to the right for decode fps."""
        return self._prev_cumsum_num_frames_per_half_fps

    @property
    def number_of_frames_fps(self) -> int:
        """Number of total frames in the dataset for decode fps."""
        return self._number_of_frames_fps

    @property
    def cumsum_num_frames_per_half(self):
        """Number of cumulative frames per half."""
        return self._cumsum_num_frames_per_half

    @property
    def prev_cumsum_num_frames_per_half(self):
        """Number of cumulative frames per half shifted to the right."""
        return self._prev_cumsum_num_frames_per_half

    @property
    def number_of_frames(self) -> int:
        """Number of total frames in the dataset."""
        return int(self._num_frames_per_half.sum())

    @property
    def num_videos(self) -> int:
        """Number of videos."""
        return len(self._video_paths)

    @property
    def num_halves(self) -> int:
        """Number of halves."""
        return len(self._half_paths)

    @property
    def num_actions(self) -> int:
        """Number of actions."""
        return len(self._position_annotation_per_half)

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
