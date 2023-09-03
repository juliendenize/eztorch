from __future__ import annotations

import json
import os
import pathlib

from iopath.common.file_io import g_pathmgr
from torchvision.datasets.folder import make_dataset


class LabeledVideoPaths:
    """LabeledVideoPaths contains pairs of video path and integer index label.

    Args:
        paths_and_labels: a list of tuples containing the video
            path and integer label.
    """

    @classmethod
    def from_path(cls, data_path: str) -> LabeledVideoPaths:
        """Factory function that creates a LabeledVideoPaths object depending on the path type.

        - If it is a directory path it uses the LabeledVideoPaths.from_directory function.
        - If it's a file it uses the LabeledVideoPaths.from_csv file.
        Args:
            file_path: The path to the file to be read.

        Returns:
            The LabeledVideoPaths object.
        """

        if g_pathmgr.isfile(data_path):
            if pathlib.Path(data_path).suffix == ".json":
                return LabeledVideoPaths.from_json(data_path)
            return LabeledVideoPaths.from_csv(data_path)
        elif g_pathmgr.isdir(data_path):
            return LabeledVideoPaths.from_directory(data_path)
        else:
            raise FileNotFoundError(f"{data_path} not found.")

    @classmethod
    def from_csv(cls, file_path: str) -> LabeledVideoPaths:
        """Factory function that creates a LabeledVideoPaths object by reading a file with the following format:

            <path> <integer_label>
            ...
            <path> <integer_label>

        Args:
            file_path: The path to the file to be read.

        Returns:
            The LabeledVideoPaths object.
        """
        assert g_pathmgr.exists(file_path), f"{file_path} not found."
        video_paths_and_label = []
        with g_pathmgr.open(file_path, "r") as f:
            for path_label in f.read().splitlines():
                line_split = path_label.rsplit(None, 2)

                # The video path file may not contain labels (e.g. for a test split). We
                # assume this is the case if only 1 path is found and set the label to
                # -1 if so.
                if len(line_split) == 1:
                    file_path = line_split[0]
                    label = -1
                    num_frames = None
                elif len(line_split) == 2:
                    file_path, label = line_split
                    num_frames = None
                else:
                    file_path, label, num_frames = line_split
                    num_frames = int(num_frames)
                video_paths_and_label.append((file_path, int(label), num_frames))

        assert (
            len(video_paths_and_label) > 0
        ), f"Failed to load dataset from {file_path}."
        return cls(video_paths_and_label)

    @classmethod
    def from_json(cls, file_path: str) -> LabeledVideoPaths:
        """Factory function that creates a LabeledVideoPaths object by reading a json file.

        Args:
            file_path: The path to the file to be read.

        Returns:
            The LabeledVideoPaths object.
        """
        assert g_pathmgr.exists(file_path), f"{file_path} not found."
        video_paths_and_label = []

        json_content = json.load(open(file_path))

        annotation = json_content["annotation"]
        videos_id = sorted([x for x in annotation.keys()])

        for video_id in videos_id:
            label = annotation[video_id]["class"]
            num_frames = annotation[video_id]["num_frames"]
            video_path = f"{video_id}"
            video_paths_and_label.append((video_path, int(label), int(num_frames)))

        assert (
            len(video_paths_and_label) > 0
        ), f"Failed to load dataset from {file_path}."
        return cls(video_paths_and_label)

    @classmethod
    def from_directory(cls, dir_path: str) -> LabeledVideoPaths:
        """Factory function that creates a LabeledVideoPaths object by parsing the structure of the given
        directory's subdirectories into the classification labels. It expects the directory format to be the
        following: dir_path/<class_name>/<video_name>.mp4.

        Classes are indexed from 0 to the number of classes, alphabetically.

        E.g.
            dir_path/class_x/xxx.ext
            dir_path/class_x/xxy.ext
            dir_path/class_x/xxz.ext
            dir_path/class_y/123.ext
            dir_path/class_y/nsdf3.ext
            dir_path/class_y/asd932_.ext

        Would produce two classes labeled 0 and 1 with 3 videos paths associated with each.

        Args:
            dir_path: Root directory to the video class directories.

        Returns:
            The LabeledVideoPaths object.
        """

        assert g_pathmgr.exists(dir_path), f"{dir_path} not found."

        # Find all classes based on directory names. These classes are then sorted and indexed
        # from 0 to the number of classes.
        classes = sorted(f.name for f in pathlib.Path(dir_path).iterdir() if f.is_dir())
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        video_paths_and_label = make_dataset(
            dir_path, class_to_idx, extensions=("mp4", "avi")
        )

        video_paths_and_label = [
            (video, label, None) for video, label in video_paths_and_label
        ]
        assert (
            len(video_paths_and_label) > 0
        ), f"Failed to load dataset from {dir_path}."
        return cls(video_paths_and_label)

    def __init__(
        self, paths_and_labels: list[tuple[str, int | None]], path_prefix=""
    ) -> None:
        self._paths_and_labels = paths_and_labels
        self._path_prefix = path_prefix

    @property
    def path_prefix(self):
        """The prefix to add to video paths."""
        return self._path_prefix

    @path_prefix.setter
    def path_prefix(self, value):
        self._path_prefix = value

    def __getitem__(self, index: int) -> tuple[str, int]:
        """
        Args:
            index: the path and label index.

        Returns:
            The path and label tuple for the given index.
        """
        path, label, num_frames = self._paths_and_labels[index]
        return (
            os.path.join(self._path_prefix, path),
            {"label": label, "num_frames": num_frames},
        )

    def __len__(self) -> int:
        """
        Returns:
            The number of video paths and label pairs.
        """
        return len(self._paths_and_labels)
