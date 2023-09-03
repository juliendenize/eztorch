"""
References:
- https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/ucf101.py
"""

from __future__ import annotations

import json
import os
import pathlib
from functools import partial
from pathlib import Path
from typing import Any, Callable

import torch
from iopath.common.file_io import g_pathmgr
from omegaconf import DictConfig
from pytorchvideo.data.clip_sampling import ClipSampler

from eztorch.datasets.decoders import DecoderType
from eztorch.datasets.labeled_video_dataset import LabeledVideoDataset
from eztorch.datasets.utils_fn import get_raw_video_duration, remove_suffix

try:
    import pandas
except ImportError:
    _HAS_PD = False
else:
    _HAS_PD = True


_UCF101_FILES = [
    "trainlist01.txt",
    "trainlist02.txt",
    "trainlist03.txt",
    "testlist01.txt",
    "testlist02.txt",
    "testlist03.txt",
]


def create_ucf101_files_for_frames(folder_files: str, frames_folder: str):
    """Create the UCF101 csv files for frame decoders.

    Args:
        folder_files: Path to the original ucf101 split files.
        frames_folder: Path to the frame folders.

    Raises:
        ImportError: If pandas is not installed.
    """

    if not _HAS_PD:
        raise ImportError("pandas is required to use this function.")

    classes = {}

    def get_video_class_index(video: str):
        video = Path(video)
        if video.parent.name not in classes:
            classes[video.parent.name] = len(classes)
        return classes[video.parent.name]

    for file in _UCF101_FILES:
        file = Path(folder_files) / file
        data = pandas.read_csv(file, sep=" ", header=None, names=["video", "label"])
        frames_folder = Path(frames_folder)

        data["label"] = data.video.map(get_video_class_index)

        data.video = data.video.map(remove_suffix)
        data["duration"] = data.video.map(
            partial(get_raw_video_duration, frames_folder)
        )
        data.to_csv(frames_folder / file.name, sep=" ", header=None, index=None)


class Ucf101LabeledVideoPaths:
    """Pre-processor for Ucf101 dataset mentioned here - https://www.crcv.ucf.edu/data/UCF101.php.

    This dataset consists of classwise folds with each class consisting of 3
        folds (splits).

    The videos directory is of the format,
        video_dir_path/class_x/<somevideo_name>.avi
        ...
        video_dir_path/class_y/<somevideo_name>.avi

    The splits/fold directory is of the format,
        folds_dir_path/classInd.txt
        folds_dir_path/testlist01.txt
        folds_dir_path/testlist02.txt
        folds_dir_path/testlist03.txt
        folds_dir_path/trainlist01.txt
        folds_dir_path/trainlist02.txt
        folds_dir_path/trainlist03.txt

    Args:
        paths_and_labels: a list of tuples containing the video
            path and integer label.
    """

    _allowed_splits = [1, 2, 3]

    @classmethod
    def from_path(
        cls,
        data_path: str,
        split_id: int = 1,
        split_type: str = "train",
        frames: bool = False,
    ) -> Ucf101LabeledVideoPaths:
        """Factory function that creates a LabeledVideoPaths object depending on the path type.

        - If it is a directory path it uses the LabeledVideoPaths.from_directory function.
        - If it's a file it uses the LabeledVideoPaths.from_csv file.
        Args:
            file_path: The path to the file or directory to be read.
            split_id: Split id. Used if path is a directory.
            split_type: Split type. Used if path is a directory.
            frames: If ``True``, UCF101 is loaded as a frame dataset.
        Returns:
            The Ucf101LabeledVideoPaths object.
        """

        if g_pathmgr.isfile(data_path):
            if pathlib.Path(data_path).suffix == ".json":
                return Ucf101LabeledVideoPaths.from_json(data_path)
            return Ucf101LabeledVideoPaths.from_csv(data_path)
        elif g_pathmgr.isdir(data_path):
            return Ucf101LabeledVideoPaths.from_directory(
                data_path, split_id, split_type, frames
            )
        else:
            raise FileNotFoundError(f"{data_path} not found.")

    @classmethod
    def from_csv(cls, file_path: str) -> Ucf101LabeledVideoPaths:
        """Factory function that creates a LabeledVideoPaths object by reading a file with the following format:

            <path> <integer_label>
            ...
            <path> <integer_label>

        Args:
            file_path: The path to the file to be read.
        Returns:
            The Ucf101LabeledVideoPaths object.
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
    def from_json(cls, file_path: str) -> Ucf101LabeledVideoPaths:
        """Factory function that creates a LabeledVideoPaths object by reading a json file.

        Args:
            file_path: The path to the file to be read.
        Returns:
            The Ucf101LabeledVideoPaths object.
        """
        assert g_pathmgr.exists(file_path), f"{file_path} not found."
        video_paths_and_label = []

        json_content = json.load(open(file_path))

        annotation = json_content["annotation"]
        videos_id = sorted([x for x in annotation.keys()])

        for video_id in videos_id:
            label = annotation[video_id]["class"]
            num_frames = (
                annotation[video_id]["num_frames"]
                if "num_frames" in annotation["video_id"]
                else None
            )
            video_path = f"{video_id}"
            video_paths_and_label.append((video_path, int(label), int(num_frames)))

        assert (
            len(video_paths_and_label) > 0
        ), f"Failed to load dataset from {file_path}."
        return cls(video_paths_and_label)

    @classmethod
    def from_directory(
        cls,
        data_path: str,
        split_id: int = 1,
        split_type: str = "train",
        frames: bool = False,
    ) -> Ucf101LabeledVideoPaths:
        """Factory function that creates Ucf101LabeledVideoPaths object form a splits/folds directory.

        Args:
            data_path: The path to the splits/folds directory of UCF-101.
            split_id: Fold id to be loaded. Options are: :math:`1`, :math:`2`, :math:`3`.
            split_type: Split/Fold type to be loaded. Options are: ``'train'``, ``'test'``.
            frames: If ``True``, UCF101 is loaded as a frame dataset.

        Returns:
            The Ucf101LabeledVideoPaths object.
        """
        data_path = pathlib.Path(data_path)
        if not data_path.is_dir():
            raise RuntimeError(f"{data_path} not found or is not a directory.")
        if not int(split_id) in cls._allowed_splits:
            raise RuntimeError(
                f"{split_id} not found in allowed split id's {cls._allowed_splits}."
            )
        file_name = data_path / f"{split_type}list0{split_id}.txt"
        label_file_name = data_path / "classInd.txt"
        return cls.from_csvs(file_name, label_file_name, frames)

    @classmethod
    def from_csvs(
        cls,
        file_path: pathlib.Path | str,
        label_file_name: pathlib.Path | str,
        frames: bool = False,
    ) -> Ucf101LabeledVideoPaths:
        """Factory function that creates Ucf101LabeledVideoPaths object form a list of split files of .txt type.

        Args:
            file_paths : The path to the splits/folds
                    directory of UCF-101.
            split_type: Split/Fold type to be loaded.
                - "train"
                - "test"

        Returns:
            The Ucf101LabeledVideoPaths object.
        """
        label_file_path = pathlib.Path(label_file_name)
        assert g_pathmgr.exists(label_file_path), f"{label_file_path} not found."
        if not (
            label_file_path.suffix == ".txt" and label_file_path.stem == "classInd"
        ):
            raise RuntimeError(f"Invalid file: {file_path}")
        class_dict = {}
        with g_pathmgr.open(label_file_path, "r") as f:
            for line in f.read().splitlines():
                line_split = line.rsplit(None, 1)
                class_dict[line_split[1]] = int(line_split[0]) - 1

        video_paths_and_label = []
        file_path = pathlib.Path(file_path)
        assert g_pathmgr.exists(file_path), f"{file_path} not found."
        if not (file_path.suffix == ".txt"):
            raise RuntimeError(f"Invalid file: {file_path}")
        with g_pathmgr.open(file_path, "r") as f:
            for line in f.read().splitlines():
                line_split = line.rsplit(None, 2)

                video_name = pathlib.Path(line_split[0])
                if frames:
                    video_path = (
                        pathlib.Path(file_path.parent)
                        / video_name.parent
                        / video_name.stem
                    )
                    num_frames = len(list(video_path.iterdir()))
                else:
                    video_path = pathlib.Path(file_path.parent) / video_name
                    num_frames = None
                video_label = class_dict[str(video_path.parent.stem)]

                video_paths_and_label.append((video_path, video_label, num_frames))

        assert (
            len(video_paths_and_label) > 0
        ), f"Failed to load dataset from {file_path}."
        return cls(video_paths_and_label)

    def __init__(
        self, paths_and_labels: list[tuple[str, dict | None]], path_prefix=""
    ) -> None:
        self._paths_and_labels = paths_and_labels
        self._path_prefix = path_prefix

    def path_prefix(self, prefix):
        self._path_prefix = prefix

    path_prefix = property(None, path_prefix)

    def __getitem__(self, index: int) -> tuple[str, dict]:
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


def Ucf101(
    data_path: str,
    clip_sampler: ClipSampler,
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    video_path_prefix: str = "",
    split_id: int = 1,
    split_type: str = "train",
    decode_audio: bool = True,
    decoder: str = "pyav",
    decoder_args: DictConfig = {},
) -> LabeledVideoDataset:
    """A helper function to create ``LabeledVideoDataset`` object for the Ucf101 dataset.

    Args:
        data_path: Path to the data. The path type defines how the data
            should be read:

            * For a file path, the file is read and each line is parsed into a
              video path and label.
            * For a directory, the directory structure defines the classes
              (i.e. each subdirectory is a class).

        clip_sampler: Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

        video_sampler: Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

        transform: This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations to the clips. See the ``LabeledVideoDataset`` class for clip
                output format.

        video_path_prefix: Path to root directory with the videos that are
                loaded in ``LabeledVideoDataset``. All the video paths before loading
                are prefixed with this path.

        split_id: Fold id to be loaded. Options are: :math:`1`, :math:`2` or :math:`3`.

        split_type: Split/Fold type to be loaded. Options are: ``'train'`` or ``'test'``.

        decode_audio: If ``True``, also decode audio from video.

        decoder: Defines what type of decoder used to decode a video.

        decoder_args: Arguments to configure the decoder.

    Returns:
            The dataset instantiated.
    """

    torch._C._log_api_usage_once("PYTORCHVIDEO.dataset.Ucf101")
    labeled_video_paths = Ucf101LabeledVideoPaths.from_path(
        data_path,
        split_id=split_id,
        split_type=split_type,
        frames=DecoderType(decoder) == DecoderType.FRAME,
    )

    labeled_video_paths.path_prefix = video_path_prefix
    dataset = LabeledVideoDataset(
        labeled_video_paths,
        clip_sampler,
        transform,
        decode_audio,
        decoder,
        decoder_args,
    )

    return dataset
