"""
References:
- https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/hmdb51.py
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from functools import partial
from pathlib import Path
from typing import Any, Callable

import torch
import torch.utils.data
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

logger = logging.getLogger(__name__)


def create_hmdb51_files_for_frames(
    folder_files: str, frames_folder: str, split_id: int
):
    """Create the HMDB51 csv files for frame decoders.

    Args:
        folder_files: Path to the original hmdb51 split files.
        frames_folder: Path to the frame folders.
        split_id: The split id.

    Raises:
        ImportError: If pandas is not installed.
    """
    if not _HAS_PD:
        raise ImportError("pandas is required to use this function.")

    folder_files = Path(folder_files)
    frames_folder = Path(frames_folder)

    file_name_format = "_test_split" + str(int(split_id))
    files = sorted(
        f
        for f in folder_files.iterdir()
        if f.is_file() and f.suffix == ".txt" and file_name_format in f.stem
    )

    action_dict = {}
    data = []
    for file in files:
        curr_data = pandas.read_csv(
            file, sep=" ", header=None, names=["video", "split"], usecols=[0, 1]
        )
        action_name = "_"
        action_name = action_name.join((file.stem).split("_")[:-2])
        if action_name not in action_dict:
            action_dict[action_name] = len(action_dict)
        curr_data["class"] = action_dict[action_name]
        curr_data.video = action_name + "/" + curr_data.video
        curr_data.video = curr_data.video.map(remove_suffix)
        curr_data["duration"] = curr_data.video.map(
            partial(get_raw_video_duration, frames_folder)
        )
        data.append(curr_data)
    data = pandas.concat(data, axis=0)

    set_to_num = {"train": 1, "test": 2, "unlabeled": 0}
    for set in ["train", "test", "unlabeled"]:
        output_file = frames_folder / f"{set}list{split_id:02d}.txt"
        output_data = data.loc[set_to_num[set] == data.split]
        output_data = output_data.drop(columns=["split"])
        output_data.to_csv(output_file, sep=" ", header=None, index=None)


class Hmdb51LabeledVideoPaths:
    """Pre-processor for Hmbd51 dataset mentioned here - https://serre-lab.clps.brown.edu/resource/hmdb-a-large-
    human-motion-database/

    This dataset consists of classwise folds with each class consisting of 3
        folds (splits).

    The videos directory is of the format,
        video_dir_path/class_x/<somevideo_name>.avi
        ...
        video_dir_path/class_y/<somevideo_name>.avi

    The splits/fold directory is of the format,
        folds_dir_path/class_x_test_split_1.txt
        folds_dir_path/class_x_test_split_2.txt
        folds_dir_path/class_x_test_split_3.txt
        ...
        folds_dir_path/class_y_test_split_1.txt
        folds_dir_path/class_y_test_split_2.txt
        folds_dir_path/class_y_test_split_3.txt

    And each text file in the splits directory class_x_test_split_<1 or 2 or 3>.txt
        <a video as in video_dir_path/class_x> <0 or 1 or 2>
        where 0,1,2 corresponds to unused, train split respectively.

    Each video has name of format
        <some_name>_<tag1>_<tag2>_<tag_3>_<tag4>_<tag5>_<some_id>.avi
    For more details on tags -
        https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
    """

    _allowed_splits = [1, 2, 3]
    _split_type_dict = {"train": 1, "test": 2, "unused": 0}

    @classmethod
    def from_path(
        cls,
        data_path: str,
        split_id: int = 1,
        split_type: str = "train",
        frames: bool = False,
    ) -> Hmdb51LabeledVideoPaths:
        """Factory function that creates a Hmdb51LabeledVideoPaths object depending on the path type.

            - If it is a directory path it uses the Hmdb51LabeledVideoPaths.from_directory function.
            - If it's a file it uses the Hmdb51LabeledVideoPaths.from_csv file.
        Args:
            file_path: The path to the file or directory to be read.
            split_id: Split id. Used if path is a directory.
            split_type: Split type. Used if path is a directory.
            frames: If ``True``, UCF101 is loaded as a frame dataset.
        Returns:
            The Hmdb51LabeledVideoPaths object.
        """

        if g_pathmgr.isfile(data_path):
            if pathlib.Path(data_path).suffix == ".json":
                return Hmdb51LabeledVideoPaths.from_json(data_path)
            return Hmdb51LabeledVideoPaths.from_csv(data_path)
        elif g_pathmgr.isdir(data_path):
            split_list = os.path.join(data_path, f"{split_type}list0{split_id}.txt")
            if g_pathmgr.isfile(split_list):
                return Hmdb51LabeledVideoPaths.from_csv(split_list)
            return Hmdb51LabeledVideoPaths.from_directory(
                data_path, split_id, split_type, frames
            )
        else:
            raise FileNotFoundError(f"{data_path} not found.")

    @classmethod
    def from_csv(cls, file_path: str) -> Hmdb51LabeledVideoPaths:
        """Factory function that creates a Hmdb51LabeledVideoPaths object by reading a file with the following
        format:

            <path> <integer_label>
            ...
            <path> <integer_label>

        Args:
            file_path: The path to the file to be read.

        Returns:
            The Hmdb51LabeledVideoPaths object.
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
    def from_json(cls, file_path: str) -> Hmdb51LabeledVideoPaths:
        """Factory function that creates a Hmdb51LabeledVideoPaths object by reading a json file.

        Args:
            file_path: The path to the file to be read.

        Returns:
            The Hmdb51LabeledVideoPaths object.
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
    ) -> Hmdb51LabeledVideoPaths:
        """Factory function that creates Hmdb51LabeledVideoPaths object form a splits/folds directory.

        Args:
            data_path (str): The path to the splits/folds directory of HMDB51.
            split_id (int): Fold id to be loaded. Belongs to [1,2,3]
            split_type (str): Split/Fold type to be loaded. It belongs to one of the
                following,
                - "train"
                - "test"
                - "unused" (This is a small set of videos that are neither
                of part of test or train fold.)
            frames (bool): If True, UCF101 is loaded as a frame dataset.

        Returns:
            The Hmdb51LabeledVideoPaths object.
        """
        data_path = pathlib.Path(data_path)
        if not data_path.is_dir():
            raise RuntimeError(f"{data_path} not found or is not a directory.")
        if not int(split_id) in cls._allowed_splits:
            raise RuntimeError(
                f"{split_id} not found in allowed split id's {cls._allowed_splits}."
            )
        file_name_format = "_test_split" + str(int(split_id))
        file_paths = sorted(
            f
            for f in data_path.iterdir()
            if f.is_file() and f.suffix == ".txt" and file_name_format in f.stem
        )
        return cls.from_csvs(file_paths, split_type, frames)

    @classmethod
    def from_csvs(
        cls,
        file_paths: list[pathlib.Path | str],
        split_type: str = "train",
        frames: bool = False,
    ) -> Hmdb51LabeledVideoPaths:
        """Factory function that creates Hmdb51LabeledVideoPaths object form a list of split files of .txt type.

        Args:
            file_paths (List[Union[pathlib.Path, str]]) : The path to the splits/folds
                    directory of HMDB51.
            split_type (str): Split/Fold type to be loaded.
                - "train"
                - "test"
                - "unused"
            frames: If True, search for the number of frames for each videos.

        Returns:
            The LabeledVideoPaths object.
        """
        action_name_to_class = {}
        video_paths_and_label = []
        for file_path in file_paths:
            file_path = pathlib.Path(file_path)
            assert g_pathmgr.exists(file_path), f"{file_path} not found."
            if not (file_path.suffix == ".txt" and "_test_split" in file_path.stem):
                raise RuntimeError(f"Invalid file: {file_path}")

            action_name = "_"
            action_name = action_name.join((file_path.stem).split("_")[:-2])

            if action_name not in action_name_to_class:
                action_name_to_class[action_name] = len(action_name_to_class)

            with g_pathmgr.open(file_path, "r") as f:
                for path_label in f.read().splitlines():
                    line_split = path_label.rsplit(None, 1)

                    if not int(line_split[1]) == cls._split_type_dict[split_type]:
                        continue

                    video_name = pathlib.Path(line_split[0])

                    if frames:
                        file_path = pathlib.Path(action_name) / video_name.stem
                        num_frames = len(list(file_path.iterdir()))
                    else:
                        file_path = pathlib.Path(action_name) / video_name
                        num_frames = None

                    video_paths_and_label.append(
                        (str(file_path), action_name_to_class[action_name], num_frames)
                    )

        assert (
            len(video_paths_and_label) > 0
        ), f"Failed to load dataset from {file_path}."
        return cls(video_paths_and_label)

    def __init__(
        self, paths_and_labels: list[tuple[str, dict | None]], path_prefix=""
    ) -> None:
        """
        Args:
            paths_and_labels [(str, int)]: a list of tuples containing the video
                path and integer label.
        """
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


def Hmdb51(
    data_path: pathlib.Path,
    clip_sampler: ClipSampler,
    transform: Callable[[dict], Any] | None = None,
    video_path_prefix: str = "",
    split_id: int = 1,
    split_type: str = "train",
    decode_audio=True,
    decoder: str = "pyav",
    decoder_args: DictConfig = {},
) -> LabeledVideoDataset:
    """A helper function to create ``LabeledVideoDataset`` object for HMDB51 dataset.

    Args:
        data_path (pathlib.Path): Path to the data. The path type defines how the data
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
            augmentations to the clips. See the ``LabeledVideoDataset`` class for
            clip output format.

        video_path_prefix: Path to root directory with the videos that are
            loaded in LabeledVideoDataset. All the video paths before loading
            are prefixed with this path.

        split_id: Fold id to be loaded. Options are 1, 2 or 3

        split_type: Split/Fold type to be loaded. Options are ("train", "test" or
            "unused")

        decoder: Defines which backend should be used to decode videos.

        decoder_args: Arguments to configure the decoder.

    Returns:
        The dataset instantiated.
    """

    torch._C._log_api_usage_once("PYTORCHVIDEO.dataset.Hmdb51")

    labeled_video_paths = Hmdb51LabeledVideoPaths.from_path(
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
        decode_audio=decode_audio,
        decoder=decoder,
        decoder_args=decoder_args,
    )

    return dataset
