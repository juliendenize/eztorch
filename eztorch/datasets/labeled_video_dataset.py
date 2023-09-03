from __future__ import annotations

import logging
import random
import warnings
from pathlib import Path
from typing import Any, Callable

from omegaconf import DictConfig
from pytorchvideo.data.clip_sampling import ClipSampler
from torch.utils.data import Dataset

from eztorch.datasets.labeled_video_paths import LabeledVideoPaths
from eztorch.datasets.utils_fn import get_raw_video_duration
from eztorch.datasets.video_path_handler import VideoPathHandler

try:
    import pandas
except ImportError:
    _HAS_PD = False
else:
    _HAS_PD = True


logger = logging.getLogger(__name__)


def create_video_files_from_folder(
    folder: str, output_folder: str, output_filename: str = "train.csv"
):
    """Create the csv files for the dataset.

    Args:
        folder: Path to the video folder.
        output_folder: Path to the frame folders.
        output_filename: Name of the output csv file.

    Raises:
        ImportError: If pandas is not installed.
    """

    if not _HAS_PD:
        raise ImportError("pandas is required to use this function.")

    folder = Path(folder)
    output_file = Path(output_folder) / output_filename

    classes = sorted(f.name for f in folder.iterdir() if f.is_dir())

    class_to_idx = {classes[i]: i for i in range(len(classes))}

    data = [
        [f"{class_folder.name}/{video_file.name}", class_to_idx[class_folder.name]]
        for class_folder in sorted(folder.iterdir())
        if class_folder.is_dir()
        for video_file in sorted(class_folder.iterdir())
        if video_file.is_file()
    ]
    df = pandas.DataFrame(data, columns=["video", "class"])

    df.to_csv(output_file, sep=" ", header=None, index=None)


def create_frames_files_from_folder(
    folder: str, output_folder: str, output_filename: str = "train.csv"
):
    """Create the dataset csv files for frame decoders.

    Args:
        folder: Path to the video folder.
        output_folder: Path to the frame folders.
        output_filename: Name of the output csv file.

    Raises:
        ImportError: If pandas is not installed.
    """

    if not _HAS_PD:
        raise ImportError("pandas is required to use this function.")

    folder = Path(folder)
    output_file = Path(output_folder) / output_filename

    classes = sorted(f.name for f in folder.iterdir() if f.is_dir())

    class_to_idx = {classes[i]: i for i in range(len(classes))}

    data = [
        [
            f"{class_folder.name}/{video_folder.name}",
            class_to_idx[class_folder.name],
            get_raw_video_duration("", video_folder),
        ]
        for class_folder in sorted(folder.iterdir())
        if class_folder.is_dir()
        for video_folder in sorted(class_folder.iterdir())
        if video_folder.is_dir()
    ]
    df = pandas.DataFrame(data, columns=["video", "class", "duration"])

    df.to_csv(output_file, sep=" ", header=None, index=None)


class LabeledVideoDataset(Dataset):
    """LabeledVideoDataset handles the storage, loading, decoding and clip sampling for a video dataset. It assumes
    each video is stored as either an encoded video (e.g. mp4, avi) or a frame video (e.g. a folder of jpg, or png)

    Args:
        labeled_video_paths: List containing
                video file paths and associated labels. If video paths are a folder
                it's interpreted as a frame video, otherwise it must be an encoded
                video.
        clip_sampler: Defines how clips should be sampled from each
            video.
        transform: This callable is evaluated on the clip output before
            the clip is returned. It can be used for user defined preprocessing and
            augmentations on the clips.
        decode_audio: If True, also decode audio from video.
        decoder: Defines what type of decoder used to decode a video.
        decoder_args: Arguments to configure the decoder.
    """

    _MAX_CONSECUTIVE_FAILURES = 10

    def __init__(
        self,
        labeled_video_paths: list[tuple[str, dict | None]],
        clip_sampler: ClipSampler,
        transform: Callable[[dict], Any] | None = None,
        decode_audio: bool = True,
        decoder: str = "pyav",
        decoder_args: DictConfig = {},
    ) -> None:
        self._decode_audio = decode_audio
        self._transform = transform
        self._clip_sampler = clip_sampler
        self._labeled_videos = labeled_video_paths
        self._decoder = decoder
        self._decoder_args = decoder_args

        self._database = None

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video_label = None
        self._loaded_clip = None
        self._next_clip_start_time = 0.0
        self.video_path_handler = VideoPathHandler()

    def __len__(self):
        """
        Returns:
            Number of videos in dataset.
        """
        return len(self._labeled_videos)

    def __getitem__(self, idx: int) -> dict:
        """Retrieves the next clip based on the clip sampling strategy and video sampler.

        Returns:
            A dictionary with the following format.

            .. code-block:: text

                {
                    'input': <video_tensor>,
                    'label': <index_label>,
                    'video_label': <index_label>
                    'idx': <idx>,
                    'clip_index': <clip_index>,
                    'aug_index': <aug_index>,
                }
        """

        for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
            # Reuse previously stored video if there are still clips to be sampled from
            # the last loaded video.
            try:
                video_path, info_dict = self._labeled_videos[idx]
                video = self.video_path_handler.video_from_path(
                    video_path,
                    decode_audio=self._decode_audio,
                    decoder=self._decoder,
                    num_frames=info_dict["num_frames"],
                    **self._decoder_args,
                )
                self._loaded_video_label = (video, info_dict)
            except Exception as e:
                old_idx = idx
                idx = random.randint(0, len(self._labeled_videos) - 1)
                warnings.warn(
                    "Failed to load video idx {} with error: {}; trial {}".format(
                        old_idx,
                        e,
                        i_try,
                    )
                )
                continue

            sample_dicts = self._load_clips_recursively(video, info_dict, idx, i_try)

            self._loaded_video_label[0].close()
            self._loaded_video_label = None
            self._next_clip_start_time = 0.0
            self._clip_sampler.reset()

            if sample_dicts is None:
                idx = random.randint(0, len(self._labeled_videos) - 1)
                continue

            return sample_dicts

        else:
            raise RuntimeError(
                f"Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries."
            )

    def _load_clips_recursively(
        self,
        video: Any,
        info_dict: dict[str, Any],
        idx: int,
        i_try: int,
    ) -> Any | list[Any] | None:
        is_last_clip = False
        is_first_clip = True
        sample_dicts = []

        while not is_last_clip:
            (
                clip_start,
                clip_end,
                clip_index,
                aug_index,
                is_last_clip,
            ) = self._clip_sampler(
                self._next_clip_start_time, video.duration, info_dict
            )

            sample_dict = self._load_clip(
                video,
                clip_start,
                clip_end,
                clip_index,
                aug_index,
                info_dict,
                idx,
                i_try,
            )

            if sample_dict is None:
                return None

            is_last_clip = (
                is_last_clip[-1] if isinstance(is_last_clip, list) else is_last_clip
            )

            if is_last_clip:
                if is_first_clip:
                    return sample_dict
                else:
                    if type(sample_dict) is list:
                        sample_dicts.extend(sample_dict)
                    else:
                        sample_dicts.append(sample_dict)
                    return sample_dicts
            else:
                if type(sample_dict) is list:
                    sample_dicts.extend(sample_dict)
                else:
                    sample_dicts.append(sample_dict)
            is_first_clip = False

    def _load_clip(
        self,
        video: Any,
        clip_start: float | list[float],
        clip_end: float | list[float],
        clip_index: int,
        aug_index: int,
        info_dict: dict[str, Any],
        idx: int,
        i_try: int,
    ) -> dict[str, Any] | None:
        if isinstance(clip_start, list):  # multi-clip in each sample
            # Only load the clips once and reuse previously stored clips if there are multiple
            # views for augmentations to perform on the same clips.
            if aug_index[0] == 0:
                self._loaded_clip = {}
                loaded_clip_list = []
                for i in range(len(clip_start)):
                    clip_dict = video.get_clip(clip_start[i], clip_end[i])
                    if clip_dict is None or clip_dict["video"] is None:
                        self._loaded_clip = None
                        break
                    loaded_clip_list.append(clip_dict)

                if self._loaded_clip is not None:
                    for key in loaded_clip_list[0].keys():
                        self._loaded_clip[key] = [x[key] for x in loaded_clip_list]

        else:  # single clip case
            # Only load the clip once and reuse previously stored clip if there are multiple
            # views for augmentations to perform on the same clip.
            if aug_index == 0:
                self._loaded_clip = video.get_clip(clip_start, clip_end)

        self._next_clip_start_time = clip_end

        video_is_null = self._loaded_clip is None or self._loaded_clip["video"] is None
        if video_is_null:
            # Close the loaded encoded video and reset the last sampled clip time ready
            # to sample a new video on the next iteration.
            if video_is_null:
                warnings.warn(
                    "Failed to load clip {} idx {}; trial {}".format(
                        video.name,
                        idx,
                        i_try,
                    )
                )
                return None

        frames = self._loaded_clip["video"]
        audio_samples = self._loaded_clip["audio"]
        sample_dict = {
            "input": frames,
            "video_name": video.name,
            "idx": idx,
            "clip_index": clip_index,
            "aug_index": aug_index,
            **info_dict,
            **({"audio": audio_samples} if audio_samples is not None else {}),
        }
        if self._transform is not None:
            sample_dict = self._transform(sample_dict)

        return sample_dict


def labeled_video_dataset(
    data_path: str,
    clip_sampler: ClipSampler,
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    video_path_prefix: str = "",
    decode_audio: bool = True,
    decoder: str = "pyav",
    decoder_args: DictConfig = {},
) -> LabeledVideoDataset:
    """A helper function to create ``LabeledVideoDataset`` object for HMDB51, Ucf101 and Kinetics datasets.

    Args:
        data_path: Path to the data. The path type defines how the data
            should be read:

            * For a file path, the file is read and each line is parsed into a
              video path and label.
            * For a directory, the directory structure defines the classes
              (i.e. each subdirectory is a class).

        clip_sampler: Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

        transform: This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations to the clips. See the ``LabeledVideoDataset`` class for clip
                output format.

        video_path_prefix: Path to root directory with the videos that are
                loaded in ``LabeledVideoDataset``. All the video paths before loading
                are prefixed with this path.

        decode_audio: If True, also decode audio from video.

        decoder: Defines what type of decoder used to decode a video.

        decoder_args: Arguments to configure the decoder.

    Returns:
        The dataset instantiated.
    """

    labeled_video_paths = LabeledVideoPaths.from_path(data_path)
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
