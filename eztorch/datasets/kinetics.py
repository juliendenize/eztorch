"""
References:
- https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/kinetics.py
"""
from typing import Any, Callable, Dict, Optional

import torch
from omegaconf import DictConfig
from pytorchvideo.data.clip_sampling import ClipSampler

from eztorch.datasets.labeled_video_dataset import (LabeledVideoDataset,
                                                    labeled_video_dataset)

"""
    Action recognition video dataset for Kinetics-{400,600,700}
    <https://deepmind.com/research/open-source/open-source-datasets/kinetics/>
"""


def Kinetics(
    data_path: str,
    clip_sampler: ClipSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = True,
    decoder: str = "pyav",
    decoder_args: DictConfig = {},
) -> LabeledVideoDataset:
    """A helper function to create ``LabeledVideoDataset`` object for the Kinetics dataset.

    Args:
        data_path: Path to the data. The path type defines how the data should be read:

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

    torch._C._log_api_usage_once("PYTORCHVIDEO.dataset.Kinetics")

    return labeled_video_dataset(
        data_path,
        clip_sampler,
        transform,
        video_path_prefix,
        decode_audio,
        decoder,
        decoder_args,
    )
