from abc import ABC
from pathlib import Path
from typing import Dict, Optional

import hydra
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import DictConfig

from eztorch.datamodules.base import BaseDataModule
from eztorch.datasets.utils_fn import get_subsample_fn


class VideoBaseDataModule(BaseDataModule, ABC):
    """Abstract class that inherits from BaseDataModule to follow standardized preprocessing for video datamodules.

    Args:
        datadir: Path to the data (eg: csv, folder, ...).
        train: Configuration for the training data to define the loading of data, the transforms and the dataloader.
        val Configuration for the validation data to define the loading of data, the transforms and the dataloader.
        test: Configuration for the testing data to define the loading of data, the transforms and the dataloader.
        video_path_prefix: Path to root directory where the videos are stored. All the video paths before loading are prefixed with this path.
        decode_audio: If ``True``, decode audio.
        decoder: Defines which backend should be used to decode videos by default.
        decoder_args: Arguments to configure the default decoder.

    .. warning::
            The loader subconfigurations must not contain 'batch_size' that is automatically computed from the 'global_batch_size' specified in the configuration.
    """

    def __init__(
        self,
        datadir: str,
        train: Optional[DictConfig] = None,
        val: Optional[DictConfig] = None,
        test: Optional[DictConfig] = None,
        video_path_prefix: str = "",
        decode_audio: bool = False,
        decoder: str = "pyav",
        decoder_args: DictConfig = {},
    ) -> None:
        super().__init__(datadir=datadir, train=train, val=val, test=test)
        self.datadir = Path(datadir)
        self.video_path_prefix = video_path_prefix

        self.decode_audio = decode_audio
        self.decoder = decoder
        # Remove dependence to OmegaConf to be able to instantiate functions after.
        self.decoder_args = dict(decoder_args)
        _instantiate_decoder_args(self.decoder_args, "default")

        if train is not None:
            train_decoder = train.get("decoder", None)
            train_decoder_args = train.get("decoder_args", None)
            self.train_decoder = (
                train_decoder if train_decoder is not None else self.decoder
            )

            if train_decoder_args is not None:
                self.train_decoder_args = dict(train_decoder_args)
                _instantiate_decoder_args(self.train_decoder_args, "train")
            else:
                self.train_decoder_args = self.decoder_args

            self.train_video_path_prefix = (
                self.train.dataset.video_path_prefix
                if self.train.get("dataset")
                and self.train.dataset.get("video_path_prefix")
                else self.video_path_prefix
            )

        if val is not None:
            val_decoder = val.get("decoder", None)
            val_decoder_args = val.get("decoder_args", None)
            self.val_decoder = val_decoder if val_decoder is not None else self.decoder

            if val_decoder_args is not None:
                self.val_decoder_args = dict(val_decoder_args)
                _instantiate_decoder_args(self.val_decoder_args, "val")
            else:
                self.val_decoder_args = self.decoder_args

            self.val_video_path_prefix = (
                self.val.dataset.video_path_prefix
                if self.val.get("dataset") and self.val.dataset.get("video_path_prefix")
                else self.video_path_prefix
            )

        if test is not None:
            test_decoder = test.get("decoder", None)
            test_decoder_args = test.get("decoder_args", None)
            self.test_decoder = (
                test_decoder if test_decoder is not None else self.decoder
            )
            if test_decoder_args is not None:
                self.test_decoder_args = dict(test_decoder_args)
                _instantiate_decoder_args(self.test_decoder_args, "test")
            else:
                self.test_decoder_args = self.decoder_args

            self.test_video_path_prefix = (
                self.test.dataset.video_path_prefix
                if self.test.get("dataset")
                and self.test.dataset.get("video_path_prefix")
                else self.video_path_prefix
            )


def _instantiate_decoder_args(decoder_args: Dict | None, split: str = "train") -> None:
    for key, value in decoder_args.items():
        if key == "frame_filter" and not callable(value):
            # recursive is False so manual instantiation
            # hydra 1.1 does not support partial/function instantiation so manually calling the function
            decoder_args[key] = get_subsample_fn(
                decoder_args["frame_filter"]["subsample_type"],
                decoder_args["frame_filter"]["num_samples"],
            )
        elif key == "transform" and not callable(value):
            decoder_args[key] = hydra.utils.instantiate(decoder_args[key])
            rank_zero_info(f"Decoder {split} transform: {decoder_args[key]}")
