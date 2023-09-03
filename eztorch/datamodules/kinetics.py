import os
from abc import ABC
from typing import Optional

import hydra
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import DictConfig

from eztorch.datamodules.video import VideoBaseDataModule
from eztorch.datasets.kinetics import Kinetics


class KineticsDataModule(VideoBaseDataModule, ABC):
    """Base datamodule for the Kinetics datasets.

    Args:
        datadir: Path to the data (eg: csv, folder, ...).
        train: Configuration for the training data to define the loading of data, the transforms and the dataloader.
        val: Configuration for the validation data to define the loading of data, the transforms and the dataloader.
        test: Configuration for the testing data to define the loading of data, the transforms and the dataloader.
        video_path_prefix: Path to root directory where the videos are stored. All the video paths before loading are prefixed with this path.
        decode_audio: If ``True``, decode audio.
        decoder: Defines which backend should be used to decode videos.
        decoder_args: Arguments to configure the decoder.
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
        super().__init__(
            datadir=datadir,
            train=train,
            val=val,
            test=test,
            video_path_prefix=video_path_prefix,
            decode_audio=decode_audio,
            decoder=decoder,
            decoder_args=decoder_args,
        )

    def _verify_classes(self, split: str = "train") -> None:
        split_dir = self.datadir / split
        dirs = [dir.stem for dir in split_dir.iterdir() if dir.is_dir()]

        assert (
            len(dirs) == self.num_classes
        ), f"{len(dirs)}/{self.num_classes} classes found: {dirs}"

    def _verify_split(self, split: str) -> None:
        dirs = [dir.stem for dir in self.datadir.iterdir()]

        if split not in dirs:
            raise FileNotFoundError(
                f"the split '{split}' was not found in {os.path.abspath(self.datadir)},"
                f" make sure the folder contains a subfolder named {split}"
            )

    def prepare_data(self) -> None:
        if self.train is not None:
            self._verify_split("train")
            self._verify_classes("train")

        if self.val is not None or self.test is not None:
            self._verify_split("val")
            self._verify_classes("val")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            if self.train is None:
                raise RuntimeError("No training configuration has been passed.")

            self.train_transform = hydra.utils.instantiate(self.train.transform)
            self.train_clip_sampler = hydra.utils.instantiate(self.train.clip_sampler)

            rank_zero_info(f"Train transform: {self.train_transform}")

            self.train_dataset = Kinetics(
                self.traindir,
                clip_sampler=self.train_clip_sampler,
                transform=self.train_transform,
                video_path_prefix=self.train_video_path_prefix,
                decode_audio=self.decode_audio,
                decoder=self.train_decoder,
                decoder_args=self.train_decoder_args,
            )

            if self.val is not None:
                self.val_transform = hydra.utils.instantiate(self.val.transform)
                self.val_clip_sampler = hydra.utils.instantiate(self.val.clip_sampler)

                rank_zero_info(f"Val transform: {self.val_transform}")

                self.val_dataset = Kinetics(
                    self.valdir,
                    clip_sampler=self.val_clip_sampler,
                    transform=self.val_transform,
                    video_path_prefix=self.val_video_path_prefix,
                    decode_audio=self.decode_audio,
                    decoder=self.val_decoder,
                    decoder_args=self.val_decoder_args,
                )

        elif stage == "test":
            if self.test is None:
                raise RuntimeError("No testing configuration has been passed.")

            self.test_transform = hydra.utils.instantiate(self.test.transform)
            self.test_clip_sampler = hydra.utils.instantiate(self.test.clip_sampler)

            rank_zero_info(f"Test transform: {self.test_transform}")

            self.test_dataset = Kinetics(
                self.testdir,
                clip_sampler=self.test_clip_sampler,
                transform=self.test_transform,
                video_path_prefix=self.test_video_path_prefix,
                decode_audio=self.decode_audio,
                decoder=self.test_decoder,
                decoder_args=self.test_decoder_args,
            )


class Kinetics400DataModule(KineticsDataModule):
    """Datamodule for the Kinetics400 datasets.

    Args:
        datadir: Path to the data (eg: csv, folder, ...).
        train: Configuration for the training data to define the loading of data, the transforms and the dataloader.
        val: Configuration for the validation data to define the loading of data, the transforms and the dataloader.
        test: Configuration for the testing data to define the loading of data, the transforms and the dataloader.
        video_path_prefix: Path to root directory where the videos are stored. All the video paths before loading are prefixed with this path.
        decode_audio: If ``True``, decode audio.
        decoder: Defines which backend should be used to decode videos.
        decoder_args: Arguments to configure the decoder.
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
        super().__init__(
            datadir=datadir,
            train=train,
            val=val,
            test=test,
            video_path_prefix=video_path_prefix,
            decode_audio=decode_audio,
            decoder=decoder,
            decoder_args=decoder_args,
        )

    @property
    def num_classes(self) -> int:
        return 400


class Kinetics200DataModule(KineticsDataModule):
    """Datamodule for the Mini-Kinetics200 dataset.

    Args:
        datadir: Path to the data (eg: csv, folder, ...).
        train: Configuration for the training data to define the loading of data, the transforms and the dataloader.
        val: Configuration for the validation data to define the loading of data, the transforms and the dataloader.
        test: Configuration for the testing data to define the loading of data, the transforms and the dataloader.
        video_path_prefix: Path to root directory with the videos that are loaded in LabeledVideoDataset. All the video paths before loading are prefixed with this path.
        decode_audio : If True, decode audio.
        decoder: Defines which backend should be used to decode videos.
        decoder_args: Arguments to configure the decoder.
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
        super().__init__(
            datadir=datadir,
            train=train,
            val=val,
            test=test,
            video_path_prefix=video_path_prefix,
            decode_audio=decode_audio,
            decoder=decoder,
            decoder_args=decoder_args,
        )

    @property
    def num_classes(self) -> int:
        """Number of classes."""
        return 200


class Kinetics600DataModule(KineticsDataModule):
    """Datamodule for the Kinetics600 datasets.

    Args:
        datadir: Path to the data (eg: csv, folder, ...).
        train: Configuration for the training data to define the loading of data, the transforms and the dataloader.
        val: Configuration for the validation data to define the loading of data, the transforms and the dataloader.
        test: Configuration for the testing data to define the loading of data, the transforms and the dataloader.
        video_path_prefix: Path to root directory where the videos are stored. All the video paths before loading are prefixed with this path.
        decode_audio: If ``True``, decode audio.
        decoder: Defines which backend should be used to decode videos.
        decoder_args: Arguments to configure the decoder.
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
        super().__init__(
            datadir=datadir,
            train=train,
            val=val,
            test=test,
            video_path_prefix=video_path_prefix,
            decode_audio=decode_audio,
            decoder=decoder,
            decoder_args=decoder_args,
        )

    @property
    def num_classes(self) -> int:
        """Number of classes."""
        return 600


class Kinetics700DataModule(KineticsDataModule):
    """Datamodule for the Kinetics700 datasets.

    Args:
        datadir: Path to the data (eg: csv, folder, ...).
        train: Configuration for the training data to define the loading of data, the transforms and the dataloader.
        val: Configuration for the validation data to define the loading of data, the transforms and the dataloader.
        test: Configuration for the testing data to define the loading of data, the transforms and the dataloader.
        video_path_prefix: Path to root directory where the videos are stored. All the video paths before loading are prefixed with this path.
        decode_audio: If ``True``, decode audio.
        decoder: Defines which backend should be used to decode videos.
        decoder_args: Arguments to configure the decoder.
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
        super().__init__(
            datadir=datadir,
            train=train,
            val=val,
            test=test,
            video_path_prefix=video_path_prefix,
            decode_audio=decode_audio,
            decoder=decoder,
            decoder_args=decoder_args,
        )

    @property
    def num_classes(self) -> int:
        """Number of classes."""
        return 700
