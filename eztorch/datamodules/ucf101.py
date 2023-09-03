from typing import Optional

import hydra
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import DictConfig

from eztorch.datamodules.video import VideoBaseDataModule
from eztorch.datasets.ucf101 import Ucf101


class Ucf101DataModule(VideoBaseDataModule):
    """Datamodule for the HMDB51 dataset.

    Args:
        datadir: Path to the data (eg: csv, folder, ...).
        train: Configuration for the training data to define the loading of data, the transforms and the dataloader.
        val: Configuration for the validation data to define the loading of data, the transforms and the dataloader.
        test: Configuration for the testing data to define the loading of data, the transforms and the dataloader.
        video_path_prefix: Path to root directory where the videos are stored. All the video paths before loading are prefixed with this path.
        decode_audio: If ``True``, decode audio.
        decoder: Defines which backend should be used to decode videos.
        decoder_args: Arguments to configure the decoder.
        split_id: Split used for training and testing.
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
        split_id: int = 1,
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

        self.split_id = split_id

    @property
    def num_classes(self) -> int:
        """Number of classes."""
        return 101

    def _verify_classes(self) -> None:
        dirs = [dir.stem for dir in self.datadir.iterdir() if dir.is_dir()]

        assert (
            len(dirs) == self.num_classes
        ), f"{len(dirs)}/{self.num_classes} classes found: {dirs}"

    def prepare_data(self) -> None:
        self._verify_classes()

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            if self.train is None:
                raise RuntimeError("No training configuration has been passed.")

            self.train_transform = hydra.utils.instantiate(self.train.transform)
            self.train_clip_sampler = hydra.utils.instantiate(self.train.clip_sampler)

            rank_zero_info(f"Train transform: {self.train_transform}")

            self.train_dataset = Ucf101(
                self.traindir,
                clip_sampler=self.train_clip_sampler,
                transform=self.train_transform,
                video_path_prefix=self.train_video_path_prefix,
                split_id=self.split_id,
                split_type="train",
                decode_audio=self.decode_audio,
                decoder=self.train_decoder,
                decoder_args=self.train_decoder_args,
            )

            if self.val is not None:
                self.val_transform = hydra.utils.instantiate(self.val.transform)
                self.val_clip_sampler = hydra.utils.instantiate(self.val.clip_sampler)

                rank_zero_info(f"Val transform: {self.val_transform}")

                self.val_dataset = Ucf101(
                    self.valdir,
                    clip_sampler=self.val_clip_sampler,
                    transform=self.val_transform,
                    video_path_prefix=self.val_video_path_prefix,
                    split_id=self.split_id,
                    split_type="test",
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

            self.test_dataset = Ucf101(
                self.testdir,
                clip_sampler=self.test_clip_sampler,
                transform=self.test_transform,
                video_path_prefix=self.test_video_path_prefix,
                split_id=self.split_id,
                split_type="test",
                decode_audio=self.decode_audio,
                decoder=self.test_decoder,
                decoder_args=self.test_decoder_args,
            )
