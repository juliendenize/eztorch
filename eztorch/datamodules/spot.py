from abc import ABC
from typing import Optional

import hydra
import torch.distributed as dist
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from eztorch.datamodules.video import VideoBaseDataModule
from eztorch.datasets.clip_samplers.spot.distributed_sampler_wrapper import \
    SpotClipSamplerDistributedSamplerWrapper
from eztorch.datasets.spot import spot_dataset
from eztorch.datasets.spot_utils.parse_utils import SpotDatasets


class SpotDataModule(VideoBaseDataModule, ABC):
    """Base datamodule for the SoccerNet datasets.

    Args:
        datadir: Path to the data (eg: csv, folder, ...).
        train: Configuration for the training data to define the loading of data, the transforms and the dataloader.
        val: Configuration for the validation data to define the loading of data, the transforms and the dataloader.
        test: Configuration for the testing data to define the loading of data, the transforms and the dataloader.
        video_path_prefix: Path to root directory where the videos are stored. All the video paths before loading are prefixed with this path.
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
        decoder: str = "frame",
        decoder_args: DictConfig = {},
    ) -> None:
        super().__init__(
            datadir=datadir,
            train=train,
            val=val,
            test=test,
            video_path_prefix=video_path_prefix,
            decoder=decoder,
            decode_audio=False,
            decoder_args=decoder_args,
        )

        self.train_clip_sampler = None
        self.val_clip_sampler = None
        self.test_clip_sampler = None

    @property
    def num_classes(self) -> int:
        return -1

    def prepare_data(self) -> None:
        # TODO: verify that all files are present for the dataset.
        super().prepare_data()

    @property
    def train_num_samples(self) -> int:
        """Number of samples in the training dataset."""
        if type(self.train_clip_sampler) is SpotClipSamplerDistributedSamplerWrapper:
            return len(self.train_clip_sampler.dataset)
        return len(self.train_clip_sampler) if self.train_clip_sampler else 0

    @property
    def val_num_samples(self) -> int:
        """Number of samples in the validation dataset."""
        if type(self.val_clip_sampler) is SpotClipSamplerDistributedSamplerWrapper:
            return len(self.val_clip_sampler.dataset)
        return len(self.val_clip_sampler) if self.val_clip_sampler else 0

    @property
    def test_num_samples(self) -> int:
        """Number of samples in the testing dataset."""
        if type(self.test_clip_sampler) is SpotClipSamplerDistributedSamplerWrapper:
            return len(self.test_clip_sampler.dataset)
        return len(self.test_clip_sampler) if self.test_clip_sampler else 0

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            if self.train is None:
                raise RuntimeError("No training configuration has been passed.")

            self.train_transform = hydra.utils.instantiate(self.train.transform)

            rank_zero_info(f"Train transform: {self.train_transform}")

            self.train_dataset = spot_dataset(
                self.traindir,
                transform=self.train_transform,
                video_path_prefix=self.train_video_path_prefix,
                decoder=self.train_decoder,
                decoder_args=self.train_decoder_args,
                label_args=self.train.dataset.get("label_args", {}),
                features_args=self.train.dataset.get("feature_args", None),
                dataset=self.train.dataset.get("dataset", SpotDatasets.TENNIS),
            )

            self.train_clip_sampler = hydra.utils.instantiate(
                self.train.clip_sampler, data_source=self.train_dataset
            )

            if dist.is_available() and dist.is_initialized():
                self.train_clip_sampler = SpotClipSamplerDistributedSamplerWrapper(
                    self.train_clip_sampler
                )

            rank_zero_info(
                f"Use {self.train_clip_sampler} for train sampler, make sure you correctly configured the sampler."
            )

            if self.val is not None:
                self.val_transform = hydra.utils.instantiate(self.val.transform)

                rank_zero_info(f"Val transform: {self.val_transform}")

                self.val_dataset = spot_dataset(
                    self.valdir,
                    transform=self.val_transform,
                    video_path_prefix=self.val_video_path_prefix,
                    decoder=self.val_decoder,
                    decoder_args=self.val_decoder_args,
                    label_args=self.val.dataset.get("label_args", {}),
                    features_args=self.val.dataset.get("feature_args", None),
                    dataset=self.val.dataset.get("dataset", SpotDatasets.TENNIS),
                )

                self.val_clip_sampler = hydra.utils.instantiate(
                    self.val.clip_sampler, data_source=self.val_dataset
                )

                if dist.is_available() and dist.is_initialized():
                    self.val_clip_sampler = SpotClipSamplerDistributedSamplerWrapper(
                        self.val_clip_sampler
                    )

                rank_zero_info(
                    f"Use {self.val_clip_sampler} for val sampler, make sure you correctly configured the sampler."
                )

        elif stage == "test":
            if self.test is None:
                raise RuntimeError("No testing configuration has been passed.")

            self.test_transform = hydra.utils.instantiate(self.test.transform)

            rank_zero_info(f"Test transform: {self.test_transform}")

            self.test_dataset = spot_dataset(
                self.testdir,
                transform=self.test_transform,
                video_path_prefix=self.test_video_path_prefix,
                decoder=self.test_decoder,
                decoder_args=self.test_decoder_args,
                label_args=self.test.dataset.get("label_args", {}),
                features_args=self.test.dataset.get("feature_args", None),
                dataset=self.test.dataset.get("dataset", SpotDatasets.TENNIS),
            )

            self.test_clip_sampler = hydra.utils.instantiate(
                self.test.clip_sampler, data_source=self.test_dataset
            )

            if dist.is_available() and dist.is_initialized():
                self.test_clip_sampler = SpotClipSamplerDistributedSamplerWrapper(
                    self.test_clip_sampler
                )

            rank_zero_info(
                f"Use {self.test_clip_sampler} for test sampler, make sure you correctly configured the sampler."
            )

    def train_dataloader(self) -> DataLoader:
        if self.train is None:
            raise RuntimeError(
                "No passed training configuration so dataloader cannot be retrieved."
            )

        loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_local_batch_size,
            collate_fn=self.train_collate_fn,
            sampler=self.train_clip_sampler,
            **self.train.loader,
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        if self.val is None:
            raise RuntimeError(
                "No passed validation configuration so dataloader cannot be retrieved."
            )

        loader = DataLoader(
            self.val_dataset,
            batch_size=self.val_local_batch_size,
            collate_fn=self.val_collate_fn,
            sampler=self.val_clip_sampler,
            **self.val.loader,
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        if self.test is None:
            raise RuntimeError(
                "No passed testing configuration so dataloader cannot be retrieved."
            )

        loader = DataLoader(
            self.test_dataset,
            batch_size=self.test_local_batch_size,
            collate_fn=self.test_collate_fn,
            sampler=self.test_clip_sampler,
            **self.test.loader,
        )
        return loader
