from abc import ABC, abstractproperty
from typing import Optional

import hydra
import torch
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import DictConfig
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10, CIFAR100, VisionDataset
from torchvision.transforms import PILToTensor

from eztorch.datamodules.base import BaseDataModule
from eztorch.datasets.dict_dataset import DictDataset


class CIFARDataModule(BaseDataModule, ABC):
    """Base datamodule for the CIFAR datasets.

    Args:
        datadir: Where to save/load the data.
        train: Configuration for the training data to define the loading of data, the transforms and the dataloader.
        val: Configuration for the validation data to define the loading of data, the transforms and the dataloader.
        test: Configuration for the testing data to define the loading of data, the transforms and the dataloader.
        num_classes_kept: Number of classes to use.
        split_train_ratio: If not ``None`` randomly split the train dataset in two with split_train_ration ratio for train.
        seed_for_split: Seed for the split.
    """

    def __init__(
        self,
        datadir: str,
        train: Optional[DictConfig] = None,
        val: Optional[DictConfig] = None,
        test: Optional[DictConfig] = None,
        num_classes_kept: Optional[int] = None,
        split_train_ratio: Optional[float] = None,
        seed_for_split: int = 42,
    ) -> None:
        super().__init__(datadir=datadir, train=train, val=val, test=test)

        self.num_classes_kept = num_classes_kept
        self.split_train_ratio = split_train_ratio
        self.seed_for_split = seed_for_split

        assert num_classes_kept is None or num_classes_kept <= self.num_classes

    @abstractproperty
    def DATASET(self) -> VisionDataset:
        """Dataset class that should be defined by subclasses."""
        return None

    def prepare_data(self) -> None:
        if self.train is not None:
            self.DATASET(
                self.datadir, train=True, download=True, transform=PILToTensor()
            )

        if self.val is not None or self.test is not None:
            self.DATASET(
                self.datadir, train=False, download=True, transform=PILToTensor()
            )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            if self.train is None:
                raise RuntimeError("No training configuration has been passed.")

            self.train_transform = hydra.utils.instantiate(self.train.transform)
            rank_zero_info(f"Train transform: {self.train_transform}")

            if self.split_train_ratio is None:
                self.train_dataset = DictDataset(
                    self.DATASET(
                        self.datadir,
                        train=True,
                        download=False,
                        transform=self.train_transform,
                    )
                )

                if self.val is not None:
                    self.val_transform = hydra.utils.instantiate(self.val.transform)
                    rank_zero_info(f"Val transform: {self.val_transform}")
                    self.val_dataset = DictDataset(
                        self.DATASET(
                            self.datadir,
                            train=False,
                            download=False,
                            transform=self.val_transform,
                        )
                    )

            else:
                train_dataset = self.DATASET(
                    self.datadir,
                    train=True,
                    download=False,
                    transform=self.train_transform,
                )
                train_length = round(len(train_dataset) * self.split_train_ratio)
                val_length = len(train_dataset) - train_length
                train_dataset, val_dataset = random_split(
                    train_dataset,
                    [train_length, val_length],
                    torch.Generator().manual_seed(self.seed_for_split),
                )
                self.train_dataset = DictDataset(train_dataset)

                if self.val is not None:
                    self.val_transform = hydra.utils.instantiate(self.val.transform)
                    rank_zero_info(f"Val transform: {self.val_transform}")
                    self.val_dataset = val_dataset
                    self.val_dataset.transform = self.val_transform
                    self.val_dataset = DictDataset(val_dataset)

            if self.num_classes_kept is not None:
                targets = torch.tensor(self.train_dataset.source_dataset.targets)
                indices_to_keep = targets < self.num_classes_kept
                self.train_dataset = torch.utils.data.Subset(
                    self.train_dataset, indices_to_keep.nonzero()
                )

            if self.val is not None and self.num_classes_kept is not None:
                targets = torch.tensor(self.val_dataset.source_dataset.targets)
                indices_to_keep = targets < self.num_classes_kept
                self.val_dataset = torch.utils.data.Subset(
                    self.val_dataset, indices_to_keep.nonzero()
                )

        elif stage == "test":
            if self.test is None:
                raise RuntimeError("No testing configuration has been passed.")

            self.test_transform = hydra.utils.instantiate(self.test.transform)
            rank_zero_info(f"Test transform: {self.test_transform}")
            self.test_dataset = DictDataset(
                self.DATASET(
                    self.datadir,
                    train=False,
                    download=False,
                    transform=self.test_transform,
                )
            )

            if self.test_dataset is not None:
                targets = torch.tensor(self.test_dataset.source_dataset.targets)
                indices_to_keep = targets < self.num_classes_kept
                self.test_dataset = torch.utils.data.Subset(
                    self.test_dataset, indices_to_keep.nonzero()
                )


class CIFAR10DataModule(CIFARDataModule):
    """Datamodule for the CIFAR10 dataset.

    Args:
        datadir: Where to save/load the data.
        train: Configuration for the training data to define the loading of data, the transforms and the dataloader.
        val: Configuration for the validation data to define the loading of data, the transforms and the dataloader.
        test: Configuration for the testing data to define the loading of data, the transforms and the dataloader.
        num_classes_kept: Number of classes to use.
        split_train_ratio: If not ``None`` randomly split the train dataset in two with split_train_ration ratio for train.
        seed_for_split: Seed for the split.

    Example::

        datamodule = CIFAR10DataModule(datadir)
    """

    def __init__(
        self,
        datadir: str,
        train: Optional[DictConfig] = None,
        val: Optional[DictConfig] = None,
        test: Optional[DictConfig] = None,
        num_classes_kept: Optional[int] = None,
        split_train_ratio: Optional[float] = None,
        seed_for_split: int = 42,
    ) -> None:
        super().__init__(
            datadir=datadir,
            train=train,
            val=val,
            test=test,
            num_classes_kept=num_classes_kept,
            split_train_ratio=split_train_ratio,
            seed_for_split=seed_for_split,
        )

    @property
    def DATASET(self) -> VisionDataset:
        """Dataset class."""
        return CIFAR10

    @property
    def num_classes(self) -> int:
        """Number of classes."""
        return 10 if self.num_classes_kept is None else self.num_classes_kept


class CIFAR100DataModule(CIFARDataModule):
    """Datamodule for the CIFAR100 dataset.

    Args:
        datadir: Where to save/load the data.
        train: Configuration for the training data to define the loading of data, the transforms and the dataloader.
        val: Configuration for the validation data to define the loading of data, the transforms and the dataloader.
        test: Configuration for the testing data to define the loading of data, the transforms and the dataloader.
        num_classes_kept: Number of classes to use.
        split_train_ratio: If not ``None`` randomly split the train dataset in two with split_train_ration ratio for train.
        seed_for_split: Seed for the split.

    Example::

        datamodule = CIFAR100DataModule(datadir)
    """

    def __init__(
        self,
        datadir: str,
        train: Optional[DictConfig] = None,
        val: Optional[DictConfig] = None,
        test: Optional[DictConfig] = None,
        num_classes_kept: Optional[int] = None,
        split_train_ratio: Optional[float] = None,
        seed_for_split: int = 42,
    ) -> None:
        super().__init__(
            datadir=datadir,
            train=train,
            val=val,
            test=test,
            num_classes_kept=num_classes_kept,
            split_train_ratio=split_train_ratio,
            seed_for_split=seed_for_split,
        )

    @property
    def DATASET(self) -> VisionDataset:
        """Dataset class."""
        return CIFAR100

    @property
    def num_classes(self) -> int:
        """Number of classes."""
        return 100 if self.num_classes_kept is None else self.num_classes_kept
