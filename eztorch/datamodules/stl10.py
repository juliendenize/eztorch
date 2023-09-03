from typing import Optional

import hydra
from omegaconf import DictConfig
from torchvision import transforms
from torchvision.datasets import STL10

from eztorch.datamodules.base import BaseDataModule
from eztorch.datasets.dict_dataset import DictDataset


class STL10DataModule(BaseDataModule):
    """Datamodule for the STL10 dataset in SSL setting.

    Args:
        datadir: Where to save/load the data.
        train: Configuration for the training data to define the loading of data, the transforms and the dataloader.
        val: Configuration for the validation data to define the loading of data, the transforms and the dataloader.
        test: Configuration for the testing data to define the loading of data, the transforms and the dataloader.
        folds: One of :math:`{0-9}` or ``None``. For training, loads one of the 10 pre-defined folds of 1k samples for the standard evaluation procedure. If no value is passed, loads the 5k samples.
        training_split: Split used for the training dataset.
    """

    def __init__(
        self,
        datadir: str,
        train: Optional[DictConfig] = None,
        val: Optional[DictConfig] = None,
        test: Optional[DictConfig] = None,
        folds: Optional[int] = None,
        training_split: str = "unlabeled",
    ) -> None:
        super().__init__(datadir=datadir, train=train, val=val, test=test)
        self.folds = folds
        self.training_split = training_split

    @property
    def num_classes(self) -> int:
        """Number of classeS."""
        return 10

    def prepare_data(self):
        if self.train is not None:
            STL10(
                self.datadir,
                split="unlabeled",
                download=True,
                transform=transforms.PILToTensor(),
            )
            STL10(
                self.datadir,
                folds=self.folds,
                split="train",
                download=True,
                transform=transforms.PILToTensor(),
            )

        if self.val is not None or self.test is not None:
            STL10(
                self.datadir,
                split="test",
                download=True,
                transform=transforms.PILToTensor(),
            )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            assert self.training_split in ["train", "train+unlabeled", "unlabeled"]

            self.train_transform = hydra.utils.instantiate(self.train.transform)
            self.train_dataset = DictDataset(
                STL10(
                    self.datadir,
                    folds=self.folds,
                    split=self.training_split,
                    download=False,
                    transform=self.train_transform,
                )
            )

            if self.val is not None:
                self.val_transform = hydra.utils.instantiate(self.val.transform)
                self.val_dataset = DictDataset(
                    STL10(
                        self.datadir,
                        split="test",
                        download=False,
                        transform=self.val_transform,
                    )
                )

        elif stage == "test":
            self.test_transform = hydra.utils.instantiate(self.test.transform)
            self.test_dataset = DictDataset(
                STL10(self.datadir, split="test", transform=self.test_transform)
            )
