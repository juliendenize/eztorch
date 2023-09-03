from typing import Optional

from omegaconf import DictConfig

from eztorch.datamodules.base import BaseDataModule
from eztorch.datasets.dumb_dataset import DumbDataset


class DumbDataModule(BaseDataModule):
    """Dumb data module for testing models with random data.

    Args:
        train: Configuration for the training data to define the loading
            of data and the dataloader.
        val: Configuration for the validation data to define the loading
            of data and the dataloader.
        test: Configuration for the testing data to define the loading
            of data and the dataloader.
    """

    def __init__(
        self,
        train: Optional[DictConfig] = None,
        val: Optional[DictConfig] = None,
        test: Optional[DictConfig] = None,
        num_classes: int = 10,
    ) -> None:
        super().__init__(datadir="", train=train, val=val, test=test)
        self.classes = num_classes

    @property
    def num_classes(self) -> int:
        return self.classes

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            if self.train is None:
                raise RuntimeError("No training configuration has been passed.")
            self.train_dataset = DumbDataset(**self.train.dataset)

            if self.val is not None:
                self.val_dataset = DumbDataset(**self.val.dataset)

        elif stage == "test":
            if self.test is None:
                raise RuntimeError("No testing configuration has been passed.")

            if self.test is not None:
                self.test_dataset = DumbDataset(**self.test.dataset)
