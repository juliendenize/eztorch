import os
from typing import List, Optional

import hydra
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import DictConfig

from eztorch.datamodules.base import BaseDataModule
from eztorch.datasets.folder_dataset import ImageFolder


class FolderDataModule(BaseDataModule):
    """Base datamodule for folder datasets.

    Args:
        datadir: Where to save/load the data.
        train: Configuration for the training data to define the loading of data, the transforms and the dataloader.
        val: Configuration for the validation data to define the loading of data, the transforms and the dataloader.
        test: Configuration for the testing data to define the loading of data, the transforms and the dataloader.
    """

    def __init__(
        self,
        datadir: str,
        train: Optional[DictConfig] = None,
        val: Optional[DictConfig] = None,
        test: Optional[DictConfig] = None,
    ) -> None:
        super().__init__(datadir=datadir, train=train, val=val, test=test)

    @property
    def class_list(self) -> Optional[List[str]]:
        """If not None, list of class selected."""
        return None

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

        if self.val is not None:
            self._verify_split("val")
            self._verify_classes("val")

        if self.test is not None:
            self._verify_split("test")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            if self.train is None:
                raise RuntimeError("No training configuration has been passed.")

            traindir = self.datadir / "train"

            self.train_transform = hydra.utils.instantiate(self.train.transform)

            rank_zero_info(f"Train transform: {self.train_transform}")

            if self.train.get("dataset"):
                self.train_dataset = ImageFolder(
                    traindir,
                    transform=self.train_transform,
                    class_list=self.class_list,
                    **self.train.dataset,
                )
            else:
                self.train_dataset = ImageFolder(
                    traindir, transform=self.train_transform, class_list=self.class_list
                )

            if self.val is not None:
                valdir = self.datadir / "val"

                self.val_transform = hydra.utils.instantiate(self.val.transform)
                rank_zero_info(f"Val transform: {self.val_transform}")
                if "dataset" in self.val:
                    self.val_dataset = ImageFolder(
                        valdir, transform=self.val_transform, **self.val.dataset
                    )
                else:
                    self.val_dataset = ImageFolder(valdir, transform=self.val_transform)

        elif stage == "test":
            if self.test is None:
                raise RuntimeError("No testing configuration has been passed.")

            testdir = self.datadir / "test"

            self.test_transform = hydra.utils.instantiate(self.test.transform)
            rank_zero_info(f"Test transform: {self.test_transform}")
            if "dataset" in self.test:
                self.test_dataset = ImageFolder(
                    testdir, transform=self.test_transform, **self.test.dataset
                )
            else:
                self.test_dataset = ImageFolder(testdir, transform=self.test_transform)
