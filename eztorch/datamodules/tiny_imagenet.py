from typing import Optional

from omegaconf import DictConfig

from eztorch.datamodules.imagenet import ImagenetDataModule


class TinyImagenetDataModule(ImagenetDataModule):
    """Base datamodule for the Tiny Imagenet dataset.

    Args:
        datadir: Where to load the data.
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
    def num_classes(self) -> int:
        """Number of classes."""
        return 200
