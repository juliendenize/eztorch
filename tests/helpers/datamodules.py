from typing import Optional

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from tests.helpers.datasets import RandomDataset


class BoringDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        dataset=RandomDataset((32, 64 * 4)),
        val_dataset=RandomDataset((32, 64 * 4)),
        batch_size: int = 1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.non_picklable = None
        self.checkpoint_state: Optional[str] = None
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

    @property
    def train_num_samples(self):
        return len(self.dataset)

    @property
    def val_num_samples(self):
        return len(self.val_dataset)

    @property
    def train_global_batch_size(self) -> int:
        return self.batch_size

    @property
    def val_global_batch_size(self) -> int:
        return self.batch_size

    @property
    def train_local_batch_size(self) -> int:
        return self.batch_size

    @property
    def val_local_batch_size(self) -> int:
        return self.batch_size

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
