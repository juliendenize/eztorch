from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


class DumbDataset(Dataset):
    """Dumb dataset that always provide random data. Useful for testing models or pipelines.

    Args:
        shape: shape of data to generate.
        len_dataset: length of the dataset. Used by dataloaders.
    """

    def __init__(self, shape: List[int], len_dataset: int) -> None:
        super().__init__()
        self.shape = list(shape)
        self.len_dataset = len_dataset

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = torch.randn(self.shape)
        label = 0

        return {"input": data, "label": label, "idx": idx}

    def __len__(self):
        return self.len_dataset
