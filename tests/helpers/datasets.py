from typing import Iterable, Optional

import torch
from torch.nn import Module
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset


class RandomDataset(Dataset):
    def __init__(self, size: Iterable[int]):
        self.length = size[0]
        self.data = torch.randn(size)

    def __getitem__(self, index):
        return {"input": self.data[index], "idx": index}

    def __len__(self):
        return self.length


class RandomLabeledDataset(Dataset):
    def __init__(self, size: Iterable[int], num_classes: int = 10):
        self.length = size[0]
        self.data = torch.randn(size)
        self.labels = torch.randint(num_classes, size=(size[0], 1))

    def __getitem__(self, index):
        return {"input": self.data[index], "label": self.labels[index], "idx": index}

    def __len__(self):
        return self.length


class RandomVisionLabeledDataset(VisionDataset):
    def __init__(
        self,
        size: Iterable[int],
        num_classes: int = 10,
        transform: Optional[Module] = None,
    ):
        super().__init__("data/", transform=transform)
        self.length = size[0]
        self.data = torch.randn(size)
        self.labels = torch.randint(num_classes, size=(size[0], 1))

    def __getitem__(self, index):
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)
        return {"input": data, "label": self.labels[index]}

    def __len__(self):
        return self.length
