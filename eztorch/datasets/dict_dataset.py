from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100


class DictDataset(Dataset):
    """Wrapper around a Dataset to have a dictionary as input for models.

    Args:
        dataset: dataset to wrap around.
    """

    def __init__(self, dataset: Dataset) -> None:
        super().__init__()
        self.source_dataset = dataset

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        super_output = self.source_dataset[idx]
        if isinstance(super_output, Mapping):
            if "idx" in super_output:
                return super_output
            else:
                super_output["idx"] = idx
                return super_output
        elif isinstance(super_output, Iterable):
            if len(super_output) == 1:
                return {"input": super_output[0], "idx": idx}
            elif len(super_output) == 2:
                return {"input": super_output[0], "label": super_output[1], "idx": idx}
            else:
                raise NotImplementedError(
                    "Impossible to know what is in the list of super_ouput."
                )
        else:
            return {"input": super_output, "idx": idx}

    def __len__(self):
        return len(self.source_dataset)


class DictCIFAR10(DictDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ dict dataset.

    Args:
        root: Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to ``True``.
        train: If ``True``, creates dataset from training set, otherwise
            creates from test set.
        transform: A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform: A function/transform that takes in the
            target and transforms it.
        download: If ``True``, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        dataset = CIFAR10(root, train, transform, target_transform, download)
        super().__init__(dataset)


class DictCIFAR100(DictDataset):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ dict dataset.

    Args:
        root: Root directory of dataset where directory
            ``cifar-100-batches-py`` exists or will be saved to if download is set to ``True``.
        train: If ``True``, creates dataset from training set, otherwise
            creates from test set.
        transform: A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform: A function/transform that takes in the
            target and transforms it.
        download: If ``True``, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        dataset = CIFAR100(root, train, transform, target_transform, download)
        super().__init__(dataset)
