from typing import Any, Iterable, List

import torch
from torch import Tensor
from torch.nn import Module, ModuleList


class ApplyTransformOnList(Module):
    """Apply transform to a list of input.

    Args:
        transform: A transform for the list of input.
        list_len: len of the input.
    """

    def __init__(self, transform: Module, list_len: int = 2) -> None:
        super().__init__()

        self.list_len = list_len
        self.transform = transform

    def forward(self, X: Iterable[Tensor]) -> List[Tensor]:
        assert self.list_len == len(X)

        X = [self.transform(X[i]) for i in range(self.list_len)]
        return X

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + f"(transform={self.transform})"
        return format_string


class ApplyTransformsOnList(Module):
    """Apply transform to a list of input.

    Args:
        transform: A transform for the list of input.
        list_len: len of the input.
    """

    def __init__(
        self,
        transforms: List[Module],
    ) -> None:
        super().__init__()

        self.list_len = len(transforms)
        self.transforms = ModuleList(transforms)

    def forward(self, X: Iterable[Tensor]) -> List[Tensor]:
        assert self.list_len == len(X)

        X = [self.transforms[i](X[i]) for i in range(self.list_len)]
        return X

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + f"(transforms={self.transforms})"
        return format_string


class ApplySameTransformOnList(Module):
    """Apply same transform to a list of input by concatenating the inputs and splitting them after.

    Args:
        transform: A transform for the list of input.
        list_len: len of the input.
        dim: The dimension to retrieve the various elements of the list.
    """

    def __init__(self, transform: Any, list_len: int = 2, dim: int = 1) -> None:
        super().__init__()

        self.list_len = list_len
        self.transform = transform
        self.dim = dim

    def forward(self, X: Iterable[Tensor]) -> List[Tensor]:
        assert self.list_len == len(X)

        X = torch.cat(X, dim=self.dim)
        X = self.transform(X)

        X = list(X.split(X.shape[self.dim] // self.list_len, dim=self.dim))
        return X

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + f"(transform={self.transform})"
        return format_string
