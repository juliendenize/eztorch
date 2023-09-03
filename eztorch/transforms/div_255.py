from typing import List

import torch
from torch import Tensor, dtype
from torch.nn import Module


def div_255(
    input: Tensor, inplace: bool = True, dtype: dtype = torch.get_default_dtype()
) -> Tensor:
    """Divide the given tensor x by 255.

    Args:
        input: The input tensor.
        inplace: Whether to perform the operation inplace. Performed after dtype conversion.
        dtype: dtype to convert the tensor before applying division.

    Returns:
        Scaled tensor by dividing 255.
    """

    input = input.to(dtype=dtype)
    if inplace:
        input /= 255.0
    else:
        input = input / 255.0

    return input


class Div255Input(Module):
    """Perform Division by 255 on a tensor or list of tensor."""

    def __init__(
        self,
        inplace: bool = True,
        dtype: dtype = torch.get_default_dtype(),
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.dtype = dtype

    def forward(
        self,
        x: Tensor | List[Tensor],
    ) -> Tensor:
        if type(x) is Tensor:
            return div_255(x, inplace=self.inplace, dtype=self.dtype)

        return [div_255(el, inplace=self.inplace, dtype=self.dtype) for el in x]

    def __repr__(self):
        return f"{__class__.__name__}(inplace={self.inplace}, dtype={self.dtype})"
