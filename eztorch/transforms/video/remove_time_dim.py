from torch import Tensor, nn


class RemoveTimeDim(nn.Module):
    """Remove time dimension from tensor.

    Suppose the tensor shape is [C,T,H,W].
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, tensor: Tensor):
        c, t, h, w = tensor.shape
        return tensor.view(c * t, h, w)

    def __repr__(self):
        return f"{self.__class__.__name__}()"
