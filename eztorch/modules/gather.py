import torch
import torch.distributed as dist
from torch import Tensor


class GatherLayer(torch.autograd.Function):
    """Gather tensor across devices with grad."""

    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(inp) for _ in range(dist.get_world_size())]
            dist.all_gather(output, inp)
        else:
            output = [inp]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (inp,) = ctx.saved_tensors
        if dist.is_available() and dist.is_initialized():
            grad_out = torch.zeros_like(inp)
            grad_out[:] = grads[dist.get_rank()]
        else:
            grad_out = grads[0]
        return grad_out


def concat_all_gather_with_backprop(x: Tensor, dim: int = 0) -> Tensor:
    """Gather tensor across devices with grad.

    Args:
        x: Tensor to gather.
        dim: Dimension to concat.

    Returns:
        Gathered tensor.
    """
    return torch.cat(GatherLayer.apply(x), dim=dim)


@torch.no_grad()
def concat_all_gather_without_backprop(x: Tensor, dim: int = 0) -> Tensor:
    """Gather tensor across devices without grad.

    Args:
        x: Tensor to gather.
        dim: Dimension to concat.

    Returns:
        Gathered tensor.
    """
    if dist.is_available() and dist.is_initialized():
        tensors_gather = [
            torch.ones_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(tensors_gather, x, async_op=False)
        output = torch.cat(tensors_gather, dim=dim)
    else:
        output = x
    return output


@torch.no_grad()
def get_world_size() -> int:
    """Returns the world size.

    Returns:
        The world size.
    """
    if dist.is_available() and dist.is_initialized():
        return torch.distributed.get_world_size()
    return 1
