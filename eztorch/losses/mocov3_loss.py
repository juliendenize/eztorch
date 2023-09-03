import torch
from torch import Tensor, nn


def compute_mocov3_loss(
    q: Tensor, k: Tensor, temp: float = 1.0, rank: int = 0
) -> Tensor:
    """Compute the MoCov3 loss.

    Args:
        q: The representations of the queries.
        k: The global representations of the keys.
        temp: Temperature for softmax.
        rank: Rank of the device for positive labels.

    Returns:
        The loss.
    """
    batch_size = q.shape[0]

    labels = (
        torch.arange(batch_size, dtype=torch.long, device=q.device) + batch_size * rank
    )
    sim = torch.einsum("nc,mc->nm", [q, k])

    logits = sim / temp
    loss = nn.functional.cross_entropy(logits, labels)

    return loss * (2 * temp)
