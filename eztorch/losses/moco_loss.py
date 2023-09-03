import torch
from torch import Tensor, nn


def compute_moco_loss(
    q: Tensor,
    k: Tensor,
    k_global: Tensor,
    use_keys: bool,
    queue: Tensor,
    temp: float = 0.2,
    rank: int = 0,
) -> Tensor:
    """Compute the SCE loss.

    Args:
        q: The representations of the queries.
        k: The representations of the keys.
        k_global: The global representations of the keys.
        use_keys: Whether to use the non-positive elements from key.
        temp: Temperature applied to the query similarities.
        rank: Rank of the device for positive labels.

    Returns:
        The loss.
    """

    batch_size = q.shape[0]

    if use_keys:
        labels = (
            torch.arange(batch_size, dtype=torch.long, device=q.device)
            + batch_size * rank
        )
        sim_k = torch.einsum("nc,mc->nm", [q, k_global])

        if queue is not None:
            sim_queue = torch.einsum("nc,ck->nk", [q, queue])
            sim = torch.cat([sim_k, sim_queue], dim=1)
        else:
            sim = sim_k

    else:
        sim_k = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        sim_queue = torch.einsum("nc,ck->nk", [q, queue])
        sim = torch.cat([sim_k, sim_queue], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=q.device)

    logits = sim / temp
    loss = nn.functional.cross_entropy(logits, labels)

    return loss
