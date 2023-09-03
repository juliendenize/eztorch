from typing import Any

import torch
from torch import Tensor, nn


def compute_sce_mask(
    batch_size: int,
    num_negatives: int,
    use_keys: bool = True,
    rank: int = 0,
    world_size: int = 1,
    device: Any = "cpu",
) -> Tensor:
    """Precompute the mask for SCE.

    Args:
        batch_size: The local batch size.
        num_negatives: The number of negatives besides the non-positive key elements.
        use_keys: Whether to use the non-positive elements from the key as negatives.
        rank: Rank of the current process.
        world_size: Number of processes that perform training.
        device: Device that performs training.

    Returns:
        The mask.
    """

    if use_keys:
        target_batch_size = batch_size * world_size
        labels = (
            torch.arange(batch_size, dtype=torch.long, device=device)
            + batch_size * rank
        )

    else:
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        target_batch_size = 1

    mask = nn.functional.one_hot(labels, target_batch_size + num_negatives)

    return mask


def compute_sce_loss(
    q: Tensor,
    k: Tensor,
    k_global: Tensor,
    use_keys: bool,
    queue: Tensor,
    mask: Tensor,
    coeff: Tensor,
    temp: float = 0.1,
    temp_m: float = 0.07,
    LARGE_NUM: float = 1e9,
) -> Tensor:
    """Compute the SCE loss.

    Args:
        q: The representations of the queries.
        k: The representations of the keys.
        k_global: The global representations of the keys.
        use_keys: Whether to use the non-positive elements from key.
        queue: The queue of representations.
        mask: Mask of positives for the query.
        coeff: Coefficient between the contrastive and relational aspects.
        temp: Temperature applied to the query similarities.
        temp_m: Temperature applied to the keys similarities.
        LARGE_NUM: Large number to mask elements.

    Returns:
        The loss.
    """

    batch_size = q.shape[0]

    if use_keys:
        sim_k_kglobal = torch.einsum("nc,kc->nk", [k, k_global])
        sim_q_kglobal = torch.einsum("nc,kc->nk", [q, k_global])
    else:
        sim_q_kglobal = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        sim_k_kglobal = torch.zeros(batch_size, device=q.device).unsqueeze(-1)

    if queue is not None:
        sim_k_queue = torch.einsum("nc,ck->nk", [k, queue])
        sim_q_queue = torch.einsum("nc,ck->nk", [q, queue])

        logits_k = torch.cat([sim_k_kglobal, sim_k_queue], dim=1)
        logits_q = torch.cat([sim_q_kglobal, sim_q_queue], dim=1)

    else:
        logits_k = sim_k_kglobal
        logits_q = sim_q_kglobal

    if use_keys:
        logits_k -= LARGE_NUM * mask

    logits_q /= temp
    logits_k /= temp_m

    prob_k = nn.functional.softmax(logits_k, dim=1)
    prob_q = nn.functional.normalize(coeff * mask + (1 - coeff) * prob_k, p=1, dim=1)

    loss = -torch.sum(prob_q * nn.functional.log_softmax(logits_q, dim=1), dim=1).mean(
        dim=0
    )

    return loss
