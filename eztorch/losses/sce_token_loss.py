from typing import Any, Tuple

import torch
from torch import Tensor, nn


def compute_sce_token_masks(
    batch_size: int,
    num_tokens: int,
    num_negatives: int,
    positive_radius: int = 0,
    keep_aligned_positive: bool = True,
    use_keys: bool = True,
    use_all_keys: bool = False,
    rank: int = 0,
    world_size: int = 1,
    device: Any = "cpu",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Precompute the masks for SCE with tokens.

    Args:
        batch_size: The local batch size.
        num_tokens: The number of tokens per instance.
        num_negatives: The number of negatives besides the non-positive key elements.
        positive_radius: The radius of adjacent tokens to consider as positives.
        keep_aligned_positive: Whether to keep the aligned token as positive.
        use_keys: Whether to use the non-positive elements from the aligned key as negatives.
        use_all_keys: Whether to use the non-positive elements from all the gathered keys as negatives.
        rank: Rank of the current process.
        world_size: Number of processes that perform training.
        device: Device that performs training.

    Returns:
        - Mask to ignore similarities for the query.
        - Mask to ignore similarities for the keys.
        - Mask of positives for the query.
        - Mask to keep log values.
        - The number of positives for the specific token.
    """

    if use_keys and use_all_keys:
        raise NotImplementedError(
            "Only one of use_keys or use_all_keys should be True."
        )

    if positive_radius == 0:
        mask_positives = torch.eye(num_tokens, device=device)
    else:
        mask_positives = torch.ones((num_tokens, num_tokens), device=device)

        if not keep_aligned_positive:
            mask_positives -= torch.eye(num_tokens, device=device)

        mask_positives -= torch.triu(mask_positives, diagonal=positive_radius + 1).to(
            device=device
        ) + torch.tril(mask_positives, diagonal=-positive_radius - 1).to(device=device)

    mask_positives = [mask_positives] * batch_size
    mask_positives = torch.block_diag(*mask_positives)

    if use_all_keys:
        global_mask_key_positives = [None for _ in range(world_size)]
        global_mask_key_negatives = [None for _ in range(world_size)]
        for i in range(world_size):
            if i == rank:
                global_mask_key_positives[i] = mask_positives
                global_mask_key_negatives[i] = 1 - mask_positives
            else:
                global_mask_key_positives[i] = torch.zeros(
                    (num_tokens * batch_size, num_tokens * batch_size), device=device
                )
                global_mask_key_negatives[i] = torch.ones(
                    (num_tokens * batch_size, num_tokens * batch_size), device=device
                )
        true_positive_indices = (
            torch.arange(batch_size * num_tokens, device=device)
            + batch_size * num_tokens * rank
        )
        total_number_of_indices = batch_size * num_tokens * world_size
    elif use_keys:
        ones_batch_size = [
            torch.ones((num_tokens, num_tokens), device=device)
        ] * batch_size
        mask_batch_size = torch.block_diag(*ones_batch_size)

        global_mask_key_positives = [mask_positives]
        global_mask_key_negatives = [1 - mask_batch_size]
        true_positive_indices = torch.arange(batch_size * num_tokens, device=device)
        total_number_of_indices = batch_size * num_tokens
    else:
        global_mask_key_positives = [mask_positives]
        global_mask_key_negatives = [1 - mask_positives]
        true_positive_indices = torch.arange(batch_size * num_tokens, device=device)
        total_number_of_indices = batch_size * num_tokens

    mask_negatives = torch.zeros(
        (num_tokens * batch_size, num_negatives), device=device
    )
    global_mask_key_negatives = torch.cat(global_mask_key_negatives, dim=1)

    global_mask_true_positives = torch.nn.functional.one_hot(
        true_positive_indices, total_number_of_indices
    ).to(dtype=mask_negatives.dtype)

    global_mask_key_positives = torch.cat(global_mask_key_positives, dim=1)

    if not keep_aligned_positive and not use_keys and not use_all_keys:
        mask_sim_q = global_mask_key_negatives
        mask_sim_k = global_mask_key_positives + global_mask_key_negatives
    elif not keep_aligned_positive and use_all_keys:
        mask_sim_q = global_mask_true_positives
        mask_sim_k = global_mask_key_positives + global_mask_true_positives
    elif not keep_aligned_positive and use_keys:
        mask_sim_q = global_mask_true_positives + global_mask_key_negatives
        mask_sim_k = (
            global_mask_key_positives
            + global_mask_true_positives
            + global_mask_key_negatives
        )
    elif keep_aligned_positive and not use_keys and not use_all_keys:
        mask_sim_q = global_mask_key_negatives
        mask_sim_k = global_mask_key_positives + global_mask_key_negatives
    elif keep_aligned_positive and use_keys:
        mask_sim_q = global_mask_key_negatives
        mask_sim_k = global_mask_key_positives + global_mask_key_negatives
    else:
        mask_sim_q = None
        mask_sim_k = global_mask_key_positives

    global_mask_positives = torch.cat(
        [global_mask_key_positives, mask_negatives], dim=1
    )
    num_positives_per_token = global_mask_positives.sum(axis=1, keepdims=True)
    mask_prob_q = global_mask_positives / num_positives_per_token

    if mask_sim_q is not None:
        mask_log_q = torch.ones_like(mask_prob_q)
        mask_log_q[:, : mask_sim_q.shape[1]] = (1 - mask_sim_q).clone()
        mask_log_q = mask_log_q.to(dtype=torch.bool)
    else:
        mask_log_q = None

    return mask_sim_q, mask_sim_k, mask_prob_q, mask_log_q, num_positives_per_token


def compute_sce_token_loss(
    q: Tensor,
    k: Tensor,
    k_global: Tensor,
    queue: Tensor | None,
    mask_sim_q: Tensor | None,
    mask_sim_k: Tensor,
    mask_prob_q: Tensor,
    mask_log_q: Tensor | None,
    coeff: Tensor,
    temp: float = 0.1,
    temp_m: float = 0.07,
    LARGE_NUM: float = 1e9,
) -> Tensor:
    """Compute the SCE loss for several tokens as output.

    Args:
        q: The representations of the queries.
        k: The representations of the keys.
        k_global: The global representations of the keys.
        queue: The queue of representations.
        mask_sim_q: Mask to ignore similarities for the query.
        mask_sim_k: Mask to ignore similarities for the keys.
        mask_prob_q: Mask of positives for the query.
        mask_log_q: Mask of elements to keep after applying log to the query distribution.
        coeff: Coefficient between the contrastive and relational aspects.
        temp: Temperature applied to the query similarities.
        temp_m: Temperature applied to the keys similarities.
        LARGE_NUM: Large number to mask elements.

    Returns:
        The loss.
    """
    sim_k_k = torch.einsum("nc,kc->nk", [k, k_global])
    sim_q_k = torch.einsum("nc,kc->nk", [q, k_global])

    if mask_sim_q is not None:
        sim_q_k -= LARGE_NUM * mask_sim_q
    sim_k_k -= LARGE_NUM * mask_sim_k

    if queue is not None:
        sim_k_queue = torch.einsum("nc,ck->nk", [k, queue])
        sim_q_queue = torch.einsum("nc,ck->nk", [q, queue])

        logits_k = torch.cat([sim_k_k, sim_k_queue], dim=1)
        logits_q = torch.cat([sim_q_k, sim_q_queue], dim=1)
    else:
        logits_k = sim_k_k
        logits_q = sim_q_k

    logits_q /= temp
    logits_k /= temp_m

    prob_k = nn.functional.softmax(logits_k, dim=1)
    prob_q = nn.functional.normalize(
        coeff * mask_prob_q + (1 - coeff) * prob_k, p=1.0, dim=1
    )
    log_q = nn.functional.log_softmax(logits_q, dim=1)

    if mask_log_q is not None:
        log_q = torch.where(mask_log_q, log_q, 0.0)

    loss = -torch.sum(prob_q * log_q, dim=1).mean(dim=0)

    return loss
