import torch
from torch import Tensor, nn


def compute_simclr_masks(
    batch_size: int,
    num_crops: int = 2,
    rank: int = 0,
    world_size: int = 1,
    device: torch.device = "cpu",
) -> tuple[Tensor, Tensor]:
    """Compute positive and negative masks for SimCLR.

    Args:
        batch_size: The local batch size per iteration.
        num_crops: The number of crops per instance.
        rank: Rank of the current process.
        world_size: Number of processes that perform training.
        device: Device that performs training.

    Returns:
        The positive and negative masks.
    """

    crops_batch_size = batch_size * num_crops
    global_crops_batch_size = batch_size * num_crops * world_size

    pos_labels = torch.arange(batch_size, dtype=torch.long, device=device)
    local_pos_mask = nn.functional.one_hot(pos_labels, batch_size)
    # Repeat local_mask for each crop and Remove same element diagonal
    local_pos_mask = local_pos_mask.repeat(num_crops, num_crops)
    local_pos_mask.fill_diagonal_(0)

    pos_mask = torch.zeros((crops_batch_size, global_crops_batch_size), device=device)
    # Put local mask in right position
    pos_mask[
        :, crops_batch_size * rank : crops_batch_size * (rank + 1)
    ] = local_pos_mask

    neg_mask = 1 - pos_mask
    # Remove same element diagonal
    neg_mask[:, crops_batch_size * rank : crops_batch_size * (rank + 1)].fill_diagonal_(
        0
    )

    return pos_mask, neg_mask


def compute_simclr_loss(
    z: Tensor, z_global: Tensor, pos_mask, neg_mask, temp: float
) -> Tensor:
    """Compute the simCLR loss.

    Args:
        z: The local representations.
        z_global: The gathered representations.
        pos_mask: Positives mask.
        neg_mask: Negative masks.
        temp: Temperature for softmax.

    Returns:
        The loss.
    """

    sim = torch.einsum("nc,kc->nk", [z, z_global])
    logits = torch.exp(sim / temp)
    pos = torch.sum(logits * pos_mask, 1)
    neg = torch.sum(logits * neg_mask, 1)
    loss = -(torch.mean(torch.log(pos / (neg + pos))))

    return loss
