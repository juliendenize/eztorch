from math import floor

import torch


def mask_tube_in_sequence(
    mask_ratio: float,
    tube_size: int,
    len_sequence: int,
    device: str | torch.device = "cpu",
):
    """Generate indices to mask tubes from a sequence.

    Args:
        mask_ratio: Ratio for the masking.
        tube_size: Tube size for the masking.
        len_sequence (int): Length of the sequence to mask.
        device: Device for the mask.

    Returns:
        Tuple:
            - The indices to mask.
            - The indices to keep.
            - The reversed order for temporal masking.
            - The number of tokens masked.
    """
    num_masked = floor(len_sequence * mask_ratio)

    indices_permuted = (
        torch.randperm(len_sequence // tube_size, device=device) * tube_size
    ).repeat_interleave(tube_size) + torch.arange(tube_size, device=device).repeat(
        len_sequence // tube_size
    )

    indices_not_kept: torch.Tensor = indices_permuted[:num_masked].sort()[0]
    indices_kept: torch.Tensor = indices_permuted[num_masked:].sort()[0]

    indices = torch.cat((indices_not_kept, indices_kept))
    inversed_temporal_masked_indices = torch.argsort(indices)

    return indices_not_kept, indices_kept, inversed_temporal_masked_indices, num_masked


def batch_mask_tube_in_sequence(
    mask_ratio: float,
    tube_size: int,
    len_sequence: int,
    batch_size: int,
    device: str | torch.device = "cpu",
):
    """Generate indices to mask tubes from a batch of sequences.

    Args:
        mask_ratio: Ratio for the masking.
        tube_size: Tube size for the masking.
        len_sequence: Length of the sequence to mask.
        batch_size: The size of the batch.
        device: Device for the mask.

    Returns:
        Tuple:
            - The indices to mask.
            - The indices to keep.
            - The reversed order for temporal masking.
            - The number of tokens masked.
    """
    tot_indices_not_kept = [None for i in range(batch_size)]
    tot_indices_kept = [None for i in range(batch_size)]
    tot_inversed_temporal_masked_indices = [None for i in range(batch_size)]
    tot_num_masked = 0

    expected_num_masked = floor(mask_ratio * len_sequence)

    tot_indices_not_kept = torch.empty(
        (batch_size, expected_num_masked),
        device=device,
        dtype=torch.long,
    )

    tot_indices_kept = torch.empty(
        (batch_size, len_sequence - expected_num_masked),
        device=device,
        dtype=torch.long,
    )

    tot_inversed_temporal_masked_indices = torch.empty(
        (batch_size, len_sequence),
        device=device,
        dtype=torch.long,
    )

    for i in range(batch_size):
        (
            indices_not_kept,
            indices_kept,
            inversed_temporal_masked_indices,
            num_masked,
        ) = mask_tube_in_sequence(mask_ratio, tube_size, len_sequence, device)

        tot_indices_not_kept[i] = indices_not_kept
        tot_indices_kept[i] = indices_kept
        tot_inversed_temporal_masked_indices[i] = inversed_temporal_masked_indices
        tot_num_masked += num_masked

    return (
        tot_indices_not_kept,
        tot_indices_kept,
        tot_inversed_temporal_masked_indices,
        tot_num_masked,
    )
