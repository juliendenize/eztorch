from functools import partial
from typing import Callable, Tuple

import torch
from torch import Tensor


def get_test_time_augmentation_fn(name: str, **kwargs) -> Callable:
    """Retrieve the test time augmentation given its name and optional arguments.

    Args:
        name: Name of the testing time augmentation. Options are: ``'avg_group'``, ``'same_avg'``, ``'same_max'``.

    Raises:
        NotImplementedError: If ``name`` given is not supported.

    Returns:
        The test time augmentation function.
    """
    if name in _TEST_AUG_FUNCTIONS:
        return partial(_TEST_AUG_FUNCTIONS[name], **kwargs)
    else:
        raise NotImplementedError(
            f"{name} not supported: try a name in {_TEST_AUG_FUNCTIONS.keys()}"
        )


def average_group(values: Tensor, labels: Tensor, ids: Tensor) -> Tuple[Tensor]:
    """Perform average from values and aggregation of labels based on different ids.

    Args:
        values: The values to average.
        labels: The labels associated to the values.
        ids: The different ids to group the values and labels.

    Returns:
        The average of values associated to its labels and ids.
    """

    ids_rolled = ids.roll(1)
    ids_rolled[0] = ids[0]
    change_ids = ids != ids_rolled

    new_ids = torch.cumsum(change_ids, dim=0)
    new_ids_expanded = new_ids.view(new_ids.size(0), 1).expand(-1, values.size(1))

    new_unique_ids, new_ids_count = new_ids_expanded.unique(dim=0, return_counts=True)

    res = torch.zeros_like(new_unique_ids, dtype=torch.float).scatter_add_(
        0, new_ids_expanded, values
    )
    res = res / new_ids_count.float().unsqueeze(1)

    change_ids[0] = True

    return res, labels[change_ids], ids[change_ids]


def average_same_num_aug(
    values: Tensor, labels: Tensor, ids: Tensor, num_aug: int = 30
) -> Tuple[Tensor]:
    """Perform average from values and aggregation of labels and ids on split tensors by specified number of
    augmentations.

    Args:
        values: The values to average.
        labels: The labels associated to the values.
        ids: The different ids to group the values and labels.
        num_aug: The number of augmentations performed for test time augmentation.

    Returns:
        The average of values associated to its labels and ids.
    """

    shape = values.shape

    bsz = torch.div(shape[0], num_aug, rounding_mode="floor")

    new_shape = (bsz, num_aug, *shape[1:])

    reshaped_values = values.view(new_shape)
    reshaped_labels = labels.view(new_shape[:2])
    reshaped_ids = ids.view(new_shape[:2])

    max_values = reshaped_values.mean(1)
    labels = reshaped_labels[:, 0]
    ids = reshaped_ids[:, 0]

    return max_values, labels, ids


def max_same_num_aug(
    values: Tensor, labels: Tensor, ids: Tensor, num_aug: int = 30
) -> Tuple[Tensor]:
    """Keep maximum from values and aggregation of labels and ids on split tensors by specified number of
    augmentations.

    Args:
        values: The values to average.
        labels: The labels associated to the values.
        ids: The different ids to group the values and labels.
        num_aug: The number of augmentations performed for test time augmentation.

    Returns:
        The average of values associated to its labels and ids.
    """

    shape = values.shape

    bsz = torch.div(shape[0], num_aug, rounding_mode="floor")

    new_shape = (bsz, num_aug, *shape[1:])

    reshaped_values = values.view(new_shape)
    reshaped_labels = labels.view(new_shape[:2])
    reshaped_ids = ids.view(new_shape[:2])

    max_values = reshaped_values.max(1).values
    labels = reshaped_labels[:, 0]
    ids = reshaped_ids[:, 0]

    return max_values, labels, ids


_TEST_AUG_FUNCTIONS = {
    "avg_group": average_group,
    "same_avg": average_same_num_aug,
    "same_max": max_same_num_aug,
}
