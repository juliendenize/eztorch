from math import ceil
from typing import List, Tuple

import torch
from torch import Tensor


def perform_all_classes_NMS(
    predictions: Tensor,
    step_timestamp: float,
    window: int | list[int] = 60,
    threshold: float = 0.5,
    min_weight: float = 0.0,
    nms_type: str = "hard",
) -> Tuple[List[Tensor], List[Tensor]]:
    """Perform NMS on all classes.

    Args:
        predictions: Predictions on which to perform NMS.
        step_timestamp: The step between each timestamps in the video.
        window: The window used for NMS.
        threshold: The threshold used for NMS.
        min_weight: Minimum weight for decay function in soft nms.
        nms_type: Type of NMS to apply, `'soft'` or `'hard'`.

    Returns:
        The list of predictions per class aswell as their corresponding timestamp.
    """

    num_elements, num_classes = predictions.shape
    kept_predictions_per_class = [None for _ in range(num_classes)]
    kept_timestamps_per_class = [None for _ in range(num_classes)]

    if nms_type == "hard":
        if isinstance(window, int):
            kept_indexes = perform_hard_NMS(predictions, window, threshold)
        else:
            kept_indexes = perform_hard_NMS_per_class(predictions, window, threshold)
        possible_indexes = torch.arange(num_elements, device=predictions.device)

        for c in range(num_classes):
            kept_predictions_per_class[c] = predictions[:, c][kept_indexes[:, c]]
            kept_timestamps_per_class[c] = (
                possible_indexes[kept_indexes[:, c]] * step_timestamp
            )

    elif nms_type == "soft":
        if isinstance(window, int):
            window = [window for _ in range(num_classes)]

        for c in range(num_classes):
            nms_predictions_class = perform_soft_NMS(
                predictions[:, c], window[c], threshold, min_weight
            )
            idx_predictions_to_keep = nms_predictions_class > 0
            kept_predictions_per_class[c] = nms_predictions_class[
                idx_predictions_to_keep
            ]
            kept_timestamps_per_class[c] = (
                torch.arange(num_elements, device=predictions.device) * step_timestamp
            )[idx_predictions_to_keep]

    return kept_predictions_per_class, kept_timestamps_per_class


def perform_hard_NMS(values: Tensor, window: int = 60, threshold: float = 0.001):
    """Perform hard NMS on some values.

    Args:
        values: Values on which to perform hard NMS.
        window: The window used for NMS.
        threshold: The threshold used for hard NMS.

    Returns:
        The boolean tensor of values to keep.
    """

    nb_values, *rest = values.shape

    keep_indexes = torch.zeros((nb_values, *rest), dtype=torch.bool)

    for i in range(nb_values):
        end_window = min(i + int(window / 2), nb_values)
        start_window = max(i - int(window / 2), 0)

        after_prediction = values[i : end_window + 1]
        before_prediction = values[start_window : i + 1]

        value = values[i]

        keep_indexes[i] = torch.logical_and(
            torch.logical_and(
                torch.ge(value, torch.max(after_prediction, 0)[0]),
                torch.ge(value, torch.max(before_prediction, 0)[0]),
            ),
            torch.ge(value, threshold),
        )

    return keep_indexes


def perform_hard_NMS_per_class(
    values: Tensor, windows: list[int], threshold: float = 0.001
):
    """Perform hard NMS per class on some values.

    Args:
        values: Values on which to perform hard NMS.
        windows: The windows used for NMS for each class.
        threshold: The threshold used for hard NMS.

    Returns:
        The boolean tensor of values to keep.
    """

    nb_values, c, *rest = values.shape

    assert len(windows) == c

    keep_indexes = torch.zeros((nb_values, c, *rest), dtype=torch.bool)

    for i in range(nb_values):
        for j in range(c):
            end_window = min(i + int(windows[j] / 2), nb_values)
            start_window = max(i - int(windows[j] / 2), 0)

            after_prediction = values[i : end_window + 1, j]
            before_prediction = values[start_window : i + 1, j]

            value = values[i, j]

            keep_indexes[i, j] = torch.logical_and(
                torch.logical_and(
                    torch.ge(value, torch.max(after_prediction, 0)[0]),
                    torch.ge(value, torch.max(before_prediction, 0)[0]),
                ),
                torch.ge(value, threshold),
            )

    return keep_indexes


def perform_soft_NMS(
    values: Tensor, window: int, threshold: float = 0.001, min_weight: float = 0.0
):
    """Perform soft NMS on some values.

    Args:
        values: Values on which to perform soft NMS.
        window: The window used for NMS.
        threshold: The threshold used for soft NMS.
        min_weight: Minimum weight to decay the values.

    Returns:
        The decayed values.
    """
    values_nms = torch.zeros_like(values)
    values_tmp = values.clone()
    max_index = torch.argmax(values_tmp)
    max_score = values_tmp[max_index]

    radius = window / 2.0
    radius_ceil = ceil(radius)

    while max_score >= threshold:
        values_nms[max_index] = max_score

        start = max(max_index - radius_ceil, 0)
        end = min(max_index + radius_ceil + 1, values_tmp.shape[0])
        frame_range = torch.arange(start, end, device=values.device)

        weights = min_weight + (1.0 - min_weight) * torch.abs(
            (frame_range - max_index) / radius
        )

        clipped_weights = torch.clip(weights, 0.0, 1.0)
        values_tmp[frame_range] *= clipped_weights
        # Remove the max_index.
        values_tmp[max_index] = -1
        # Find the maximum from the remaining values.
        max_index = torch.argmax(values_tmp)
        max_score = values_tmp[max_index]
    return values_nms
