from typing import Any, Dict

import torch
from torch.nn import Module


class BatchReduceTimestamps(Module):
    """Aggregate successive timestamps of soccernet batch of clips information."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: Dict[str, Any]):
        timestamps = batch["timestamps"]

        labels = batch.get("labels", None)
        reduced_timestamps = (timestamps[:, ::2] + timestamps[:, 1::2]) / 2

        if labels is not None:
            has_label = batch["has_label"]
            ignore_class = batch["ignore_class"]

            reduced_labels = torch.logical_or(
                labels[:, ::2] != 0, labels[:, 1::2] != 0
            ).to(dtype=labels.dtype)

            reduced_has_label = torch.logical_or(
                has_label[:, ::2], has_label[:, 1::2]
            ).to(dtype=has_label.dtype)

            batch_size, num_timestamps, *rest_shape = labels.shape
            step_timestamps = timestamps.clone()
            step_timestamps = step_timestamps - step_timestamps.roll(
                shifts=(0, 1), dims=(0, 1)
            )
            step_timestamps[:, 0] = step_timestamps[:, 2]
            step_timestamps = step_timestamps.unsqueeze(-1).expand(-1, -1, *rest_shape)

            reduced_ignore_class = torch.logical_or(
                ignore_class[:, ::2],
                ignore_class[:, 1::2],
            )

            new_batch = {
                "labels": reduced_labels,
                "timestamps": reduced_timestamps,
                "has_label": reduced_has_label,
                "ignore_class": reduced_ignore_class,
            }

        else:
            new_batch = {"timestamps": reduced_timestamps}

        for key in batch.keys():
            if key not in new_batch:
                new_batch[key] = batch[key]

        return new_batch


class BatchMiddleTimestamps(Module):
    """Select the middle timestamps of soccernet batch of clips information."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: Dict[str, Any]):
        timestamps = batch["timestamps"]
        middle = timestamps.shape[1] // 2

        labels = batch.get("labels", None)
        reduced_timestamps = timestamps[:, middle : middle + 1]

        if labels is not None:
            new_batch = {
                "labels": batch["labels"][:, middle : middle + 1],
                "timestamps": reduced_timestamps,
                "has_label": batch["has_label"][:, middle : middle + 1],
                "ignore_class": batch["ignore_class"][:, middle : middle + 1],
            }

        else:
            new_batch = {"timestamps": reduced_timestamps}

        for key in batch.keys():
            if key not in new_batch:
                new_batch[key] = batch[key]

        return new_batch


class ReduceTimestamps(Module):
    """Aggregate successive timestamps of soccernet clip information."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: Dict[str, Any]):
        timestamps = batch["timestamps"]

        labels = batch.get("labels", None)

        if labels is not None:
            has_label = batch["has_label"]
            ignore_class = batch["ignore_class"]

            reduced_labels = torch.logical_or(labels[::2] != 0, labels[1::2] != 0).to(
                dtype=labels.dtype
            )

            reduced_has_label = torch.logical_or(has_label[::2], has_label[1::2]).to(
                dtype=has_label.dtype
            )

            num_timestamps, *rest_shape = labels.shape
            step_timestamps = timestamps.clone()
            step_timestamps = step_timestamps - step_timestamps.roll(
                shifts=(1), dims=(0)
            )
            step_timestamps[0] = step_timestamps[2]
            step_timestamps = step_timestamps.unsqueeze(-1).expand(-1, *rest_shape)

            reduced_timestamps = (timestamps[::2] + timestamps[1::2]) / 2

            reduced_ignore_class = torch.logical_or(
                ignore_class[::2],
                ignore_class[1::2],
            )

            new_batch = {
                "labels": reduced_labels,
                "has_label": reduced_has_label,
                "ignore_class": reduced_ignore_class,
                "timestamps": reduced_timestamps,
            }

        else:
            new_batch = {"timestamps": reduced_timestamps}

        for key in batch.keys():
            if key not in new_batch:
                new_batch[key] = batch[key]

        return new_batch


class MiddleTimestamps(Module):
    """Select the middle timestamps of soccernet clip information."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: Dict[str, Any]):
        timestamps = batch["timestamps"]
        middle = timestamps.shape[0] // 2

        labels = batch.get("labels", None)
        reduced_timestamps = timestamps[middle]

        if labels is not None:
            new_batch = {
                "labels": batch["labels"][middle],
                "timestamps": reduced_timestamps,
                "has_label": batch["has_label"][middle],
                "ignore_class": batch["ignore_class"][middle],
            }

        else:
            new_batch = {"timestamps": reduced_timestamps}

        for key in batch.keys():
            if key not in new_batch:
                new_batch[key] = batch[key]

        return new_batch
