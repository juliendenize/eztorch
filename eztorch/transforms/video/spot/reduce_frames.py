from typing import Any, Dict

from torch.nn import Module


class BatchMiddleFrames(Module):
    """Select the middle frame of spot batch of clips information."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: Dict[str, Any]):
        frame_indices = batch["frame_indices"]
        middle = frame_indices.shape[1] // 2

        labels = batch.get("labels", None)
        reduced_frame_indices = frame_indices[:, middle : middle + 1]

        if labels is not None:
            new_batch = {
                "labels": batch["labels"][:, middle : middle + 1],
                "frame_indices": reduced_frame_indices,
                "has_label": batch["has_label"][:, middle : middle + 1],
                "ignore_class": batch["ignore_class"][:, middle : middle + 1],
            }
        else:
            new_batch = {"frame_indices": reduced_frame_indices}

        for key in batch.keys():
            if key not in new_batch:
                new_batch[key] = batch[key]

        return new_batch


class MiddleFrames(Module):
    """Select the middle frame of spot clip information."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: Dict[str, Any]):
        frame_indices = batch["frame_indices"]
        middle = frame_indices.shape[0] // 2

        labels = batch.get("labels", None)
        reduced_frame_indices = frame_indices[middle]

        if labels is not None:
            new_batch = {
                "labels": batch["labels"][middle],
                "frame_indices": reduced_frame_indices,
                "has_label": batch["has_label"][middle],
                "ignore_class": batch["ignore_class"][middle],
            }

        else:
            new_batch = {"frame_indices": reduced_frame_indices}

        for key in batch.keys():
            if key not in new_batch:
                new_batch[key] = batch[key]

        return new_batch
