import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from eztorch.datasets.soccernet_utils.parse_utils import (
    REVERSE_ACTION_SPOTTING_LABELS, REVERSE_BALL_SPOTTING_LABELS,
    SoccerNetTask)
from eztorch.evaluation.nms import perform_all_classes_NMS
from eztorch.modules.gather import concat_all_gather_without_backprop


def aggregate_predictions(
    predictions: Tensor,
    timestamps: Tensor,
    video_num_timestamps: int,
    step_timestamp: float,
):
    """Aggregate by returning the maximum prediction per timestamp.

    Args:
        predictions: The predictions to aggregate.
        timestamps: The original timestamps of the predictions.
        video_num_timestamps: The number of timestamps in the video.
        step_timestamp: The step between each timestamps in the video. Used for rounding the timestamps.

    Returns:
        The predictions shifted and aggregated aswell as the timestamps indexes of the predictions in the video.
    """
    num_classes = predictions.shape[1]

    index_rounded_shifted_timestamps = get_timestamps_indexes(
        timestamps, step_timestamp
    )

    index_rounded_shifted_timestamps = torch.maximum(
        index_rounded_shifted_timestamps,
        torch.tensor(0, device=index_rounded_shifted_timestamps.device),
    )

    index_rounded_shifted_timestamps = torch.minimum(
        index_rounded_shifted_timestamps,
        torch.tensor(
            video_num_timestamps - 1, device=index_rounded_shifted_timestamps.device
        ),
    ).to(dtype=torch.long)

    min_index_timestamp = torch.min(index_rounded_shifted_timestamps)
    max_index_timestamp = torch.max(index_rounded_shifted_timestamps)

    predictions_to_keep = torch.ones(
        index_rounded_shifted_timestamps.shape,
        dtype=torch.bool,
        device=predictions.device,
    )

    # TODO: See if possible to eliminate synchro.
    num_timestamps = (max_index_timestamp - min_index_timestamp + 1).item()

    output_predictions = torch.zeros(
        (num_timestamps, num_classes),
        dtype=predictions.dtype,
        device=predictions.device,
    )

    for c in range(num_classes):
        class_index_timestamps = index_rounded_shifted_timestamps[:, c][
            predictions_to_keep[:, c]
        ]

        class_preds = predictions[:, c][predictions_to_keep[:, c]]

        unique_index_timestamps_class: Tensor = class_index_timestamps.unique()

        for unique_index in unique_index_timestamps_class:
            has_unique_index = class_index_timestamps == unique_index
            max_prediction = torch.max(class_preds[has_unique_index])

            output_predictions[unique_index - min_index_timestamp, c] = max_prediction

    return output_predictions, torch.arange(
        min_index_timestamp,
        max_index_timestamp + 1,
        device=predictions.device,
        dtype=torch.long,
    )


def get_rounded_timestamps(timestamps: Tensor, step_timestamp: float) -> Tensor:
    """Get the rounded timestamps.

    Args:
        timestamps: Timestamps to round.
        step_timestamp: Step for rounding.

    Returns:
        The rounded timestamps.
    """
    return torch.round(timestamps / step_timestamp) * step_timestamp


def get_timestamps_indexes(timestamps: Tensor, step_timestamp: float) -> Tensor:
    """Get the timestamps indexes.

    Args:
        timestamps: Timestamps to get indexes.
        step_timestamp: Step for indexing timestamps.

    Returns:
        The timestamps indexes.
    """
    return torch.round(timestamps / step_timestamp).to(dtype=torch.int)


def initialize_predictions(
    dataset: Dataset,
    step_timestamp: int,
    max_video_index: int,
    min_video_index: int,
    device: str = "cpu",
) -> Dict[int, Dict[int, Tensor]]:
    """Initialize predictions for videos that have indexes between [min_video_index, max_video_index].

    Args:
        dataset: The dataset that contains the videos.
        step_timestamp: Step of timestamp between each predictions to initialize the predictions storage.
        max_video_index: Max video index to keep.
        min_video_index: Min video index to keep.
        device: The device to store predictions.

    Returns:
        The initialized predictions.
    """

    predictions = {
        video_idx: {
            half_idx: torch.zeros(
                (
                    round(
                        dataset.get_half_duration(video_idx, half_idx) / step_timestamp
                    )
                    - 1,
                    dataset.num_classes,
                ),
                device=device,
            )
            for half_idx in range(2)
        }
        for video_idx in range(min_video_index, max_video_index + 1)
    }
    return predictions


def aggregate_and_filter_clips(
    class_preds: Tensor,
    timestamps: Tensor,
    num_timestamps: Tensor,
    video_indexes: Tensor,
    halves_indexes: Tensor,
    max_video_index: Tensor,
    min_video_index: Tensor,
) -> Tuple[Tensor] | None:
    """Aggregate and filter only clips that have indexes between [min_video_index, max_video_index]. If none have
    been kept, returns None.

    Args:
        class_preds: Predictions to add.
        timestamps: Timestamps of the predictions.
        num_timestamps: Number of timestamps for the half.
        video_indexes: Indexes of the videos.
        halves_indexes: Indexes of the halves.
        max_video_index: Max video index to keep.
        min_video_index: Min video index to keep.

    Returns:
        The filtered tensors or None.
    """
    class_preds = concat_all_gather_without_backprop(class_preds)

    timestamps = concat_all_gather_without_backprop(timestamps.contiguous())
    num_timestamps = concat_all_gather_without_backprop(num_timestamps.contiguous())
    video_indexes = concat_all_gather_without_backprop(video_indexes.contiguous())
    halves_indexes = concat_all_gather_without_backprop(halves_indexes.contiguous())

    if min_video_index == -1:
        return

    shard_preds = torch.logical_and(
        video_indexes <= max_video_index, video_indexes >= min_video_index
    )

    if not torch.any(shard_preds):
        return

    shard_preds = torch.nonzero(shard_preds, as_tuple=True)

    class_preds = class_preds[shard_preds]
    timestamps = timestamps[shard_preds]
    num_timestamps = num_timestamps[shard_preds]
    video_indexes = video_indexes[shard_preds]
    halves_indexes = halves_indexes[shard_preds]

    num_timestamps = num_timestamps.cpu()
    video_indexes = video_indexes.cpu()
    halves_indexes = halves_indexes.cpu()

    b, t, c = class_preds.shape
    timestamps = timestamps.view(b, t, 1).expand(-1, -1, c)

    return (
        class_preds,
        timestamps,
        num_timestamps,
        video_indexes,
        halves_indexes,
    )


def add_clip_prediction(
    predictions: Dict[int, Dict[int, Tensor]],
    class_preds: Tensor,
    timestamp_indexes: Tensor,
    video_index: int,
    half_index: int,
    merge_predictions_type: str = "max",
) -> None:
    """Add the given predictions of classes of the particular timestamps of a video to the stored predictions.

    Args:
        predictions: Current predictions of the halves stored in a dictionary.
        class_preds: Predictions to add.
        timestamp_indexes: Timestamp indexes to update.
        video_index: Index of the video.
        half_index: Index of the half.
        merge_predictions_type: Strategy to merge the predictions at same place.
    """

    prev_class_preds = predictions[video_index][half_index][timestamp_indexes]
    class_preds = class_preds.to(dtype=prev_class_preds.dtype)

    if merge_predictions_type == "max":
        replace_predictions = torch.gt(class_preds, prev_class_preds)

        predictions[video_index][half_index][timestamp_indexes] = torch.where(
            replace_predictions, class_preds, prev_class_preds
        )
    elif merge_predictions_type == "average":
        average_predictions = torch.gt(prev_class_preds, 0)
        predictions[video_index][half_index][timestamp_indexes] = torch.where(
            average_predictions,
            torch.mean(torch.stack((class_preds, prev_class_preds)), 0),
            class_preds,
        )

    return


def add_clips_predictions(
    predictions: Dict[int, Dict[int, Tensor]],
    class_preds: Tensor,
    timestamps: Tensor,
    num_timestamps: Tensor,
    video_indexes: Tensor,
    halves_indexes: Tensor,
    eval_step_timestamp: Tensor,
    remove_seconds_predictions: float | Tensor = 0.0,
    merge_predictions_type: str = "max",
) -> None:
    """Add the given predictions of classes of the particular timestamps of the batch to the stored predictions.

    Args:
        predictions: Current predictions of the halves stored in a dictionary.
        class_preds: Predictions to add.
        timestamps: Timestamps of the predictions.
        num_timestamps: Number of timestamps for the half.
        video_indexes: Indexes of the videos.
        halves_indexes: Indexes of the halves.
        eval_step_timestamp: Step of timestamps for validation to estimate the indexes of each predictions.
        remove_seconds_predictions: Do not keep first seconds and last seconds of the predictions.
        merge_predictions_type: Strategy to merge the predictions at same place.
    """

    b, t, c = class_preds.shape

    for i in range(b):
        half_class_preds = class_preds[i]
        half_timestamps = timestamps[i]
        half_num_timestamps = int(num_timestamps[i])
        video_index = int(video_indexes[i])
        half_index = int(halves_indexes[i])

        if remove_seconds_predictions > 0.0 and half_timestamps.shape[0] > 1:
            half_timestamps_one = half_timestamps[:, 0]
            step_timestamp = half_timestamps_one[1] - half_timestamps_one[0]

            remove_before = torch.logical_and(
                (half_timestamps_one - half_timestamps_one[0])
                < remove_seconds_predictions,
                half_timestamps_one > remove_seconds_predictions,
            )
            remove_after = torch.logical_and(
                (half_timestamps_one[-1] - half_timestamps_one)
                < remove_seconds_predictions,
                (half_num_timestamps * step_timestamp - half_timestamps_one)
                > remove_seconds_predictions,
            )

            keep_predictions = torch.logical_not(
                torch.logical_or(remove_before, remove_after)
            )

            # TODO: See if possible to eliminate synchro.
            keep_predictions = keep_predictions.nonzero(as_tuple=True)

            half_class_preds = half_class_preds[keep_predictions]
            half_timestamps: Tensor = half_timestamps[keep_predictions]

        half_class_preds, timestamps_indexes = aggregate_predictions(
            half_class_preds,
            half_timestamps,
            half_num_timestamps,
            eval_step_timestamp,
        )

        add_clip_prediction(
            predictions,
            half_class_preds,
            timestamps_indexes,
            video_index,
            half_index,
            merge_predictions_type,
        )


def postprocess_spotting_half_predictions(
    predictions: Tensor,
    half_id: str | int,
    step_timestamp: int,
    NMS_args: Dict[Any, Any],
    task: SoccerNetTask = SoccerNetTask.ACTION,
) -> List[Dict[str, Any]]:
    """Postprocess the half predictions for action spotting.

    Args:
        predictions: The half predictions.
        half_id: The id of the half.
        step_timestamp: Step between each timestamps, used for aggregating predictions in NMS.
        NMS_args: Arguments to configure the `perform_all_classes_NMS` function.
        task: The SoccerNet task.

    Returns:
        The predictions of the half in the correct format.
    """
    (kept_predictions_per_class, kept_timestamps_per_class,) = perform_all_classes_NMS(
        predictions,
        step_timestamp,
        **NMS_args,
    )

    kept_predictions_per_class = [t.cpu() for t in kept_predictions_per_class]
    kept_timestamps_per_class = [t.cpu() for t in kept_timestamps_per_class]

    if SoccerNetTask(task) == SoccerNetTask.ACTION:
        reverse_labels = REVERSE_ACTION_SPOTTING_LABELS
    elif SoccerNetTask(task) == SoccerNetTask.BALL:
        reverse_labels = REVERSE_BALL_SPOTTING_LABELS

    half_predictions = [
        {
            "gameTime": f"{half_id} - {int(timestamp_c / 60):02d}:{int(timestamp_c % 60):02d}",
            "label": reverse_labels[c],
            "position": f"{int(timestamp_c * 1000)}",
            "half": str(half_id),
            "confidence": float(prediction_c),
        }
        for c, (kept_predictions_c, kept_timestamps_c) in enumerate(
            zip(kept_predictions_per_class, kept_timestamps_per_class)
        )
        for prediction_c, timestamp_c in zip(kept_predictions_c, kept_timestamps_c)
    ]

    return half_predictions


def save_spotting_predictions(
    predictions: Dict[str, Any],
    saving_path: str | Path,
    dataset: Dataset,
    step_timestamp: int,
    NMS_args: Dict[Any, Any],
    make_zip: bool = True,
) -> None:
    """Save the predictions for action spotting.

    Args:
        predictions: The predictions to save as a dictionary following this format:
            ```
            predictions = {
                <video_name>: {
                    <half_name>: torch.tensor(preds),
                    ...
                },
                ...
            }
            ```
        saving_path: Path to the saving directory.
        dataset: Dataset from which to retrieve metadata.
        step_timestamp: Step between each timestamps, used for aggregating predictions in NMS.
        NMS_args: Arguments to configure the `perform_all_classes_NMS` function.
        make_zip: Whether to make a zip of the predictions.
    """
    saving_path = Path(saving_path)

    for video_index in predictions:
        json_output = {}
        video_metadata = dataset.get_video_metadata(video_index)
        json_output["UrlLocal"] = video_metadata["url_local"]

        predictions_json = []
        for half_index in predictions[video_index]:
            predictions_half = postprocess_spotting_half_predictions(
                predictions[video_index][half_index],
                video_metadata["half_id"][half_index],
                step_timestamp,
                NMS_args,
                dataset.task,
            )
            predictions_json.extend(predictions_half)

        predictions_json.sort(key=lambda x: int(x["position"]))
        predictions_json.sort(key=lambda x: int(x["half"]))
        json_output["predictions"] = predictions_json

        json_path: Path = saving_path / video_metadata["url_local"]
        json_path.mkdir(exist_ok=True, parents=True)

        with open(json_path / "results_spotting.json", "w+") as out:
            json.dump(json_output, out)

    if make_zip:
        shutil.make_archive(str(saving_path), "zip", saving_path)
        shutil.rmtree(saving_path)

    return


def save_raw_spotting_predictions(
    predictions: Dict[str, Any],
    saving_path: str | Path,
    make_zip: bool = True,
) -> None:
    """Save the raw predictions for action spotting.

    Args:
        predictions: The predictions to save.
        saving_path: Path to the saving directory.
        make_zip: Whether to make a zip of the predictions.
    """

    saving_path = Path(saving_path)
    saving_path.mkdir(exist_ok=True, parents=True)

    for video_index in predictions:
        for half_index in predictions[video_index]:
            pred_path = saving_path / f"preds_video{video_index}_half{half_index}.pth"
            torch.save(predictions[video_index][half_index].cpu(), pred_path)

    if make_zip:
        shutil.make_archive(str(saving_path), "zip", saving_path)
        shutil.rmtree(saving_path)

    return


def load_raw_action_spotting_predictions(
    saved_path: Path | str,
    video_indexes: List[int],
    device: Any = "cpu",
) -> Dict[int, Dict[int, Tensor]]:
    """Load the raw predictions for action spotting.

    Args:
        saved_path: Where the predictions are saved.
        video_indexes: Indexes of the video to load.
        device: Device to load the predictions.
    """
    predictions = {
        video_index: {half_index: None for half_index in range(2)}
        for video_index in video_indexes
    }

    saved_path = Path(saved_path)
    from_zip = zipfile.is_zipfile(saved_path)

    if from_zip:
        with zipfile.ZipFile(saved_path, "r") as z:
            for video_index in video_indexes:
                for half_index in range(2):
                    with z.open(f"preds_video{video_index}_half{half_index}.pth") as f:
                        predictions[video_index][half_index] = torch.load(
                            f, map_location="cpu"
                        ).to(device=device)
    else:
        for video_index in video_indexes:
            for half_index in range(2):
                predictions[video_index][half_index] = torch.load(
                    str(saved_path / f"preds_video{video_index}_half{half_index}.pth"),
                    map_location="cpu",
                ).to(device=device)

    return predictions


def merge_predictions(
    saving_path: str | Path,
    saved_paths: List[str | Path],
    video_indexes: List[int],
    kind_merge: str = "average",
    device: Any = "cpu",
    make_zip: bool = True,
):
    """Merge several predictions for action spotting.

    Args:
        saving_path: The path to save the predictions.
        saved_paths: Paths to the saved predictions.
        video_indexes: Indexes of the predictions' videos.
        kind_merge: How to merge the predictions.
        device: Device to merge the features.
        make_zip: Whether to make a zip of the merged predictions.
    """
    loaded_predictions = [
        load_raw_action_spotting_predictions(Path(saved_path), video_indexes, device)
        for saved_path in saved_paths
    ]

    if kind_merge == "average":
        fn_merge = torch.mean
        idx_keep = None
    elif kind_merge == "max":
        fn_merge = torch.max
        idx_keep = 0
    elif kind_merge == "min":
        fn_merge = torch.min
        idx_keep = 0
    else:
        raise NotImplementedError(f"{kind_merge} not defined.")

    merged_predictions = {}
    for video_idx in loaded_predictions[0]:
        merged_predictions[video_idx] = {
            half_idx: None for half_idx in loaded_predictions[0][video_idx]
        }
        for half_idx in loaded_predictions[0][video_idx]:
            merged_prediction = torch.stack(
                [
                    loaded_prediction[video_idx][half_idx]
                    for loaded_prediction in loaded_predictions
                ]
            )
            merged_prediction = fn_merge(merged_prediction, dim=0)
            if idx_keep is not None:
                merged_prediction = merged_prediction[idx_keep]
            merged_predictions[video_idx][half_idx] = merged_prediction

    save_raw_spotting_predictions(merged_predictions, saving_path, make_zip)

    return
