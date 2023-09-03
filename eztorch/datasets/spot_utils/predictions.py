import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from eztorch.datasets.spot_utils.parse_utils import (
    REVERSE_LABELS_SPOT_DATASETS, SpotDatasets)
from eztorch.evaluation.nms import perform_all_classes_NMS
from eztorch.modules.gather import concat_all_gather_without_backprop


def initialize_predictions(
    dataset: Dataset,
    max_video_index: int,
    min_video_index: int,
    device: str = "cpu",
) -> Dict[int, Dict[int, Tensor]]:
    """Initialize predictions for videos that have indexes between [min_video_index, max_video_index].

    Args:
        dataset: The dataset that contains the videos.
        max_video_index: Max video index to keep.
        min_video_index: Min video index to keep.
        device: The device to store predictions.

    Returns:
        The initialized predictions.
    """

    predictions = {
        video_idx: torch.zeros(
            (
                dataset._annotated_videos[video_idx]["num_frames"],
                dataset.num_classes,
            ),
            device=device,
        )
        for video_idx in range(min_video_index, max_video_index + 1)
    }
    return predictions


def aggregate_and_filter_clips(
    class_preds: Tensor,
    frames: Tensor,
    num_frames: Tensor,
    video_indexes: Tensor,
    max_video_index: Tensor,
    min_video_index: Tensor,
) -> Tuple[Tensor] | None:
    """Aggregate and filter only clips that have indexes between [min_video_index, max_video_index]. If none have
    been kept, returns None.

    Args:
        class_preds: Predictions to add.
        frames: Frames of the predictions.
        num_frames: Number of frames for the video.
        video_indexes: Indexes of the videos.
        max_video_index: Max video index to keep.
        min_video_index: Min video index to keep.

    Returns:
        The filtered tensors or None.
    """
    class_preds = concat_all_gather_without_backprop(class_preds)
    frames = concat_all_gather_without_backprop(frames.contiguous())
    num_frames = concat_all_gather_without_backprop(num_frames.contiguous())
    video_indexes = concat_all_gather_without_backprop(video_indexes.contiguous())

    shard_preds = torch.logical_and(
        video_indexes <= max_video_index, video_indexes >= min_video_index
    )

    if not torch.any(shard_preds):
        return

    shard_preds = torch.nonzero(shard_preds, as_tuple=True)

    class_preds = class_preds[shard_preds]
    frames = frames[shard_preds]
    num_frames = num_frames[shard_preds]
    video_indexes = video_indexes[shard_preds]

    num_frames = num_frames.cpu()
    video_indexes = video_indexes.cpu()

    return (
        class_preds,
        frames,
        num_frames,
        video_indexes,
    )


def add_clip_prediction(
    predictions: Dict[int, Dict[int, Tensor]],
    class_preds: Tensor,
    frames: Tensor,
    video_index: int,
    merge_predictions_type: str = "max",
) -> None:
    """Add the given predictions of classes of the particular timestamps of a video to the stored predictions.

    Args:
        predictions: Current predictions of the halves stored in a dictionary.
        class_preds: Predictions to add.
        frames: Timestamp indexes to update.
        video_index: Index of the video.
        merge_predictions_type: Strategy to merge the predictions at same place.
    """
    prev_class_preds = predictions[video_index][frames]
    class_preds = class_preds.to(dtype=prev_class_preds.dtype)

    if merge_predictions_type == "max":
        replace_predictions = torch.gt(class_preds, prev_class_preds)

        predictions[video_index][frames] = torch.where(
            replace_predictions, class_preds, prev_class_preds
        )
    elif merge_predictions_type == "average":
        average_predictions = torch.gt(prev_class_preds, 0)
        predictions[video_index][frames] = torch.where(
            average_predictions,
            torch.mean(torch.stack((class_preds, prev_class_preds)), 0),
            class_preds,
        )

    return


def add_clips_predictions(
    predictions: Dict[int, Dict[int, Tensor]],
    class_preds: Tensor,
    frames: Tensor,
    num_frames: Tensor,
    video_indexes: Tensor,
    remove_frames_predictions: int | Tensor = 0,
    merge_predictions_type: str = "max",
) -> None:
    """Add the given predictions of classes of the particular timestamps of the batch to the stored predictions.

    Args:
        predictions: Current predictions of the halves stored in a dictionary.
        class_preds: Predictions to add.
        frames: Frames of the predictions.
        num_frames: Number of frames for the half.
        video_indexes: Indexes of the videos.
        remove_frames_predictions: Do not keep first frames and last frames of the predictions.
        merge_predictions_type: Strategy to merge the predictions at same place.
    """
    b, t, c = class_preds.shape

    for i in range(b):
        video_class_preds = class_preds[i]
        video_frames = frames[i]
        video_num_frames = int(num_frames[i])
        video_index = int(video_indexes[i])

        if remove_frames_predictions > 0 and video_frames.shape[0] > 1:

            remove_before = torch.logical_and(
                (video_frames - video_frames[0]) < remove_frames_predictions,
                video_frames[0] > remove_frames_predictions,
            )
            remove_after = torch.logical_and(
                (video_frames[-1] - video_frames) < remove_frames_predictions,
                (video_num_frames - video_frames) > remove_frames_predictions,
            )

            keep_predictions = torch.logical_not(
                torch.logical_or(remove_before, remove_after)
            )

            # TODO: See if possible to eliminate synchro.
            keep_predictions = keep_predictions.nonzero(as_tuple=True)

            video_class_preds = video_class_preds[keep_predictions]
            video_frames: Tensor = video_frames[keep_predictions]

        add_clip_prediction(
            predictions,
            video_class_preds,
            video_frames,
            video_index,
            merge_predictions_type,
        )


def postprocess_spotting_video_predictions(
    predictions: Tensor,
    NMS_args: Dict[Any, Any],
    dataset: SpotDatasets = SpotDatasets.TENNIS,
) -> List[Dict[str, Any]]:
    """Postprocess the half predictions for action spotting.

    Args:
        predictions: The half predictions.
        half_id: The id of the half.
        step_timestamp: Step between each timestamps, used for aggregating predictions in NMS.
        NMS_args: Arguments to configure the `perform_all_classes_NMS` function.

    Returns:
        The predictions of the half in the correct format.
    """
    (kept_predictions_per_class, kept_frames_per_class,) = perform_all_classes_NMS(
        predictions,
        1,
        **NMS_args,
    )

    kept_predictions_per_class = [t.cpu() for t in kept_predictions_per_class]
    kept_frames_per_class = [t.cpu() for t in kept_frames_per_class]

    reverse_labels = REVERSE_LABELS_SPOT_DATASETS[dataset]

    video_predictions = [
        {
            "label": reverse_labels[c],
            "frame": int(frame_c),
            "score": float(prediction_c),
        }
        for c, (kept_predictions_c, kept_frames_c) in enumerate(
            zip(kept_predictions_per_class, kept_frames_per_class)
        )
        for prediction_c, frame_c in zip(kept_predictions_c, kept_frames_c)
    ]

    return video_predictions


def save_spotting_predictions(
    predictions: Dict[str, Any],
    saving_path: str | Path,
    dataset: Dataset,
    NMS_args: Dict[Any, Any],
) -> None:
    """Save the predictions for spotting.

    Args:
        predictions: The predictions to save as a dictionary following this format:
            ```
            predictions = {
                <video_name>: torch.tensor(preds),
                ...
            }
            ```
        saving_path: Path to the saving directory.
        dataset: Dataset from which to retrieve metadata.
        NMS_args: Arguments to configure the `perform_all_classes_NMS` function.
    """
    saving_path = Path(saving_path)

    json_output = []
    for video_index in predictions:
        predictions_json = {}
        video_metadata = dataset.get_video_metadata(video_index)
        predictions_json["video"] = video_metadata["video_name"]

        predictions_video = postprocess_spotting_video_predictions(
            predictions[video_index],
            NMS_args,
            dataset.dataset,
        )

        predictions_video.sort(key=lambda x: int(x["frame"]))
        predictions_json["events"] = predictions_video

        json_output.append(predictions_json)

    saving_path.mkdir(exist_ok=True, parents=True)
    json.dumps(json_output, indent=2)

    with open(saving_path / "predictions.json", "w+") as out:
        json.dump(json_output, out)

    return


def save_raw_spotting_predictions(
    predictions: Dict[str, Any],
    saving_path: str | Path,
    make_zip: bool = True,
) -> None:
    """Save the raw predictions for spotting.

    Args:
        predictions: The predictions to save.
        saving_path: Path to the saving directory.
        make_zip: Whether to make a zip of the predictions.
    """

    saving_path = Path(saving_path)
    saving_path.mkdir(exist_ok=True, parents=True)

    for video_index in predictions:
        pred_path = saving_path / f"preds_video{video_index}.pth"
        torch.save(predictions[video_index].cpu(), pred_path)

    if make_zip:
        shutil.make_archive(str(saving_path), "zip", saving_path)
        shutil.rmtree(saving_path)

    return


def load_raw_spotting_predictions(
    saved_path: Path | str,
    video_indexes: List[int],
    device: Any = "cpu",
) -> Dict[int, Dict[int, Tensor]]:
    """Load the raw predictions for spotting.

    Args:
        saved_path: Where the predictions are saved.
        video_indexes: Indexes of the video to load.
        device: Device to load the predictions.
    """
    predictions = {video_index: None for video_index in video_indexes}

    saved_path = Path(saved_path)
    from_zip = zipfile.is_zipfile(saved_path)

    if from_zip:
        with zipfile.ZipFile(saved_path, "r") as z:
            for video_index in video_indexes:
                with z.open(f"preds_video{video_index}.pth") as f:
                    predictions[video_index] = torch.load(f, map_location="cpu").to(
                        device=device
                    )
    else:
        for video_index in video_indexes:
            predictions[video_index] = torch.load(
                str(saved_path / f"preds_video{video_index}.pth"),
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
    """Merge several predictions for spotting.

    Args:
        saving_path: The path to save the predictions.
        saved_paths: Paths to the saved predictions.
        video_indexes: Indexes of the predictions' videos.
        kind_merge: How to merge the predictions.
        device: Device to merge the features.
        make_zip: Whether to make a zip of the merged predictions.
    """
    loaded_predictions = [
        load_raw_spotting_predictions(Path(saved_path), video_indexes, device)
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
        merged_prediction = torch.stack(
            [loaded_prediction[video_idx] for loaded_prediction in loaded_predictions]
        )
        merged_prediction = fn_merge(merged_prediction, dim=0)
        if idx_keep is not None:
            merged_prediction = merged_prediction[idx_keep]
        merged_predictions[video_idx] = merged_prediction

    save_raw_spotting_predictions(merged_predictions, saving_path, make_zip)

    return
