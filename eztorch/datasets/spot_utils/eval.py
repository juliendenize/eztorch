###
#   Adapted from E2E spot.
###

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(fpath: str | Path):
    """Load a JSON file.

    Args:
        fpath: Path to the JSON file.

    Returns:
        The JSON content.
    """
    with open(fpath) as fp:
        return json.load(fp)


def parse_ground_truth(truth: dict) -> dict:
    """Parse the ground truth labels.

    Args:
        truth: The JSON dataset content.

    Returns:
        The parsed labels.
    """
    label_dict = defaultdict(lambda: defaultdict(list))
    for x in truth:
        for e in x["events"]:
            label_dict[e["label"]][x["video"]].append(e["frame"])
    return label_dict


def get_predictions(pred: dict, label: str | None = None) -> list:
    """Get the label predictions.

    Args:
        pred: All the predictions.
        label: The label to look for.

    Returns:
        The predictions for the label.
    """
    flat_pred = []
    for x in pred:
        for e in x["events"]:
            if label is None or e["label"] == label:
                flat_pred.append((x["video"], e["frame"], e["score"]))
    flat_pred.sort(key=lambda x: x[-1], reverse=True)
    return flat_pred


def compute_average_precision(
    pred: list,
    truth: np.array,
    tolerance: int = 0,
    min_precision: int = 0,
) -> float:
    """Compute the average precision.

    Args:
        pred (list): The label predictions.
        truth (np.array): The truth labels.
        tolerance: The frame tolerance.
        min_precision: The minimum precision.

    Returns:
        The average precision.
    """
    total = sum([len(x) for x in truth.values()])
    recalled = set()

    # The full precision curve has TOTAL number of bins, when recall increases
    # by in increments of one
    pc = []
    _prev_score = 1
    for i, (video, frame, score) in enumerate(pred, 1):
        assert score <= _prev_score
        _prev_score = score

        # Find the ground truth frame that is closest to the prediction
        gt_closest = None
        for gt_frame in truth.get(video, []):
            if (video, gt_frame) in recalled:
                continue
            if gt_closest is None or (abs(frame - gt_closest) > abs(frame - gt_frame)):
                gt_closest = gt_frame

        # Record precision each time a true positive is encountered
        if gt_closest is not None and abs(frame - gt_closest) <= tolerance:
            recalled.add((video, gt_closest))
            p = len(recalled) / i
            pc.append(p)

            # Stop evaluation early if the precision is too low.
            # Not used, however when nin_precision is 0.
            if p < min_precision:
                break

    interp_pc = []
    max_p = 0
    for p in pc[::-1]:
        max_p = max(p, max_p)
        interp_pc.append(max_p)
    interp_pc.reverse()  # Not actually necessary for integration

    # Compute AUC by integrating up to TOTAL bins
    return sum(interp_pc) / total


def compute_mAPs(truth: dict, pred: dict, tolerances: list[int] = [0, 1, 2, 4]):
    """Compute the mAPs at different tolerances.

    Args:
        truth: The truth labels.
        pred: The label predictions.
        tolerances: The tolerances to compute the mAPs.

    Returns:
        The computed mAPs.
    """
    assert {v["video"] for v in truth} == {
        v["video"] for v in pred
    }, "Video set mismatch!"

    truth_by_label = parse_ground_truth(truth)

    fig, axes = None, None

    class_aps_for_tol = []
    mAPs = []
    for i, tol in enumerate(tolerances):
        class_aps = []
        for j, (label, truth_for_label) in enumerate(sorted(truth_by_label.items())):
            ap = compute_average_precision(
                get_predictions(pred, label=label),
                truth_for_label,
                tolerance=tol,
                plot_ax=axes[j, i] if axes is not None else None,
            )
            class_aps.append((label, ap))
        mAP = np.mean([x[1] for x in class_aps])
        mAPs.append(mAP)
        class_aps.append(("mAP", mAP))
        class_aps_for_tol.append(class_aps)

    header = ["AP @ tol"] + tolerances
    rows = []
    for c, _ in class_aps_for_tol[0]:
        row = [c]
        for class_aps in class_aps_for_tol:
            for c2, val in class_aps:
                if c2 == c:
                    row.append(val * 100)
        rows.append(row)

    return mAPs, tolerances, header, rows
