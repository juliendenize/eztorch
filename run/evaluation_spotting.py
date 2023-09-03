from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.utilities import rank_zero_info
from tabulate import tabulate

from eztorch.datasets.spot import spot_dataset
from eztorch.datasets.spot_utils.eval import compute_mAPs, load_json
from eztorch.datasets.spot_utils.parse_utils import SpotDatasets
from eztorch.datasets.spot_utils.predictions import (
    load_raw_spotting_predictions, save_spotting_predictions)

if __name__ == "__main__":

    parser = ArgumentParser(
        description="Evaluation for Spotting",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Path to the truth dataset",
    )
    parser.add_argument(
        "--truth-path",
        required=True,
        type=str,
        help="Path to the truth dataset",
    )
    parser.add_argument(
        "--predictions-path",
        required=True,
        type=str,
        help="Path to the predictions folder (or zipped file) with prediction",
    )
    parser.add_argument(
        "--process-predictions",
        required=False,
        help="Whether to process predictions before making evaluation.",
        action="store_true",
    )
    parser.add_argument(
        "--preprocess-predictions-path",
        required=False,
        type=str,
        help="Path to the predictions to process.",
        default="",
    )
    parser.add_argument(
        "--nms-threshold",
        required=False,
        type=float,
        help="Threshold for the NMS.",
        default=0.5,
    )
    parser.add_argument(
        "--nms-window",
        required=False,
        type=int,
        nargs="+",
        help="Windows in seconds for the NMS.",
        default=10,
    )
    parser.add_argument(
        "--nms-type",
        required=False,
        type=str,
        help="Type of NMS.",
        default="hard",
    )
    parser.add_argument(
        "--tolerances",
        required=False,
        type=int,
        nargs="+",
        help="Tolerances.",
        default=[0, 1, 2, 3, 4],
    )

    args = parser.parse_args()

    predictions_path = Path(args.predictions_path)
    if len(args.nms_window) == 1:
        args.nms_window = args.nms_window[0]

    rank_zero_info(args)

    if args.process_predictions:
        dataset = spot_dataset(
            data_path=args.truth_path,
            transform=None,
            video_path_prefix="",
            decoder="frame",
            decoder_args={},
            label_args=None,
            dataset=SpotDatasets(args.dataset),
        )

        predictions = load_raw_spotting_predictions(
            args.preprocess_predictions_path,
            video_indexes=list(range(dataset.num_videos)),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        save_spotting_predictions(
            predictions=predictions,
            saving_path=args.predictions_path,
            dataset=dataset,
            NMS_args={
                "threshold": args.nms_threshold,
                "window": args.nms_window,
                "type": args.nms_type,
            },
        )

    predictions_path = Path(args.predictions_path)
    if not predictions_path.exists():
        predictions_path = str(
            predictions_path.parent / (str(predictions_path.name) + ".zip")
        )

    pred = load_json(predictions_path)
    truth = load_json(args.truth_path)

    maps, tolerances, header, rows = compute_mAPs(
        truth,
        pred,
        args.tolerances,
        False,
    )

    rank_zero_info(tabulate(rows, headers=header, floatfmt="0.2f"))
    rank_zero_info(f"Avg mAP (across tolerances): {np.mean(maps) * 100:0.2f}")
