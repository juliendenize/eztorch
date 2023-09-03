from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import torch
from lightning.pytorch.utilities import rank_zero_info
from SoccerNet.Evaluation.ActionSpotting import evaluate

from eztorch.datasets.soccernet import soccernet_dataset
from eztorch.datasets.soccernet_utils.parse_utils import (
    REVERSE_ACTION_SPOTTING_LABELS, REVERSE_BALL_SPOTTING_LABELS,
    SoccerNetTask)
from eztorch.datasets.soccernet_utils.predictions import (
    load_raw_action_spotting_predictions, save_spotting_predictions)

if __name__ == "__main__":

    parser = ArgumentParser(
        description="Evaluation for Action Spotting",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--soccernet-path",
        required=True,
        type=str,
        help="Path to the SoccerNet-V2 dataset folder (or zipped file) with labels",
    )
    parser.add_argument(
        "--predictions-path",
        required=True,
        type=str,
        help="Path to the predictions folder (or zipped file) with prediction",
    )
    parser.add_argument(
        "--prediction_file",
        required=False,
        type=str,
        help="Name of the prediction files as stored in folder (or zipped file) [None=try to infer it]",
        default="results_spotting.json",
    )
    parser.add_argument(
        "--split",
        required=False,
        type=str,
        help="Set on which to evaluate the performances",
        default="test",
    )
    parser.add_argument(
        "--version",
        required=False,
        type=int,
        help="Version of SoccerNet [1,2]",
        default=2,
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
        "--dataset-path",
        required=False,
        type=str,
        help="Path to the dataset to process.",
        default="",
    )
    parser.add_argument(
        "--fps",
        required=False,
        type=int,
        help="FPS of the dataset to decode.",
        default=2,
    )
    parser.add_argument(
        "--step-timestamp",
        required=False,
        type=float,
        help="Step in seconds between each timestamps.",
        default=1,
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
        "--task",
        required=False,
        type=str,
        help="Task.",
        default="action",
    )

    args = parser.parse_args()
    args.task = SoccerNetTask(args.task)
    if len(args.nms_window) == 1:
        args.nms_window = args.nms_window[0]

    predictions_path = Path(args.predictions_path)

    rank_zero_info(args)

    if args.process_predictions:
        dataset = soccernet_dataset(
            data_path=args.dataset_path,
            transform=None,
            video_path_prefix="",
            decoder="frame",
            decoder_args={"fps": args.fps},
            label_args=None,
            features_args=None,
            task=args.task,
        )

        predictions = load_raw_action_spotting_predictions(
            saved_path=args.preprocess_predictions_path,
            video_indexes=list(range(dataset.num_videos)),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        save_spotting_predictions(
            predictions=predictions,
            saving_path=args.predictions_path,
            dataset=dataset,
            step_timestamp=args.step_timestamp,
            NMS_args={
                "threshold": args.nms_threshold,
                "window": args.nms_window,
                "nms_type": args.nms_type,
            },
            make_zip=True,
        )

    predictions_path = Path(args.predictions_path)
    if not predictions_path.exists():
        predictions_path = str(
            predictions_path.parent / (str(predictions_path.name) + ".zip")
        )

    if args.task == SoccerNetTask.ACTION:
        dataset = "SoccerNet"
        eval_task = "spotting"
        num_classes = 17
        label_files = "Labels-v2.json"
        framerate = 2
        REVERSE_LABELS = REVERSE_ACTION_SPOTTING_LABELS
        metrics = ["tight", "at1", "at2", "at3", "at4", "at5", "loose"]
    elif args.task == SoccerNetTask.BALL:
        dataset = "Ball"
        eval_task = "spotting"
        num_classes = 2
        label_files = "Labels-ball.json"
        framerate = 25
        REVERSE_LABELS = REVERSE_BALL_SPOTTING_LABELS
        metrics = ["tight", "at1", "at2", "at3", "at4", "at5"]

    for metric in metrics:
        results = evaluate(
            SoccerNet_path=args.soccernet_path,
            Predictions_path=predictions_path,
            split=args.split,
            version=args.version,
            prediction_file=args.prediction_file,
            label_files=label_files,
            num_classes=num_classes,
            framerate=framerate,
            metric=metric,
            dataset=dataset,
            task=eval_task,
        )

        rank_zero_info(
            f"{metric}_Average_mAP/mAP: {results['a_mAP']:.04f}",
        )

        for c, map_class in enumerate(results["a_mAP_per_class"]):
            rank_zero_info(
                f"{metric}_Average_mAP/mAP_{REVERSE_LABELS[c]}: {map_class:.04f}",
            )

        rank_zero_info(
            f"{metric}_Average_mAP_visible/mAP: {results['a_mAP_visible']:.04f}",
        )

        for c, map_class in enumerate(results["a_mAP_per_class_visible"]):
            rank_zero_info(
                f"{metric}_Average_mAP_visible/mAP_{REVERSE_LABELS[c]}: {map_class:.04f}",
            )

        rank_zero_info(
            f"{metric}_Average_mAP_unshown/mAP: {results['a_mAP_unshown']:.04f}",
        )

        for c, map_class in enumerate(results["a_mAP_per_class_unshown"]):
            rank_zero_info(
                f"{metric}_Average_mAP_unshown/mAP_{REVERSE_LABELS[c]}: {map_class:.04f}",
            )
