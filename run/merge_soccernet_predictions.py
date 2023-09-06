import argparse
from pathlib import Path

from lightning.pytorch.utilities import rank_zero_info

from eztorch.datasets.soccernet import soccernet_dataset
from eztorch.datasets.soccernet_utils.parse_utils import SoccerNetTask
from eztorch.datasets.soccernet_utils.predictions import merge_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge SoccerNet Action Spotting predictions."
    )
    parser.add_argument("--dataset-path", type=str, help="Path to the dataset.")
    parser.add_argument("--fps", type=int, help="FPS for the dataset.")
    parser.add_argument(
        "--predictions-path",
        type=str,
        nargs="+",
        help="Paths to the predictions to merge.",
    )
    parser.add_argument(
        "--output-folder", type=str, help="Output merged predictions folder."
    )
    parser.add_argument(
        "--kind-merge",
        type=str,
        choices=["average", "min", "max"],
        default="average",
        help="Kind of merge for predictions.",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task.",
        default="action",
    )

    args = parser.parse_args()

    rank_zero_info(args)

    dataset = soccernet_dataset(
        data_path=Path(args.dataset_path),
        transform=None,
        video_path_prefix="",
        decoder="frame",
        decoder_args={"fps": args.fps},
        label_args=None,
        features_args=None,
        task=SoccerNetTask(args.task),
    )

    merge_predictions(
        saving_path=args.output_folder,
        saved_paths=args.predictions_path,
        video_indexes=list(range(dataset.num_videos)),
        kind_merge=args.kind_merge,
        device="cpu",
        make_zip=True,
    )
