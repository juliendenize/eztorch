import argparse

from eztorch.datasets.soccernet import soccernet_dataset
from eztorch.datasets.soccernet_utils.parse_utils import SoccerNetTask

parser = argparse.ArgumentParser(description="Precompute soccernet labels.")
parser.add_argument("--radius-label", type=float, default=0.5)
parser.add_argument("--data-path", type=str, default="")
parser.add_argument("--path-prefix", type=str, default="")
parser.add_argument("--fps", type=int, default=2)
parser.add_argument("--cache-dir", type=str, required=True)
parser.add_argument("--task", type=str, default="action")


def main():
    args = parser.parse_args()

    soccernet_dataset(
        data_path=args.data_path,
        transform=None,
        video_path_prefix=args.path_prefix,
        decoder_args={"fps": args.fps},
        features_args=None,
        label_args={
            "radius_label": args.radius_label,
            "cache_dir": args.cache_dir,
        },
        task=SoccerNetTask(args.task),
    )


if __name__ == "__main__":
    main()
