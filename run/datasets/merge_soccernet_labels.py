import argparse
from pathlib import Path

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge SoccerNet Action Spotting labels cache."
    )
    parser.add_argument(
        "--splits-folders", type=str, nargs="+", help="The folders' cache to merge."
    )
    parser.add_argument("--output-folder", type=str, help="Output file folder.")

    args = parser.parse_args()

    splits = [Path(split) for split in args.splits_folders]
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    labels = torch.cat([torch.load(split / "labels.pt") for split in splits])
    num_timestamps_per_half = torch.cat(
        [torch.load(split / "num_timestamps_per_half.pt") for split in splits]
    )

    torch.save(labels, output_folder / "labels.pt")

    torch.save(num_timestamps_per_half, output_folder / "num_timestamps_per_half.pt")
