import argparse
import json
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge SoccerNet Action Spotting files."
    )
    parser.add_argument(
        "--splits-files", type=str, nargs="+", help="The files to merge."
    )
    parser.add_argument("--output-folder", type=str, help="Output file folder.")
    parser.add_argument("--output-filename", type=str, help="Output filename.")

    args = parser.parse_args()

    json_contents = {}

    print(f"Merge splits file: {args.splits_files}")
    for split in args.splits_file:
        split_file = Path(split)
        with open(split_file) as f:
            json_split = json.load(f)
        print(json_split)

        print(f"Split {split} has {len(json_split)} videos.")
        json_contents.update(json_split)

    print(f"Total annotations: {len(json_contents)} videos.")

    with open(Path(args.output_folder) / f"{args.output_filename}.json", "w+") as f:
        json.dump(json_contents, f)
