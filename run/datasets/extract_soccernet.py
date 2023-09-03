import argparse
from pathlib import Path

from eztorch.datasets.soccernet_utils import (
    extract_frames_from_annotated_videos_ffmpeg, make_annotations_ffmpeg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract soccernet frames.")
    parser.add_argument("--input-folder", type=str, help="Location raw dataset.")
    parser.add_argument("--output-folder", type=str, help="Location to output folder.")
    parser.add_argument("--fps", type=int, help="FPS for extraction.")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "valid", "test", "challenge"],
        help="Which split to extract.",
    )

    args = parser.parse_args()

    source_dir = Path(args.input_folder)
    extraction_dir = Path(args.output_folder)

    split = args.split
    split_socc = split if split != "val" else "valid"

    raw_annotations_dir = extraction_dir / "raw_annotations"
    is_challenge_split = split == "challenge"

    raw_annotations_dir.mkdir(parents=True, exist_ok=True)
    raw_annotations_file = raw_annotations_dir / f"{split}.json"
    make_annotations_ffmpeg(source_dir / split_socc, raw_annotations_file)
    extract_frames_from_annotated_videos_ffmpeg(
        raw_annotations_file,
        source_dir / split_socc,
        extraction_dir / split,
        extraction_dir,
        args.fps,
        1,
    )
