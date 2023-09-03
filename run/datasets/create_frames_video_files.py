import argparse

from eztorch.datasets.hmdb51 import create_hmdb51_files_for_frames
from eztorch.datasets.labeled_video_dataset import \
    create_frames_files_from_folder
from eztorch.datasets.ucf101 import create_ucf101_files_for_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create frames video files.")
    parser.add_argument("--input-folder", type=str, help="Location raw dataset.")
    parser.add_argument("--output-folder", type=str, help="Location to output folder.")
    parser.add_argument(
        "--output-filename",
        type=str,
        default="train.csv",
        help="Name of the output file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["folder", "hmdb51", "ucf101"],
        default="folder",
        help="Type of dataset to extract.",
    )

    args = parser.parse_args()

    if args.dataset == "folder":
        create_frames_files_from_folder(
            folder=args.input_folder,
            out_folder=args.output_folder,
            output_filename=args.output_filename,
        )
    elif args.dataset == "hmdb51":
        for split in [1, 2, 3]:
            create_hmdb51_files_for_frames(
                folder_files=args.input_folder,
                frames_folder=args.output_folder,
                split_id=split,
            )
    elif args.dataset == "ucf101":
        create_ucf101_files_for_frames(
            folder_files=args.input_folder, frames_folder=args.output_folder
        )
    else:
        raise ValueError(f"Wrong dataset specified: got {args.dataset}.")
