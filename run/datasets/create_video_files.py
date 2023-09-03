import argparse

from eztorch.datasets.labeled_video_dataset import \
    create_video_files_from_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create video files.")
    parser.add_argument("--input-folder", type=str, help="Location raw dataset.")
    parser.add_argument("--output-folder", type=str, help="Location to output folder.")
    parser.add_argument("--output-filename", type=str, help="Name of the output file.")

    args = parser.parse_args()

    create_video_files_from_folder(
        folder=args.input_folder,
        output_folder=args.output_folder,
        output_filename=args.output_filename,
    )
