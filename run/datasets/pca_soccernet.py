import argparse
from pathlib import Path

from eztorch.datasets.soccernet import soccernet_dataset
from eztorch.datasets.soccernet_utils.features import (load_features,
                                                       pca_features,
                                                       save_features)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform PCA on SoccerNet features.")
    parser.add_argument("--dataset-json", type=str, help="Location dataset.")
    parser.add_argument(
        "--task",
        type=str,
        choices=["action", "ball", "all"],
        help="Task of the dataset.",
    )
    parser.add_argument(
        "--video-zip-prefix", type=str, default="", help="Prefix of features in zip."
    )
    parser.add_argument(
        "--features-path",
        type=str,
        help="Path to the input features.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        help="Path to output the PCA features.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        help="Name of the features files.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        help="Dimension of the PCA ouput.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        help="FPS of the stored features.",
    )

    args = parser.parse_args()

    dataset = soccernet_dataset(
        data_path=args.dataset_json,
        transform=None,
        video_path_prefix="",
        decoder="frame",
        decoder_args={"fps": args.fps},
        label_args=None,
        features_args=None,
        task=args.task,
    )

    video_paths = [
        video_path.decode() for video_path in dataset._annotated_videos._video_paths
    ]

    print("load_features")
    video_features = load_features(
        features_dir=args.features_path,
        video_paths=video_paths,
        filename=args.filename,
        video_zip_prefix=args.video_zip_prefix,
        as_tensor=False,
    )
    print("features loaded")

    print("pca")

    features_pca = pca_features(features=video_features, dim=args.dim, standardize=True)
    print("end pca")

    print("save")

    save_features(
        dataset=dataset,
        saving_path=Path(args.save_path) / f"pca_{args.dim}_std_{args.filename}",
        features=features_pca,
        filename=f"pca_{args.dim}_std_{args.filename}",
        make_zip=True,
    )
    print("end save")
