import shutil
import zipfile
from pathlib import Path
from typing import List

import numpy as np
import torch
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from torch import Tensor
from torch.utils.data import Dataset


def load_features(
    features_dir: str | Path,
    video_paths: List[str | Path],
    filename: str,
    video_zip_prefix: str = "",
    as_tensor: bool = False,
) -> dict[int, dict[int, np.ndarray | Tensor]]:
    """Load SoccerNet features.

    Args:
        features_dir: Directory or zip where the features are stored.
        video_paths: Path to the videos in directory.
        filename: Name of the files containing the features.
        video_zip_prefix: Video zip prefix if features stored in Zip.
        as_tensor: Whether to decode the features as tensor or numpy.

    Returns:
        The loaded features for each video and half.
    """
    features_dir = Path(features_dir)
    video_zip_prefix = Path(video_zip_prefix)

    video_features = {video_index: {} for video_index in range(len(video_paths))}
    from_zip = zipfile.is_zipfile(features_dir)

    for video_index, video_path in enumerate(video_paths):
        if from_zip:
            with zipfile.ZipFile(features_dir, "r") as z:
                for half_index in range(1, 3):
                    with z.open(
                        str(
                            video_zip_prefix
                            / video_path
                            / f"{half_index}_{filename}.npy"
                        )
                    ) as f:
                        video_features[video_index][half_index] = np.load(f)
                        if as_tensor:
                            video_features[video_index][half_index] = torch.from_numpy(
                                video_features[video_index][half_index]
                            )
        else:
            for half_index in range(1, 3):
                video_features[video_index][half_index] = np.load(
                    features_dir / video_path / f"{half_index}_{filename}.npy"
                )
                if as_tensor:
                    video_features[video_index][half_index] = torch.from_numpy(
                        video_features[video_index][half_index]
                    )

    return video_features


def save_features(
    dataset: Dataset,
    saving_path: str | Path,
    features: dict[int, dict[int, Tensor]],
    filename: str,
    make_zip: bool,
) -> None:
    """Save the features, one file per half per match.

    Args:
        dataset: Dataset to save the features from.
        saving_path: Path to save the features.
        features: The features to save.
        filename: The filename for each file containing the stored features.
        make_zip: Store the features as a Zip file.
    """
    saving_path = Path(saving_path)

    for video_index in features:
        video_path = saving_path / str(
            dataset.get_video_metadata(video_index)["url_local"]
        )
        video_path.mkdir(exist_ok=True, parents=True)
        for half_index in features[video_index]:
            half_path = f"{half_index}_{filename}.npy"
            np.save(video_path / half_path, features[video_index][half_index])

    if make_zip:
        shutil.make_archive(str(saving_path), "zip", saving_path)
        shutil.rmtree(saving_path)

    return


def pca_features(
    features: dict[int, dict[int, np.ndarray]],
    dim: int,
    standardize: bool = True,
    **kwargs,
):
    """Apply PCA on the given SoccerNet features.

    Args:
        features: The features to apply PCA on.
        dim: The output dimension of the PCA.
        standardize: Whether to standardize features before PCA.

    Returns:
        The dimensionally reduced features.
    """
    features_all = np.concatenate(
        [
            features[video_index][half_index]
            for video_index in features
            for half_index in features[video_index]
        ]
    )
    pca = PCA(n_components=dim, **kwargs)
    if standardize:
        features_all = (
            features_all - np.mean(features_all, 0, keepdims=True)
        ) / np.std(features_all, 0, keepdims=True)
    features_all = pca.fit_transform(features_all)

    features_pca = {video_index: {} for video_index in features}

    slice_min: int = 0

    for video_index in features:
        for half_index in features[video_index]:
            slice_max = slice_min + features[video_index][half_index].shape[0]
            features_pca[video_index][half_index] = features_all[slice_min:slice_max]
            slice_min = slice_max

    return features_pca
