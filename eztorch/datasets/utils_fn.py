from functools import partial
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from iopath.common.file_io import g_pathmgr
from numpy.typing import NDArray

from eztorch.utils.utils import get_world_size


def get_time_difference_indices(
    indices: NDArray, add_last_indice: bool = False, max_last_indice: int = -1
) -> Tuple[NDArray, NDArray]:
    """Get the indices of frames when time difference is desired. Time difference recquires to have subsequent
    indices to work. This function retrieves lacking indices and avoid duplicated.

    Args:
        indices: The base indices.
        add_last_indice: Whether to add last indice make the approximation.
        max_last_indice: In case to add last indice, make sure to not have an indice that goes further than this maximum.

    Returns:
        The new indices and the boolean mask of indices to keep after time difference is computed.
    """

    indices = np.asarray(indices)

    recquire_new_indice = indices != np.roll(indices, -1) - 1
    recquire_new_indice[-1] = add_last_indice

    num_new_indices = recquire_new_indice.sum() + (1 if not add_last_indice else 0)

    new_indices = [None for i in range(len(indices) + num_new_indices)]
    keep_indices = [None for i in range(len(indices) + num_new_indices)]
    curr_element = 0
    for i in range(len(indices)):
        new_indices[curr_element] = indices[i]
        keep_indices[curr_element] = True
        curr_element += 1
        if recquire_new_indice[i]:
            new_indices[curr_element] = indices[i] + 1
            keep_indices[curr_element] = False
            curr_element += 1

    if add_last_indice:
        keep_last = indices[-1] + 1 > max_last_indice
        new_indices[-1] = indices[-1] + (1 if not keep_last else 0)
        if not keep_last:
            keep_indices[-1] = False
        else:
            keep_indices[-1], keep_indices[-2] = True, False
    else:
        keep_indices[-1], keep_indices[-2] = True, False
        new_indices[-1], new_indices[-2] = new_indices[-2], new_indices[-2] - 1

    indices = np.array(new_indices)
    keep_indices = np.array(keep_indices)

    return indices, keep_indices


def get_video_to_frame_path_fn(
    fn_type: str = "idx", zeros: int = 8, incr: int = 1
) -> Callable:
    """Get the function to get video frame paths.

    Args:
        fn_type: The function to use. Options are:

            * idx: it retrieves a video path frame from video path and the index of the frame.

        zeros: The number of zeroes used to name the frame path.

    Raises:
        NotImplementedError: If ``fn_type`` is not supported.

    Returns:
        The function to retrieve frames from a video path.
    """

    if fn_type == "idx":

        def fn(video_path, frame_idx):
            return f"{video_path}/{frame_idx+incr:0{zeros}d}.jpg"

        return fn
    else:
        raise NotImplementedError(f"{fn_type} unknown.")


def random_subsample(
    x: List, num_samples: int = 8, time_difference: bool = False
) -> Tuple[NDArray]:
    """Randomly subsample a list of indices.

    Args:
        x: The list to subsample
        num_samples: The number of samples to keep.
        time_difference: If ``True``, retrieve indices to be able to apply time difference.

    Returns:
        The indices and the boolean mask of indices to keep after time difference is computed.
    """

    t = len(x)
    assert num_samples > 0 and t > 0 and t >= num_samples
    # Sample by nearest neighbor interpolation if num_samples > t.
    indices = np.linspace(0, t - 1, num_samples)
    indices = np.clip(indices, 0, t - 1).astype(int)
    indices = np.sort(np.random.choice(indices, replace=False))

    indices = np.array(x)[indices]

    if time_difference:
        indices, keep_indices = get_time_difference_indices(indices)
    else:
        keep_indices = np.array([True for i in range(len(indices))])

    return indices, keep_indices


def uniform_subsample(
    x: List, num_samples: int = 8, time_difference: bool = False
) -> Tuple[NDArray]:
    """Unformly subsample a list of indices.

    Args:
        x: The list to subsample
        num_samples: The number of samples to keep.
        time_difference: If ``True``, retrieve indices to be able to apply time_difference.

    Returns:
        The indices and the boolean mask of indices to keep after time difference is computed.
    """

    t = len(x)
    assert num_samples > 0 and t > 0
    # Sample by nearest neighbor interpolation if num_samples > t.
    indices = np.linspace(0, t - 1, num_samples)
    indices = np.clip(indices, 0, t - 1).astype(int)
    indices = np.array(x)[indices]

    if time_difference:
        indices, keep_indices = get_time_difference_indices(indices)
    else:
        keep_indices = np.array([True for i in range(len(indices))])

    return indices, keep_indices


def get_subsample_fn(
    subsample_type: str = "uniform", num_samples: int = 8, time_difference: bool = False
) -> Callable:
    """Get the function to subsample video frame indices.

    Args:
        subsample_type: The function to use. Options are: ``'uniform'``, ``'random'``.
        num_samples: The number of samples to keep.
        time_difference: If ``True``, retrieve indices to be able to apply time_difference.

    Raises:
        NotImplementedError: If ``subsample_type`` is not supported.

    Returns:
        The function to subsamples indices.
    """

    if subsample_type == "uniform":
        return partial(
            uniform_subsample, num_samples=num_samples, time_difference=time_difference
        )
    elif subsample_type == "random":
        return partial(
            random_subsample, num_samples=num_samples, time_difference=time_difference
        )
    else:
        raise NotImplementedError(f"{subsample_type} unknown.")


def remove_suffix(file: Path) -> Path:
    """Remove the suffix from a path.

    Args:
        file: The path.

    Returns:
        The path without the suffix.
    """

    return Path(file).with_suffix("")


def get_raw_video_duration(root_folder: str, raw_video: str) -> int:
    """Get the video duration from a video extracted in frames folder.

    Args:
        root_folder: The root folders of the videos.
        raw_video: The video path.

    Returns:
        The video duration.
    """

    root_folder = Path(root_folder)
    video_path = remove_suffix(root_folder / raw_video)
    duration = len(g_pathmgr.ls(video_path))

    return duration


def get_shard_indices(
    dataset_size: int, shuffle_shards=True, seed: int = 42
) -> List[int]:
    """Retrieve the indices for the shard.

    Args:
        dataset_size: Complete size of the dataset.
        shuffle_shards: Whether to shuffle before sharding.
        seed: Seed for sharding.

    Raises:
        NotImplementedError: If called without distributing initialized.

    Returns:
        The shard indices.
    """

    if get_world_size() == 1:
        return list(torch.arange(dataset_size, dtype=torch.int32))

    if not dist.is_available() or not dist.is_initialized():
        raise NotImplementedError(
            "Sharding should only be performed during distributed training."
        )

    num_shards = dist.get_world_size()
    shard_id = dist.get_rank()

    if shuffle_shards:
        g = torch.Generator()
        g.manual_seed(seed)

        global_indices = torch.randperm(dataset_size, generator=g, dtype=torch.int32)

    else:
        global_indices = torch.arange(dataset_size, dtype=torch.int32)

    indices_per_shard = dataset_size // num_shards
    remainder_indices = dataset_size % num_shards

    start_indice_shard = 0
    end_indice_shard = indices_per_shard - 1 + (1 if remainder_indices > 0 else 0)

    for id in range(0, shard_id):
        start_indice_shard = end_indice_shard + 1
        end_indice_shard = (
            start_indice_shard
            + indices_per_shard
            - 1
            + (1 if (id + 1) < remainder_indices else 0)
        )

    indices = global_indices[torch.arange(start_indice_shard, end_indice_shard + 1)]

    return list(indices)
