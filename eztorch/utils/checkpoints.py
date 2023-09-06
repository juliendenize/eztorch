import re
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch


def get_matching_files_in_dir(dir: str, file_pattern: str) -> List[Path]:
    """Retrieve files in directory matching a pattern.

    Args:
        dir: Directory path.
        file_pattern: Pattern for the files.

    Raises:
        NotADirectoryError: If `dir` does not exist or is not a directory.

    Returns:
        List of files matching the pattern
    """

    dir = Path(dir)
    if dir.exists() and dir.is_dir():
        files = list(dir.glob(file_pattern))
        return files
    else:
        raise NotADirectoryError(
            f'Directory "{dir}" does not exist or is not a directory'
        )


def get_ckpts_in_dir(dir: str, ckpt_pattern: str = "*.ckpt") -> List[Path]:
    """Get all checkpoints in a directory.

    Args:
        dir: Directory path containing the checkpoints.
        ckpt_pattern: Checkpoint glob pattern.

    Returns:
        List of checkpoints paths in directory.
    """

    try:
        files = get_matching_files_in_dir(dir, ckpt_pattern)
    except NotADirectoryError:
        warnings.warn(f"No checkpoint found in: {dir}", category=RuntimeWarning)
        files = []
    return files


def get_last_ckpt_in_dir(
    dir: str,
    ckpt_pattern: str = "*.ckpt",
    key_sort: Callable = lambda x: x.stat().st_mtime,
) -> Optional[Path]:
    """Get last ckpt in directory following a sorting function.

    Args:
        dir: Directory path containing the checkpoints.
        ckpt_pattern: Checkpoint glob pattern.
        key_sort: Function to sort the checkpoints.

    Returns:
       Last checkpoint in `dir`, if it exists, according to `key_sort`.
    """
    ckpts = get_ckpts_in_dir(dir, ckpt_pattern)
    if ckpts == []:
        return None
    ckpts.sort(key=key_sort, reverse=False)

    return ckpts[-1]


def get_last_ckpt_in_path_or_dir(
    checkpoint_file: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    ckpt_pattern: str = "*.ckpt",
    key_sort: Callable = lambda x: x.stat().st_mtime,
) -> Optional[Path]:
    """Get checkpoint from file or from last checkpoint in directory following a sorting function.

    Args:
        checkpoint_file: Checkpoint file path containing the checkpoint.
        checkpoint_dir: Directory path containing the checkpoints.
        ckpt_pattern: Checkpoint glob pattern.
        key_sort: Function to sort the checkpoints.

    Returns:
        Checkpoint file if it exists or last checkpoint in `dir` according to `key_sort`.
    """
    if checkpoint_file is not None:
        checkpoint_file_path = Path(checkpoint_file)
        if checkpoint_file_path.exists() and checkpoint_file_path.is_file():
            return checkpoint_file_path
        else:
            warnings.warn(
                f"{checkpoint_file} is not a file or do not exist.",
                category=RuntimeWarning,
            )
    if checkpoint_dir is not None:
        return get_last_ckpt_in_dir(
            checkpoint_dir, ckpt_pattern=ckpt_pattern, key_sort=key_sort
        )
    return None


def get_ckpt_by_callback_mode(
    checkpoint_path: str,
    checkpoint_mode: str,
) -> List[Path]:
    """Get checkpoint from ModelCheckpoint callback based on the mode: ``'best'``, ``'last'``, or ``'both'``.

    Args:
        checkpoint_path: Checkpoint file path containing the callback checkpoint.
        checkpoint_mode: Mode to read the callback checkpoint. Can be either ``'best'``, ``'last'`` or ``'both'``.

    Returns:
        Checkpoint paths based on the mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_checkpoint_str = str(checkpoint["callbacks"])

    paths: List[Path] = []
    if checkpoint_mode == "best" or checkpoint_mode == "both":
        regex = r"'best_model_path':\s'([a-zA-Z/0-9=_\-\.]+\.ckpt)'"
        paths.append(Path(re.search(regex, model_checkpoint_str).group(1)))
    elif checkpoint_mode == "last" or checkpoint_mode == "both":
        regex = r"'last_model_path':\s'([a-zA-Z/0-9=_\-\.]+\.ckpt)'"
        paths.append(Path(re.search(regex, model_checkpoint_str).group(1)))
    else:
        raise NotImplementedError(f"Checkpoint mode '{checkpoint_mode}' not supported.")

    new_paths = []
    checkpoint_dir = Path(checkpoint_path).parent
    for path in paths:
        if path.exists():
            new_paths.append(path)
        else:
            new_path = checkpoint_dir / path.name
            assert (
                new_path.exists()
            ), f"The checkpoint {path} is not available and not found at {new_path}."
            new_paths.append(new_path)
    return new_paths


def get_sub_state_dict_from_pl_ckpt(
    checkpoint_path: str, pattern: str = r"^(trunk\.)"
) -> Dict[Any, Any]:
    """Retrieve sub state dict from a pytorch lightning checkpoint.

    Args:
        checkpoint_path: Pytorch lightning checkpoint path.
        pattern: Pattern to filter the keys for the sub state dictionary.
            If value is ``""`` keep all keys.

    Returns:
        Sub state dict from the checkpoint following the pattern.
    """
    model = torch.load(checkpoint_path)
    if "state_dict" in model:
        state_dict = {
            k: v
            for k, v in model["state_dict"].items()
            if pattern == "" or re.match(pattern, k)
        }
    else:
        state_dict = {
            k: v for k, v in model.items() if pattern == "" or re.match(pattern, k)
        }

    return state_dict


def remove_pattern_in_keys_from_dict(d: Dict[Any, Any], pattern: str) -> Dict[Any, Any]:
    """Remove the pattern from keys in a dictionary.

    Args:
        d: The dictionary.
        pattern: Pattern to remove from the keys.
            If value is ``""`` keep all keys.

    Returns:
        Input dictionary with updated keys.
    """
    if pattern == "":
        return d
    return {re.sub(pattern, "", k): v for k, v in d.items()}
