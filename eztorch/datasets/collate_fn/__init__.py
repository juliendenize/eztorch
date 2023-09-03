from typing import Callable

from torch.utils.data._utils.collate import default_collate

from eztorch.datasets.collate_fn.multiple_samples_collate import \
    multiple_samples_collate

_COLLATE_FUNCTIONS = {
    "default": default_collate,
    "multiple_samples_collate": multiple_samples_collate,
}


def get_collate_fn(name: str) -> Callable:
    """Get a Collate function from its name through the _COLLATE_FUNCTIONS dictionary.

    Args:
        name: The collate function name.

    Raises:
        NotImplementedError: If the name is not supported.

    Returns:
        Callable: the collate function
    """
    if name in _COLLATE_FUNCTIONS:
        return _COLLATE_FUNCTIONS[name]
    else:
        raise NotImplementedError(
            f"{name} not supported: try a name in {_COLLATE_FUNCTIONS.keys()}"
        )
