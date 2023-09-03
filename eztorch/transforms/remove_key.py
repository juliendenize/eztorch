from typing import Dict

from torch import Tensor
from torch.nn import Module


class RemoveKey(Module):
    """Removes the given key from the input dict. Useful for removing modalities from a video clip that aren't
    needed.

    Args:
        key: The dictionary key to remove.
    """

    def __init__(self, key: str):
        super().__init__()
        self._key = key

    def __call__(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if self._key in x:
            del x[self._key]
        return x


class RemoveInputKey(RemoveKey):
    """Remove video key from sample dictionary."""

    def __init__(self):
        super().__init__("input")


class RemoveAudioKey(RemoveKey):
    """Remove audio key from sample dictionary."""

    def __init__(self):
        super().__init__("audio")
