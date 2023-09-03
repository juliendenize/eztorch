from typing import Dict, List

from torch import Tensor


class DictKeepKeys:
    """Keep specified keys in dict.

    References:
        - https://github.com/kalyanvasudev/pytorchInput-1/blob/export-D33431232/pytorchInput_trainer/pytorchInput_trainer/datamodule/transforms.py

    Args:
        keys: The list of keys to keep.
    """

    def __init__(self, keys: List[str]):
        self.keys = keys

    def __call__(self, x: Dict[str, Tensor]) -> List[Tensor]:
        x = {key: value for key, value in x.items() if key in self.keys}
        return x

    def __repr__(self):
        return f"{self.__class__.__name__ }(keys={self.keys})"


class DictKeepInputLabel(DictKeepKeys):
    """Transform dict to list containing values of only ``'input'`` and ``'label'``."""

    def __init__(self):
        super().__init__(["input", "label"])


class DictKeepInputLabelIdx(DictKeepKeys):
    """Transform dict to list containing values of only ``'input'``, ``'label'``, ``'index'`` and
    ``'aug_index'``."""

    def __init__(self):
        super().__init__(["input", "label", "idx", "aug_index"])
