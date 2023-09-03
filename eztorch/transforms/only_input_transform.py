from typing import Any, Callable, Dict, List

from torchvision.transforms import Compose

from eztorch.transforms.apply_key import (ApplySameTransformInputKeyOnList,
                                          ApplyTransformInputKey,
                                          ApplyTransformInputKeyOnList,
                                          ApplyTransformOnDict)
from eztorch.transforms.dict_keep_keys import DictKeepInputLabelIdx


class OnlyInputListTransform(Compose):
    """Apply Transform to only the key ``'input'`` in a list of sample dictionary.

    Args:
        transform: The transform to apply.
    """

    def __init__(self, transform: Callable) -> None:
        transforms = [ApplyTransformInputKeyOnList(transform), DictKeepInputLabelIdx()]

        super().__init__(transforms=transforms)


class OnlyInputTransform(Compose):
    """Apply Transform to only the key ``'input'`` in a sample dictionary.

    Args:
        transform: The transform to apply.
    """

    def __init__(self, transform: Callable) -> None:
        transforms = [ApplyTransformInputKey(transform), DictKeepInputLabelIdx()]

        super().__init__(transforms=transforms)


class OnlyInputListSameTransform(Compose):
    """Apply the same transform to only the key ``'input'`` in a list of sample dictionary.

    Args:
        transform: The transform to apply.
    """

    def __init__(self, transform: Callable) -> None:
        transforms = [
            ApplySameTransformInputKeyOnList(transform),
            DictKeepInputLabelIdx(),
        ]

        super().__init__(transforms=transforms)


class OnlyInputTransformWithDictTransform(Compose):
    """Apply Transform to only the key ``'input'`` in a sample dictionary with a transformation on the dictionary
    afterwards.

    Args:
        transform: The transform to apply to the input.
        dict_transform: The transform to apply to the dictionary.
        first_dict: If ``True``, first apply the transformation on the dict, else, first apply the transformation on the input.
    """

    def __init__(
        self, transform: Callable, dict_transform: Callable, first_dict: bool = False
    ) -> None:
        if first_dict:
            transforms = [
                ApplyTransformInputKey(transform),
                ApplyTransformOnDict(dict_transform),
                DictKeepInputLabelIdx(),
            ]
        else:
            transforms = [
                ApplyTransformInputKey(transform),
                ApplyTransformOnDict(dict_transform),
                DictKeepInputLabelIdx(),
            ]

        super().__init__(transforms=transforms)


class OnlyInputListTransformWithDictTransform:
    """Apply Transform to only the key ``'input'`` in a list of sample dictionary with a transformation on the
    dictionary afterwards.

    Args:
        transform: The transform to apply to the input.
        dict_transform: The transform to apply to the dictionary.
        first_dict: If ``True``, first apply the transformation on the dict, else, first apply the transformation on the input.
    """

    def __init__(
        self, transform: Callable, dict_transform: Callable, first_dict: bool = False
    ) -> None:
        self.transform = OnlyInputTransformWithDictTransform(
            transform, dict_transform, first_dict
        )

    def __call__(self, x: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.transform(sample) for sample in x]

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += "\n"
        format_string += f"    {self.transform}"
        format_string += "\n)"
        return format_string
