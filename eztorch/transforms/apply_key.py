from typing import Dict, List

import torch
from torch import Tensor
from torch.nn import Module


class ApplyTransformToKey(Module):
    """Applies transform to key of dictionary input.

    Args:
        key: The dictionary key the transform is applied to.
        transform: The transform that is applied.
    """

    def __init__(self, key: str, transform: Module):
        super().__init__()
        self._key = key
        self._transform = transform

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x[self._key] = self._transform(x[self._key])
        return x


class ApplyTransformToKeyOnList(Module):
    """
    Applies transform to key of dictionary input where input is a list
    Args:
        key: the dictionary key the transform is applied to.
        transform: the transform that is applied.

    Example::
        >>>  transforms.ApplyTransformToKeyOnList(
        >>>       key='input',
        >>>       transform=UniformTemporalSubsample(num_video_samples),
        >>>   )
    """

    def __init__(self, key: str, transform: Module) -> None:  # pyre-ignore[24]
        super().__init__()
        self._key = key
        self._transform = transform

    def forward(self, x: Dict[str, List[Tensor]]) -> Dict[str, List[Tensor]]:
        x[self._key] = [self._transform(a) for a in x[self._key]]
        return x

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(key={self._key}, transform={self._transform})"
        )


class ApplySameTransformToKeyOnList(Module):
    """Applies the same transform to key of dictionary input where input is a list.

    Args:
        key: the dictionary key the transform is applied to.
        transform: the transform that is applied.
        dim: The dimension to retrieve the various elements of the list.
    """

    def __init__(
        self, key: str, transform: Module, dim: int = 1
    ) -> None:  # pyre-ignore[24]
        super().__init__()
        self._key = key
        self._transform = transform
        self._dim = dim

    def forward(self, x: Dict[str, List[Tensor]]) -> Dict[str, List[Tensor]]:
        data = x[self._key]
        len_data = len(data)

        data = torch.cat(data, dim=self._dim)
        data = self._transform(data)
        data = list(data.split(data.shape[self._dim] // len_data, dim=self._dim))

        x[self._key] = data
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(key={self._key}, transform={self._transform}, dim={self._dim})"


class ApplyTransformInputKeyOnList(ApplyTransformToKeyOnList):
    """Apply Transform to the input key.

    Args:
        transform: The transform to apply.
    """

    def __init__(self, transform: Module):
        super().__init__("input", transform=transform)

    def __repr__(self):
        return f"{self.__class__.__name__}(transform={self._transform})"


class ApplySameTransformInputKeyOnList(ApplySameTransformToKeyOnList):
    """Apply same transform to the input list key.

    Args:
        transform: The transform to apply.
        dim: The dimension to retrieve the various elements of the list.
    """

    def __init__(self, transform: Module, dim: int = 1):
        super().__init__("input", transform=transform, dim=dim)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(transform={self._transform}, dim={self._dim})"
        )


class ApplyTransformAudioKeyOnList(ApplyTransformToKeyOnList):
    """Apply Transform to the audio key.

    Args:
        transform: The transform to apply.
    """

    def __init__(self, transform: Module):
        super().__init__("audio", transform=transform)

    def __repr__(self):
        return f"{self.__class__.__name__}(transform={self._transform})"


class ApplyTransformInputKey(ApplyTransformToKey):
    """Apply Transform to the input key.

    Args:
        transform: The transform to apply.
    """

    def __init__(self, transform: Module):
        super().__init__("input", transform=transform)

    def __repr__(self):
        return f"{self.__class__.__name__}(transform={self._transform})"


class ApplyTransformAudioKey(ApplyTransformToKey):
    """Apply Transform to the audio key.

    Args:
        transform: The transform to apply.
    """

    def __init__(self, transform: Module):
        super().__init__("audio", transform=transform)


class ApplyTransformOnDict(Module):
    """Apply Transform to the audio key.

    Args:
        transform: The transform to apply.
    """

    def __init__(self, transform: Module):
        super().__init__()
        self._transform = transform

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = self._transform(x)

        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(transform={self._transform})"
