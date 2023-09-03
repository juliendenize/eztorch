# ------------------------------------------------------------------------
# Modified from pytorch-lightning https://github.com/Lightning-AI/lightning/#
# Licensed under the Apache License 2.0
# ------------------------------------------------------------------------

from typing import Any, Iterator, List, Optional

from torch.utils.data import Dataset, DistributedSampler

from eztorch.datasets.clip_samplers.soccernet.soccernet_clip_sampler import \
    SoccerNetClipSampler


class _DatasetSamplerWrapper(Dataset):
    """Dataset to create indexes from `SoccerNetClipSampler`."""

    def __init__(self, sampler: SoccerNetClipSampler) -> None:
        self._sampler = sampler
        self._sampler_list: Optional[List[Any]] = None

    def __getitem__(self, index: int) -> Any:
        if self._sampler_list is None:
            self._sampler_list = list(self._sampler)
        return self._sampler_list[index]

    def __len__(self) -> int:
        return len(self._sampler)

    def set_epoch(self, epoch: int) -> None:
        self._sampler.set_epoch(epoch)
        return

    def reset(self) -> None:
        """Reset the sampler list in order to get new sampling."""
        self._sampler_list = list(self._sampler)

    def __repr__(self) -> str:
        return str(self._sampler)


class SoccerNetClipSamplerDistributedSamplerWrapper(DistributedSampler):
    """Wrapper over ``Sampler`` for distributed training.

    Note:
        The purpose of this wrapper is to take care of sharding the sampler indices. It is up to the underlying
        sampler to handle randomness and shuffling. The ``shuffle`` and ``seed`` arguments on this wrapper won't
        have any effect.
    """

    def __init__(
        self, sampler: SoccerNetClipSampler, *args: Any, **kwargs: Any
    ) -> None:
        shuffle = sampler.shuffle
        sampler.set_shuffle(False)
        super().__init__(
            _DatasetSamplerWrapper(sampler),
            seed=sampler.seed,
            shuffle=shuffle,
            *args,
            **kwargs,
        )

    def __iter__(self) -> Iterator:
        self.dataset.reset()
        return (self.dataset[index] for index in super().__iter__())

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        self.dataset.set_epoch(epoch)
        return

    def __repr__(self) -> str:
        return f"{__class__.__name__}(sampler={self.dataset}, shuffle={self.shuffle}, seed={self.seed})"
