from abc import ABC

from torch.utils.data import Sampler

from eztorch.datasets.soccernet import SoccerNet
from eztorch.utils.utils import get_default_seed


class SoccerNetClipSampler(Sampler, ABC):
    """Base class for SoccerNet clip samplers.

    Args:
        data_source: SoccerNet dataset.
        shuffle: Whether to shuffle indices.
    """

    def __init__(
        self,
        data_source: SoccerNet,
        shuffle: bool = False,
    ) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.epoch = 0
        self.seed = get_default_seed()
        self._shuffle = shuffle

    @property
    def shuffle(self) -> bool:
        return self._shuffle

    def set_shuffle(self, shuffle: bool) -> None:
        """Set shuffle value.

        Args:
            shuffle: Value for shuffle.
        """
        self._shuffle = shuffle

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        This ensures that at each epoch the windows are not the same for relevant subclass samplers.

        Args:
            epoch: Epoch number.
        """
        self.epoch = epoch
