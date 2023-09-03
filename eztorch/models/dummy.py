import math
from abc import ABC
from typing import Any, Dict, Iterable, Optional

import hydra
import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from eztorch.models.eztorch_base_module import EztorchBaseModule


class DummyModel(EztorchBaseModule, ABC):
    """Dummy model to perform test such as profiling dataloading.

    Args:
        input_shape: The input shape of the data.
        transform: The configuration of a transform to apply to the data.
    """

    def __init__(
        self,
        input_shape: int,
        transform: Optional[DictConfig] = None,
    ) -> None:
        super().__init__()

        self.transform = (
            hydra.utils.instantiate(transform) if transform is not None else None
        )

        self.save_hyperparameters()

        input_dim = math.prod(input_shape)
        self.layer = nn.Linear(input_dim, 1)

    def configure_optimizers(self) -> Dict[Any, Any]:
        return torch.optim.Adam(
            self.parameters(),
            1e-4,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = torch.flatten(x, 1)
        x = self.layer(x)
        return x

    def training_step(self, batch: Iterable[Any], batch_idx: int):
        x = batch["input"]

        if self.transform is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    x = self.transform(x)
        if type(x) is list:
            pred = self.forward(x[0])
        else:
            pred = self.forward(x)

        return torch.nn.functional.binary_cross_entropy_with_logits(
            pred, torch.ones_like(pred)
        )
