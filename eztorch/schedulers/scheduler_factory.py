from typing import Any, Dict, Optional

from omegaconf import DictConfig
from torch.optim import Optimizer

from eztorch.optimizers.utils import scale_learning_rate
from eztorch.schedulers.utils import _SCHEDULERS


def scheduler_factory(
    optimizer: Optimizer,
    name: str,
    params: DictConfig = {},
    interval: str = "epoch",
    num_steps_per_epoch: Optional[int] = None,
    scaler: Optional[str] = None,
    batch_size: Optional[int] = None,
    multiply_lr: float = 1.0,
) -> Dict[str, Any]:
    """Scheduler factory.

    Args:
        optimizer: Optimizer to wrap around.
        name: Name of the scheduler to retrieve the scheduler constructor from the ``_SCHEDULERS`` dict.
        params: Scheduler parameters for the scheduler constructor.
        interval: Interval to call step, if ``'epoch'`` call` :func:`.step()` at each epoch.
        num_steps_per_epoch: Number of steps per epoch. Useful for some schedulers.
        scaler: Scaler rule for the initial learning rate.
        batch_size: Batch size for the input of the model.
        multiply_lr: Multiply the learning rate by factor. Applied for warmup and minimum learning rate aswell.

    Returns:
        Scheduler configuration for pytorch lightning.
    """

    if interval == "step":
        if name == "linear_warmup_cosine_annealing_lr":
            params.max_epochs = num_steps_per_epoch * params.max_epochs
            params.warmup_epochs = num_steps_per_epoch * params.warmup_epochs

    if params.get("eta_min"):
        params.eta_min = scale_learning_rate(
            params.eta_min, scaler, batch_size, multiply_lr
        )
    if params.get("warmup_start_lr"):
        params.warmup_start_lr = scale_learning_rate(
            params.warmup_start_lr, scaler, batch_size, multiply_lr
        )

    scheduler = _SCHEDULERS[name](optimizer=optimizer, **params)

    return {"scheduler": scheduler, "interval": interval}
