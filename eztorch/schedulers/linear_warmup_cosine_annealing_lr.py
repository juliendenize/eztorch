# -------------------------------------------------------------------------------
# Modified from lightning-bolts (https://github.com/Lightning-AI/lightning-bolts)
# Licensed under the Apache License, Version 2.0
# -------------------------------------------------------------------------------

import math
import warnings
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Sets the learning rate of each parameter group to follow a linear warmup schedule between warmup_start_lr
    and base_lr followed by a cosine annealing schedule between base_lr and eta_min.

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Maximum number of iterations for linear warmup.
        max_epochs: Maximum number of iterations.
        warmup_start_lr: Learning rate to start the linear warmup.
        eta_min: Minimum learning rate.
        last_epoch: The index of last epoch.

    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        ``warmup_start_lr`` for the first epoch which is 0 in most cases.

    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.

    Example::
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        >>> #
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> #
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        has_layer_lr_decay = (
            optimizer.param_groups[0].get("layer_lr_decay", None) is not None
        )

        if has_layer_lr_decay:
            self.layer_lr_decay_values = [
                group["layer_lr_decay"] for group in optimizer.param_groups
            ]
        else:
            self.layer_lr_decay_values = [1.0 for _ in optimizer.param_groups]

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0 and self.warmup_epochs != 0:
            return [
                self.warmup_start_lr * decay_value
                for decay_value in self.layer_lr_decay_values
            ]

        elif self.last_epoch == 0:
            return [
                base_lr * layer_lr_decay
                for base_lr, layer_lr_decay in zip(
                    self.base_lrs, self.layer_lr_decay_values
                )
            ]

        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"]
                + (base_lr - self.warmup_start_lr)
                / (self.warmup_epochs - 1)
                * layer_lr_decay
                for base_lr, group, layer_lr_decay in zip(
                    self.base_lrs,
                    self.optimizer.param_groups,
                    self.layer_lr_decay_values,
                    strict=True,
                )
            ]

        elif self.last_epoch == self.warmup_epochs:
            return [
                base_lr * layer_lr_decay
                for base_lr, layer_lr_decay in zip(
                    self.base_lrs, self.layer_lr_decay_values, strict=True
                )
            ]

        elif (self.last_epoch - 1 - self.max_epochs) % (
            2 * (self.max_epochs - self.warmup_epochs)
        ) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                * layer_lr_decay
                for base_lr, group, layer_lr_decay in zip(
                    self.base_lrs,
                    self.optimizer.param_groups,
                    self.layer_lr_decay_values,
                    strict=True,
                )
            ]

        return [
            (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            / (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs - 1)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min * layer_lr_decay)
            + self.eta_min * layer_lr_decay
            for group, layer_lr_decay in zip(
                self.optimizer.param_groups, self.layer_lr_decay_values, strict=True
            )
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if self.last_epoch < self.warmup_epochs:
            return [
                layer_lr_decay
                * (
                    self.warmup_start_lr
                    + self.last_epoch
                    * (base_lr - self.warmup_start_lr)
                    / (self.warmup_epochs - 1)
                )
                for base_lr, layer_lr_decay in zip(
                    self.base_lrs, self.layer_lr_decay_values, strict=True
                )
            ]

        return [
            layer_lr_decay
            * (
                self.eta_min
                + 0.5
                * (base_lr - self.eta_min)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.last_epoch - self.warmup_epochs)
                        / (self.max_epochs - self.warmup_epochs)
                    )
                )
            )
            for base_lr, layer_lr_decay in zip(
                self.base_lrs, self.layer_lr_decay_values, strict=True
            )
        ]
