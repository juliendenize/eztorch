from typing import Any, Optional, Union

import lightning.pytorch as pl
from lightning.pytorch.utilities import GradClipAlgorithmType
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from torch.optim.optimizer import Optimizer


class EztorchBaseModule(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        return self.clip_gradients(
            optimizer, gradient_clip_val, gradient_clip_algorithm
        )

    def clip_gradients(
        self,
        optimizer: Optimizer,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        """Handles gradient clipping internally.

        Note:
            - Do not override this method. If you want to customize gradient clipping, consider using
              :meth:`configure_gradient_clipping` method.
            - For manual optimization (``self.automatic_optimization = False``), if you want to use
              gradient clipping, consider calling
              ``self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")``
              manually in the training step.

        Args:
            optimizer: Current optimizer being used.
            gradient_clip_val: The value at which to clip gradients.
            gradient_clip_algorithm: The gradient clipping algorithm to use. Pass ``gradient_clip_algorithm="value"``
                to clip by value, and ``gradient_clip_algorithm="norm"`` to clip by norm.
        """

        if self.fabric is not None:
            self.fabric.clip_gradients(
                self,
                optimizer,
                clip_val=gradient_clip_val
                if gradient_clip_algorithm == GradClipAlgorithmType.VALUE
                else None,
                max_norm=None
                if gradient_clip_algorithm == GradClipAlgorithmType.VALUE
                else gradient_clip_val,
                error_if_nonfinite=False,
            )
            return

        if gradient_clip_val is None:
            gradient_clip_val = self.trainer.gradient_clip_val or 0.0
        elif (
            self.trainer.gradient_clip_val is not None
            and self.trainer.gradient_clip_val != gradient_clip_val
        ):
            raise MisconfigurationException(
                f"You have set `Trainer(gradient_clip_val={self.trainer.gradient_clip_val!r})`"
                f" and have passed `clip_gradients(gradient_clip_val={gradient_clip_val!r})`."
                " Please use only one of them."
            )

        if gradient_clip_algorithm is None:
            gradient_clip_algorithm = self.trainer.gradient_clip_algorithm or "norm"
        else:
            gradient_clip_algorithm = gradient_clip_algorithm.lower()
            if (
                self.trainer.gradient_clip_algorithm is not None
                and self.trainer.gradient_clip_algorithm != gradient_clip_algorithm
            ):
                raise MisconfigurationException(
                    f"You have set `Trainer(gradient_clip_algorithm={self.trainer.gradient_clip_algorithm.value!r})`"
                    f" and have passed `clip_gradients(gradient_clip_algorithm={gradient_clip_algorithm!r})"
                    " Please use only one of them."
                )

        if not isinstance(gradient_clip_val, (int, float)):
            raise TypeError(
                f"`gradient_clip_val` should be an int or a float. Got {gradient_clip_val}."
            )

        if not GradClipAlgorithmType.supported_type(gradient_clip_algorithm.lower()):
            raise MisconfigurationException(
                f"`gradient_clip_algorithm` {gradient_clip_algorithm} is invalid."
                f" Allowed algorithms: {GradClipAlgorithmType.supported_types()}."
            )

        gradient_clip_algorithm = GradClipAlgorithmType(gradient_clip_algorithm)
        self.trainer.precision_plugin.clip_gradients(
            optimizer, gradient_clip_val, gradient_clip_algorithm
        )
