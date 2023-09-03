import math
from abc import ABC
from typing import Dict, List, Optional, Sequence, Union

import hydra
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn import Module

from eztorch.models.siamese.base import SiameseBaseModel


class MomentumSiameseBaseModel(SiameseBaseModel, ABC):
    """Abstract class to represent siamese models with a momentum branch.

    Subclasses should implement training_step method.

    Args:
        trunk: Config to build a trunk.
        optimizer: Config to build optimizers and schedulers.
        projector: Config to build a project.
        predictor: Config to build a predictor.
        train_transform: Config to perform transformation on train input.
        val_transform: Config to perform transformation on val input.
        test_transform: Config to perform transformation on test input.
        normalize_outputs: If ``True``, normalize outputs.
        num_global_crops: Number of global crops which are the first elements of each batch.
        num_local_crops: Number of local crops which are the last elements of each batch.
        num_splits: Number of splits to apply to each crops.
        num_splits_per_combination: Number of splits used for combinations of features of each split.
        mutual_pass: If ``True``, perform one pass per branch per crop resolution.
        initial_momentum: Initial value for the momentum update.
        scheduler_momentum: Rule to update the momentum value.
    """

    def __init__(
        self,
        trunk: DictConfig,
        optimizer: DictConfig,
        projector: Optional[DictConfig] = None,
        predictor: Optional[DictConfig] = None,
        train_transform: Optional[DictConfig] = None,
        val_transform: Optional[DictConfig] = None,
        test_transform: Optional[DictConfig] = None,
        normalize_outputs: bool = True,
        num_global_crops: int = 2,
        num_local_crops: int = 0,
        num_splits: int = 0,
        num_splits_per_combination: int = 2,
        mutual_pass: bool = False,
        initial_momentum: int = 0.996,
        scheduler_momentum: str = "cosine",
    ) -> None:
        super().__init__(
            trunk=trunk,
            optimizer=optimizer,
            projector=projector,
            predictor=predictor,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            normalize_outputs=normalize_outputs,
            num_global_crops=num_global_crops,
            num_local_crops=num_local_crops,
            num_splits=num_splits,
            num_splits_per_combination=num_splits_per_combination,
            mutual_pass=mutual_pass,
        )

        self.save_hyperparameters()
        self.momentum_trunk = hydra.utils.instantiate(trunk)
        self.momentum_projector = (
            hydra.utils.instantiate(projector) if projector is not None else None
        )

        self.initial_momentum = initial_momentum
        self.scheduler_momentum = scheduler_momentum
        self.current_momentum = initial_momentum

        for param in self.momentum_trunk.parameters():
            param.requires_grad = False

        self.momentum_trunk.load_state_dict(self.trunk.state_dict())

        if self.momentum_projector is not None:
            for param in self.momentum_projector.parameters():
                param.requires_grad = False
            self.momentum_projector.load_state_dict(self.projector.state_dict())

    def _update_momentum(self) -> float:
        if self.scheduler_momentum == "constant":
            return self.current_momentum
        # Cosine rule that increase value from initial value to 1.
        elif self.scheduler_momentum == "cosine":
            max_steps = (
                self.training_steps_per_epoch * self.trainer.max_epochs - 1
            )  # -1 because self.global_step starts at 0
            momentum = (
                1
                - (1 - self.initial_momentum)
                * (math.cos(math.pi * self.global_step / max_steps) + 1)
                / 2
            )
            return momentum
        elif self.scheduler_momentum == "cosine_epoch":
            # -1 because self.trainer.current_epoch starts at 0
            max_steps = self.trainer.max_epochs - 1
            momentum = (
                1
                - (1 - self.initial_momentum)
                * (math.cos(math.pi * self.current_epoch / max_steps) + 1)
                / 2
            )
            return momentum
        else:
            raise NotImplementedError(f"{self.scheduler_momentum} is not supported.")

    @torch.no_grad()
    def _update_weights(
        self, online: Union[Module, Tensor], target: Union[Module, Tensor]
    ) -> None:
        for (_, online_p), (_, target_p) in zip(
            online.named_parameters(),
            target.named_parameters(),
        ):
            target_p.data = (
                self.current_momentum * target_p.data
                + (1 - self.current_momentum) * online_p.data
            )

    @torch.no_grad()
    def momentum_shared_step(self, x: Tensor) -> Dict[str, Tensor]:
        """Momentum shared step that pass the input tensor in the momentum trunk and momentum projector.

        Args:
            x: The input tensor.

        Returns:
            The computed representations.
        """
        h = self.momentum_trunk(x)
        z = self.momentum_projector(h) if self.momentum_projector is not None else h
        if self.normalize_outputs:
            z = nn.functional.normalize(z, dim=1)

        return {"h": h, "z": z}

    @torch.no_grad()
    def multi_crop_momentum_shared_step(
        self, x: List[Tensor]
    ) -> List[Dict[str, Tensor]]:
        if self.mutual_pass:
            outputs = [{} for _ in range(len(x))]

            global_tensor = torch.cat(x)
            global_output = self.momentum_shared_step(global_tensor)

            dict_keys = global_output.keys()

            for key in dict_keys:
                if key == "z" and self.momentum_projector is None:
                    continue
                global_key_output = global_output[key]
                chunked_global_key_output = global_key_output.chunk(len(x))
                for i in range(len(x)):
                    outputs[i][key] = chunked_global_key_output[i]

            if self.momentum_projector is None:
                for i in range(len(x)):
                    outputs[i]["z"] = outputs[i]["h"]

            return outputs
        else:
            return [self.momentum_shared_step(x_i) for x_i in x]

    def on_train_batch_end(
        self,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
    ) -> None:
        self._update_weights(self.trunk, self.momentum_trunk)
        if self.projector is not None:
            self._update_weights(self.projector, self.momentum_projector)

        # log momentum value used to update the weights
        self.log(
            "pretrain/momentum_value",
            self.current_momentum,
            on_step=True,
            on_epoch=False,
        )

        # update momentum value
        self.current_momentum = self._update_momentum()
