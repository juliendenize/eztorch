from typing import Any, Dict, Iterable, Optional

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from eztorch.losses.simclr_loss import (compute_simclr_loss,
                                        compute_simclr_masks)
from eztorch.models.siamese.base import SiameseBaseModel
from eztorch.modules.gather import concat_all_gather_with_backprop


class SimCLRModel(SiameseBaseModel):
    """SimCLR model with version 1, 2 that can be configured.

    References:
        - SimCLR: https://arxiv.org/abs/2002.05709
        - SimCLRv2: https://arxiv.org/abs/2006.10029

    Args:
        trunk: Config tu build a trunk.
        optimizer: Config tu build optimizers and schedulers.
        projector: Config to build a project.
        train_transform: Config to perform transformation on train input.
        val_transform: Config to perform transformation on val input.
        test_transform: Config to perform transformation on test input.
        normalize_outputs: If ``True``, normalize outputs.
        num_global_crops: Number of global crops which are the first elements of each batch.
        num_local_crops: Number of local crops which are the last elements of each batch.
        num_splits: Number of splits to apply to each crops.
        num_splits_per_combination: Number of splits used for combinations of features of each split.
        mutual_pass: If ``True``, perform one pass per crop resolution.
        temp: Temperature parameter to scale the online similarities.
    """

    def __init__(
        self,
        trunk: DictConfig,
        optimizer: DictConfig,
        projector: Optional[DictConfig] = None,
        train_transform: Optional[DictConfig] = None,
        val_transform: Optional[DictConfig] = None,
        test_transform: Optional[DictConfig] = None,
        normalize_outputs: bool = True,
        num_global_crops: int = 2,
        num_local_crops: int = 0,
        num_splits: int = 0,
        num_splits_per_combination: int = 2,
        mutual_pass: bool = False,
        temp: float = 0.1,
    ) -> None:
        super().__init__(
            trunk=trunk,
            optimizer=optimizer,
            projector=projector,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            num_global_crops=num_global_crops,
            num_local_crops=num_local_crops,
            num_splits=num_splits,
            num_splits_per_combination=num_splits_per_combination,
            mutual_pass=mutual_pass,
            predictor=None,
            normalize_outputs=normalize_outputs,
        )

        assert not self.use_split, "Splits not supported for SimCLR"

        self.temp = temp

        self.save_hyperparameters()

    def _precompute_mask(self) -> None:
        batch_size = self.trainer.datamodule.train_local_batch_size

        self.pos_mask, self.neg_mask = compute_simclr_masks(
            batch_size=batch_size,
            num_crops=self.num_crops,
            rank=self.global_rank,
            world_size=self.trainer.world_size,
            device=self.device,
        )

    def on_fit_start(self) -> None:
        super().on_fit_start()

        self._precompute_mask()

    def compute_loss(self, z: Tensor, z_global: Tensor) -> Tensor:
        """Compute the SimCLR loss.

        z_global is provided and not computed in the loss to prevent multiple gathering of z that require synchronisation among processes.

        Args:
            z: The representations of all crops.
            z_global: The global representations of all crops. Aggregated on all devices.

        Returns:
            The loss.
        """

        return compute_simclr_loss(z, z_global, self.pos_mask, self.neg_mask, self.temp)

    def training_step(self, batch: Iterable[Any], batch_idx: int) -> Dict[str, Tensor]:
        X = batch["input"]

        assert len(X) == self.num_crops

        if self.train_transform is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    X = self.train_transform(X)

        outs_online = self.multi_crop_shared_step(X)

        z = torch.cat([out_online["z"] for out_online in outs_online])
        z_global = concat_all_gather_with_backprop(z)

        loss = self.compute_loss(z, z_global)

        outputs = {"loss": loss}
        # Only compute stats for first crop to avoid unnecessary computations
        outputs.update(outs_online[0])

        for name_output, output in outputs.items():
            if name_output != "loss":
                outputs[name_output] = output.detach()

        self.log(
            "pretrain/loss", outputs["loss"], prog_bar=True, on_step=True, on_epoch=True
        )

        return outputs
