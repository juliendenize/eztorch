from typing import Any, Dict, Iterable, Optional

import torch
from omegaconf import DictConfig
from torch import Tensor

from eztorch.losses.mocov3_loss import compute_mocov3_loss
from eztorch.models.siamese.momentum_base import MomentumSiameseBaseModel
from eztorch.modules.gather import concat_all_gather_without_backprop


class MoCov3Model(MomentumSiameseBaseModel):
    """MoCov3 that can be configured as in the paper.

    References:
        - MoCov3: https://arxiv.org/abs/2104.02057

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
        temp: Temperature parameter to scale the online similarities.
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
        initial_momentum: int = 0.99,
        scheduler_momentum: str = "cosine",
        temp: float = 1.0,
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
            initial_momentum=initial_momentum,
            scheduler_momentum=scheduler_momentum,
        )

        self.save_hyperparameters()

        assert not self.use_split, "Splits not supported for MoCov3"

        self.temp = temp

    def compute_loss(self, q: Tensor, k: Tensor) -> Tensor:
        """Compute the MoCo loss.

        Args:
            q: The representations of the queries.
            k: The representations of the keys.

        Returns:
            The loss.
        """

        k_global = concat_all_gather_without_backprop(k)

        return compute_mocov3_loss(
            q, k_global, self.device, self.temp, self.global_rank
        )

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()

        self.log("pretrain/temp", self.temp, on_step=False, on_epoch=True)

    def training_step(self, batch: Iterable[Any], batch_idx: int) -> Dict[str, Tensor]:
        X = batch["input"]
        X = [X] if isinstance(X, Tensor) else X

        assert len(X) == self.num_crops

        if self.train_transform is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    X = self.transform(X)

        outs_online = self.multi_crop_shared_step(X)
        outs_momentum = self.multi_crop_momentum_shared_step(X[: self.num_global_crops])

        tot_loss = 0
        for i in range(self.num_global_crops):
            for j in range(self.num_crops):
                if i == j:
                    continue
                loss = self.compute_loss(outs_online[j]["q"], outs_momentum[i]["z"])
                tot_loss += loss

        outputs = {"loss": tot_loss}
        # Only keep outputs from first computation to avoid unnecessary time and memory cost.
        outputs.update(outs_online[0])
        for name_output, output in outputs.items():
            if name_output != "loss":
                outputs[name_output] = output.detach()

        self.log(
            "pretrain/loss", outputs["loss"], prog_bar=True, on_step=True, on_epoch=True
        )

        return outputs
