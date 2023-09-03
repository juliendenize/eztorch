from typing import Optional

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from eztorch.losses.moco_loss import compute_moco_loss
from eztorch.models.siamese.shuffle_momentum_queue_base import \
    ShuffleMomentumQueueBaseModel


class MoCoModel(ShuffleMomentumQueueBaseModel):
    """MoCo model with version 1, 2, 2+, 3 that can be configured.

    References:
        - MoCo: https://arxiv.org/abs/1911.05722
        - MoCov2: https://arxiv.org/abs/2003.04297
        - MoCov2+: https://arxiv.org/abs/2011.10566
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
        initial_momentum: initial value for the momentum update.
        scheduler_momentum: rule to update the momentum value.
        shuffle_bn: If ``True``, apply shuffle normalization trick from MoCo.
        num_devices: Number of devices used to train the model in each node.
        simulate_n_devices: Number of devices to simulate to apply shuffle trick. Requires ``shuffle_bn`` to be ``True`` and ``num_devices`` to be :math:`1`.
        queue: Config to build a queue.
        sym: If ``True``, symmetrised the loss.
        use_keys: If ``True``, add keys to negatives.
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
        initial_momentum: int = 0.999,
        scheduler_momentum: str = "constant",
        shuffle_bn: bool = True,
        num_devices: int = 1,
        simulate_n_devices: int = 8,
        queue: Optional[DictConfig] = None,
        sym: bool = False,
        use_keys: bool = False,
        temp: float = 0.2,
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
            shuffle_bn=shuffle_bn,
            num_devices=num_devices,
            simulate_n_devices=simulate_n_devices,
            queue=queue,
            sym=sym,
            use_keys=use_keys,
        )

        self.save_hyperparameters()

        self.temp = temp

    def compute_loss(
        self, q: Tensor, k: Tensor, k_global: Tensor, queue: Tensor | None
    ) -> Tensor:
        """Compute the MoCo loss.

        Args:
            q: The representations of the queries.
            k: The representations of the keys.
            k_global: The global representations of the keys.
            queue: The queue of representations.

        Returns:
            The loss.
        """

        return compute_moco_loss(
            q,
            k,
            k_global,
            self.use_keys,
            queue,
            self.temp,
            self.global_rank,
        )

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()

        self.log("pretrain/temp", self.temp, on_step=False, on_epoch=True)
