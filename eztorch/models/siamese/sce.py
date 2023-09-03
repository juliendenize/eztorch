from typing import Any, Optional

from omegaconf import DictConfig
from torch import Tensor

from eztorch.losses.sce_loss import compute_sce_loss, compute_sce_mask
from eztorch.models.siamese.shuffle_momentum_queue_base import \
    ShuffleMomentumQueueBaseModel
from eztorch.utils.utils import scheduler_value

LARGE_NUM = 1e9


class SCEModel(ShuffleMomentumQueueBaseModel):
    """SCE model.

    References:
        - SCE: https://arxiv.org/pdf/2111.14585.pdf

    Args:
        trunk: Config tu build a trunk.
        optimizer: Config tu build optimizers and schedulers.
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
        temp_m: Temperature parameter to scale the target similarities. Initial value if warmup applied.
        start_warmup_temp_m: Initial temperature parameter to scale the target similarities in case of warmup.
        warmup_epoch_temp_m: Number of warmup epochs for the target temperature.
        warmup_scheduler_temp_m: Type of scheduler for warming up the target temperature. Options are: ``'linear'``, ``'cosine'``.
        coeff: Coeff parameter between InfoNCE and relational aspects.
        warmup_scheduler_coeff: Type of scheduler for warming up the coefficient. Options are: ``'linear'``, ``'cosine'``.
        warmup_epoch_coeff: Number of warmup epochs for coefficient.
        start_warmup_coeff: Starting value of coefficient for warmup.
        scheduler_coeff: Type of scheduler for coefficient after warmup. Options are: ``'linear'``, ``'cosine'``.
        final_scheduler_coeff: Final value of scheduler coefficient.
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
        temp: float = 0.1,
        temp_m: float = 0.05,
        start_warmup_temp_m: float = 0.05,
        warmup_epoch_temp_m: int = 0,
        warmup_scheduler_temp_m: Optional[int] = "cosine",
        coeff: float = 0.5,
        warmup_scheduler_coeff: Optional[int] = "linear",
        warmup_epoch_coeff: int = 0,
        start_warmup_coeff: float = 1.0,
        scheduler_coeff: Optional[str] = None,
        final_scheduler_coeff: float = 0.0,
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
        self.temp_m = temp_m

        self.start_warmup_temp_m = start_warmup_temp_m
        self.final_temp_m = temp_m
        self.warmup_scheduler_temp_m = warmup_scheduler_temp_m
        self.warmup_epoch_temp_m = warmup_epoch_temp_m

        self.coeff = coeff
        self.initial_coeff = coeff

        self.warmup_scheduler_coeff = warmup_scheduler_coeff
        self.warmup_epoch_coeff = warmup_epoch_coeff
        self.start_warmup_coeff = start_warmup_coeff

        self.scheduler_coeff = scheduler_coeff
        self.final_scheduler_coeff = final_scheduler_coeff

    def _precompute_mask(self) -> None:
        batch_size = self.trainer.datamodule.train_local_batch_size

        self.mask = compute_sce_mask(
            batch_size=batch_size,
            num_negatives=self.queue.shape[1] if self.queue is not None else 0,
            use_keys=self.use_keys,
            rank=self.global_rank,
            world_size=self.trainer.world_size,
            device=self.device,
        )

    def on_fit_start(self) -> None:
        super().on_fit_start()

        self._precompute_mask()

    def compute_loss(
        self, q: Tensor, k: Tensor, k_global: Tensor, queue: Tensor | None
    ) -> Tensor:
        """Compute the SCE loss for several tokens as output.

        Args:
            q: The representations of the queries.
            k: The representations of the keys.
            k_global: The global representations of the keys.
            queue: The queue of representations if not None.

        Returns:
            The loss.
        """

        k_loss = k_global if self.use_keys else k

        loss = compute_sce_loss(
            q=q,
            k=k,
            k_global=k_loss,
            use_keys=self.use_keys,
            queue=queue,
            mask=self.mask,
            coeff=self.coeff,
            temp=self.temp,
            temp_m=self.temp_m,
            LARGE_NUM=LARGE_NUM,
        )

        return loss

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        if self.warmup_epoch_temp_m > 0:
            if self.current_epoch >= self.warmup_epoch_temp_m:
                self.temp_m = self.final_temp_m
            else:
                self.temp_m = scheduler_value(
                    self.warmup_scheduler_temp_m,
                    self.start_warmup_temp_m,
                    self.final_temp_m,
                    self.global_step,
                    self.warmup_epoch_temp_m * self.training_steps_per_epoch - 1,
                )

        if self.warmup_epoch_coeff > 0:
            if self.current_epoch >= self.warmup_epoch_coeff:
                self.coeff = self.initial_coeff
            else:
                self.coeff = scheduler_value(
                    self.warmup_scheduler_coeff,
                    self.start_warmup_coeff,
                    self.initial_coeff,
                    self.global_step,
                    self.warmup_epoch_coeff * self.training_steps_per_epoch - 1,
                )

        if self.scheduler_coeff is not None:
            if self.warmup_epoch_coeff > 0:
                if self.current_epoch >= self.warmup_epoch_coeff:
                    self.coeff = scheduler_value(
                        self.scheduler_coeff,
                        self.initial_coeff,
                        self.final_scheduler_coeff,
                        self.global_step
                        - self.warmup_epoch_coeff * self.training_steps_per_epoch,
                        (self.trainer.max_epochs - self.warmup_epoch_coeff)
                        * self.training_steps_per_epoch
                        - 1,
                    )
            else:
                self.coeff = scheduler_value(
                    self.scheduler_coeff,
                    self.initial_coeff,
                    self.final_scheduler_coeff,
                    self.global_step,
                    self.trainer.max_epochs * self.training_steps_per_epoch - 1,
                )

        self.log("pretrain/temp", self.temp, on_step=True, on_epoch=True)
        self.log("pretrain/temp_m", self.temp_m, on_step=True, on_epoch=True)
        self.log("pretrain/coeff", self.coeff, on_step=True, on_epoch=True)

        return
