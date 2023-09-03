from typing import Any, Dict, Iterable, Optional

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from eztorch.losses.sce_token_loss import (compute_sce_token_loss,
                                           compute_sce_token_masks)
from eztorch.models.siamese.base import SiameseBaseModel
from eztorch.models.siamese.sce import SCEModel
from eztorch.modules.gather import concat_all_gather_without_backprop

LARGE_NUM = 1e9


class SCETokensModel(SCEModel):
    """SCE model for tokens output.

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
        mutual_pass: If ``True``, perform one pass per branch per crop resolution.
        initial_momentum: initial value for the momentum update.
        scheduler_momentum: rule to update the momentum value.
        queue: Config to build a queue.
        sym: If ``True``, symmetrised the loss.
        use_keys: If ``True``, add aligned keys to negatives.
        use_all_keys: If ``True``, add all keys to negatives.
        num_out_tokens: Number of expected output tokens.
        positive_radius: Number of adjacent tokens to consider as positives.
        keep_aligned_positive: Whether to keep the aligned token as positive.
        temp: Temperature parameter to scale the online similarities.
        temp_m: Temperature parameter to scale the target similarities. Initial value if warmup applied.
        start_warmup_temp_m: Initial temperature parameter to scale the target similarities in case of warmup.
        warmup_epoch_temp_m: Number of warmup epochs for the target temperature.
        warmup_scheduler_temp_m: Type of scheduler for warming up the target temperature. Options are: ``'linear'``, ``'cosine'``.
        coeff: Coeff parameter between InfoNCE and relational aspects.
        normalize_positive_coeff: Whether to use the `coeff` argument or multiply it by normalized mask over number of positives.
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
        mutual_pass: bool = False,
        initial_momentum: int = 0.999,
        scheduler_momentum: str = "constant",
        queue: DictConfig = {},
        sym: bool = False,
        use_keys: bool = True,
        use_all_keys: bool = False,
        num_prefix_tokens: int = 0,
        num_out_tokens: int = 32,
        positive_radius: int = 0,
        keep_aligned_positive: bool = True,
        temp: float = 0.1,
        temp_m: float = 0.05,
        start_warmup_temp_m: float = 0.05,
        warmup_epoch_temp_m: int = 0,
        warmup_scheduler_temp_m: Optional[int] = "cosine",
        coeff: float = 0.5,
        normalize_positive_coeff: bool = False,
        warmup_scheduler_coeff: Optional[int] = "linear",
        warmup_epoch_coeff: int = 0,
        start_warmup_coeff: float = 1,
        scheduler_coeff: Optional[str] = None,
        final_scheduler_coeff: float = 0,
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
            num_splits=0,
            num_splits_per_combination=0,
            mutual_pass=mutual_pass,
            initial_momentum=initial_momentum,
            scheduler_momentum=scheduler_momentum,
            shuffle_bn=False,
            num_devices=1,
            simulate_n_devices=8,
            queue=queue,
            sym=sym,
            use_keys=use_keys,
            temp=temp,
            temp_m=temp_m,
            start_warmup_temp_m=start_warmup_temp_m,
            warmup_epoch_temp_m=warmup_epoch_temp_m,
            warmup_scheduler_temp_m=warmup_scheduler_temp_m,
            coeff=coeff,
            warmup_scheduler_coeff=warmup_scheduler_coeff,
            warmup_epoch_coeff=warmup_epoch_coeff,
            start_warmup_coeff=start_warmup_coeff,
            scheduler_coeff=scheduler_coeff,
            final_scheduler_coeff=final_scheduler_coeff,
        )

        self.use_all_keys = use_all_keys
        self.num_prefix_tokens = num_prefix_tokens

        if num_prefix_tokens > 0:
            raise NotImplementedError("For now prefix tokens is not implemented.")

        self.num_out_tokens = num_out_tokens
        self.positive_radius = positive_radius
        self.keep_aligned_positive = keep_aligned_positive

        self.normalize_positive_coeff = normalize_positive_coeff

        if self.use_keys and self.use_all_keys:
            raise NotImplementedError(
                "Only one of use_keys or use_all_keys should be True."
            )

    def _precompute_masks(self) -> None:
        batch_size = self.trainer.datamodule.train_local_batch_size
        num_tokens = self.num_out_tokens - self.num_prefix_tokens

        (
            self.mask_sim_q,
            self.mask_sim_k,
            self.mask_prob_q,
            self.mask_log_q,
            self.num_positives_per_token,
        ) = compute_sce_token_masks(
            batch_size=batch_size,
            num_tokens=num_tokens,
            num_negatives=self.queue.shape[1],
            positive_radius=self.positive_radius,
            keep_aligned_positive=self.keep_aligned_positive,
            use_keys=self.use_keys,
            use_all_keys=self.use_all_keys,
            rank=self.global_rank,
            world_size=self.trainer.world_size,
            device=self.device,
        )

    def on_fit_start(self) -> None:
        super().on_fit_start()

        self._precompute_masks()

        self.coeff = (
            self.initial_coeff * (1 - 1 / (self.num_positives_per_token + 1))
            if self.normalize_positive_coeff
            else torch.tensor(self.initial_coeff, device=self.device)
        )

    def compute_loss(
        self, q: Tensor, k: Tensor, k_global: Tensor, queue: Tensor | None
    ) -> Tensor:
        """Compute the SCE loss for several tokens as output.

        Args:
            q: The representations of the queries.
            k: The representations of the keys.
            k_global: The global representations of the keys.
            queue: The queue of representations if queue is not None.

        Returns:
            The loss.
        """

        k_loss = k_global if self.use_all_keys else k

        loss = compute_sce_token_loss(
            q=q,
            k=k,
            k_global=k_loss,
            queue=queue,
            mask_sim_q=self.mask_sim_q,
            mask_sim_k=self.mask_sim_k,
            mask_prob_q=self.mask_prob_q,
            mask_log_q=self.mask_log_q,
            coeff=self.coeff,
            temp=self.temp,
            temp_m=self.temp_m,
            LARGE_NUM=LARGE_NUM,
        )

        return loss

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        if self.warmup_epoch_temp_m > 0:
            raise NotImplementedError(
                "Not implemented for now, until scheduler per epoch is supported."
            )

        if self.warmup_epoch_coeff > 0:
            raise NotImplementedError(
                "Not implemented for now, until scheduler per epoch is supported."
            )

        if self.scheduler_coeff is not None:
            raise NotImplementedError(
                "Not implemented for now, until scheduler per epoch is supported."
            )

        self.log(
            "pretrain/temp", self.temp, on_step=True, on_epoch=True, sync_dist=False
        )
        self.log(
            "pretrain/temp_m", self.temp_m, on_step=True, on_epoch=True, sync_dist=False
        )
        self.log(
            "pretrain/coeff",
            self.coeff.mean(),
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )

        return


class SCEDistillTokens(SiameseBaseModel):
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
        mutual_pass: bool = False,
        queue: DictConfig | None = None,
        use_keys: bool = True,
        use_all_keys: bool = False,
        num_prefix_tokens: int = 0,
        num_out_tokens: int = 32,
        positive_radius: int = 0,
        keep_aligned_positive: bool = True,
        temp: float = 0.1,
        temp_m: float = 0.05,
        coeff: float = 0.5,
        normalize_positive_coeff: bool = False,
    ) -> None:
        """SCE model for tokens output with distillation of features.

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
            mutual_pass: If ``True``, perform one pass per branch per crop resolution.
            queue: Config to build a queue.
            use_keys: If ``True``, add aligned keys to negatives.
            use_all_keys: If ``True``, add all keys to negatives.
            num_out_tokens: Number of expected output tokens.
            positive_radius: Number of adjacent tokens to consider as positives.
            keep_aligned_positive: Whether to keep the aligned token as positive.
            temp: Temperature parameter to scale the online similarities.
            temp_m: Temperature parameter to scale the target similarities. Initial value if warmup applied.
            coeff: Coeff parameter between InfoNCE and relational aspects.
            normalize_positive_coeff: Whether to use the `coeff` argument or multiply it by normalized mask over number of positives.
        """

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
            num_splits=0,
            num_splits_per_combination=0,
            mutual_pass=mutual_pass,
        )

        self.coeff = coeff
        self.temp = temp
        self.temp_m = temp_m
        self.use_keys = use_keys

        self.use_all_keys = use_all_keys
        self.num_prefix_tokens = num_prefix_tokens

        if num_prefix_tokens > 0:
            raise NotImplementedError("For now prefix tokens is not implemented.")

        self.num_out_tokens = num_out_tokens
        self.positive_radius = positive_radius
        self.keep_aligned_positive = keep_aligned_positive

        self.normalize_positive_coeff = normalize_positive_coeff

        if self.use_keys and self.use_all_keys:
            raise NotImplementedError(
                "Only one of use_keys or use_all_keys should be True."
            )

        if queue is not None:
            self.register_buffer(
                "queue",
                nn.functional.normalize(
                    torch.randn(queue.feature_dim, queue.size), dim=0
                ),
            )
            self.queue_size = queue.size
            self.queue_feature_dim = queue.feature_dim
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        else:
            self.queue = None
        self.accumulated_keys = []

    @torch.no_grad()
    def update_queue(self, x: Tensor) -> None:
        """Update the queue using a fifo Procedure.

        Args:
            x: The tensor to add to the queue.
        """

        def _do_update_queue(x: Tensor) -> None:
            batch_size = x.shape[1]

            ptr = int(self.queue_ptr)
            # for simplicity
            assert self.queue_size % batch_size == 0

            # replace the keys at ptr (dequeue and enqueue)
            self.queue[:, ptr : ptr + batch_size] = x

            # move pointer
            ptr = (ptr + batch_size) % self.queue_size
            self.queue_ptr[0] = ptr

        if self.trainer.accumulate_grad_batches > 1:
            should_accumulate = (
                not self.trainer.fit_loop.epoch_loop._accumulated_batches_reached()
                and not self.trainer.fit_loop.epoch_loop._num_ready_batches_reached()
            )
            self.accumulated_keys.append(x.T)
            if not should_accumulate:
                accumulated_keys = torch.cat(self.accumulated_keys, 1)
                _do_update_queue(accumulated_keys)
                self.accumulated_keys = []
        else:
            _do_update_queue(x.T)

    def _precompute_masks(self) -> None:
        batch_size = self.trainer.datamodule.train_local_batch_size
        num_tokens = self.num_out_tokens - self.num_prefix_tokens

        (
            self.mask_sim_q,
            self.mask_sim_k,
            self.mask_prob_q,
            self.mask_log_q,
            self.num_positives_per_token,
        ) = compute_sce_token_masks(
            batch_size=batch_size,
            num_tokens=num_tokens,
            num_negatives=self.queue.shape[1],
            positive_radius=self.positive_radius,
            keep_aligned_positive=self.keep_aligned_positive,
            use_keys=self.use_keys,
            use_all_keys=self.use_all_keys,
            rank=self.global_rank,
            world_size=self.trainer.world_size,
            device=self.device,
        )

    def on_fit_start(self) -> None:
        super().on_fit_start()

        self._precompute_masks()

        self.coeff = (
            self.coeff * (1 - 1 / (self.num_positives_per_token + 1))
            if self.normalize_positive_coeff
            else torch.tensor(self.coeff, device=self.device)
        )

    def compute_loss(
        self, q: Tensor, k: Tensor, k_global: Tensor, queue: Tensor | None
    ) -> Tensor:
        """Compute the SCE loss for several tokens as output.

        Args:
            q: The representations of the queries.
            k: The representations of the keys.
            k_global: The global representations of the keys.
            queue: The queue of representations if queue is not None.

        Returns:
            The loss.
        """

        k_global = k_global if self.use_all_keys else k

        loss = compute_sce_token_loss(
            q=q,
            k=k,
            k_global=k_global,
            queue=queue,
            mask_sim_q=self.mask_sim_q,
            mask_sim_k=self.mask_sim_k,
            mask_prob_q=self.mask_prob_q,
            mask_log_q=self.mask_log_q,
            coeff=self.coeff,
            temp=self.temp,
            temp_m=self.temp_m,
            LARGE_NUM=LARGE_NUM,
        )

        return loss

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        self.log(
            "pretrain/temp", self.temp, on_step=True, on_epoch=True, sync_dist=False
        )
        self.log(
            "pretrain/temp_m", self.temp_m, on_step=True, on_epoch=True, sync_dist=False
        )
        self.log(
            "pretrain/coeff",
            self.coeff.mean(),
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )

        return

    def training_step(self, batch: Iterable[Any], batch_idx: int) -> Dict[str, Tensor]:
        X = batch["input"]
        X = [X] if isinstance(X, Tensor) else X

        assert len(X) == self.num_crops

        if self.train_transform is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    X = self.train_transform(X)

        if self.num_crops > 2:
            outs_online = self.multi_crop_shared_step(X)
        else:
            outs_online = [self.shared_step(X[0])]

        features = batch["features"]
        b, t, *dims = features.shape
        features = features.view((b * t, *dims))

        if self.hparams.normalize_outputs:
            features = nn.functional.normalize(features, dim=1)

        features_global = concat_all_gather_without_backprop(features)

        if self.queue is not None:
            queue = self.queue.clone().detach()
        else:
            queue = None

        tot_loss = 0
        for j in range(self.num_crops):
            loss = self.compute_loss(
                outs_online[j]["q"], features, features_global, queue
            )
            tot_loss += loss

        tot_loss /= self.num_crops

        if self.queue is not None:
            self.update_queue(features_global)

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
