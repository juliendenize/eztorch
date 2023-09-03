from abc import ABC
from math import comb
from typing import Any, Dict, Iterable, Optional

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from eztorch.models.siamese.shuffle_momentum_base import \
    ShuffleMomentumSiameseBaseModel
from eztorch.modules.gather import concat_all_gather_without_backprop


class ShuffleMomentumQueueBaseModel(ShuffleMomentumSiameseBaseModel, ABC):
    """SCE model.

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
        initial_momentum: Initial value for the momentum update.
        scheduler_momentum: Rule to update the momentum value.
        shuffle_bn: If ``True``, apply shuffle normalization trick from MoCo.
        num_devices: Number of devices used to train the model in each node.
        simulate_n_devices: Number of devices to simulate to apply shuffle trick. Requires ``shuffle_bn`` to be ``True`` and ``num_devices`` to be :math:`1`.
        queue: Config to build a queue.
        sym: If ``True``, symmetrised the loss.
        use_keys: If ``True``, add keys to negatives.
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
        shuffle_bn: bool = False,
        num_devices: int = 1,
        simulate_n_devices: int = 8,
        queue: Optional[DictConfig] = None,
        sym: bool = False,
        use_keys: bool = False,
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
        )

        self.save_hyperparameters()

        self.sym = sym
        self.use_keys = use_keys

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

    def training_step(self, batch: Iterable[Any], batch_idx: int) -> Dict[str, Tensor]:
        X = batch["input"]
        X = [X] if isinstance(X, Tensor) else X

        assert len(X) == self.num_crops

        if self.train_transform is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    X = self.train_transform(X)

        if self.queue is not None:
            queue = self.queue.clone().detach()
        else:
            queue = None

        if self.sym:
            if self.shuffle_bn:
                outs_momentum = [
                    self.momentum_shared_step(x) for x in X[: self.num_global_crops]
                ]
            else:
                outs_momentum = self.multi_crop_momentum_shared_step(
                    X[: self.num_global_crops]
                )

            outs_online = self.multi_crop_shared_step(X)

            k_globals = [
                concat_all_gather_without_backprop(outs_momentum[i]["z"])
                for i in range(self.num_global_crops)
            ]

            tot_loss = 0
            for i in range(self.num_global_crops):
                if self.use_split:
                    for j in range(self.num_crops):
                        if i == j:
                            continue
                        for q in outs_online[j]["q_split"]:
                            loss = self.compute_loss(
                                q, outs_momentum[i]["z"], k_globals[i], queue
                            )
                            tot_loss += loss
                else:
                    for j in range(self.num_crops):
                        if i == j:
                            continue
                        loss = self.compute_loss(
                            outs_online[j]["q"],
                            outs_momentum[i]["z"],
                            k_globals[i],
                            queue,
                        )
                        tot_loss += loss
            if self.use_split:
                tot_loss /= (
                    (self.num_crops - 1)
                    * self.num_global_crops
                    * comb(self.num_splits**2, self.num_splits_per_combination)
                )
            else:
                tot_loss /= (self.num_crops - 1) * self.num_global_crops

            if self.queue is not None:
                k_global = torch.cat(k_globals, dim=0)
                self.update_queue(k_global)

        else:
            if self.num_crops > 2:
                outs_online = self.multi_crop_shared_step(X[1:])
            else:
                outs_online = [self.shared_step(X[1])]

            outs_momentum = self.momentum_shared_step(X[0])
            k_global = concat_all_gather_without_backprop(outs_momentum["z"])

            tot_loss = 0
            for j in range(self.num_crops - 1):
                if self.use_split:
                    for q in outs_online[j]["q_split"]:
                        loss = self.compute_loss(q, outs_momentum["z"], k_global, queue)
                        tot_loss += loss
                else:
                    loss = self.compute_loss(
                        outs_online[j]["q"], outs_momentum["z"], k_global, queue
                    )
                    tot_loss += loss
            if self.use_split:
                tot_loss /= (self.num_crops - 1) * comb(
                    self.num_splits**2, self.num_splits_per_combination
                )
            else:
                tot_loss /= self.num_crops - 1

            if self.queue is not None:
                self.update_queue(k_global)

        outputs = {"loss": tot_loss}
        # Only keep outputs from first computation to avoid unnecessary time and memory cost.
        outputs.update(outs_online[0])
        for name_output, output in outputs.items():
            if name_output not in ["loss", "q_split"]:
                outputs[name_output] = output.detach()

        self.log(
            "pretrain/loss", outputs["loss"], prog_bar=True, on_step=True, on_epoch=True
        )

        return outputs
