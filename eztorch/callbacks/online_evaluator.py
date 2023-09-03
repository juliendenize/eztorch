# ------------------------------------------------------------------------
# Modified from lightning-bolts (https://github.com/Lightning-AI/lightning-bolts)
# Licensed under the Apache License, Version 2.0
# -----------------------------

from typing import Any, Dict, List, Sequence

import hydra
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.utilities import rank_zero_warn
from omegaconf import DictConfig
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parameter import Parameter
from torchmetrics.functional import accuracy

from eztorch.utils.strategies import is_strategy_ddp


class OnlineEvaluator(Callback):
    """Attaches a classifier to evaluate a specific representation from the model during training.

    Args:
        optimizer: Config to instantiate an optimizer and optionally a scheduler.
        classifier: Config to instantiate a classifier.
        input_name: Name of the representation to evaluate from the model outputs.
        precision: Precision for the classifier that must match the model, if :math:`16` use automatic mixed precision.

    Example::

        optimizer = {...} # config to build an optimizer
        classifier = {...} # config to build a classifier
        trainer = Trainer(callbacks=[OnlineEvaluator(optimizer, classifier)])
    """

    def __init__(
        self,
        optimizer: DictConfig,
        classifier: DictConfig,
        input_name: str = "h",
        precision: int = 32,
    ):
        super().__init__()

        self.input_name = input_name
        self.classifier = hydra.utils.instantiate(classifier)
        self.optimizer, self.scheduler = hydra.utils.instantiate(
            optimizer, model=self.classifier
        )
        self.precision = precision

        assert precision in [16, 32]

        self.use_amp = self.precision == 16
        self.scaler = GradScaler(enabled=self.use_amp)
        self._recovered_callback_state = None

    @property
    def learnable_params(self) -> List[Parameter]:
        """List of learnable parameters."""
        params = list(self.classifier.parameters())
        return params

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # must move to device after setup, as during setup, pl_module is still on cpu
        self.classifier = self.classifier.to(pl_module.device)

        # switch for PL compatibility reasons
        accel = (
            trainer.accelerator_connector
            if hasattr(trainer, "accelerator_connector")
            else trainer._accelerator_connector
        )
        if accel.is_distributed:
            if is_strategy_ddp(accel.strategy):
                from torch.nn.parallel import DistributedDataParallel as DDP

                self.classifier = DDP(self.classifier, device_ids=[pl_module.device])
            else:
                rank_zero_warn(
                    "Does not support this type of distributed accelerator. The online evaluator will not sync."
                )

        if self._recovered_callback_state is not None:
            self.classifier.load_state_dict(
                self._recovered_callback_state["state_dict"]
            )
            self.optimizer.load_state_dict(
                self._recovered_callback_state["optimizer_state"]
            )
            self.scaler.load_state_dict(self._recovered_callback_state["scaler"])

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
    ) -> None:

        targets = batch["label"]

        representations = outputs[self.input_name].clone().detach()

        mask = targets != -1

        with autocast(enabled=self.use_amp):
            logits = self.classifier(representations[mask])
            loss = nn.functional.cross_entropy(logits, targets[mask])

        num_classes = trainer.datamodule.num_classes
        task = "binary" if num_classes <= 2 else "multiclass"
        online_acc_1 = accuracy(logits, targets, num_classes=num_classes, task=task)
        online_acc_5 = accuracy(
            logits, targets, num_classes=num_classes, task=task, top_k=5
        )

        # update finetune weights
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        if self.scheduler is not None:
            self.scheduler.step()

        pl_module.log(
            "online/train_acc_1",
            online_acc_1,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        pl_module.log("online/train_acc_5", online_acc_5, on_step=True, on_epoch=True)
        pl_module.log("online/train_loss", loss, on_step=True, on_epoch=True)

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.classifier.eval()

    def on__validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.classifier.train()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        targets = batch["label"]

        representations = outputs[self.input_name].clone()

        mask = targets != -1

        with autocast(enabled=self.use_amp):
            logits = self.classifier(representations[mask])
            loss = nn.functional.cross_entropy(logits, targets[mask])

        num_classes = trainer.datamodule.num_classes
        task = "binary" if num_classes <= 2 else "multiclass"
        val_acc_1 = accuracy(logits, targets, num_classes=num_classes, task=task)
        val_acc_5 = accuracy(
            logits, targets, num_classes=num_classes, task=task, top_k=5
        )

        pl_module.log(
            "online/val_acc_1",
            val_acc_1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        pl_module.log(
            "online/val_acc_5", val_acc_5, on_step=False, on_epoch=True, sync_dist=True
        )
        pl_module.log(
            "online/val_loss", loss, on_step=False, on_epoch=True, sync_dist=True
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "state_dict": self.classifier.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
    ) -> None:
        self._recovered_callback_state = state_dict
