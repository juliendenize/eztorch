from typing import Any, Dict, List, Optional

import hydra
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from torchmetrics.classification.accuracy import Accuracy

from eztorch.evaluation.test_time_augmentation_fn import \
    get_test_time_augmentation_fn
from eztorch.models.eztorch_base_module import EztorchBaseModule


class SupervisedModel(EztorchBaseModule):
    """Supervised model.

    Args:
        model: Config to build a model.
        optimizer: Config to build optimizers and schedulers.
        train_transform: Config to perform transformation on train input.
        val_transform: Config to perform transformation on val input.
        test_transform: Config to perform transformation on test input.
        val_time_augmentation: Ensembling method for test time augmentation used at validation.
        test_time_augmentation: Ensembling method for test time augmentation used at test.
    """

    def __init__(
        self,
        model: DictConfig,
        optimizer: DictConfig,
        train_transform: Optional[DictConfig] = None,
        val_transform: Optional[DictConfig] = None,
        test_transform: Optional[DictConfig] = None,
        val_time_augmentation: Optional[DictConfig] = None,
        test_time_augmentation: Optional[DictConfig] = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.model = hydra.utils.instantiate(model)
        self.optimizer_cfg = optimizer

        self.train_transform = (
            hydra.utils.instantiate(train_transform)
            if train_transform is not None
            else None
        )

        self.val_transform = (
            hydra.utils.instantiate(val_transform)
            if val_transform is not None
            else None
        )

        self.test_transform = (
            hydra.utils.instantiate(test_transform)
            if test_transform is not None
            else None
        )

        self.val_time_augmentation = (
            get_test_time_augmentation_fn(**val_time_augmentation)
            if val_time_augmentation
            else None
        )
        self.test_time_augmentation = (
            get_test_time_augmentation_fn(**test_time_augmentation)
            if test_time_augmentation
            else None
        )

    @property
    def learnable_params(self) -> List[Parameter]:
        """Learnable parameters of the model."""
        return list(self.model.parameters())

    @property
    def training_steps_per_epoch(self) -> Optional[int]:
        """Total training steps inferred from datamodule and devices."""
        if (
            self.trainer.datamodule is not None
            and self.trainer.datamodule.train_num_samples > 0
        ):
            return (
                self.trainer.datamodule.train_num_samples
                // self.trainer.datamodule.train_global_batch_size
            )
        else:
            return None

    def on_fit_start(self) -> None:
        num_classes = self.trainer.datamodule.num_classes
        task = "binary" if num_classes <= 2 else "multiclass"
        self.train_acc_1 = Accuracy(task=task, num_classes=num_classes, top_k=1).to(
            self.device
        )
        self.train_acc_5 = Accuracy(task=task, num_classes=num_classes, top_k=5).to(
            self.device
        )
        self.val_acc_1 = Accuracy(task=task, num_classes=num_classes, top_k=1).to(
            self.device
        )
        self.val_acc_5 = Accuracy(task=task, num_classes=num_classes, top_k=5).to(
            self.device
        )

    def on_test_start(self) -> None:
        num_classes = self.trainer.datamodule.num_classes
        task = "binary" if num_classes <= 2 else "multiclass"
        self.test_acc_1 = Accuracy(task=task, num_classes=num_classes, top_k=1).to(
            self.device
        )
        self.test_acc_5 = Accuracy(task=task, num_classes=num_classes, top_k=5).to(
            self.device
        )

    def configure_optimizers(self) -> Dict[Any, Any]:
        optimizer, scheduler = hydra.utils.instantiate(
            self.optimizer_cfg,
            num_steps_per_epoch=self.optimizer_cfg.get(
                "num_steps_per_epoch", self.training_steps_per_epoch
            ),
            model=self,
        )

        if scheduler is None:
            return optimizer

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return x

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, targets = batch["input"], batch["label"]

        if self.train_transform is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    x = self.train_transform(x)

        preds = self(x)
        loss = nn.functional.cross_entropy(preds, targets)

        acc_1 = self.train_acc_1(preds, targets)
        acc_5 = self.train_acc_5(preds, targets)

        self.log("train/loss", loss, on_epoch=True)
        self.log("train/acc_1", acc_1, on_epoch=True, prog_bar=True)
        self.log("train/acc_5", acc_5, on_epoch=True)

        return loss

    @property
    def num_layers(self) -> int:
        """Number of layers of the model."""
        return self.model.num_layers

    def get_param_layer_id(self, name: str) -> int:
        """Get the layer id of the named parameter.

        Args:
            name: The name of the parameter.
        """
        if name.startswith("model."):
            return self.model.get_param_layer_id(name[len("model.") :])
        else:
            raise NotImplementedError(f"{name} should not have been used.")

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        if self.val_time_augmentation is not None:
            x, targets, idx = batch["input"], batch["label"], batch["idx"]

            if self.val_transform is not None:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        x = self.val_transform(x)

            preds = self(x)
            preds = preds.softmax(-1)
            preds, targets, idx = self.val_time_augmentation(preds, targets, idx)
        else:
            x, targets = batch["input"], batch["label"]

            if self.val_transform is not None:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        x = self.val_transform(x)

            preds = self(x)

        loss = nn.functional.cross_entropy(preds, targets)

        self.val_acc_1(preds, targets)
        self.val_acc_5(preds, targets)

        self.log("val/loss", loss)
        self.log("val/acc_1", self.val_acc_1, prog_bar=True)
        self.log("val/acc_5", self.val_acc_5)

        return loss

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        if self.test_time_augmentation is not None:
            x, targets, idx = batch["input"], batch["label"], batch["idx"]

            if self.test_transform is not None:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        x = self.test_transform(x)

            preds = self(x)
            preds = preds.softmax(-1)
            preds, targets, idx = self.test_time_augmentation(preds, targets, idx)
        else:
            x, targets = batch["input"], batch["label"]

            if self.test_transform is not None:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        x = self.test_transform(x)

            preds = self(x)

        loss = nn.functional.cross_entropy(preds, targets)

        self.test_acc_1(preds, targets)
        self.test_acc_5(preds, targets)

        self.log("test/loss", loss)
        self.log("test/acc_1", self.test_acc_1, prog_bar=True)
        self.log("test/acc_5", self.test_acc_5)

        return loss
