from typing import Any, Dict, List, Optional

import hydra
import lightning.pytorch as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch.nn.parameter import Parameter
from torchmetrics.classification.accuracy import Accuracy

from eztorch.evaluation.test_time_augmentation_fn import \
    get_test_time_augmentation_fn
from eztorch.utils.checkpoints import (get_sub_state_dict_from_pl_ckpt,
                                       remove_pattern_in_keys_from_dict)


class LinearClassifierEvaluation(pl.LightningModule):
    """Linear classifier evaluation for self-supervised learning.

    Args:
        trunk: Config to build a trunk.
        classifier: Config to build a classifier.
        optimizer: Config to build an optimizer.
        pretrained_trunk_path: Path to the pretrained trunk file.
        trunk_pattern: Pattern to retrieve the trunk model in checkpoint state_dict and delete the key.
        train_transform: Config to perform transformation on train input.
        val_transform: Config to perform transformation on val input.
        test_transform: Config to perform transformation on test input.
        val_time_augmentation: If not ``None``, ensembling method for test time augmentation used at validation.
        test_time_augmentation: If not ``None``, ensembling method for test time augmentation used at test.
    """

    def __init__(
        self,
        trunk: DictConfig,
        classifier: DictConfig,
        optimizer: DictConfig,
        pretrained_trunk_path: str,
        trunk_pattern: str = r"^(trunk\.)",
        train_transform: Optional[DictConfig] = None,
        val_transform: Optional[DictConfig] = None,
        test_transform: Optional[DictConfig] = None,
        val_time_augmentation: Optional[DictConfig] = None,
        test_time_augmentation: Optional[DictConfig] = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.optimizer_cfg = optimizer

        trunk_state_dict = get_sub_state_dict_from_pl_ckpt(
            checkpoint_path=pretrained_trunk_path, pattern=trunk_pattern
        )
        trunk_state_dict = remove_pattern_in_keys_from_dict(
            d=trunk_state_dict, pattern=trunk_pattern
        )

        self.trunk = hydra.utils.instantiate(trunk)
        self.trunk.load_state_dict(trunk_state_dict)

        for param in self.trunk.parameters():
            param.requires_grad = False

        self.classifier = hydra.utils.instantiate(classifier)

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
        """List of learnable parameters."""
        params = list(self.classifier.parameters())
        return params

    @property
    def num_layers(self) -> int:
        """Number of layers of the model."""
        return self.classifier.num_layers

    def get_param_layer_id(self, name: str) -> int:
        """Get the layer id of the named parameter.

        Args:
            name: The name of the parameter.
        """
        self.classifier.get_param_layer_id(name[len("classifier.") :])

    @property
    def training_steps_per_epoch(self) -> Optional[int]:
        """Total training steps inferred from datamodule and devices."""
        # TODO use pl.__version__ when the train_dataloader is initialized in trainer before configure_optimizers is called.
        # if pl.__version__ >= '1.x.0': return len(self.trainer.train_dataloader)
        if self.trainer.datamodule is not None:
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

    def forward(self, x: Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            h = self.trunk(x)
        preds = self.classifier(h)
        return {"preds": preds, "h": h}

    def configure_optimizers(self) -> Dict[Any, Any]:
        optimizer, scheduler = hydra.utils.instantiate(
            self.optimizer_cfg,
            num_steps_per_epoch=self.training_steps_per_epoch,
            model=self,
        )

        if scheduler is None:
            return optimizer

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def shared_step(self, x: Tensor):
        with torch.no_grad():
            h = self.trunk(x)

        preds = self.classifier(h)

        return preds

    def on_train_epoch_start(self) -> None:
        self.trunk.eval()

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, targets = batch["input"], batch["label"]

        if self.train_transform is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    x = self.train_transform(x)

        preds = self.shared_step(x)
        loss = nn.functional.cross_entropy(preds, targets)

        acc_1 = self.train_acc_1(preds, targets)
        acc_5 = self.train_acc_5(preds, targets)

        self.log("train/loss", loss, on_epoch=True)
        self.log("train/acc_1", acc_1, on_epoch=True, prog_bar=True)
        self.log("train/acc_5", acc_5, on_epoch=True)

        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        if self.val_time_augmentation is not None:
            x, targets, ids = batch["input"], batch["label"], batch["idx"]

            if self.val_transform is not None:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        x = self.val_transform(x)

            preds = self.shared_step(x)
            preds = preds.softmax(-1)
            preds, targets, ids = self.val_time_augmentation(preds, targets, ids)
        else:
            x, targets = batch["input"], batch["label"]

            if self.val_transform is not None:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        x = self.val_transform(x)

            preds = self.shared_step(x)

        loss = nn.functional.cross_entropy(preds, targets)

        self.val_acc_1(preds, targets)
        self.val_acc_5(preds, targets)

        self.log("val/loss", loss)
        self.log("val/acc_1", self.val_acc_1, prog_bar=True)
        self.log("val/acc_5", self.val_acc_5)

        return loss

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        if self.test_time_augmentation is not None:
            x, targets, ids = batch["input"], batch["label"], batch["idx"]

            if self.test_transform is not None:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        x = self.test_transform(x)

            preds = self.shared_step(x)
            preds = preds.softmax(-1)
            preds, targets, ids = self.test_time_augmentation(preds, targets, ids)
        else:
            x, targets = batch["input"], batch["label"]

            if self.test_transform is not None:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        x = self.test_transform(x)

            preds = self.shared_step(x)

        loss = nn.functional.cross_entropy(preds, targets)

        self.test_acc_1(preds, targets)
        self.test_acc_5(preds, targets)

        self.log("test/loss", loss)
        self.log("test/acc_1", self.test_acc_1, prog_bar=True)
        self.log("test/acc_5", self.test_acc_5)

        return loss
