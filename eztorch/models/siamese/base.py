from abc import ABC, abstractmethod
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional

import hydra
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from eztorch.models.eztorch_base_module import EztorchBaseModule


class SiameseBaseModel(EztorchBaseModule, ABC):
    """Abstract class to represent siamese models.

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
        num_global_crops: Number of global crops which are the first elements
        of each batch.
        num_local_crops: Number of local crops which are the last elements
        of each batch.
        num_splits: Number of splits to apply to each crops.
        num_splits_per_combination: Number of splits used for combinations of features of each split.
        mutual_pass: If ``True``, perform one pass per crop resolution.
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
    ) -> None:
        super().__init__()

        assert not (
            mutual_pass and num_splits > 0
        ), "mutual_pass is not supported with num_splits > 0."

        self.save_hyperparameters()

        self.trunk = hydra.utils.instantiate(trunk)

        self.projector = (
            hydra.utils.instantiate(projector) if projector is not None else None
        )

        self.predictor = (
            hydra.utils.instantiate(predictor)
            if predictor is not None and self.projector is not None
            else None
        )

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

        self.optimizer_cfg = optimizer

        self.normalize_outputs = normalize_outputs

        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops
        self.num_crops = self.num_global_crops + self.num_local_crops

        self.num_splits = num_splits
        self.num_splits_per_combination = num_splits_per_combination
        self.use_split = num_splits > 0

        self.mutual_pass = mutual_pass

    @property
    def learnable_params(self) -> List[Parameter]:
        """List of learnable parameters."""

        params = []
        params.extend(self.trunk.parameters())

        if self.projector is not None:
            params.extend(self.projector.parameters())

        if self.predictor is not None:
            params.extend(self.predictor.parameters())
        return params

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
        h = self.trunk(x)
        z = self.projector(h) if self.projector is not None else h
        q = self.predictor(z) if self.predictor is not None else z
        return q

    @torch.no_grad()
    def local_split(self, x):
        side_indent = x.size(-2) // self.num_splits, x.size(-1) // self.num_splits
        col_splits = x.split(side_indent[1], dim=-1)
        x = [
            split
            for col_split in col_splits
            for split in col_split.split(side_indent[0], dim=-2)
        ]
        x = torch.cat(x, dim=0)
        return x

    def multi_crop_shared_step(self, x: List[Tensor]) -> List[Dict[str, Tensor]]:
        if self.mutual_pass:
            if self.num_local_crops > 0:
                return self.multi_crop_with_local_shared_step(x)
            else:
                return self.multi_crop_global_shared_step(x)
        else:
            return [self.shared_step(x_i) for x_i in x]

    def multi_crop_global_shared_step(self, x: List[Tensor]) -> List[Dict[str, Tensor]]:
        outputs = [{} for _ in range(len(x))]

        global_tensor = torch.cat(x)
        global_output = self.shared_step(global_tensor)

        dict_keys = global_output.keys()

        for key in dict_keys:
            if (
                key == "z"
                and self.projector is None
                or key == "q"
                and self.predictor is None
            ):
                continue

            global_key_output = global_output[key]
            chunked_global_key_output = global_key_output.chunk(len(x))

            for i in range(len(x)):
                outputs[i][key] = chunked_global_key_output[i]

        if self.projector is None:
            for i in range(len(x)):
                outputs[i]["z"] = outputs[i]["h"]

        if self.predictor is None:
            for i in range(len(x)):
                outputs[i]["q"] = outputs[i]["z"]

        return outputs

    def multi_crop_with_local_shared_step(
        self, x: List[Tensor]
    ) -> List[Dict[str, Tensor]]:
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )

        start_idx, h = 0, torch.empty(0, device=x[0].device)
        for _, end_idx in enumerate(idx_crops):
            h_output = self.trunk(torch.cat(x[start_idx:end_idx]))
            start_idx = end_idx
            h = torch.cat((h, h_output))

        z = self.projector(h) if self.projector is not None else h

        if self.predictor is not None:
            q = self.predictor(z)
            if self.hparams.normalize_outputs:
                # We need to normalize both representations in order to use them properly in the loss.
                z = nn.functional.normalize(z, dim=1)
                q = nn.functional.normalize(q, dim=1)
        else:
            if self.hparams.normalize_outputs:
                z = nn.functional.normalize(z, dim=1)
            q = z

        outputs = [{} for _ in range(len(x))]
        global_output = {"h": h, "z": z, "q": q}

        dict_keys = global_output.keys()

        for key in dict_keys:
            if (
                key == "z"
                and self.projector is None
                or key == "q"
                and self.predictor is None
            ):
                continue

            global_key_output = global_output[key]
            chunked_global_key_output = global_key_output.chunk(len(x))

            for i in range(len(x)):
                outputs[i][key] = chunked_global_key_output[i]

        if self.projector is None:
            for i in range(len(x)):
                outputs[i]["z"] = outputs[i]["h"]

        if self.predictor is None:
            for i in range(len(x)):
                outputs[i]["q"] = outputs[i]["z"]

        return outputs

    def shared_step(self, x: Tensor) -> Dict[str, Tensor]:
        """Shared step that pass the input tensor in transforms, the trunk, projector and predictor.

        Args:
            x: The input tensor.

        Returns:
            The computed representations.
        """

        if self.use_split:
            batch_size = x.size(0)
            x = self.local_split(x)

        h = self.trunk(x)

        if self.use_split:
            h_splits = list(h.split(h.size(0) // self.num_splits**2, dim=0))

            h = torch.cat(
                list(
                    map(
                        lambda x: sum(x) / self.num_splits_per_combination,
                        list(combinations(h_splits, r=self.num_splits_per_combination)),
                    )
                ),
                dim=0,
            )

        z = self.projector(h) if self.projector is not None else h

        if self.predictor is not None:
            q = self.predictor(z)
            if self.hparams.normalize_outputs:
                # We need to normalize both representations in order to use them properly in the loss.
                z = nn.functional.normalize(z, dim=1)
                q = nn.functional.normalize(q, dim=1)
            if self.use_split:
                q_split = q.split(batch_size, dim=0)
        else:
            if self.hparams.normalize_outputs:
                z = nn.functional.normalize(z, dim=1)
            if self.use_split:
                q_split = z.split(batch_size, dim=0)
            q = z

        if self.use_split:
            return {"h": h, "z": z, "q": q, "q_split": q_split}

        return {"h": h, "z": z, "q": q}

    def val_shared_step(self, x: Tensor) -> Dict[str, Tensor]:
        """Validation shared step that pass the input tensor in the trunk, projector and predictor.

        Args:
            x: The input tensor.

        Returns:
            The computed representations.
        """
        h = self.trunk(x)
        z = self.projector(h) if self.projector is not None else h

        if self.predictor is not None:
            q = self.predictor(z)
            if self.hparams.normalize_outputs:
                # We need to normalize both representations in order to use them properly in the loss.
                z = nn.functional.normalize(z, dim=1)
                q = nn.functional.normalize(q, dim=1)
        else:
            if self.hparams.normalize_outputs:
                z = nn.functional.normalize(z, dim=1)
            q = z

        return {"h": h, "z": z, "q": q}

    def test_shared_step(self, x: Tensor) -> Dict[str, Tensor]:
        """Test shared step that pass the input tensor in the trunk, projector and predictor.

        Args:
            x: The input tensor.

        Returns:
            The computed representations.
        """
        h = self.trunk(x)
        z = self.projector(h) if self.projector is not None else h

        if self.predictor is not None:
            q = self.predictor(z)
            if self.hparams.normalize_outputs:
                # We need to normalize both representations in order to use them properly in the loss.
                z = nn.functional.normalize(z, dim=1)
                q = nn.functional.normalize(q, dim=1)
        else:
            if self.hparams.normalize_outputs:
                z = nn.functional.normalize(z, dim=1)
            q = z

        return {"h": h, "z": z, "q": q}

    def up_to_projector_shared_step(self, x: Tensor) -> Dict[str, Tensor]:
        """Shared step that pass the input tensor in the trunk and projector.

        Args:
            x: The input tensor.

        Returns:
            The computed representations.
        """
        h = self.trunk(x)
        z = self.projector(h) if self.projector is not None else h
        if self.hparams.normalize_outputs:
            z = nn.functional.normalize(z, dim=1)

        return {"h": h, "z": z}

    @abstractmethod
    def compute_loss(self):
        pass

    @abstractmethod
    def training_step(self, batch: Iterable[Any], batch_idx: int):
        pass

    def validation_step(self, batch: Iterable[Any], batch_idx: int):
        x = batch["input"]

        if self.val_transform is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    x = self.val_transform(x)

        return self.val_shared_step(x)

    def test_step(self, batch: Iterable[Any], batch_idx: int):
        x = batch["input"]

        if self.test_transform is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    x = self.test_transform(x)

        return self.test_shared_step(x)

    @property
    def num_layers(self) -> int:
        """Number of layers of the model."""
        return (
            self.trunk.num_layers
            + self.projector.num_layers
            + (self.predictor.num_layers if self.predictor is not None else 0)
        )

    def get_param_layer_id(self, name: str) -> int:
        """Get the layer id of the named parameter.

        Args:
            name: The name of the parameter.
        """
        if name.startswith("trunk."):
            return self.trunk.get_param_layer_id(name[len("trunk.") :])
        elif name.startswith("projector."):
            return self.trunk.num_layers + self.projector.get_param_layer_id(
                name[len("projector.") :]
            )
        elif name.startswith("predictor."):
            return (
                self.trunk.num_layers
                + self.projector.num_layers
                + self.predictor.get_param_layer_id(name[len("predictor.") :])
            )
        else:
            raise NotImplementedError(f"{name} not found.")
