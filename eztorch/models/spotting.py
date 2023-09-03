import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.distributed as dist
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import DictConfig
from tabulate import tabulate
from torch import Tensor, nn
from torch.nn import Parameter

from eztorch.datasets.spot_utils.eval import compute_mAPs, load_json
from eztorch.datasets.spot_utils.predictions import (
    add_clips_predictions, aggregate_and_filter_clips, initialize_predictions,
    load_raw_spotting_predictions, save_raw_spotting_predictions,
    save_spotting_predictions)
from eztorch.datasets.utils_fn import get_shard_indices
from eztorch.losses.spot_loss import compute_spot_loss_fn
from eztorch.models.eztorch_base_module import EztorchBaseModule
from eztorch.utils.checkpoints import (get_sub_state_dict_from_pl_ckpt,
                                       remove_pattern_in_keys_from_dict)


class SpottingModel(EztorchBaseModule):
    r"""Model to perform spotting.

    Args:
        trunk: Config to build a trunk.
        head_class: Config to build a head for classification.
        optimizer: Config to build an optimizer for trunk.
        pretrained_trunk_path: Path to the pretrained trunk file.
        pretrained_path: Path to the pretrained model.
        prediction_args: Arguments to configure predictions.
        loss_fn_args: Arguments for the loss function.
        trunk_pattern: Pattern to retrieve the trunk model in checkpoint state_dict and delete the key.
        freeze_trunk: Whether to freeze the trunk.
        train_transform: Config to perform transformation on train input.
        val_transform: Config to perform transformation on val input.
        test_transform: Config to perform transformation on test input.
        save_val_preds_path: Path to store the validation predictions.
        save_test_preds_path: Path to store the test predictions.
        NMS_args: Arguments to configure the NMS.
        evaluation_args: Arguments to configure the evaluation.

    Example::

        trunk = {...} # config to build a trunk
        head_class = {...} # config to build a head for classification
        optimizer = {...} # config to build an optimizer
        pretrained_trunk_path = ... # path where the trunk has been saved

        model = SpottingModel(trunk, head_class, optimizer, pretrained_trunk_path)
    """

    def __init__(
        self,
        trunk: DictConfig,
        head_class: DictConfig,
        optimizer: DictConfig,
        pretrained_path: str | None = None,
        pretrained_trunk_path: str | None = None,
        loss_fn_args: DictConfig = DictConfig({}),
        prediction_args: DictConfig = DictConfig({}),
        trunk_pattern: str = r"^(trunk\.)",
        freeze_trunk: bool = False,
        train_transform: DictConfig | None = None,
        val_transform: DictConfig | None = None,
        test_transform: DictConfig | None = None,
        save_val_preds_path: str | Path = "val_preds/",
        save_test_preds_path: str | Path = "test_preds/",
        NMS_args: DictConfig = DictConfig({"window": 10, "threshold": 0.001}),
        evaluation_args: DictConfig = DictConfig({}),
        do_compile: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.optimizer_cfg = optimizer
        self.freeze_trunk = freeze_trunk

        self.trunk: nn.Module = hydra.utils.instantiate(trunk)

        if pretrained_trunk_path is not None:
            trunk_state_dict = get_sub_state_dict_from_pl_ckpt(
                checkpoint_path=pretrained_trunk_path, pattern=trunk_pattern
            )
            trunk_state_dict = remove_pattern_in_keys_from_dict(
                d=trunk_state_dict, pattern=trunk_pattern
            )

            missing_keys, unexpected_keys = self.trunk.load_state_dict(
                trunk_state_dict, strict=False
            )
            rank_zero_info(
                f"Loaded {__class__.__name__} from pretrained trunk weights model.\n"
                f"missing_keys:{missing_keys}\n"
                f"unexpected_keys:{unexpected_keys}"
            )

        self.head_class: nn.Module = hydra.utils.instantiate(head_class)
        self.loss_fn = compute_spot_loss_fn

        if do_compile:
            self.trunk = torch.compile(self.trunk)
            self.head_class = torch.compile(self.head_class)
            self.loss_fn = torch.compile(self.loss_fn)

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

        self.loss_fn_args = dict(loss_fn_args)

        self.remove_frames_predictions = prediction_args.get(
            "remove_frames_predictions", -1.0
        )

        self.merge_predictions_type = prediction_args.get(
            "merge_predictions_type", "max"
        )

        self.save_val_preds_path = Path(save_val_preds_path)
        self.save_test_preds_path = Path(save_test_preds_path)

        self.NMS_args = NMS_args
        self.evaluation_args = evaluation_args

        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location="cpu")["state_dict"]

            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False
            )
            rank_zero_info(
                f"Loaded {__class__.__name__} from pretrained torch model.\n"
                f"missing_keys:{missing_keys}\n"
                f"unexpected_keys:{unexpected_keys}"
            )

            rank_zero_info(
                f"{__class__.__name__} loaded trunk weights from {pretrained_trunk_path}."
            )

        if self.freeze_trunk:
            for param in self.trunk.parameters():
                param.requires_grad = False

    @property
    def learnable_params(self) -> List[Tuple[str, Parameter]]:
        """Learnable parameters of the model."""
        params = []
        if not self.freeze_trunk:
            params.extend(self.trunk.parameters())
        params.extend(self.head_class.parameters())

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

    def on_validation_epoch_start(self) -> None:
        local_shard_indices = get_shard_indices(
            self.trainer.datamodule.val_dataset.num_videos, shuffle_shards=False
        )
        self.min_video_index = min(local_shard_indices)
        self.max_video_index = max(local_shard_indices)

        self.predictions = initialize_predictions(
            self.trainer.datamodule.val_dataset,
            self.max_video_index,
            self.min_video_index,
            self.device,
        )
        return

    def on_test_start(self) -> None:
        local_shard_indices = get_shard_indices(
            self.trainer.datamodule.test_dataset.num_videos, shuffle_shards=False
        )
        self.min_video_index = min(local_shard_indices)
        self.max_video_index = max(local_shard_indices)

        self.predictions = initialize_predictions(
            self.trainer.datamodule.test_dataset,
            self.max_video_index,
            self.min_video_index,
            self.device,
        )

        return

    def on_train_epoch_start(self) -> None:
        if self.freeze_trunk:
            self.trunk.eval()

    def forward(self, x: Tensor) -> Dict[str, Any]:
        h = self.trunk(x)
        class_preds = self.head_class(h)
        return {
            "class_preds": class_preds,
            "h": h,
        }

    @property
    def num_layers(self) -> int:
        """Number of layers of the model."""
        return self.num_layers_class

    @property
    def num_layers_class(self) -> int:
        return self.trunk.num_layers + self.head_class.num_layers

    def get_param_layer_id(self, name: str) -> int:
        """Get the layer id of the named parameter.

        Args:
            name: The name of the parameter.
        """
        if name.startswith("trunk."):
            return self.trunk.get_param_layer_id(name[len("trunk.") :])
        elif name.startswith("head_class."):
            return self.trunk.num_layers + self.head_class.get_param_layer_id(
                name[len("head_class.") :]
            )
        else:
            raise NotImplementedError(f"{name} should not have been used.")

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

    def shared_step(self, x: Tensor, inversed_temporal_masked_indices: Tensor | None):
        if self.freeze_trunk:
            with torch.no_grad():
                if inversed_temporal_masked_indices is not None:
                    h: Tensor = self.trunk(
                        x,
                        inversed_temporal_masked_indices=inversed_temporal_masked_indices,
                    )
                else:
                    h = self.trunk(x)
        else:
            if inversed_temporal_masked_indices is not None:
                h = self.trunk(
                    x, inversed_temporal_masked_indices=inversed_temporal_masked_indices
                )
            else:
                h = self.trunk(x)
        if h.ndim == 2:
            h = h.reshape(h.shape[0], 1, h.shape[1])
        class_preds: Tensor = self.head_class(h)

        return class_preds

    def training_step(self, batch: Dict[Any, Any], batch_idx: int) -> Tensor:
        if self.train_transform is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    batch = self.train_transform(batch)

        (x, labels, has_label, ignore_class,) = (
            batch["input"],
            batch["labels"],
            batch["has_label"],
            batch["ignore_class"],
        )

        inversed_temporal_masked_indices: Tensor | None = batch.get(
            "inversed_temporal_masked_indices", None
        )

        class_preds = self.shared_step(x, inversed_temporal_masked_indices)

        loss = self.loss_fn(
            class_preds=class_preds,
            class_target=labels,
            ignore_class=ignore_class,
            has_label=has_label,
            mixup_weights=batch.get("mixup_weights", None),
            **self.loss_fn_args,
        )

        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )

        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        if self.val_transform is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    batch = self.val_transform(batch)

        (x, frames, labels, has_label, ignore_class, video_indexes, num_frames,) = (
            batch["input"],
            batch["frame_indices"],
            batch["labels"],
            batch["has_label"],
            batch["ignore_class"],
            batch["video_index"],
            batch["num_frames"],
        )

        class_preds = self.shared_step(x, inversed_temporal_masked_indices=None)

        loss = self.loss_fn(
            class_preds=class_preds,
            class_target=labels,
            ignore_class=ignore_class,
            has_label=has_label,
            mixup_weights=batch.get("mixup_weights", None),
            **self.loss_fn_args,
        )

        class_preds = class_preds.sigmoid()

        kept_tensors = aggregate_and_filter_clips(
            class_preds,
            frames,
            num_frames,
            video_indexes,
            self.max_video_index,
            self.min_video_index,
        )

        if kept_tensors is not None:
            (
                class_preds,
                frames,
                num_frames,
                video_indexes,
            ) = kept_tensors

            add_clips_predictions(
                self.predictions,
                class_preds,
                frames,
                num_frames,
                video_indexes,
                self.remove_frames_predictions,
                self.merge_predictions_type,
            )

        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        if self.test_transform is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    batch = self.test_transform(batch)

        x, frames, num_frames, video_indexes = (
            batch["input"],
            batch["frame_indices"],
            batch["num_frames"],
            batch["video_index"],
        )

        class_preds = self.shared_step(x, inversed_temporal_masked_indices=None)

        class_preds = class_preds.sigmoid()

        kept_tensors = aggregate_and_filter_clips(
            class_preds,
            frames,
            num_frames,
            video_indexes,
            self.max_video_index,
            self.min_video_index,
        )

        if kept_tensors is not None:
            (
                class_preds,
                frames,
                num_frames,
                video_indexes,
            ) = kept_tensors

            add_clips_predictions(
                self.predictions,
                class_preds,
                frames,
                num_frames,
                video_indexes,
                self.remove_frames_predictions,
                self.merge_predictions_type,
            )

        return 0

    def _make_evaluation(
        self,
        predictions_path: str | Path,
        logger: bool = False,
    ) -> None:

        pred = load_json(predictions_path)
        truth = load_json(self.evaluation_args.get("truth_path"))

        maps, tolerances, header, rows = compute_mAPs(
            truth,
            pred,
            self.evaluation_args["tolerances"],
            self.evaluation_args.get("plot_pr", False),
        )

        rank_zero_info(tabulate(rows, headers=header, floatfmt="0.2f"))
        rank_zero_info(f"Avg mAP (across tolerances): {np.mean(maps) * 100:0.2f}")
        if logger:
            self.log(
                "eval/Avg-mAP",
                np.mean(maps) * 100,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
            for map, tolerance in zip(maps, tolerances):
                self.log(
                    f"eval/mAP@{tolerance}",
                    map * 100,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return

        preds_path = self.save_val_preds_path / str(self.trainer.current_epoch)
        raw_preds_path = self.save_val_preds_path / f"{self.trainer.current_epoch}_raw"

        save_raw_spotting_predictions(self.predictions, raw_preds_path, make_zip=False)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if self.global_rank == 0:
            shutil.make_archive(str(raw_preds_path), "zip", raw_preds_path)
            shutil.rmtree(raw_preds_path)

            predictions = load_raw_spotting_predictions(
                str(raw_preds_path) + ".zip",
                list(range(self.trainer.datamodule.val_dataset.num_videos)),
                self.device,
            )

            save_spotting_predictions(
                predictions,
                preds_path,
                self.trainer.datamodule.val_dataset,
                self.NMS_args,
            )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        self._make_evaluation(
            preds_path / f"predictions.json",
            logger=True,
        )

        return super().on_validation_epoch_end()

    def on_test_end(self) -> None:
        raw_preds_path = (
            self.save_test_preds_path.parent / f"{self.save_test_preds_path.name}_raw"
        )

        save_raw_spotting_predictions(self.predictions, raw_preds_path, make_zip=False)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if self.global_rank == 0:
            shutil.make_archive(str(raw_preds_path), "zip", raw_preds_path)
            shutil.rmtree(raw_preds_path)

            predictions = load_raw_spotting_predictions(
                str(raw_preds_path) + ".zip",
                list(range(self.trainer.datamodule.test_dataset.num_videos)),
                self.device,
            )

            save_spotting_predictions(
                predictions,
                self.save_test_preds_path,
                self.trainer.datamodule.test_dataset,
                self.NMS_args,
            )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        self._make_evaluation(
            self.save_test_preds_path / f"predictions.json",
            logger=False,
        )

        return super().on_test_end()
