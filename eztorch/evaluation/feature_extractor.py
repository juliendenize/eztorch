from fractions import Fraction
from typing import Any, Dict, List, Optional

import hydra
import lightning.pytorch as pl
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Identity
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset

from eztorch.datasets.soccernet_utils.features import save_features
from eztorch.datasets.spot_utils.features import \
    save_features as spot_save_features
from eztorch.evaluation.test_time_augmentation_fn import \
    get_test_time_augmentation_fn
from eztorch.utils.checkpoints import (get_sub_state_dict_from_pl_ckpt,
                                       remove_pattern_in_keys_from_dict)


class FeatureExtractor(pl.LightningModule):
    """Feature extractor.

    Args:
        trunk: Config to build a trunk.
        pretrained_trunk_path: Path to the pretrained trunk file.
        head: Config to build a head such as for pooling, normalizing, ....
        trunk_pattern: Pattern to retrieve the trunk model in checkpoint state_dict and delete the key.
        train_transform: Config to perform transformation on train input.
        val_transform: Config to perform transformation on val input.
        test_transform: Config to perform transformation on test input.
        train_time_augmentation: If not ``None``, ensembling method for test time augmentation used for the training data.
        val_time_augmentation: If not ``None``, ensembling method for test time augmentation used for the validation data.
        test_time_augmentation: If not ``None``, ensembling method for test time augmentation used for the test data.
        storage_prefix_name: Prefix of the files to store features, labels and ids. The split is added before the prefix, example for train split: `f"train{prefix}_features.pth"`.
    """

    def __init__(
        self,
        trunk: DictConfig,
        pretrained_trunk_path: str,
        head: Optional[DictConfig] = None,
        trunk_pattern: str = r"^(trunk\.)",
        train_transform: Optional[DictConfig] = None,
        val_transform: Optional[DictConfig] = None,
        test_transform: Optional[DictConfig] = None,
        train_time_augmentation: Optional[DictConfig] = None,
        val_time_augmentation: Optional[DictConfig] = None,
        test_time_augmentation: Optional[DictConfig] = None,
        storage_prefix_name: str = "",
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        trunk_state_dict = get_sub_state_dict_from_pl_ckpt(
            checkpoint_path=pretrained_trunk_path, pattern=trunk_pattern
        )
        trunk_state_dict = remove_pattern_in_keys_from_dict(
            d=trunk_state_dict, pattern=trunk_pattern
        )

        self.trunk = hydra.utils.instantiate(trunk)
        self.trunk.load_state_dict(trunk_state_dict)

        self.head = hydra.utils.instantiate(head) if head is not None else Identity()

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

        self.train_time_augmentation = (
            get_test_time_augmentation_fn(**train_time_augmentation)
            if train_time_augmentation
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

        self.storage_prefix_name = storage_prefix_name
        self.automatic_optimization = False

    @property
    def learnable_params(self) -> List[Parameter]:
        """List of learnable parameters."""
        params = list()
        return params

    def configure_optimizers(self):
        return

    def forward(self, x: Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            h = self.trunk(x)
            h = self.head(h)
        return {"features": h}

    def shared_step(self, x: Tensor):
        with torch.no_grad():
            h = self.trunk(x)
            h = self.head(h)

        return h

    def on_train_epoch_start(self) -> None:
        self.eval()
        self.train_outputs = {"features": [], "labels": [], "ids": []}
        return

    def on_validation_epoch_start(self) -> None:
        self.val_outputs = {"features": [], "labels": [], "ids": []}
        return

    def on_test_epoch_start(self) -> None:
        self.test_outputs = {"features": [], "labels": [], "ids": []}
        return

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        if self.train_time_augmentation is not None:
            x, labels, ids = batch["input"], batch["label"], batch["idx"]

            if self.train_transform is not None:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        x = self.train_transform(x)

            features = self.shared_step(x)
            features, labels, ids = self.train_time_augmentation(features, labels, ids)
        else:
            x, labels, ids = batch["input"], batch["label"], batch["idx"]

            if self.train_transform is not None:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        x = self.train_transform(x)

            features = self.shared_step(x)

        self.train_outputs["features"].append(features.cpu())
        self.train_outputs["labels"].append(labels.cpu())
        self.train_outputs["ids"].append(ids.cpu())

        return torch.tensor(0)

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        if self.val_time_augmentation is not None:
            x, labels, ids = batch["input"], batch["label"], batch["idx"]

            if self.val_transform is not None:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        x = self.val_transform(x)

            features = self.shared_step(x)
            features, labels, ids = self.val_time_augmentation(features, labels, ids)
        else:
            x, labels, ids = batch["input"], batch["label"], batch["idx"]

            if self.val_transform is not None:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        x = self.val_transform(x)

            features = self.shared_step(x)

        self.val_outputs["features"].append(features.cpu())
        self.val_outputs["labels"].append(labels.cpu())
        self.val_outputs["ids"].append(ids.cpu())

        return torch.tensor(0)

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        if self.test_time_augmentation is not None:
            x, labels, ids = batch["input"], batch["label"], batch["idx"]

            if self.test_transform is not None:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        x = self.test_transform(x)

            features = self.shared_step(x)
            features, labels, ids = self.test_time_augmentation(features, labels, ids)
        else:
            x, labels, ids = batch["input"], batch["label"], batch["idx"]

            if self.test_transform is not None:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        x = self.test_transform(x)

            features = self.shared_step(x)

        self.test_outputs["features"].append(features.cpu())
        self.test_outputs["labels"].append(labels.cpu())
        self.test_outputs["ids"].append(ids.cpu())

        return torch.tensor(0)

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        features = torch.cat(self.train_outputs["features"])
        labels = torch.cat(self.train_outputs["labels"])
        ids = torch.cat(self.train_outputs["ids"])

        torch.save(features, f"train{self.storage_prefix_name}_features.pth")
        torch.save(labels, f"train{self.storage_prefix_name}_labels.pth")
        torch.save(ids, f"train{self.storage_prefix_name}_ids.pth")

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        features = torch.cat(self.val_outputs["features"])
        labels = torch.cat(self.val_outputs["labels"])
        ids = torch.cat(self.val_outputs["ids"])

        torch.save(features, f"val{self.storage_prefix_name}_features.pth")
        torch.save(labels, f"val{self.storage_prefix_name}_labels.pth")
        torch.save(ids, f"val{self.storage_prefix_name}_ids.pth")

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()
        features = torch.cat(self.test_outputs["features"])
        labels = torch.cat(self.test_outputs["labels"])
        ids = torch.cat(self.test_outputs["ids"])

        torch.save(features, f"test{self.storage_prefix_name}_features.pth")
        torch.save(labels, f"test{self.storage_prefix_name}_labels.pth")
        torch.save(ids, f"test{self.storage_prefix_name}_ids.pth")


class SoccerNetFeatureExtractor(pl.LightningModule):
    """Feature extractor.

    Args:
        trunk: Config to build a trunk.
        dim_features: Dimension of the features.
        filename: Filename of the files to store features following the naming convention of SoccerNet Baidu features.
        pretrained_trunk_path: Path to the pretrained trunk file.
        head: Config to build a head such as for pooling, normalizing, ....
        trunk_pattern: Pattern to retrieve the trunk model in checkpoint state_dict and delete the key.
        train_transform: Config to perform transformation on train input.
        val_transform: Config to perform transformation on val input.
        test_transform: Config to perform transformation on test input.
        fps_in: Fps of the input.
        fps_out: Fps of extraction.
    """

    def __init__(
        self,
        trunk: DictConfig,
        dim_features: int,
        filename: str,
        pretrained_trunk_path: str,
        head: Optional[DictConfig] = None,
        trunk_pattern: str = r"^(trunk\.)",
        train_transform: Optional[DictConfig] = None,
        val_transform: Optional[DictConfig] = None,
        test_transform: Optional[DictConfig] = None,
        fps_in: int = 4,
        fps_out: int = 2,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        trunk_state_dict = get_sub_state_dict_from_pl_ckpt(
            checkpoint_path=pretrained_trunk_path, pattern=trunk_pattern
        )
        trunk_state_dict = remove_pattern_in_keys_from_dict(
            d=trunk_state_dict, pattern=trunk_pattern
        )

        self.trunk = hydra.utils.instantiate(trunk)
        self.trunk.load_state_dict(trunk_state_dict)

        self.head = hydra.utils.instantiate(head) if head is not None else Identity()

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

        self.filename = filename
        self.automatic_optimization = False
        self.dim_features = dim_features
        self.fps_in = fps_in
        self.fps_out = fps_out

    @property
    def learnable_params(self) -> List[Parameter]:
        """List of learnable parameters."""
        params = list()
        return params

    def configure_optimizers(self):
        return

    def forward(self, x: Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            h = self.trunk(x)
            h = self.head(h)
        return {"features": h}

    def shared_step(self, x: Tensor):
        with torch.no_grad():
            h = self.trunk(x)
            h = self.head(h)

        return h

    def _initialize_features(
        self,
        dataset: Dataset,
    ) -> Dict[int, Dict[int, Tensor]]:
        frac_fps = Fraction(self.fps_out)
        over_frac_fps = Fraction(1, self.fps_out)

        features = {}

        for i in range(dataset.num_videos):
            video_metadata = dataset.get_video_metadata(i)
            features[i] = {}
            for h_id, h_idx in zip(
                video_metadata["half_id"], video_metadata["half_idx"]
            ):
                end_sec = Fraction(
                    Fraction(int(video_metadata["duration"][h_idx % 2] * self.fps_out)),
                    frac_fps,
                )
                num_features = len(
                    torch.arange(
                        0,
                        float(end_sec),
                        float(over_frac_fps),
                    )
                )

                features[i][h_id] = torch.zeros(
                    (num_features, self.dim_features), device="cpu"
                )

        return features

    def _add_features(
        self,
        features: Dict[int, Dict[int, Tensor]],
        predictions: Tensor,
        video_index: Tensor,
        half_id: Tensor,
        indices: Tensor,
    ):
        for p, v, h, i in zip(predictions, video_index, half_id, indices):
            features[v][h][i] = p
        return

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()

        self.train_features = self._initialize_features(
            self.trainer.datamodule.train_dataset,
        )

        return

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()

        self.val_features = self._initialize_features(
            self.trainer.datamodule.val_dataset,
        )
        return

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()

        self.eval()

        self.test_features = self._initialize_features(
            self.trainer.datamodule.test_dataset,
        )
        return

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        (x, video_idx, half_idx, frame_indices) = (
            batch["input"],
            batch["video_index"].tolist(),
            batch["half"].tolist(),
            batch["frame_indices"].cpu(),
        )

        if self.train_transform is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    x = self.train_transform(x)

        features = self.shared_step(x)
        indices = frame_indices[:, frame_indices.shape[1] // 2].tolist()

        self._add_features(
            self.train_features, features.cpu(), video_idx, half_idx, indices
        )

        return torch.tensor(0)

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        (x, video_idx, half_idx, frame_indices) = (
            batch["input"],
            batch["video_index"].tolist(),
            batch["half"].tolist(),
            batch["frame_indices"].cpu(),
        )

        if self.val_transform is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    x = self.val_transform(x)

        features = self.shared_step(x)
        indices = frame_indices[:, frame_indices.shape[1] // 2].tolist()

        self._add_features(
            self.val_features, features.cpu(), video_idx, half_idx, indices
        )

        return torch.tensor(0)

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        (x, video_idx, half_idx, frame_indices) = (
            batch["input"],
            batch["video_index"].tolist(),
            batch["half"].tolist(),
            batch["frame_indices"].cpu(),
        )

        if self.test_transform is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    x = self.test_transform(x)

        features = self.shared_step(x)

        frame_indices = frame_indices[:, frame_indices.shape[1] // 2]
        if self.fps_out < self.fps_in:
            if not self.fps_in % self.fps_out == 0:
                raise NotImplementedError

            frame_indices = frame_indices // self.fps_out

        indices = frame_indices.tolist()

        self._add_features(
            self.test_features, features.cpu(), video_idx, half_idx, indices
        )

        return torch.tensor(0)

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()

        save_features(
            self.trainer.datamodule.train_dataset,
            f"{self.filename}/",
            self.train_features,
            self.filename,
            True,
        )

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()

        save_features(
            self.trainer.datamodule.val_dataset,
            f"{self.filename}/",
            self.val_features,
            self.filename,
            True,
        )

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()

        save_features(
            self.trainer.datamodule.test_dataset,
            f"{self.filename}/",
            self.test_features,
            self.filename,
            True,
        )


class SpotFeatureExtractor(pl.LightningModule):
    """Feature extractor.

    Args:
        trunk: Config to build a trunk.
        dim_features: Dimension of the features.
        filename: Filename of the files to store features following the naming convention of SoccerNet Baidu features.
        pretrained_trunk_path: Path to the pretrained trunk file.
        head: Config to build a head such as for pooling, normalizing, ....
        trunk_pattern: Pattern to retrieve the trunk model in checkpoint state_dict and delete the key.
        train_transform: Config to perform transformation on train input.
        val_transform: Config to perform transformation on val input.
        test_transform: Config to perform transformation on test input.
        window_duration: Duration of each window. Used to initialize features dimension.

    Example::

        trunk = {...} # config to build a trunk
        pretrained_trunk_path = ... # path where the trunk has been saved

        model = FeatureExtractor(trunk, pretrained_trunk_path)
    """

    def __init__(
        self,
        trunk: DictConfig,
        dim_features: int,
        filename: str,
        pretrained_trunk_path: str,
        head: Optional[DictConfig] = None,
        trunk_pattern: str = r"^(trunk\.)",
        train_transform: Optional[DictConfig] = None,
        val_transform: Optional[DictConfig] = None,
        test_transform: Optional[DictConfig] = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        trunk_state_dict = get_sub_state_dict_from_pl_ckpt(
            checkpoint_path=pretrained_trunk_path, pattern=trunk_pattern
        )
        trunk_state_dict = remove_pattern_in_keys_from_dict(
            d=trunk_state_dict, pattern=trunk_pattern
        )

        self.trunk = hydra.utils.instantiate(trunk)
        self.trunk.load_state_dict(trunk_state_dict)

        self.head = hydra.utils.instantiate(head) if head is not None else Identity()

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

        self.filename = filename
        self.automatic_optimization = False
        self.dim_features = dim_features

    @property
    def learnable_params(self) -> List[Parameter]:
        """List of learnable parameters."""
        params = list()
        return params

    def configure_optimizers(self):
        return

    def forward(self, x: Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            h = self.trunk(x)
            h = self.head(h)
        return {"features": h}

    def shared_step(self, x: Tensor):
        with torch.no_grad():
            h = self.trunk(x)
            h = self.head(h)

        return h

    def _initialize_features(
        self,
        dataset: Dataset,
    ) -> Dict[int, Dict[int, Tensor]]:

        features = {}

        for i in range(dataset.num_videos):
            video_metadata = dataset.get_video_metadata(i)
            features[i] = torch.zeros(
                (video_metadata["num_frames"], self.dim_features), device="cpu"
            )

        return features

    def _add_features(
        self,
        features: Dict[int, Dict[int, Tensor]],
        predictions: Tensor,
        video_index: Tensor,
        indices: Tensor,
    ):
        for p, v, i in zip(predictions, video_index, indices):
            features[v][i] = p
        return

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()

        self.train_features = self._initialize_features(
            self.trainer.datamodule.train_dataset,
        )

        return

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()

        self.val_features = self._initialize_features(
            self.trainer.datamodule.val_dataset,
        )
        return

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()

        self.eval()

        self.test_features = self._initialize_features(
            self.trainer.datamodule.test_dataset,
        )
        return

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        (x, video_idx, frame_indices) = (
            batch["input"],
            batch["video_index"].tolist(),
            batch["frame_indices"].cpu(),
        )

        if self.train_transform is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    x = self.train_transform(x)

        features = self.shared_step(x)
        indices = frame_indices[:, frame_indices.shape[1] // 2].tolist()

        self._add_features(self.train_features, features.cpu(), video_idx, indices)

        return torch.tensor(0)

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        (x, video_idx, frame_indices) = (
            batch["input"],
            batch["video_index"].tolist(),
            batch["frame_indices"].cpu(),
        )

        if self.val_transform is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    x = self.val_transform(x)

        features = self.shared_step(x)
        indices = frame_indices[:, frame_indices.shape[1] // 2].tolist()

        self._add_features(self.val_features, features.cpu(), video_idx, indices)

        return torch.tensor(0)

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        (x, video_idx, frame_indices) = (
            batch["input"],
            batch["video_index"].tolist(),
            batch["frame_indices"].cpu(),
        )

        if self.test_transform is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    x = self.test_transform(x)

        features = self.shared_step(x)

        frame_indices = frame_indices[:, frame_indices.shape[1] // 2]

        indices = frame_indices.tolist()

        self._add_features(self.test_features, features.cpu(), video_idx, indices)

        return torch.tensor(0)

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()

        spot_save_features(
            self.trainer.datamodule.train_dataset,
            f"{self.filename}/",
            self.train_features,
            self.filename,
            True,
        )

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()

        spot_save_features(
            self.trainer.datamodule.val_dataset,
            f"{self.filename}/",
            self.val_features,
            self.filename,
            True,
        )

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()

        spot_save_features(
            self.trainer.datamodule.test_dataset,
            f"{self.filename}/",
            self.test_features,
            self.filename,
            True,
        )
