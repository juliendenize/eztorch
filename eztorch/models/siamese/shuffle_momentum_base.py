from abc import ABC
from typing import Dict, Optional, Tuple

import torch
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import DictConfig
from torch import Tensor, nn

from eztorch.models.siamese.momentum_base import MomentumSiameseBaseModel
from eztorch.modules.gather import concat_all_gather_without_backprop
from eztorch.modules.split_batch_norm import convert_to_split_batchnorm
from eztorch.utils.strategies import get_num_devices_in_trainer


class ShuffleMomentumSiameseBaseModel(MomentumSiameseBaseModel, ABC):
    """Abstract class to represent siamese models with a momentum branch and possibility to shuffle input elements
    in momentum branch to apply normalization trick from MoCo.

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
        initial_momentum: int = 0.996,
        scheduler_momentum: str = "cosine",
        shuffle_bn: bool = True,
        num_devices: int = 1,
        simulate_n_devices: int = 8,
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
        )

        self.save_hyperparameters()

        self.num_devices = num_devices
        self.shuffle_bn = shuffle_bn
        self.simulate_n_devices = simulate_n_devices

        if self.num_devices == -1 and self.shuffle_bn:
            rank_zero_info(
                f"In {__class__.__name__} when num_devices=-1, it is assumed that there are more than one device."
            )

        elif self.num_devices <= 1 and self.shuffle_bn:
            if self.simulate_n_devices <= 1:
                AttributeError(
                    "if num_devices is 1 and shuffle_bn is True, the simulate_n_devices attribute should be superior to 1"
                )
            self.momentum_trunk = convert_to_split_batchnorm(
                self.momentum_trunk, self.simulate_n_devices
            )
            if self.momentum_projector is not None:
                self.momentum_projector = convert_to_split_batchnorm(
                    self.momentum_projector, self.simulate_n_devices
                )

    def on_train_start(self):
        old_num_devices = self.num_devices
        self.num_devices = get_num_devices_in_trainer(self.trainer)
        if old_num_devices != self.num_devices:
            rank_zero_info(
                f"Num devices passed to {__class__.__name__}: {old_num_devices} has been updated to {self.num_devices}."
            )

    @torch.no_grad()
    def momentum_shared_step(self, x: Tensor) -> Dict[str, Tensor]:
        """Momentum shared step that call either '_momentum_shared_step_n_devices' or
        '_momentum_shared_step_single_device' depending the number of devices used for training.

        Args:
            x: The input tensor.

        Returns:
            The computed representations.
        """

        if self.num_devices > 1:
            return self._momentum_shared_step_n_devices(x)

        else:
            return self._momentum_shared_step_single_device(x)

    @torch.no_grad()
    def _momentum_shared_step_n_devices(self, x: Tensor) -> Dict[str, Tensor]:
        """Momentum shared step with several devices passing input tensor in momentum trunk and momentum projector.
        If shuffle_bn is True, it gathers and shuffles the input tensors across devices following MoCo batch norm
        trick.

        *** Only support DistributedDataParallel (DDP) model. ***

        Args:
            x: The input tensor.

        Returns:
            The computed representations.
        """

        if self.shuffle_bn:
            x, idx_unshuffle = self._batch_shuffle_ddp(x)

        h = self.momentum_trunk(x)
        z = self.momentum_projector(h) if self.momentum_projector is not None else h

        if self.shuffle_bn:
            z = self._batch_unshuffle_ddp(z, idx_unshuffle)

        if self.normalize_outputs:
            z = nn.functional.normalize(z, dim=1)

        return {"h": h, "z": z}

    @torch.no_grad()
    def _momentum_shared_step_single_device(self, x: Tensor) -> Dict[str, Tensor]:
        """Momentum shared step with one device passing input tensor in momentum trunk and momentum projector. If
        shuffle_bn is True, it shuffles the input tensor across device following MoCo batch norm trick which is
        simulated in the trunk.

        *** Only support DistributedDataParallel (DDP) model. ***

        Args:
            x: The input tensor.

        Returns:
            The computed representations.
        """
        if self.shuffle_bn:
            x, idx_unshuffle = self._batch_shuffle_single_device(x)

        h = self.momentum_trunk(x)
        z = self.momentum_projector(h) if self.momentum_projector is not None else h

        if self.shuffle_bn:
            z = self._batch_unshuffle_single_device(z, idx_unshuffle)

        if self.normalize_outputs:
            z = nn.functional.normalize(z, dim=1)

        return {"h": h, "z": z}

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Unshuffle the shuffled tensor along first dimension across devices.

        *** Only support DistributedDataParallel (DDP) model. ***

        Args:
            x: The shuffled tensor.
            idx_unshuffle: The unshuffle indices to retrieve original tensor before its shuffling.

        Returns:
            The shuffled tensor and the unshuffle indices.
        """

        # gather from all devices
        x_gather = concat_all_gather_without_backprop(x)
        batch_size_all = x_gather.shape[0]

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all, device=self.device)

        # broadcast to all devices
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this device
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(self.num_devices, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x: Tensor, idx_unshuffle: Tensor) -> Tensor:
        """Unshuffle the shuffled tensor along first dimension across devices.

        *** Only support DistributedDataParallel (DDP) model. ***

        Args:
            x: The shuffled tensor.
            idx_unshuffle: The unshuffle indices to retrieve original tensor before its shuffling.

        Returns:
            The unshuffled tensor.
        """

        # gather from all devices
        x_gather = concat_all_gather_without_backprop(x)

        # restored index for this device
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(self.num_devices, -1)[gpu_idx]

        return x_gather[idx_this]

    @torch.no_grad()
    def _batch_shuffle_single_device(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Shuffle the input tensor along first dimension on current device.

        Args:
            x: The input tensor.

        Returns:
            The shuffled tensor and the unshuffle indices.
        """

        # random shuffle index
        batch_size = x.shape[0]
        idx_shuffle = torch.randperm(batch_size, device=self.device)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_device(
        self, x: Tensor, idx_unshuffle: Tensor
    ) -> Tensor:
        """Unshuffle the shuffled tensor along first dimension on current device.

        Args:
            x: The shuffled tensor.
            idx_unshuffle: The unshuffle indices to retrieve original tensor before its shuffling.

        Returns:
            The unshuffled tensor.
        """

        return x[idx_unshuffle]
