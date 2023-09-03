from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from lightning.pytorch.utilities import rank_zero_info
from torch.nn import (BatchNorm1d, BatchNorm2d, BatchNorm3d, LayerNorm, Module,
                      SyncBatchNorm)
from torch.nn.parameter import Parameter

from eztorch.modules.split_batch_norm import SplitBatchNorm2D
from eztorch.optimizers.lars import LARS

_NORM_LAYERS = (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    SyncBatchNorm,
    SplitBatchNorm2D,
    LayerNorm,
)
_OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "lars": LARS,
}


def retrieve_model_params(
    model: Module,
    modules_to_filter: Iterable[Module] = [],
    keys_to_filter: Iterable[str] = [],
) -> Tuple[List[Parameter], List[Parameter]]:
    """Retrieve sets of filtered and not filtered parameters from a model.

    Args:
        model: Model to retrieve the params from.
        modules_to_filter: Module to filter.
        keys_to_filter: keys to filter.

    Returns:
        Filtered parameters, other parameters.
    """

    other_parameters = []
    filtered_parameters = []
    for module in model.modules():
        if type(module) in modules_to_filter:
            for param_name, param in module.named_parameters(recurse=False):
                filtered_parameters.append(param)
        else:
            for param_name, param in module.named_parameters(recurse=False):
                no_key = all([param_name != key for key in keys_to_filter])
                if no_key:
                    other_parameters.append(param)
                else:
                    filtered_parameters.append(param)
    return filtered_parameters, other_parameters


def filter_learnable_params(
    parameters: Iterable[Parameter], model: Module
) -> List[Parameter]:
    """Filter passed parameters to be in learnable parameters list from model. If model do not have
    ``learnable_params`` property defined, return all passed parameters.

    Args:
        parameters: Parameters to filter.
        model: Model to retrieve learnable parameters from.

    Returns:
        Learnable parameters.
    """

    if hasattr(model, "learnable_params"):
        return [
            param
            for param in parameters
            if param.requires_grad
            and any(
                [param is learnable_param for learnable_param in model.learnable_params]
            )
        ]
    else:
        rank_zero_info(
            f"Model of type {type(model)} has no learnable parameters defined, all passed parameters returned."
        )
        return list([param for param in parameters if param.requires_grad])


def scale_learning_rate(
    initial_lr: int,
    scaler: Optional[str] = None,
    batch_size: Optional[int] = None,
    multiply_lr: float = 1.0,
) -> int:
    """Scale the initial learning rate.

    Args:
        initial_lr: Initial learning rate.
        scaler: Scaler rule.
        batch_size: Batch size to scale the learning rate.
        multiply_lr: Multiply the learning rate by factor.

    Returns:
        Scaled initial learning rate.
    """

    if scaler is None or scaler == "none":
        return initial_lr * multiply_lr
    elif scaler == "linear":
        lr = initial_lr * batch_size / 256
    elif scaler == "sqrt":
        lr = initial_lr * np.sqrt(batch_size)
    lr *= multiply_lr
    return lr
