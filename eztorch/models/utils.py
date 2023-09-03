from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import (GELU, AdaptiveAvgPool3d, AdaptiveMaxPool3d, AvgPool3d,
                      BatchNorm1d, BatchNorm2d, BatchNorm3d, MaxPool3d, Module,
                      Parameter, ReLU, Softmax, SyncBatchNorm)
from torch.utils.data import DataLoader

_ACTIVATION_LAYERS = {"relu": ReLU, "gelu": GELU, "softmax": Softmax}

_BN_LAYERS = {
    "bn_1D": BatchNorm1d,
    "bn_2D": BatchNorm2d,
    "bn_3D": BatchNorm3d,
    "sync_bn": SyncBatchNorm,
}

_POOL_LAYERS = {
    "adaptive_avg_pool_3d": AdaptiveAvgPool3d,
    "adaptive_max_pool_3d": AdaptiveMaxPool3d,
    "avg_pool_3d": AvgPool3d,
    "max_pool_3d": MaxPool3d,
}


def extract_features(model: Module, loader: DataLoader) -> Tuple[Tensor, Tensor]:
    """Extract features from a model.

    Args:
        model: The model to extract features from.
        loader: The dataloader to retrieve features from.

    Returns:
        The features and its associated labels.
    """
    x, y = [], []
    for x_i, y_i in iter(loader):
        x.append(model(x_i))
        y.append(y_i)
    x = torch.cat(x)
    y = torch.cat(y)
    return x, y


def group_params_layer_id(model: Module) -> List[Tuple[int, Tuple[str, Parameter]]]:
    """Retrive from model the groups of parameters in the different layers.

    Args:
        model: The model to retrieve the parameters from.

    Returns:
        The list of groups of parameters in the format:
            [
                (id_layer, (name_param, param)),
                ...
            ]
    """
    group_parameters = {}
    for name, param in model.named_parameters():
        layer_id = model.get_param_layer_id(name)
        if layer_id in group_parameters:
            group_parameters[layer_id].append((name, param))
        else:
            group_parameters[layer_id] = [(name, param)]

    return list(group_parameters.items())
