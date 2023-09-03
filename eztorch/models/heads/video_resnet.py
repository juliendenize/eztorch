# ------------------------------------------------------------------------
# Modified from pytorchvideo (https://github.com/facebookresearch/pytorchvideo)
# Licensed under the Apache License, Version 2.0
# -----------------------------


from typing import Callable, List, Optional, Tuple, Union

import torch
from pytorchvideo.models.weight_init import init_net_weights
from torch import Tensor
from torch.nn import (AdaptiveAvgPool3d, AvgPool3d, Dropout, Identity, Linear,
                      Module, Parameter, Softmax)

from eztorch.models.utils import _ACTIVATION_LAYERS, _POOL_LAYERS


class VideoResNetHead(Module):
    """
    ResNet basic head. This layer performs an optional pooling operation followed by an
    optional dropout, a fully-connected projection, an optional activation layer and a
    global spatiotemporal averaging.

    ::

                                        Pool3d
                                           ↓
                                        Dropout
                                           ↓
                                       Projection
                                           ↓
                                       Activation
                                           ↓
                                       Averaging

    The builder can be found in :func:`create_video_resnet_head`.

    Args:
            pool: Pooling module.
            dropout: dropout module.
            proj: project module.
            activation: activation module.
            output_pool: pooling module for output.
            init_std: init std for weights from pytorchvideo.
    """

    def __init__(
        self,
        pool: Module = None,
        dropout: Module = None,
        proj: Module = None,
        activation: Module = None,
        output_pool: Module = None,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()

        self.pool = pool
        self.dropout = dropout
        self.proj = proj
        self.activation = activation
        self.output_pool = output_pool

        self.do_proj = proj is not None and type(proj) is not Identity

        init_net_weights(self, init_std, "resnet")

    def forward(self, x: Tensor) -> Tensor:
        # Performs pooling.
        if self.pool is not None:
            x = self.pool(x)
        # Performs dropout.
        if self.dropout is not None:
            x = self.dropout(x)
        # Performs projection.
        if self.do_proj:
            x = x.permute((0, 2, 3, 4, 1))
            x = self.proj(x)
            x = x.permute((0, 4, 1, 2, 3))
        # Performs activation.
        if self.activation is not None:
            x = self.activation(x)
        if self.output_pool is not None:
            # Performs global averaging.
            x = self.output_pool(x)
            x = x.view(x.shape[0], -1)
        return x

    @property
    def num_layers(self) -> int:
        """Number of layers of the model."""
        return 1

    def get_param_layer_id(self, name: str) -> int:
        """Get the layer id of the named parameter.

        Args:
            name: The name of the parameter.
        """
        return 0

    @property
    def learnable_params(self) -> List[Parameter]:
        """List of learnable parameters."""
        params = list(self.parameters())
        return params


class VideoResNetTemporalHead(Module):
    """ResNet basic head for keeping temporal dimension. This layer performs an initial temporal pooling and
    reshape the output.

    Args:
        init_std: init std for weights from pytorchvideo.
    """

    def __init__(
        self,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()

        init_net_weights(self, init_std, "resnet")

    def forward(self, x: Tensor) -> Tensor:
        # Performs pooling.
        b, d, t, h, w = x.shape
        x = torch.nn.functional.adaptive_avg_pool3d(x, [t, 1, 1])
        x = self.view(b, t, d)
        return x


def create_video_resnet_head(
    *,
    # Projection configs.
    in_features: int,
    num_classes: int = 400,
    # Pooling configs.
    pool: Union[str, Callable] = AvgPool3d,
    output_size: Tuple[int] = (1, 1, 1),
    pool_kernel_size: Tuple[int] = (1, 7, 7),
    pool_stride: Tuple[int] = (1, 1, 1),
    pool_padding: Tuple[int] = (0, 0, 0),
    # Dropout configs.
    dropout_rate: float = 0.5,
    # Activation configs.
    activation: Optional[Union[str, Callable]] = None,
    # Output configs.
    output_with_global_average: bool = True,
) -> Module:
    """
    Creates ResNet basic head. This layer performs an optional pooling operation
    followed by an optional dropout, a fully-connected projection, an activation layer
    and a global spatiotemporal averaging.

    ::


                                        Pooling
                                           ↓
                                        Dropout
                                           ↓
                                       Projection
                                           ↓
                                       Activation
                                           ↓
                                       Averaging

    Activation examples include: ``ReLU``, ``Softmax``, ``Sigmoid``, and ``None``.
    Pool3d examples include: ``AvgPool3d``, ``MaxPool3d``, ``AdaptiveAvgPool3d``, and ``None``.

    Args:

        in_features: Input channel size of the resnet head.
        num_classes: Output channel size of the resnet head.
        pool: A callable that constructs resnet head pooling layer,
            examples include: ``AvgPool3d``, ``MaxPool3d``, ``AdaptiveAvgPool3d``, and
            ``None`` (not applying pooling).
        pool_kernel_size: Pooling kernel size(s) when not using adaptive
            pooling.
        pool_stride: Pooling stride size(s) when not using adaptive pooling.
        pool_padding: Pooling padding size(s) when not using adaptive
            pooling.
        output_size: Spatial temporal output size when using adaptive
            pooling.

        activation: A callable that constructs resnet head activation
            layer, examples include: ``ReLU``, ``Softmax``, ``Sigmoid``, and ``None`` (not
            applying activation).

        dropout_rate: Dropout rate.

        output_with_global_average: If ``True``, perform global averaging on temporal
            and spatial dimensions and reshape output to :math:`batch\\_size \times out\\_features`.
    """

    if type(activation) is str:
        activation = _ACTIVATION_LAYERS[activation]

    if type(pool) is str:
        pool = _POOL_LAYERS[pool]

    if activation is None:
        activation_model = None
    elif activation == Softmax:
        activation_model = activation(dim=1)
    else:
        activation_model = activation()

    if pool is None:
        pool_model = None
    elif pool == AdaptiveAvgPool3d:
        pool_model = pool(output_size)
    else:
        pool_model = pool(
            kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding
        )

    if output_with_global_average:
        output_pool = AdaptiveAvgPool3d(1)
    else:
        output_pool = None

    if num_classes > 0:
        proj = Linear(in_features, num_classes)
    else:
        proj = None

    return VideoResNetHead(
        proj=proj,
        activation=activation_model,
        pool=pool_model,
        dropout=Dropout(dropout_rate) if dropout_rate > 0 else None,
        output_pool=output_pool,
    )


def create_video_resnet_temporal_head() -> Module:
    """Creates ResNet basic head for keeping temporal dimension.

    This layer performs an initial temporal pooling and reshape the output.
    """
    return VideoResNetTemporalHead()
