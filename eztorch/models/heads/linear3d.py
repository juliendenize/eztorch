"""
References:
- https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/models/head.py
"""

from typing import Callable, List, Tuple, Union

from torch import Tensor, nn
from torch.nn import (AdaptiveAvgPool3d, AvgPool3d, Dropout, Linear, Module,
                      Parameter)
from torch.nn import functional as F

from eztorch.models.utils import _BN_LAYERS, _POOL_LAYERS


class Linear3DHead(Module):
    """
    Linear 3D head. This layer performs an optional pooling operation followed by an
    optional dropout, a fully-connected projection, an optional activation layer and a
    global spatiotemporal averaging.

    ::

                                        Pool3d
                                           ↓
                                     Normalization
                                           ↓
                                        Dropout
                                           ↓
                                       Projection

    Args:
            in_features: Input channel size of the resnet head.
            pool: Pooling module.
            dropout: Dropout module.
            bn: Batch normalization module.
            proj: Project module.
            norm: If ``True``, normalize features along first dimension.
            init_std: Init std for weights from pytorchvideo.
            view: If ``True``, apply reshape view to :math:`(-1, num features)`.
    """

    def __init__(
        self,
        input_dim: int,
        pool: Module = None,
        dropout: Module = None,
        bn: Module = None,
        proj: Module = None,
        norm: bool = False,
        init_std: float = 0.01,
        view: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.pool = pool
        self.dropout = dropout
        self.bn = bn
        self.proj = proj
        self.norm = norm
        self.view = view
        self.init_std = init_std

        self._init_weights()

    def forward(self, x: Tensor) -> Tensor:
        # Performs pooling.
        if self.pool is not None:
            x = self.pool(x)

        if self.view:
            x = x.view(-1, self.input_dim)

        if self.norm:
            x = F.normalize(x)
        # Performs bn.
        if self.bn is not None:
            x = self.bn(x)
        # Performs dropout.
        if self.dropout is not None:
            x = self.dropout(x)
        # Performs proj.
        if self.proj is not None:
            # Performs global averaging.
            x = self.proj(x)

        return x

    def _init_weights(self):
        for name, param in self.proj.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.normal_(param, mean=0.0, std=self.init_std)
        if self.bn is not None:
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()

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


def create_linear3d_head(
    *,
    # Projection configs.
    in_features: int,
    num_classes: int = 400,
    bn: Union[str, Callable] = None,
    norm: bool = False,
    # Pooling configs.
    pool: Union[str, Callable] = AvgPool3d,
    pool_kernel_size: Tuple[int] = (1, 7, 7),
    pool_stride: Tuple[int] = (1, 1, 1),
    pool_padding: Tuple[int] = (0, 0, 0),
    output_size: Tuple[int] = (1, 1, 1),
    # Dropout configs.
    dropout_rate: float = 0.5,
    view: bool = True
) -> Module:
    r"""
    Creates ResNet basic head. This layer performs an optional pooling operation
    followed by an optional dropout, a fully-connected projection, an activation layer
    and a global spatiotemporal averaging.

    ::

                                        Pool3d
                                           ↓
                                     Normalization
                                           ↓
                                        Dropout
                                           ↓
                                       Projection

    Pool3d examples include: AvgPool3d, MaxPool3d, AdaptiveAvgPool3d, and None.

    Args:

        in_features: Input channel size of the resnet head.
        num_classes: Output channel size of the resnet head.
        bn: A callable that constructs a batch norm layer.
        norm: If ``True``, normalize features along first dimension.
        pool: A callable that constructs resnet head pooling layer,
            examples include: ``nn.AvgPool3d``, ``nn.MaxPool3d``, ``nn.AdaptiveAvgPool3d``, and
            ``None`` (not applying pooling).
        pool_kernel_size: Pooling kernel size(s) when not using adaptive
            pooling.
        pool_stride: Pooling stride size(s) when not using adaptive pooling.
        pool_padding: Pooling padding size(s) when not using adaptive
            pooling.
        output_size: Spatial temporal output size when using adaptive
            pooling.
        dropout_rate: Dropout rate.
        view: Whether to apply reshape view to :math:`(-1, num\ features)`.

    """

    if type(pool) is str:
        pool = _POOL_LAYERS[pool]

    if type(bn) is str:
        bn = _BN_LAYERS[bn]

    if bn is not None:
        bn = bn(in_features)

    if pool is None:
        pool_model = None
    elif pool == AdaptiveAvgPool3d:
        pool_model = pool(output_size)
    else:
        pool_model = pool(
            kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding
        )

    proj = Linear(in_features, num_classes)

    return Linear3DHead(
        in_features,
        bn=bn,
        norm=norm,
        proj=proj,
        pool=pool_model,
        dropout=Dropout(dropout_rate) if dropout_rate > 0 else None,
        view=view,
    )
