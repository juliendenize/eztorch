from typing import List, Optional, Union

from torch import Tensor
from torch.nn import (BatchNorm2d, BatchNorm3d, Dropout, Linear, Module,
                      Parameter, Sequential)

from eztorch.models.utils import _BN_LAYERS


class LinearHead(Module):
    """Build a Linear head with optional dropout and normalization.

    Args:
        affine: Use affine in normalization layer.
        bias: Use bias in linear layer. If ``norm_layer``, set to ``False``.
        dropout: Dropout probability, if :math:`0`, no dropout layer.
        dropout_inplace: Use inplace operation in dropout.
        input_dim: Input dimension for the linear head.
        norm_layer: Normalization layer after the linear layer, if ``str`` lookup for the module in ``_BN_LAYERS`` dictionary.
        output_dim: Output dimension for the linear head.
        init_normal: If ``True``, make normal initialization for linear layer.
        init_mean: Mean for the initialization.
        init_std: STD for the initialization.
        zero_bias: If ``True``, put zeros to bias for the initialization.

    Raises:
        NotImplementedError: If ``norm_layer`` is not supported.
    """

    def __init__(
        self,
        affine: bool = True,
        bias: bool = True,
        dropout: float = 0.0,
        dropout_inplace: bool = False,
        input_dim: int = 2048,
        norm_layer: Optional[Union[str, Module]] = None,
        output_dim: int = 1000,
        init_normal: bool = True,
        init_mean: float = 0.0,
        init_std: float = 0.01,
        zero_bias: bool = True,
    ) -> None:
        super().__init__()

        if norm_layer is not None:
            norm = True
            if type(norm_layer) is str:
                norm_layer = _BN_LAYERS[norm_layer]
            if norm_layer in [BatchNorm2d, BatchNorm3d]:
                raise NotImplementedError("{norm_layer} not supported in LinearHead")
        else:
            norm = False

        layers = []

        if dropout > 0.0:
            layers.append(Dropout(p=dropout, inplace=dropout_inplace))

        linear_layer = Linear(input_dim, output_dim, bias=bias and not norm)

        # init linear_layer
        if init_normal:
            linear_layer.weight.data.normal_(mean=init_mean, std=init_std)
            if zero_bias and (bias and not norm):
                linear_layer.bias.data.zero_()

        layers.append(linear_layer)

        if norm:
            layers.append(norm_layer(num_features=output_dim, affine=affine))

        self.layers = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

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
