from typing import Iterable, List, Optional, Union

from torch import Tensor
from torch.nn import (BatchNorm1d, Dropout, Linear, Module, Parameter, ReLU,
                      Sequential, SyncBatchNorm)

from eztorch.models.utils import _ACTIVATION_LAYERS, _BN_LAYERS


class MLPHead(Module):
    r"""Build a MLP head with optional dropout and normalization.

    Args:
        activation_inplace: Inplace operation for activation layers.
        activation_layer: Activation layer, if str lookup for the module in ``_ACTIVATION_LAYERS`` dictionary.
        affine: If ``True``, use affine in normalization layer.
        bias: If ``True``, use bias in linear layer. If ``norm_layer``, set to ``False``.
        dropout: Dropout probability, if :math:`0`, no dropout layer.
        dropout_inplace: If ``True``, use inplace operation in dropout.
        hidden_dims: dimension of the hidden layers :math:`(num\_layers - 1)`. If int, used for all hidden layers.
        input_dim: Input dimension for the MLP head.
        norm_layer: Normalization layer after the linear layer, if str lookup for the module in ``_BN_LAYERS`` dictionary.
        num_layers: Number of layers :math:`(number\ of\ hidden\ layers + 1)`.
        last_bias: If ``True``, use bias in output layer. If ``last_norm`` and ``norm_layer`` set to ``False``.
        last_norm: If ``True``, Apply normalization to the last layer if ``norm_layer``.
        last_affine: If ``True``, use affine in output normalization layer.
        output_dim: Output dimension for the MLP head.
        last_init_normal: If ``True``, make normal initialization for last layer.
        init_mean: Mean for the last initialization.
        init_std: STD for the last initialization.
        zero_bias: If ``True``, put zeros to bias for the last initialization.

    Raises:
        NotImplementedError: If ``norm_layer`` is not supported.
    """

    def __init__(
        self,
        activation_inplace: bool = True,
        activation_layer: Union[str, Module] = ReLU,
        affine: bool = True,
        bias: bool = True,
        dropout: Union[float, Iterable[float]] = 0.0,
        dropout_inplace: bool = False,
        hidden_dims: Union[int, Iterable[int]] = 2048,
        input_dim: int = 2048,
        norm_layer: Optional[Union[str, Module]] = None,
        num_layers: int = 2,
        last_bias: bool = True,
        last_norm: bool = False,
        last_affine: bool = False,
        output_dim: int = 128,
        last_init_normal: bool = False,
        init_mean: float = 0.0,
        init_std: float = 0.01,
        zero_bias: bool = True,
    ) -> None:
        super().__init__()

        if type(dropout) is float:
            dropout = [dropout] * num_layers

        if type(activation_layer) is str:
            activation_layer = _ACTIVATION_LAYERS[activation_layer]

        if type(hidden_dims) is int:
            hidden_dims = [hidden_dims] * (num_layers - 1)

        if norm_layer is not None:
            norm = True
            if type(norm_layer) is str:
                norm_layer = _BN_LAYERS[norm_layer]
            if norm_layer not in [BatchNorm1d, SyncBatchNorm]:
                raise NotImplementedError("{norm_layer} not supported in MLPHead")
        else:
            norm = False

        assert len(hidden_dims) == num_layers - 1
        assert len(dropout) == num_layers

        layers = []

        for i in range(num_layers):
            dim_in = input_dim if i == 0 else hidden_dims[i - 1]
            dim_out = output_dim if i == num_layers - 1 else hidden_dims[i]

            use_norm = (
                True
                if norm and (i < num_layers - 1 or last_norm and i == num_layers - 1)
                else False
            )
            use_affine = (
                True
                if affine and i < num_layers - 1 or last_affine and i == num_layers - 1
                else False
            )
            use_bias = (
                True
                if (bias and i < num_layers - 1 or last_bias and i == num_layers - 1)
                and not use_norm
                else False
            )
            use_activation = True if i < num_layers - 1 else False

            if dropout[i] > 0.0:
                layers.append(Dropout(p=dropout[i], inplace=dropout_inplace))
            layers.append(Linear(dim_in, dim_out, bias=use_bias and not use_norm))

            # init last normal layer
            if i == num_layers - 1 and last_init_normal:
                layers[-1].weight.data.normal_(mean=init_mean, std=init_std)
                if zero_bias and use_bias:
                    layers[-1].bias.data.zero_()

            if use_norm:
                layers.append(norm_layer(num_features=dim_out, affine=use_affine))
            if use_activation:
                if activation_inplace:
                    layers.append(activation_layer(inplace=True))
                else:
                    layers.append(activation_layer())

        self.layers = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

    @property
    def learnable_params(self) -> List[Parameter]:
        """List of learnable parameters."""
        params = list(self.parameters())
        return params

    @property
    def num_layers(self) -> int:
        """Number of layers of the model."""
        return len(self.layers)

    def get_param_layer_id(self, name: str) -> int:
        """Get the layer id of the named parameter.

        Args:
            name: The name of the parameter.
        """
        if name.startswith("layers."):
            layer_id = int(name.split(".")[1])
            return layer_id
        else:
            return self.num_layers - 1
