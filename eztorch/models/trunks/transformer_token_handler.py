from typing import Tuple

import hydra
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Module

from eztorch.utils.utils import all_false, is_only_one_condition_true


class TransformerTokenHandlerModel(Module):
    """A general purpose model that handles the transformer token outputs.

    Args:
        transformer: The transformer to handle.
        return_cls_token: Whether to return the class token.
        return_cls_and_other_tokens: Whether to return tuple of (cls_token, other_tokens).
        flatten_tokens: Whether to flatten the tokens.
        average_tokens: Whether to average the tokens.

    Only one of the four conditions should be True.
    """

    def __init__(
        self,
        transformer: Module,
        return_cls_token: bool = True,
        return_cls_and_other_tokens: bool = False,
        flatten_tokens: bool = False,
        average_tokens: bool = False,
    ):
        super().__init__()
        self.transformer = transformer
        self.return_cls_token = return_cls_token
        self.return_cls_and_other_tokens = return_cls_and_other_tokens
        self.flatten_tokens = flatten_tokens
        self.average_tokens = average_tokens

        assert is_only_one_condition_true(
            return_cls_token,
            return_cls_and_other_tokens,
            flatten_tokens,
            average_tokens,
        ) or all_false(
            return_cls_token,
            return_cls_and_other_tokens,
            flatten_tokens,
            average_tokens,
        ), "Only one of return_cls_token, return_cls_and_other_tokens, flatten_tokens, average_tokens should be True or all to be False."

    def forward(self, x: Tensor, **kwargs) -> Tensor | Tuple[Tensor, Tensor]:
        x = self.transformer(x, **kwargs)
        if self.return_cls_token:
            return x[:, 0]
        elif self.return_cls_and_other_tokens:
            return x

        x = x[:, 1:]
        batch_size, num_tokens, *dims = x.shape

        if self.flatten_tokens:
            x = x.reshape(batch_size * num_tokens, *dims)
        elif self.average_tokens:
            x = x.mean(1)

        return x

    @property
    def num_layers(self) -> int:
        """Number of layers of the model."""
        return self.transformer.num_layers

    def get_param_layer_id(self, name: str) -> int:
        """Get the layer id of the named parameter.

        Args:
            name: The name of the parameter.
        """
        return self.transformer.get_param_layer_id(name[len("transformer.") :])


class ViTokenHandlerModel(TransformerTokenHandlerModel):
    """A general purpose model that handles the transformer token outputs.

    Args:
        transformer: The transformer to handle.
        return_cls_token: Whether to return the class token.
        return_cls_and_other_tokens: Whether to return tuple of (cls_token, other_tokens).
        flatten_tokens: Whether to flatten the tokens.
        average_tokens: Whether to average the tokens.
        separate_spatial_temporal: Whether to separate the spatial and temporal outputs.
        detach_temporal: Whether to detach the temporal encoder from the spatial encoder.
    Only one of the four conditions should be True.
    """

    def __init__(
        self,
        transformer: Module,
        return_cls_token: bool = True,
        return_cls_and_other_tokens: bool = False,
        flatten_tokens: bool = False,
        average_tokens: bool = False,
        separate_spatial_temporal: bool = False,
        detach_temporal: bool = False,
    ):
        super().__init__(
            transformer,
            return_cls_token=return_cls_token,
            return_cls_and_other_tokens=return_cls_and_other_tokens,
            flatten_tokens=flatten_tokens,
            average_tokens=average_tokens,
        )

        self.separate_spatial_temporal = separate_spatial_temporal
        self.detach_temporal = detach_temporal

    def forward(self, x: Tensor, **kwargs) -> Tensor | Tuple[Tensor]:
        if not self.separate_spatial_temporal:
            if self.transformer.temporal_class_token:
                return super().forward(x, **kwargs)
            else:
                x = self.transformer(
                    x, return_spatial=False, detach_spatial=False, **kwargs
                )
                batch_size, num_tokens, *dims = x.shape

                if self.flatten_tokens:
                    x = x.reshape(batch_size * num_tokens, *dims)
                elif self.average_tokens:
                    x = x.mean(1)

                return x
        else:
            x_spatial, x_temporal = self.transformer(
                x, return_spatial=True, detach_spatial=self.detach_temporal, **kwargs
            )
            if self.transformer.temporal_class_token:
                if self.return_cls_token:
                    return x_spatial, x_temporal[:, 0]
                elif self.return_cls_and_other_tokens:
                    return x_spatial, x_temporal

                x_temporal = x_temporal[:, 1:]
                batch_size, num_tokens, *dims = x_temporal.shape

                if self.flatten_tokens:
                    x_temporal = x_temporal.reshape(batch_size * num_tokens, *dims)
                    x_temporal = x_temporal.reshape(batch_size * num_tokens, *dims)
                elif self.average_tokens:
                    x_temporal = x_temporal.mean(1)
            else:
                if self.flatten_tokens:
                    x_temporal = x_temporal.reshape(batch_size * num_tokens, *dims)
                    x_temporal = x_temporal.reshape(batch_size * num_tokens, *dims)
                elif self.average_tokens:
                    x_temporal = x_temporal.mean(1)
            return x_spatial, x_temporal


def create_transformer_token_handler_model(
    transformer: DictConfig,
    return_cls_token: bool = True,
    return_cls_and_other_tokens: bool = False,
    flatten_tokens: bool = False,
    average_tokens: bool = False,
) -> TransformerTokenHandlerModel:
    """Build a TransformerTokenHandlerModel.

    Args:
        return_cls_token: Whether to return the class token.
        return_cls_and_other_tokens: Whether to return tuple of (cls_token, other_tokens).
        flatten_tokens: Whether to flatten the tokens.
        average_tokens: Whether to average the tokens.
    """

    transformer = hydra.utils.instantiate(transformer)

    model = TransformerTokenHandlerModel(
        transformer,
        return_cls_token=return_cls_token,
        return_cls_and_other_tokens=return_cls_and_other_tokens,
        flatten_tokens=flatten_tokens,
        average_tokens=average_tokens,
    )

    return model


def create_vitransformer_token_handler_model(
    transformer: DictConfig,
    return_cls_token: bool = True,
    return_cls_and_other_tokens: bool = False,
    flatten_tokens: bool = False,
    average_tokens: bool = False,
    separate_spatial_temporal: bool = False,
    detach_temporal: bool = False,
) -> ViTokenHandlerModel:
    """Build a TransformerTokenHandlerModel.

    Args:
        return_cls_token: Whether to return the class token.
        return_cls_and_other_tokens: Whether to return tuple of (cls_token, other_tokens).
        flatten_tokens: Whether to flatten the tokens.
        average_tokens: Whether to average the tokens.
        separate_spatial_temporal: Whether to separate the spatial and temporal outputs.
        detach_temporal: Whether to detach the temporal encoder from the spatial encoder.
    """

    transformer = hydra.utils.instantiate(transformer)

    model = ViTokenHandlerModel(
        transformer,
        return_cls_token=return_cls_token,
        return_cls_and_other_tokens=return_cls_and_other_tokens,
        flatten_tokens=flatten_tokens,
        average_tokens=average_tokens,
        separate_spatial_temporal=separate_spatial_temporal,
        detach_temporal=detach_temporal,
    )

    return model
