from typing import List, Optional, Tuple

import hydra
from lightning.pytorch.utilities import rank_zero_info, rank_zero_warn
from omegaconf import DictConfig
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from eztorch.models.utils import group_params_layer_id
from eztorch.optimizers.utils import (_NORM_LAYERS, _OPTIMIZERS,
                                      filter_learnable_params,
                                      retrieve_model_params,
                                      scale_learning_rate)


def optimizer_factory(
    name: str,
    initial_lr: float,
    model: Module,
    batch_size: Optional[int] = None,
    num_steps_per_epoch: Optional[int] = None,
    layer_decay_lr: float | None = None,
    keys_without_decay: List[str] = [],
    exclude_wd_norm: bool = False,
    exclude_wd_bias: bool = False,
    scaler: Optional[str] = None,
    params: DictConfig = {},
    divide_wd_by_lr: bool = False,
    scheduler: Optional[DictConfig] = None,
    multiply_lr: float = 1.0,
    multiply_parameters: List[Parameter] = [],
) -> Tuple[Optimizer, Optional[_LRScheduler]]:
    """Optimizer factory to build optimizers and optionally an attached scheduler.

    Args:
        name: Name of the scheduler to retrieve the optimizer constructor from ``_OPTIMIZERS`` dict.
        initial_lr: Initial learning rate.
        model: Model to optimize.
        batch_size: Batch size for the input of the model.
        num_steps_per_epoch: Number of steps per epoch. Useful for some schedulers.
        keys_without_decay: Keys to filter parameters for weight decay.
        exclude_wd_norm: If ``True``, exclude normalization layers to be regularized by weight decay.
        exclude_wd_bias: If ``True``, exclude bias layers to be regularized by weight decay.
        scaler: Scaler rule for the initial learning rate.
        params: Parameters for the optimizer constructor.
        divide_wd_by_lr: If ``True``, divide the weight decay by the value of the learning rate.
        scheduler: Scheduler config.
        multiply_lr: Multiply the learning rate by factor. Applied for scheduler aswell.

    Returns:
        The optimizer with its optional scheduler.
    """

    optimizer_class = _OPTIMIZERS[name]

    lr = scale_learning_rate(initial_lr, scaler, batch_size, multiply_lr)

    if "weight_decay" in params and divide_wd_by_lr:
        params["weight_decay"] /= lr
        rank_zero_info(f"weight_decay has been scaled to {params['weight_decay']}")

    modules_without_decay = []

    if exclude_wd_norm:
        modules_without_decay.extend(_NORM_LAYERS)
    if exclude_wd_bias:
        keys_without_decay.append("bias")

    # Retrieve all the parameters in the model excluding the specified modules and keys.
    no_wd_parameters, wd_parameters = retrieve_model_params(
        model, modules_without_decay, keys_without_decay
    )

    # Filter learnable params as a property of the model if it is defined.
    wd_parameters = filter_learnable_params(wd_parameters, model)
    no_wd_parameters = filter_learnable_params(no_wd_parameters, model)

    named_wd_parameters = [
        name
        for name, param in model.named_parameters()
        if any([param is wd_param for wd_param in wd_parameters])
    ]
    named_no_wd_parameters = [
        name
        for name, param in model.named_parameters()
        if any([param is no_wd_param for no_wd_param in no_wd_parameters])
    ]

    if layer_decay_lr is not None:
        if not hasattr(model, "num_layers") and not hasattr(
            model, "group_params_layer_id"
        ):
            raise NotImplementedError(
                "Model should have `num_layers` and `params_layer_id` defined."
            )
        else:
            num_layers: int = model.num_layers
            params_layer_id = group_params_layer_id(model)
            layer_lr_decay_values = list(
                layer_decay_lr ** (num_layers - 1 - i) for i in range(num_layers)
            )

        group_wd_parameters = [
            (
                i,
                layer_lr_decay_values[i],
                [param for name, param in parameters if name in named_wd_parameters],
                [name for name, param in parameters if name in named_wd_parameters],
            )
            for i, parameters in params_layer_id
        ]
        group_no_wd_parameters = [
            (
                i,
                layer_lr_decay_values[i],
                [param for name, param in parameters if name in named_no_wd_parameters],
                [name for name, param in parameters if name in named_no_wd_parameters],
            )
            for i, parameters in params_layer_id
        ]

        rank_zero_info(
            f"{model._get_name()} optimizer's:" "\nLayers with weight decay:"
        )

        for i, layer_lr_decay_value, parameters, name_parameters in group_wd_parameters:
            rank_zero_info(
                f"Layer {i}: num parameters={len(parameters)}, decay={layer_lr_decay_value}, name parameters={name_parameters}"
            )

        rank_zero_info("Layers without weight decay:")

        for (
            i,
            layer_lr_decay_value,
            parameters,
            name_parameters,
        ) in group_no_wd_parameters:
            rank_zero_info(
                f"Layer {i}: num parameters={len(parameters)}, decay={layer_lr_decay_value}, name parameters={name_parameters}"
            )

        group_parameters = [
            {
                "params": parameters,
                "layer_lr_decay": layer_lr_decay,
                "layer_id": layer_id,
            }
            for layer_id, layer_lr_decay, parameters, _ in group_wd_parameters
        ]
        group_parameters += [
            {
                "params": parameters,
                "weight_decay": 0.0,
                "layer_lr_decay": layer_lr_decay,
                "layer_id": layer_id,
            }
            for layer_id, layer_lr_decay, parameters, _ in group_no_wd_parameters
            if parameters != []
        ]

    else:
        group_parameters = [
            {
                "params": [
                    param for param in wd_parameters if param not in multiply_parameters
                ]
            }
        ]
        if len(multiply_parameters) > 0:
            wd_multiply_parameters = [
                param for param in multiply_parameters if param in wd_parameters
            ]
            if len(wd_multiply_parameters) > 0:
                group_parameters.append(
                    {"params": wd_parameters, "lr": lr * multiply_lr}
                )
        if no_wd_parameters != []:
            group_parameters.append(
                {
                    "params": [
                        param
                        for param in no_wd_parameters
                        if param not in multiply_parameters
                    ],
                    "weight_decay": 0.0,
                }
            )
            if len(multiply_parameters) > 0:
                no_wd_multiply_parameters = [
                    param for param in multiply_parameters if param in no_wd_parameters
                ]
                if len(no_wd_multiply_parameters) > 0:
                    group_parameters.append(
                        {"params": wd_parameters, "lr": lr * multiply_lr}
                    )
                    group_parameters.append(
                        {
                            "params": [
                                param
                                for param in multiply_parameters
                                if param in no_wd_parameters
                            ],
                            "lr": lr * multiply_lr,
                            "weight_decay": 0.0,
                        }
                    )

        rank_zero_info(
            f"{model._get_name()} optimizer's:"
            "\n"
            f"With weight decay: num parameters={len(wd_parameters)}, name parameters: {named_wd_parameters}"
            "\n"
            f"Without weight decay: num parameters={len(no_wd_parameters)}, name parameters:{named_no_wd_parameters}"
        )

    optimizer = optimizer_class(group_parameters, lr=lr, **params)

    if scheduler is not None:
        scheduler = hydra.utils.instantiate(
            scheduler,
            num_steps_per_epoch=num_steps_per_epoch,
            optimizer=optimizer,
            scaler=scaler,
            batch_size=batch_size,
            multiply_lr=multiply_lr,
        )

    return optimizer, scheduler
