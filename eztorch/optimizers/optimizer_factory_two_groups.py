from typing import Optional, Tuple

import hydra
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import DictConfig
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from eztorch.optimizers.utils import (_NORM_LAYERS, _OPTIMIZERS,
                                      filter_learnable_params,
                                      retrieve_model_params,
                                      scale_learning_rate)


def optimizer_factory_two_groups(
    name: str,
    initial_lr1: float,
    initial_lr2: float,
    model1: Module,
    model2: Module,
    batch_size: Optional[int] = None,
    num_steps_per_epoch: Optional[int] = None,
    exclude_wd_norm: bool = False,
    exclude_wd_bias: bool = False,
    scaler: Optional[str] = None,
    params: DictConfig = {},
    scheduler: Optional[DictConfig] = None,
) -> Tuple[Optimizer, Optional[_LRScheduler]]:
    """Optimizer factory to build an optimizer for two groups of parameters and optionally an attached scheduler.

    Args:
        name: Name of the scheduler to retrieve the optimizer constructor from ``_OPTIMIZERS`` dict.
        initial_lr1: Initial learning rate for model 1.
        initial_lr2: Initial learning rate for model 2.
        model1: Model 1 to optimize.
        model2: Model 2 to optimize.
        batch_size: Batch size for the input of the model.
        num_steps_per_epoch: Number of steps per epoch. Useful for some schedulers.
        exclude_wd_norm: If  ``True``, exclude normalization layers to be regularized by weight decay.
        exclude_wd_bias: If ``True``, exclude bias layers to be regularized by weight decay.
        scaler: Scaler rule for the initial learning rate.
        params: Parameters for the optimizer constructor.
        scheduler: Scheduler config for model.

    Returns:
        The optimizer with its optional scheduler.
    """

    optimizer_class = _OPTIMIZERS[name]

    lr1 = scale_learning_rate(initial_lr1, scaler, batch_size)
    lr2 = scale_learning_rate(initial_lr2, scaler, batch_size)

    modules_without_decay = []
    keys_without_decay = []

    if exclude_wd_norm:
        modules_without_decay.extend(_NORM_LAYERS)
    if exclude_wd_bias:
        keys_without_decay.append("bias")

    # Retrieve all the parameters in the model excluding the specified modules and keys.
    no_wd_parameters1, wd_parameters1 = retrieve_model_params(
        model1, modules_without_decay, keys_without_decay
    )

    no_wd_parameters2, wd_parameters2 = retrieve_model_params(
        model2, modules_without_decay, keys_without_decay
    )

    # Filter learnable params as a property of the model if it is defined.
    wd_parameters1 = filter_learnable_params(wd_parameters1, model1)
    no_wd_parameters1 = filter_learnable_params(no_wd_parameters1, model1)

    wd_parameters2 = filter_learnable_params(wd_parameters2, model2)
    no_wd_parameters2 = filter_learnable_params(no_wd_parameters2, model2)

    named_wd_parameters1 = [
        name
        for name, param in model1.named_parameters()
        if any([param is wd_param for wd_param in wd_parameters1])
    ]
    named_no_wd_parameters1 = [
        name
        for name, param in model1.named_parameters()
        if any([param is no_wd_param for no_wd_param in no_wd_parameters1])
    ]

    named_wd_parameters2 = [
        name
        for name, param in model2.named_parameters()
        if any([param is wd_param for wd_param in wd_parameters2])
    ]
    named_no_wd_parameters2 = [
        name
        for name, param in model2.named_parameters()
        if any([param is no_wd_param for no_wd_param in no_wd_parameters2])
    ]

    list_optim = [
        {"params": wd_parameters1, "lr": lr1},
        {"params": wd_parameters2, "lr": lr2},
    ]
    if no_wd_parameters1 != []:
        list_optim.append({"params": no_wd_parameters1, "weight_decay": 0.0})
    if no_wd_parameters2 != []:
        list_optim.append({"params": no_wd_parameters2, "weight_decay": 0.0})
    optimizer = optimizer_class(list_optim, **params)

    rank_zero_info(
        f"{model1._get_name()} optimizer's:"
        "\n"
        f"With weight decay: num parameters={len(wd_parameters1)}, name parameters: {named_wd_parameters1}"
        "\n"
        f"Without weight decay: num parameters={len(no_wd_parameters1)}, name parameters:{named_no_wd_parameters1}"
    )

    rank_zero_info(
        f"{model2._get_name()} optimizer's:"
        "\n"
        f"With weight decay: num parameters={len(wd_parameters2)}, name parameters: {named_wd_parameters2}"
        "\n"
        f"Without weight decay: num parameters={len(no_wd_parameters2)}, name parameters:{named_no_wd_parameters2}"
    )

    if scheduler is not None:
        scheduler = hydra.utils.instantiate(
            scheduler,
            num_steps_per_epoch=num_steps_per_epoch,
            optimizer=optimizer,
            scaler=scaler,
            batch_size=batch_size,
        )

    return optimizer, scheduler
