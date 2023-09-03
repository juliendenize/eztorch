import logging
import math
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from hydra._internal.utils import _locate
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import rank_zero_info
from packaging import version

log = logging.getLogger(__name__)


def compile_model(
    model: LightningModule,
    do_compile: bool = False,
    fullgraph: bool = False,
    dynamic: bool = False,
    backend: Union[str, Callable] = "inductor",
    mode: Union[str, None] = None,
    options: Optional[Dict[str, Union[str, int, bool]]] = None,
    disable: bool = False,
):
    """If torch version is greater than `'2.0.0'` and users ask for it, compile the model.

    Args:
        model: Model to compile.
        do_compile: Whether to compile the model.
        fullgraph: Defined by `torch.compile`.
        dynamic: Defined by `torch.compile`.
        backend: Defined by `torch.compile`.
        mode: Defined by `torch.compile`.
        passes: Defined by `torch.compile`.

    Returns:
        The compiled model if available else the model.
    """
    if version.parse(torch.__version__) >= version.parse("2.0.0.dev") and do_compile:
        rank_zero_info(f"Compiling model {model.__class__.__name__}.")
        return torch.compile(
            model=model,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            options=options,
            disable=disable,
        )
    else:
        rank_zero_info(f"Not compiling model {model.__class__.__name__}.")
        return model


def get_default_seed(default_seed: int = 0) -> int:
    """Get the default seed if pytorch lightning did not initialize one.

    Args:
        default_seed: The default seed.

    Returns:
        Pytorch lightning seed or the default one.
    """
    return int(os.getenv("PL_GLOBAL_SEED", default_seed))


def get_global_rank() -> int:
    """Get global rank of the process."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:  # torchrun launch
        rank = int(os.environ["RANK"])
    elif int(os.environ.get("SLURM_NPROCS", 1)) > 1:  # srun launch
        rank = int(os.environ["SLURM_PROCID"])
    else:  # single gpu & process launch
        rank = 0
    return rank


def get_local_rank() -> int:
    """Get local rank of the process."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:  # torchrun launch
        local_rank = int(os.environ["LOCAL_RANK"])
    elif int(os.environ.get("SLURM_NPROCS", 1)) > 1:  # srun launch
        local_rank = int(os.environ["SLURM_LOCALID"])
    else:  # single gpu & process launch
        local_rank = 0
    return local_rank


def get_world_size() -> int:
    """Get world size or number of the processes."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:  # torchrun launch
        world_size = int(os.environ["WORLD_SIZE"])
    elif int(os.environ.get("SLURM_NPROCS", 1)) > 1:  # srun launch
        world_size = int(os.environ["SLURM_NPROCS"])
    else:  # single gpu & process launch
        world_size = 1
    return world_size


def get_local_world_size() -> int:
    """Get local world size or number of processes on the node."""
    if dist.is_available() and dist.is_initialized():
        return torch.cuda.device_count()
    else:
        return 1


def is_only_one_condition_true(*conditions: List[bool]) -> bool:
    """Test if only one of the conditions is True."""
    a = conditions[0]
    b = conditions[0]
    for condition in conditions[1:]:
        a = a ^ condition
        b = b & condition
    return a & ~b


def all_false(*conditions: List[bool]) -> bool:
    """Test that all conditions are False."""
    return all([~condition for condition in conditions])


def warmup_value(
    initial_value: float, final_value: float, step: int = 0, max_step: int = 0
) -> float:
    """Apply warmup to a value.

    Args:
        initial_value: Initial value.
        final_value: Final value.
        step: Current step.
        max_step: Max step for warming up.

    Returns:
        The value at current warmup step.
    """

    if step >= max_step:
        return final_value
    else:
        return initial_value + step * (final_value - initial_value) / (max_step)


def scheduler_value(
    scheduler: Optional[str],
    initial_value: float,
    final_value: float,
    step: int = 0,
    max_step: int = 0,
) -> float:
    """Apply scheduler to a value.

    Args:
        scheduler: The type of the scheduler.
        initial_value: The initial value.
        final_value: The final value.
        step: Current step.
        max_step: Maximum step for scheduler.

    Returns:
        The value at current step.
    """

    if scheduler is None:
        return initial_value
    elif scheduler == "linear":
        if final_value < initial_value:
            return initial_value + step * (final_value - initial_value) / (max_step)
        else:
            return final_value + step * (initial_value - final_value) / (max_step)
    elif scheduler == "cosine":
        if final_value > initial_value:
            return initial_value + 0.5 * (
                1.0 + math.cos(math.pi + math.pi * step / (max_step))
            ) * (final_value - initial_value)
        else:
            return initial_value - 0.5 * (
                1.0 + math.cos(math.pi + math.pi * step / (max_step))
            ) * (initial_value - final_value)
