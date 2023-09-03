from typing import Any, List

import lightning.pytorch as pl
from lightning.pytorch.strategies import (DDPStrategy, DeepSpeedStrategy,
                                          SingleDeviceStrategy,
                                          SingleTPUStrategy, Strategy)
from lightning.pytorch.trainer import Trainer

process_independent_strategies: List[Strategy] = [
    DDPStrategy,
    DeepSpeedStrategy,
]
fully_dependent_strategies: List[Strategy] = [SingleDeviceStrategy]

tpu_strategies: List[Strategy] = [SingleTPUStrategy]

supported_strategies: List[Strategy] = (
    process_independent_strategies + fully_dependent_strategies
)


def get_global_batch_size_in_trainer(
    local_batch_size: int,
    trainer: Trainer,
) -> int:
    """Get global batch size used by a trainer based on the local batch size.

    Args:
        local_batch_size: The local batch size used by the trainer.
        trainer: The trainer used.

    Raises:
        AttributeError: The strategy is not supported.

    Returns:
        The global batch size.
    """

    strategy = get_trainer_strategy(trainer)
    devices = trainer.num_devices
    num_nodes = trainer.num_nodes
    if not any(
        [
            isinstance(strategy, supported_strategy)
            for supported_strategy in supported_strategies
        ]
    ):
        raise AttributeError(f"Strategy {strategy} not supported.")
    else:
        if any([isinstance(strategy, tpu_strategy) for tpu_strategy in tpu_strategies]):
            return local_batch_size * devices
        elif any(
            [
                isinstance(strategy, process_independent_strategy)
                for process_independent_strategy in process_independent_strategies
            ]
        ):
            return local_batch_size * devices * num_nodes
        elif any(
            [
                isinstance(strategy, fully_dependent_strategy)
                for fully_dependent_strategy in fully_dependent_strategies
            ]
        ):
            return local_batch_size
        else:
            raise AttributeError(f"Strategy {strategy} not supported.")


def get_local_batch_size_in_trainer(
    global_batch_size: int,
    trainer: Trainer,
) -> int:
    """Get local batch size used by a trainer based on the global batch size.

    Args:
        global_batch_size: The global batch size used by the trainer.
        strategy: The trainer used.

    Raises:
        AttributeError: The strategy is not supported.

    Returns:
        The local batch size.
    """
    strategy = get_trainer_strategy(trainer)
    devices = trainer.num_devices
    num_nodes = trainer.num_nodes
    if not any(
        [
            isinstance(strategy, supported_strategy)
            for supported_strategy in supported_strategies
        ]
    ):
        raise AttributeError(f"Strategy {strategy} not supported.")
    else:
        if any([isinstance(strategy, tpu_strategy) for tpu_strategy in tpu_strategies]):
            return global_batch_size // devices
        elif any(
            [
                isinstance(strategy, process_independent_strategy)
                for process_independent_strategy in process_independent_strategies
            ]
        ):
            return global_batch_size // devices // num_nodes
        elif any(
            [
                isinstance(strategy, fully_dependent_strategy)
                for fully_dependent_strategy in fully_dependent_strategies
            ]
        ):
            return global_batch_size
        else:
            raise AttributeError(f"Strategy {strategy} not supported.")


def get_num_devices_in_trainer(
    trainer: Trainer,
) -> int:
    """Get the number of devices used by the trainer.

    Args:
        trainer: The trainer.

    Raises:
        AttributeError: The strategy used by trainer is not supported

    Returns:
        The number of devices used by trainer.
    """
    strategy = get_trainer_strategy(trainer)
    if not any(
        [
            isinstance(strategy, supported_strategy)
            for supported_strategy in supported_strategies
        ]
    ):
        raise AttributeError(f"Strategy {strategy} not supported.")
    else:
        if any([isinstance(strategy, tpu_strategy) for tpu_strategy in tpu_strategies]):
            return trainer.num_devices
        elif any(
            [
                isinstance(strategy, process_independent_strategy)
                for process_independent_strategy in process_independent_strategies
            ]
        ):
            return trainer.num_devices * trainer.num_nodes
        elif any(
            [
                isinstance(strategy, fully_dependent_strategy)
                for fully_dependent_strategy in fully_dependent_strategies
            ]
        ):
            return 1
        else:
            raise AttributeError(f"Strategy {strategy} not supported.")


def get_trainer_strategy(trainer: Trainer) -> Any:
    """Retrieve the strategy from a trainer.

    Args:
        trainer: The trainer.

    Returns:
        The strategy.
    """
    if pl.__version__ < "1.6.0":
        return trainer.training_type_plugin
    else:
        return trainer.strategy


def is_strategy_ddp(strategy: Any) -> bool:
    """Test if strategy is ddp.

    Args:
        strategy: The strategy.

    Returns:
        ``True`` if strategy is ddp.
    """
    return any(
        [
            isinstance(strategy, process_strategy)
            for process_strategy in process_independent_strategies
        ]
    )


def is_strategy_tpu(strategy: Any) -> bool:
    """Test if strategy is tpu.

    Args:
        strategy: The strategy.

    Returns:
        ``True`` if strategy is tpu.
    """

    return any([isinstance(strategy, tpu_strategy) for tpu_strategy in tpu_strategies])
