import os
import warnings
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.core.datamodule import LightningDataModule
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import DictConfig, OmegaConf

from eztorch.utils.checkpoints import (get_ckpt_by_callback_mode,
                                       get_last_ckpt_in_path_or_dir)
from eztorch.utils.utils import compile_model


@hydra.main(
    config_path="../eztorch/configs/run/evaluation/linear_classifier/sce/resnet50",
    config_name="resnet50_imagenet_mocov3",
)
def main(config: DictConfig) -> None:
    rundir = Path(to_absolute_path(config.dir.run))
    rundir.mkdir(parents=True, exist_ok=True)
    os.chdir(rundir)
    rank_zero_info(f"Run directory: {rundir}")

    hydradir = rundir / "config"
    hydradir.mkdir(parents=True, exist_ok=True)
    config_file = hydradir / "linear_classifier_evaluation.yaml"

    resolved_config = OmegaConf.to_yaml(config, resolve=True)

    # Save resolved config
    with config_file.open(mode="w") as f:
        f.write(resolved_config)

    # Fix seed, if seed everything: fix seed for python, numpy and pytorch
    if config.get("seed"):
        hydra.utils.instantiate(config.seed)
    else:
        warnings.warn("No seed fixed, the results are not reproducible.")

    # Create callbacks
    callbacks = []
    if config.get("callbacks"):
        for _, callback_cfg in config.callbacks.items():
            callback: Callback = hydra.utils.instantiate(callback_cfg)
            callbacks.append(callback)

    # Create trainer
    trainer: Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks)

    # Create datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Create model
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Search for last checkpoint if it exists
    model_ckpt_dirpath = (
        config.callbacks.model_checkpoint.dirpath
        if config.callbacks.get("model_checkpoint")
        else None
    )
    ckpt_path = get_last_ckpt_in_path_or_dir(config.ckpt_path, model_ckpt_dirpath)
    if ckpt_path is not None:
        warnings.warn(
            f"A checkpoint has been found and loaded from this file: {ckpt_path}",
            category=RuntimeWarning,
        )

    rank_zero_info(resolved_config)
    rank_zero_info(model)

    model = compile_model(model, **config.get("compile", {}))

    # Fit the trainer
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    if config.get("test"):
        if config.test.get("ckpt_by_callback_mode"):
            ckpt_paths = get_ckpt_by_callback_mode(
                config.test.ckpt_path, config.test.ckpt_by_callback_mode
            )
        else:
            ckpt_paths = [config.test.ckpt_path]

        for ckpt_path in ckpt_paths:
            trainer.test(model, ckpt_path=ckpt_path, datamodule=datamodule)


if __name__ == "__main__":
    main()
