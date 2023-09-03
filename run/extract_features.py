import os
import warnings
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.core.datamodule import LightningDataModule
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.nn import Module


@hydra.main(
    config_path="../eztorch/configs/run/evaluation/feature_extractor/resnet3d50",
    config_name="resnet3d50_ucf101",
)
def main(config: DictConfig) -> None:
    rundir = Path(to_absolute_path(config.dir.run))
    rundir.mkdir(parents=True, exist_ok=True)
    os.chdir(rundir)
    rank_zero_info(f"Run directory: {rundir}")

    hydradir = rundir / "config/"
    hydradir.mkdir(parents=True, exist_ok=True)
    config_file = hydradir / "extract_features.yaml"

    resolved_config = OmegaConf.to_yaml(config, resolve=True)

    # Save resolved config
    with config_file.open(mode="w") as f:
        f.write(resolved_config)

    # Fix seed, if seed everything: fix seed for python, numpy and pytorch
    if config.get("seed"):
        hydra.utils.instantiate(config.seed)
    else:
        warnings.warn("No seed fixed, the results are not reproducible.")

    # Create datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Create model for feature extractor
    if not config.model.get("_target_"):
        with open_dict(config):
            config.model._target_ = "eztorch.evaluation.FeatureExtractor"
            config.model._recursive_ = False

    model: Module = hydra.utils.instantiate(config.model)

    # Create callbacks
    callbacks = []
    if config.get("callbacks"):
        for _, callback_cfg in config.callbacks.items():
            callback: Callback = hydra.utils.instantiate(callback_cfg)
            callbacks.append(callback)

    # Create trainer
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, devices=1, strategy="auto"
    )

    rank_zero_info(resolved_config)
    rank_zero_info(model)

    if config.datamodule.get("train"):
        trainer.fit(model, datamodule=datamodule)
    elif config.datamodule.get("val"):
        trainer.validate(model, datamodule=datamodule)

    if config.datamodule.get("test"):
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
