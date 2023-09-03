import os
import warnings
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from lightning.pytorch import LightningModule
from lightning.pytorch.core.datamodule import LightningDataModule
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import DictConfig, OmegaConf

from eztorch.models.dummy import DummyModel


@hydra.main(
    config_path="../eztorch/configs/run/pretrain/moco", config_name="resnet18_cifar10"
)
def main(config: DictConfig) -> None:
    rundir = Path(to_absolute_path(config.dir.run))
    rundir.mkdir(parents=True, exist_ok=True)
    os.chdir(rundir)
    rank_zero_info(f"Run directory: {rundir}")

    hydradir = rundir / "config/"
    hydradir.mkdir(parents=True, exist_ok=True)
    config_file = hydradir / "dataloader.yaml"

    resolved_config = OmegaConf.to_yaml(config, resolve=True)

    # Save resolved config
    with config_file.open(mode="w") as f:
        f.write(resolved_config)

    # Fix seed, if seed everything: fix seed for python, numpy and pytorch
    if config.get("seed"):
        hydra.utils.instantiate(config.seed)
    else:
        warnings.warn("No seed fixed, the results are not reproducible.")

    # Create trainer
    trainer: Trainer = hydra.utils.instantiate(config.trainer, callbacks=[])

    # Create datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    if config.model.get("input_shape") is None:
        raise AssertionError("input_shape should be specified in model config.")

    # Create model
    if config.model.get("transform"):
        transform = config.model.train_transform
    else:
        transform = None
    model: LightningModule = DummyModel(config.model.input_shape, transform=transform)

    rank_zero_info(config.datamodule)
    rank_zero_info(model)

    # Fit the trainer
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
