import os
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    config_path="../eztorch/configs/run/evaluation/retrieval_from_bank",
    config_name="default",
)
def main(config: DictConfig) -> None:
    rundir = Path(to_absolute_path(config.dir.run))
    rundir.mkdir(parents=True, exist_ok=True)
    os.chdir(rundir)
    rank_zero_info(f"Run directory: {rundir}")

    hydradir = rundir / "config/"
    hydradir.mkdir(parents=True, exist_ok=True)
    config_file = hydradir / "retrieval_train_from_test.yaml"

    resolved_config = OmegaConf.to_yaml(config, resolve=True)

    # Save resolved config
    with config_file.open(mode="w") as f:
        f.write(resolved_config)

    rank_zero_info(resolved_config)

    ranks = config.ranks

    rank_zero_info("\nLoading query features and labels...")
    query_features = torch.load(config.query.features_path)
    query_labels = torch.load(config.query.labels_path)
    rank_zero_info(
        f"Loaded query features and labels.\nshape of features is: {query_features.shape}.\nshape of labels is: {query_labels.shape}."
    )

    rank_zero_info("\nLoading bank features and labels...")
    bank_features = torch.load(config.bank.features_path)
    bank_labels = torch.load(config.bank.labels_path)
    rank_zero_info(
        f"Loaded bank features and labels.\nshape of features is: {bank_features.shape}.\nshape of labels is: {bank_labels.shape}."
    )

    if torch.cuda.is_available():
        rank_zero_info("\nCuda available, tensors put on GPU...")

        query_features = query_features
        query_labels = query_labels

        bank_features = bank_features
        bank_labels = bank_labels

    # centering
    if config.query.center:
        rank_zero_info("\nQuery centering...")
        query_features = query_features - query_features.mean(dim=0, keepdim=True)
        rank_zero_info("Query centered...")
    if config.bank.center:
        rank_zero_info("\nBank centering...")
        bank_features = bank_features - bank_features.mean(dim=0, keepdim=True)
        rank_zero_info("Bank centered...")

    # normalize
    if config.query.normalize:
        rank_zero_info("\nQuery normalizing...")
        query_features = torch.nn.functional.normalize(query_features, p=2, dim=1)
        rank_zero_info("Query normalized...")
    if config.bank.normalize:
        rank_zero_info("\nBank normalizing...")
        bank_features = torch.nn.functional.normalize(bank_features, p=2, dim=1)
        rank_zero_info("Bank normalized...")

    # dot product
    rank_zero_info("\nComputing similarties...")
    sim = query_features.matmul(bank_features.t())
    rank_zero_info("Computed similarities...")

    rank_zero_info("\nStart computing metrics:")
    for rank in ranks:
        _, topkidx = torch.topk(sim, rank, dim=1)
        acc = (
            torch.any(bank_labels[topkidx] == query_labels.unsqueeze(1), dim=1)
            .float()
            .mean()
            .item()
        )
        rank_zero_info(f"R @ {rank} = {acc:.4f}")


if __name__ == "__main__":
    main()
