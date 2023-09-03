import unittest

import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig
from torch import nn

from eztorch.losses.simclr_loss import compute_simclr_loss
from eztorch.models.siamese import SimCLRModel
from eztorch.transforms import MultiCropTransform
from tests.helpers.datamodules import BoringDataModule
from tests.helpers.datasets import RandomVisionLabeledDataset


class TestSimCLRModel(unittest.TestCase):
    def setUp(self) -> None:
        self.trunk_cfg = DictConfig(
            {
                "_target_": "eztorch.models.trunks.create_resnet",
                "name": "resnet18",
                "num_classes": 0,
                "small_input": True,
            }
        )
        self.projector_cfg = DictConfig(
            {
                "_target_": "eztorch.models.heads.MLPHead",
                "input_dim": 512,
                "output_dim": 2,
            }
        )
        self.queue_cfg = DictConfig(
            {"_target_": "eztorch.models.queues.FIFOQueue", "size": 8, "feature_dim": 2}
        )
        self.temp = 10.0

    def test_simclr_init(self):
        SimCLRModel(trunk=self.trunk_cfg, projector=None, optimizer={}, temp=self.temp)

        SimCLRModel(trunk=self.trunk_cfg, projector=None, optimizer={}, temp=self.temp)

        SimCLRModel(trunk=self.trunk_cfg, projector=None, optimizer={}, temp=self.temp)

        SimCLRModel(
            trunk=self.trunk_cfg,
            projector=self.projector_cfg,
            optimizer={},
            temp=self.temp,
        )

    def test_simclr_loss(self):
        z = torch.tensor(
            [[1.0, 2], [3, 4], [5, 6], [7, 8], [0, 0], [1, 1], [0, 0], [0, 0]]
        )
        pos_mask = torch.tensor(
            [
                [0.0, 0, 0, 0, 1, 0, 0, 0],
                [0.0, 0, 0, 0, 0, 1, 0, 0],
                [0.0, 0, 0, 0, 0, 0, 1, 0],
                [0.0, 0, 0, 0, 0, 0, 0, 1],
                [1.0, 0, 0, 0, 0, 0, 0, 0],
                [0.0, 1, 0, 0, 0, 0, 0, 0],
                [0.0, 0, 1, 0, 0, 0, 0, 0],
                [0.0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )
        neg_mask = torch.tensor(
            [
                [0.0, 1, 1, 1, 0, 1, 1, 1],
                [1.0, 0, 1, 1, 1, 0, 1, 1],
                [1.0, 1, 0, 1, 1, 1, 0, 1],
                [1.0, 1, 1, 0, 1, 1, 1, 0],
                [0.0, 1, 1, 1, 0, 1, 1, 1],
                [1.0, 0, 1, 1, 1, 0, 1, 1],
                [1.0, 1, 0, 1, 1, 1, 0, 1],
                [1.0, 1, 1, 0, 1, 1, 1, 0],
            ]
        )
        sim = torch.tensor(
            [
                [5.0, 11, 17, 23, 0, 3, 0, 0],
                [11, 25, 39, 53, 0, 7, 0, 0],
                [17, 39, 61, 83, 0, 11, 0, 0],
                [23, 53, 83, 113, 0, 15, 0, 0],
                [0.0, 0, 0, 0, 0, 0, 0, 0],
                [3, 7, 11, 15, 0, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        logits = torch.exp(sim / self.temp)
        pos = torch.sum(logits * pos_mask, 1)
        neg = torch.sum(logits * neg_mask, 1)
        loss = -(torch.mean(torch.log(pos / (neg + pos))))

        output_loss = compute_simclr_loss(z, z, pos_mask, neg_mask, self.temp)
        assert torch.equal(loss, output_loss)

    def test_simclr_fit(self):
        optimizer_cfg = DictConfig(
            {
                "_target_": "eztorch.optimizers.optimizer_factory",
                "name": "adam",
                "scheduler": None,
                "initial_lr": 0.06,
            }
        )

        transform_cfg = [{"num_views": 2, "transform": nn.Identity()}]

        model = SimCLRModel(
            trunk=self.trunk_cfg,
            optimizer=optimizer_cfg,
            temp=self.temp,
        )

        datamodule = BoringDataModule(
            dataset=RandomVisionLabeledDataset(
                (128, 3, 32, 32), transform=MultiCropTransform(transform_cfg)
            ),
            val_dataset=RandomVisionLabeledDataset((128, 3, 32, 32), transform=None),
            batch_size=64,
        )

        trainer = Trainer(fast_dev_run=1, devices=1)
        trainer.fit(model, datamodule)

    def test_simclr_mutual_fit(self):
        optimizer_cfg = DictConfig(
            {
                "_target_": "eztorch.optimizers.optimizer_factory",
                "name": "adam",
                "scheduler": None,
                "initial_lr": 0.06,
            }
        )

        transform_cfg = [{"num_views": 2, "transform": nn.Identity()}]

        model = SimCLRModel(
            trunk=self.trunk_cfg,
            optimizer=optimizer_cfg,
            temp=self.temp,
            mutual_pass=True,
        )

        datamodule = BoringDataModule(
            dataset=RandomVisionLabeledDataset(
                (128, 3, 32, 32), transform=MultiCropTransform(transform_cfg)
            ),
            val_dataset=RandomVisionLabeledDataset((128, 3, 32, 32), transform=None),
            batch_size=64,
        )

        trainer = Trainer(fast_dev_run=1, devices=1)
        trainer.fit(model, datamodule)


if __name__ == "__main__":
    unittest.main()
