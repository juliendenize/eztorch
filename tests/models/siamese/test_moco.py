import unittest

import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig
from torch import nn

from eztorch.losses.moco_loss import compute_moco_loss
from eztorch.models.siamese import MoCoModel
from eztorch.transforms import MultiCropTransform
from tests.helpers.datamodules import BoringDataModule
from tests.helpers.datasets import RandomVisionLabeledDataset


class TestMoCoModel(unittest.TestCase):
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
        self.predictor_cfg = DictConfig(
            {
                "_target_": "eztorch.models.heads.MLPHead",
                "input_dim": 512,
                "output_dim": 2,
            }
        )
        self.queue_cfg = DictConfig(
            {"_target_": "eztorch.models.queues.FIFOQueue", "size": 8, "feature_dim": 2}
        )
        self.temp = 0.2

    def test_moco_init(self):
        MoCoModel(
            trunk=self.trunk_cfg,
            projector=None,
            predictor=None,
            optimizer={},
            queue=None,
            num_devices=1,
            simulate_n_devices=1,
            temp=self.temp,
        )

        MoCoModel(
            trunk=self.trunk_cfg,
            projector=None,
            predictor=None,
            optimizer={},
            queue=None,
            num_devices=1,
            simulate_n_devices=8,
            temp=self.temp,
        )

        MoCoModel(
            trunk=self.trunk_cfg,
            projector=None,
            predictor=None,
            optimizer={},
            queue=None,
            num_devices=2,
            simulate_n_devices=8,
            temp=self.temp,
        )

        MoCoModel(
            trunk=self.trunk_cfg,
            projector=self.projector_cfg,
            predictor=None,
            optimizer={},
            queue=None,
            num_devices=2,
            temp=self.temp,
        )

        MoCoModel(
            trunk=self.trunk_cfg,
            projector=self.projector_cfg,
            predictor=self.predictor_cfg,
            optimizer={},
            queue=None,
            num_devices=2,
            temp=self.temp,
        )

        MoCoModel(
            trunk=self.trunk_cfg,
            projector=self.projector_cfg,
            predictor=self.predictor_cfg,
            optimizer={},
            queue=self.queue_cfg,
            num_devices=2,
            temp=self.temp,
        )

    def test_moco_loss_without_key(self):
        q = torch.arange(1.0, 9.0, 1.0).view((4, 2))
        k = torch.arange(9.0, 17.0, 1.0).view((4, 2))
        queue = torch.tensor([[0.0, 0, 0, 2], [0.0, 0, 0, 2.0]])

        model = MoCoModel(
            trunk=self.trunk_cfg,
            projector=self.projector_cfg,
            predictor=self.predictor_cfg,
            optimizer={},
            queue=self.queue_cfg,
            num_devices=2,
            temp=self.temp,
        )

        model.queue = queue

        labels = torch.tensor([0, 0, 0, 0])
        sim = torch.tensor(
            [
                [29, 0, 0, 0, 2.5],
                [81, 0, 0, 0, 5.5],
                [149, 0, 0, 0, 8.5],
                [233, 0, 0, 0, 11.5],
            ]
        )
        logits = sim / self.temp
        loss = nn.functional.cross_entropy(logits, labels)

        output_loss = compute_moco_loss(q, k, k, False, queue, self.temp, 0)

        assert torch.equal(loss, output_loss)

    def test_moco_loss_with_key(self):
        q = torch.arange(1.0, 9.0, 1.0).view((4, 2))
        k = torch.tensor([[0, 0], [1, 1], [0, 0], [0.0, 0.0]])
        queue = torch.tensor([[0.0, 0, 0, 2], [0.0, 0, 0, 2.0]])

        model = MoCoModel(
            trunk=self.trunk_cfg,
            projector=self.projector_cfg,
            predictor=self.predictor_cfg,
            optimizer={},
            use_keys=True,
            queue=self.queue_cfg,
            num_devices=2,
            temp=self.temp,
        )

        model.queue = queue

        labels = torch.tensor([0, 1, 2, 3])
        sim = torch.tensor(
            [
                [0.0, 3, 0, 0, 0, 0, 0, 6],
                [0, 7, 0, 0, 0, 0, 0, 14],
                [0, 11, 0, 0, 0, 0, 0, 22],
                [0, 15, 0, 0, 0, 0, 0, 30],
            ]
        )
        logits = sim / self.temp
        loss = nn.functional.cross_entropy(logits, labels)

        output_loss = compute_moco_loss(q, k, k, True, queue, self.temp, 0)

        assert torch.equal(loss, output_loss)

    def test_moco_loss_with_key_without_queue(self):
        q = torch.arange(1.0, 9.0, 1.0).view((4, 2))
        k = torch.tensor([[0, 0], [1, 1], [0, 0], [0.0, 0.0]])

        model = MoCoModel(
            trunk=self.trunk_cfg,
            projector=self.projector_cfg,
            predictor=self.predictor_cfg,
            optimizer={},
            use_keys=True,
            queue=None,
            num_devices=2,
            temp=self.temp,
        )

        labels = torch.tensor([0, 1, 2, 3])
        sim = torch.tensor(
            [
                [0.0, 3, 0, 0],
                [0, 7, 0, 0],
                [0, 11, 0, 0],
                [0, 15, 0, 0],
            ]
        )
        logits = sim / self.temp
        loss = nn.functional.cross_entropy(logits, labels)

        output_loss = compute_moco_loss(q, k, k, True, None, self.temp, 0)

        assert torch.equal(loss, output_loss)

    def test_moco_sym_fit(self):
        optimizer_cfg = DictConfig(
            {
                "_target_": "eztorch.optimizers.optimizer_factory",
                "name": "adam",
                "scheduler": None,
                "initial_lr": 0.06,
            }
        )

        transform_cfg = [{"num_views": 2, "transform": nn.Identity()}]

        model = MoCoModel(
            trunk=self.trunk_cfg,
            optimizer=optimizer_cfg,
            use_keys=True,
            queue=None,
            num_devices=1,
            simulate_n_devices=1,
            sym=True,
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

    def test_moco_sym_mutual_fit(self):
        optimizer_cfg = DictConfig(
            {
                "_target_": "eztorch.optimizers.optimizer_factory",
                "name": "adam",
                "scheduler": None,
                "initial_lr": 0.06,
            }
        )

        transform_cfg = [{"num_views": 2, "transform": nn.Identity()}]

        model = MoCoModel(
            trunk=self.trunk_cfg,
            optimizer=optimizer_cfg,
            use_keys=True,
            queue=None,
            num_devices=1,
            simulate_n_devices=1,
            sym=True,
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

    def test_moco_fit(self):
        optimizer_cfg = DictConfig(
            {
                "_target_": "eztorch.optimizers.optimizer_factory",
                "name": "adam",
                "scheduler": None,
                "initial_lr": 0.06,
            }
        )

        transform_cfg = [{"num_views": 2, "transform": nn.Identity()}]

        model = MoCoModel(
            trunk=self.trunk_cfg,
            optimizer=optimizer_cfg,
            projector=self.projector_cfg,
            use_keys=True,
            queue=None,
            num_devices=1,
            simulate_n_devices=1,
            initial_momentum=0.98,
            scheduler_momentum="cosine",
            temp=self.temp,
        )

        datamodule = BoringDataModule(
            dataset=RandomVisionLabeledDataset(
                (128, 3, 32, 32), transform=MultiCropTransform(transform_cfg)
            ),
            val_dataset=RandomVisionLabeledDataset((128, 3, 32, 32), transform=None),
            batch_size=64,
        )

        assert model.current_momentum == 0.98

        trainer = Trainer(fast_dev_run=2, devices=1)
        trainer.fit(model, datamodule)

        assert model.current_momentum == 0.98

    def test_moco_mutual_fit(self):
        optimizer_cfg = DictConfig(
            {
                "_target_": "eztorch.optimizers.optimizer_factory",
                "name": "adam",
                "scheduler": None,
                "initial_lr": 0.06,
            }
        )

        transform_cfg = [{"num_views": 2, "transform": nn.Identity()}]

        model = MoCoModel(
            trunk=self.trunk_cfg,
            optimizer=optimizer_cfg,
            projector=self.projector_cfg,
            use_keys=True,
            queue=None,
            num_devices=1,
            simulate_n_devices=1,
            initial_momentum=0.98,
            scheduler_momentum="cosine",
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

        assert model.current_momentum == 0.98

        trainer = Trainer(fast_dev_run=2, devices=1)
        trainer.fit(model, datamodule)

        assert model.current_momentum == 0.98


if __name__ == "__main__":
    unittest.main()
