import unittest

import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig
from torch import nn

from eztorch.losses.ressl_loss import compute_ressl_loss, compute_ressl_mask
from eztorch.models.siamese import ReSSLModel
from eztorch.transforms import MultiCropTransform
from tests.helpers.datamodules import BoringDataModule
from tests.helpers.datasets import RandomVisionLabeledDataset


class TestReSSLModel(unittest.TestCase):
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
        self.temp = 0.1
        self.temp_m = 0.04

    def test_ressl_init(self):
        ReSSLModel(
            trunk=self.trunk_cfg,
            projector=None,
            predictor=None,
            optimizer={},
            queue=None,
            num_devices=1,
            simulate_n_devices=1,
            temp=self.temp,
            temp_m=self.temp_m,
        )

        ReSSLModel(
            trunk=self.trunk_cfg,
            projector=None,
            predictor=None,
            optimizer={},
            queue=None,
            num_devices=1,
            simulate_n_devices=8,
            temp=self.temp,
            temp_m=self.temp_m,
        )

        ReSSLModel(
            trunk=self.trunk_cfg,
            projector=None,
            predictor=None,
            optimizer={},
            queue=None,
            num_devices=2,
            simulate_n_devices=8,
            temp=self.temp,
            temp_m=self.temp_m,
        )

        ReSSLModel(
            trunk=self.trunk_cfg,
            projector=self.projector_cfg,
            predictor=None,
            optimizer={},
            queue=None,
            num_devices=2,
            temp=self.temp,
            temp_m=self.temp_m,
        )

        ReSSLModel(
            trunk=self.trunk_cfg,
            projector=self.projector_cfg,
            predictor=self.predictor_cfg,
            optimizer={},
            queue=None,
            num_devices=2,
            temp=self.temp,
            temp_m=self.temp_m,
        )

        ReSSLModel(
            trunk=self.trunk_cfg,
            projector=self.projector_cfg,
            predictor=self.predictor_cfg,
            optimizer={},
            queue=self.queue_cfg,
            num_devices=2,
            temp=self.temp,
            temp_m=self.temp_m,
        )

    def test_ressl_loss_without_key(self):
        q = torch.arange(1.0, 9.0, 1.0).view((4, 2))
        k = torch.tensor([[0, 0], [1, 1], [0, 0], [1.0, 1.0]])
        queue = torch.tensor([[0.0, 0, 0, 2], [0.0, 0, 0, 2.0]])
        queue

        model = ReSSLModel(
            trunk=self.trunk_cfg,
            projector=self.projector_cfg,
            predictor=self.predictor_cfg,
            optimizer={},
            use_keys=False,
            queue=self.queue_cfg,
            num_devices=2,
            temp=self.temp,
            temp_m=self.temp_m,
        )

        model.queue = queue

        sim_qqueue = torch.Tensor(
            [[0.0, 0, 0, 6], [0, 0, 0, 14], [0, 0, 0, 22], [0, 0, 0, 30]]
        )
        sim_kqueue = torch.Tensor(
            [[0.0, 0, 0, 0], [0, 0, 0, 4], [0, 0, 0, 0], [0, 0, 0, 4]]
        )

        sim_q = sim_qqueue
        sim_k = sim_kqueue

        logits_q = sim_q / self.temp
        logits_k = sim_k / self.temp_m

        loss = -torch.sum(
            nn.functional.softmax(logits_k.detach(), dim=1)
            * nn.functional.log_softmax(logits_q, dim=1),
            dim=1,
        ).mean(dim=0)

        output_loss = compute_ressl_loss(
            q, k, k, False, queue, None, self.temp, self.temp_m
        )

        assert torch.equal(loss, output_loss)

    def test_ressl_loss_with_key(self):
        q = torch.arange(1.0, 9.0, 1.0).view((4, 2))
        k = torch.tensor([[0, 0], [1, 1], [0, 0], [1.0, 1.0]])
        queue = torch.tensor([[0.0, 0, 0, 2], [0.0, 0, 0, 2.0]])

        model = ReSSLModel(
            trunk=self.trunk_cfg,
            projector=self.projector_cfg,
            predictor=self.predictor_cfg,
            optimizer={},
            use_keys=True,
            queue=self.queue_cfg,
            num_devices=2,
            temp=self.temp,
            temp_m=self.temp_m,
        )

        model.queue = queue

        sim_qk = torch.tensor(
            [
                [0, 3, 0, 3],
                [0, 7, 0, 7],
                [0, 11, 0, 11],
                [0, 15.0, 0, 15],
            ]
        )
        sim_kk = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 2, 0, 2],
                [0, 0, 0, 0],
                [0, 2.0, 0, 2],
            ]
        )
        sim_qqueue = torch.Tensor(
            [[0.0, 0, 0, 6], [0, 0, 0, 14], [0, 0, 0, 22], [0, 0, 0, 30]]
        )
        sim_kqueue = torch.Tensor(
            [[0.0, 0, 0, 0], [0, 0, 0, 4], [0, 0, 0, 0], [0, 0, 0, 4]]
        )

        mask = torch.tensor(
            [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]
        )

        sim_kk -= 1e9 * mask
        sim_qk -= 1e9 * mask

        sim_q = torch.cat([sim_qk, sim_qqueue], dim=1)
        sim_k = torch.cat([sim_kk, sim_kqueue], dim=1)

        logits_q = sim_q / self.temp
        logits_k = sim_k / self.temp_m

        loss = -torch.sum(
            nn.functional.softmax(logits_k.detach(), dim=1)
            * nn.functional.log_softmax(logits_q, dim=1),
            dim=1,
        ).mean(dim=0)

        mask = compute_ressl_mask(q.shape[0], queue.shape[1], True, 0, 1, "cuda")
        output_loss = compute_ressl_loss(
            q, k, k, True, queue, mask, self.temp, self.temp_m
        )

        assert torch.equal(loss, output_loss)

    def test_ressl_loss_with_key_without_queue(self):
        q = torch.arange(1.0, 9.0, 1.0).view((4, 2))
        k = torch.tensor([[0, 0], [1, 1], [0, 0], [1.0, 1.0]])

        model = ReSSLModel(
            trunk=self.trunk_cfg,
            projector=self.projector_cfg,
            predictor=self.predictor_cfg,
            optimizer={},
            use_keys=True,
            queue=None,
            num_devices=2,
            temp=self.temp,
            temp_m=self.temp_m,
        )

        sim_qk = torch.tensor(
            [
                [0, 3, 0, 3],
                [0, 7, 0, 7],
                [0, 11, 0, 11],
                [0, 15.0, 0, 15],
            ]
        )
        sim_kk = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 2, 0, 2],
                [0, 0, 0, 0],
                [0, 2.0, 0, 2],
            ]
        )

        mask = torch.tensor(
            [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]
        )

        sim_kk -= 1e9 * mask
        sim_qk -= 1e9 * mask

        sim_q = sim_qk
        sim_k = sim_kk

        logits_q = sim_q / self.temp
        logits_k = sim_k / self.temp_m

        loss = -torch.sum(
            nn.functional.softmax(logits_k.detach(), dim=1)
            * nn.functional.log_softmax(logits_q, dim=1),
            dim=1,
        ).mean(dim=0)

        mask = compute_ressl_mask(q.shape[0], 0, True, 0, 1, "cuda")
        output_loss = compute_ressl_loss(
            q, k, k, True, None, mask, self.temp, self.temp_m
        )

        assert torch.equal(loss, output_loss)

    def test_ressl_sym_fit(self):
        optimizer_cfg = DictConfig(
            {
                "_target_": "eztorch.optimizers.optimizer_factory",
                "name": "adam",
                "scheduler": None,
                "initial_lr": 0.06,
            }
        )

        transform_cfg = [{"num_views": 2, "transform": nn.Identity()}]

        model = ReSSLModel(
            trunk=self.trunk_cfg,
            optimizer=optimizer_cfg,
            use_keys=True,
            queue=None,
            num_devices=1,
            simulate_n_devices=1,
            sym=True,
            temp=self.temp,
            temp_m=self.temp_m,
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

    def test_ressl_sym_mutual_fit(self):
        optimizer_cfg = DictConfig(
            {
                "_target_": "eztorch.optimizers.optimizer_factory",
                "name": "adam",
                "scheduler": None,
                "initial_lr": 0.06,
            }
        )

        transform_cfg = [{"num_views": 2, "transform": nn.Identity()}]

        model = ReSSLModel(
            trunk=self.trunk_cfg,
            optimizer=optimizer_cfg,
            use_keys=True,
            queue=None,
            num_devices=1,
            simulate_n_devices=1,
            sym=True,
            temp=self.temp,
            temp_m=self.temp_m,
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

    def test_ressl_fit(self):
        optimizer_cfg = DictConfig(
            {
                "_target_": "eztorch.optimizers.optimizer_factory",
                "name": "adam",
                "scheduler": None,
                "initial_lr": 0.06,
            }
        )

        transform_cfg = [{"num_views": 2, "transform": nn.Identity()}]

        model = ReSSLModel(
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
            temp_m=self.temp_m,
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

    def test_ressl_mutual_fit(self):
        optimizer_cfg = DictConfig(
            {
                "_target_": "eztorch.optimizers.optimizer_factory",
                "name": "adam",
                "scheduler": None,
                "initial_lr": 0.06,
            }
        )

        transform_cfg = [{"num_views": 2, "transform": nn.Identity()}]

        model = ReSSLModel(
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
            temp_m=self.temp_m,
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
