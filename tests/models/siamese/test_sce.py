import unittest

import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig
from torch import nn

from eztorch.models.siamese import SCEModel
from eztorch.transforms import MultiCropTransform
from tests.helpers.datamodules import BoringDataModule
from tests.helpers.datasets import RandomVisionLabeledDataset


class TestSCEModel(unittest.TestCase):
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
        self.temp_m = 0.05
        self.coeff = 0.5

    def test_sce_init(self):
        SCEModel(
            trunk=self.trunk_cfg,
            projector=None,
            predictor=None,
            optimizer={},
            queue=None,
            num_devices=1,
            simulate_n_devices=1,
            temp=self.temp,
            temp_m=self.temp_m,
            coeff=self.coeff,
        )

        SCEModel(
            trunk=self.trunk_cfg,
            projector=None,
            predictor=None,
            optimizer={},
            queue=None,
            num_devices=1,
            simulate_n_devices=8,
            temp=self.temp,
            temp_m=self.temp_m,
            coeff=self.coeff,
        )

        SCEModel(
            trunk=self.trunk_cfg,
            projector=None,
            predictor=None,
            optimizer={},
            queue=None,
            num_devices=2,
            simulate_n_devices=8,
            temp=self.temp,
            temp_m=self.temp_m,
            coeff=self.coeff,
        )

        SCEModel(
            trunk=self.trunk_cfg,
            projector=self.projector_cfg,
            predictor=None,
            optimizer={},
            queue=None,
            num_devices=2,
            temp=self.temp,
            temp_m=self.temp_m,
            coeff=self.coeff,
        )

        SCEModel(
            trunk=self.trunk_cfg,
            projector=self.projector_cfg,
            predictor=self.predictor_cfg,
            optimizer={},
            queue=None,
            num_devices=2,
            temp=self.temp,
            temp_m=self.temp_m,
            coeff=self.coeff,
        )

        SCEModel(
            trunk=self.trunk_cfg,
            projector=self.projector_cfg,
            predictor=self.predictor_cfg,
            optimizer={},
            queue=self.queue_cfg,
            num_devices=2,
            temp=self.temp,
            temp_m=self.temp_m,
            coeff=self.coeff,
        )

    def test_sce_sym_fit(self):
        optimizer_cfg = DictConfig(
            {
                "_target_": "eztorch.optimizers.optimizer_factory",
                "name": "adam",
                "scheduler": None,
                "initial_lr": 0.06,
            }
        )

        transform_cfg = [{"num_views": 2, "transform": nn.Identity()}]

        model = SCEModel(
            trunk=self.trunk_cfg,
            optimizer=optimizer_cfg,
            use_keys=True,
            queue=None,
            num_devices=1,
            simulate_n_devices=1,
            sym=True,
            temp=self.temp,
            temp_m=self.temp_m,
            coeff=self.coeff,
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

    def test_sce_split_fit(self):
        optimizer_cfg = DictConfig(
            {
                "_target_": "eztorch.optimizers.optimizer_factory",
                "name": "adam",
                "scheduler": None,
                "initial_lr": 0.06,
            }
        )

        transform_cfg = [{"num_views": 2, "transform": nn.Identity()}]

        model = SCEModel(
            trunk=self.trunk_cfg,
            optimizer=optimizer_cfg,
            use_keys=True,
            queue=None,
            num_devices=1,
            simulate_n_devices=1,
            sym=False,
            num_splits=2,
            temp=self.temp,
            temp_m=self.temp_m,
            coeff=self.coeff,
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

    def test_sce_split_sym_fit(self):
        optimizer_cfg = DictConfig(
            {
                "_target_": "eztorch.optimizers.optimizer_factory",
                "name": "adam",
                "scheduler": None,
                "initial_lr": 0.06,
            }
        )

        transform_cfg = [{"num_views": 2, "transform": nn.Identity()}]

        model = SCEModel(
            trunk=self.trunk_cfg,
            optimizer=optimizer_cfg,
            use_keys=True,
            queue=None,
            num_devices=1,
            simulate_n_devices=1,
            sym=True,
            num_splits=2,
            temp=self.temp,
            temp_m=self.temp_m,
            coeff=self.coeff,
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

    def test_sce_split_sym_with_queue_fit(self):
        optimizer_cfg = DictConfig(
            {
                "_target_": "eztorch.optimizers.optimizer_factory",
                "name": "adam",
                "scheduler": None,
                "initial_lr": 0.06,
            }
        )

        transform_cfg = [{"num_views": 2, "transform": nn.Identity()}]

        model = SCEModel(
            trunk=self.trunk_cfg,
            optimizer=optimizer_cfg,
            use_keys=True,
            queue=DictConfig({"size": 128, "feature_dim": 512}),
            num_devices=1,
            simulate_n_devices=1,
            sym=True,
            num_splits=2,
            temp=self.temp,
            temp_m=self.temp_m,
            coeff=self.coeff,
        )

        datamodule = BoringDataModule(
            dataset=RandomVisionLabeledDataset(
                (128, 3, 32, 32), transform=MultiCropTransform(transform_cfg)
            ),
            val_dataset=RandomVisionLabeledDataset((128, 3, 32, 32), transform=None),
            batch_size=64,
        )

        trainer = Trainer(fast_dev_run=2, devices=1)
        trainer.fit(model, datamodule)

    def test_sce_sym_mutual_fit(self):
        optimizer_cfg = DictConfig(
            {
                "_target_": "eztorch.optimizers.optimizer_factory",
                "name": "adam",
                "scheduler": None,
                "initial_lr": 0.06,
            }
        )

        transform_cfg = [{"num_views": 2, "transform": nn.Identity()}]

        model = SCEModel(
            trunk=self.trunk_cfg,
            optimizer=optimizer_cfg,
            use_keys=True,
            queue=None,
            num_devices=1,
            simulate_n_devices=1,
            sym=True,
            temp=self.temp,
            temp_m=self.temp_m,
            coeff=self.coeff,
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

    def test_sce_fit(self):
        optimizer_cfg = DictConfig(
            {
                "_target_": "eztorch.optimizers.optimizer_factory",
                "name": "adam",
                "scheduler": None,
                "initial_lr": 0.06,
            }
        )

        transform_cfg = [{"num_views": 2, "transform": nn.Identity()}]

        model = SCEModel(
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
            coeff=self.coeff,
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

    def test_sce_mutual_fit(self):
        optimizer_cfg = DictConfig(
            {
                "_target_": "eztorch.optimizers.optimizer_factory",
                "name": "adam",
                "scheduler": None,
                "initial_lr": 0.06,
            }
        )

        transform_cfg = [{"num_views": 2, "transform": nn.Identity()}]

        model = SCEModel(
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
            coeff=self.coeff,
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
