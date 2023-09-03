from typing import List, Tuple

import torch
from lightning.pytorch import LightningModule
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

from tests.helpers.datasets import RandomDataset


class BoringModel(LightningModule):
    def __init__(self):
        """Testing PL Module. Use as follows:

        - subclass
        - modify the behavior for what you want
        class TestModel(BaseTestModel):
            def training_step(...):
                # do your own thing
        or:
        model = BaseTestModel()
        model.on_train_epoch_end = None
        """
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    @property
    def num_layers(self) -> int:
        return 1

    def get_param_layer_id(self, name: str) -> int:
        return 0

    def forward(self, x):
        return self.layer(x)

    def loss(self, batch, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def step(self, x):
        x = self(x)
        out = torch.nn.functional.mse_loss(x, torch.ones_like(x))
        return out

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def on_train_batch_end(self, training_step_outputs):
        return training_step_outputs

    def on_train_epoch_end(self, outputs) -> None:
        torch.stack([x["loss"] for x in outputs]).mean()

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"x": loss}

    def on_validation_epoch_end(self, outputs) -> None:
        torch.stack([x["x"] for x in outputs]).mean()

    def test_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"y": loss}

    def on_test_epoch_end(self, outputs) -> None:
        torch.stack([x["y"] for x in outputs]).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def test_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def predict_dataloader(self):
        return DataLoader(RandomDataset(32, 64))


class LargeBoringModel(LightningModule):
    def __init__(self):
        """Testing PL Module. Use as follows:

        - subclass
        - modify the behavior for what you want
        class TestModel(BaseTestModel):
            def training_step(...):
                # do your own thing
        or:
        model = BaseTestModel()
        model.on_train_epoch_end = None
        """
        super().__init__()
        self.layer1 = torch.nn.Linear(32, 32, bias=False)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.layer2 = torch.nn.Linear(32, 32, bias=False)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.layer3 = torch.nn.Linear(32, 32, bias=False)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.layer4 = torch.nn.Linear(32, 2, bias=True)
        # for sake of not having it in learnable_params
        self.useless_layer = torch.nn.Linear(32, 2)

    @property
    def learnable_params(self) -> List[Parameter]:
        params = [
            param
            for layer in [
                self.layer1,
                self.bn1,
                self.layer2,
                self.bn2,
                self.layer3,
                self.bn3,
                self.layer4,
            ]
            for param in layer.parameters()
        ]
        return params

    @property
    def num_layers(self) -> int:
        """Number of layers of the model."""
        return 4

    def get_param_layer_id(self, name: str) -> int:
        """Get the layer id of the named parameter.

        Args:
            name: The name of the parameter.
        """
        if name.startswith("layer1") or name.startswith("bn1"):
            return 0
        elif name.startswith("layer2") or name.startswith("bn2"):
            return 1
        elif name.startswith("layer3") or name.startswith("bn3"):
            return 2
        elif name.startswith("layer4"):
            return 3
        elif name.startswith("useless_layer"):
            return 3

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.layer3(x)
        x = self.bn3(x)
        return x

    def loss(self, batch, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def step(self, x):
        x = self(x)
        out = torch.nn.functional.mse_loss(x, torch.ones_like(x))
        return out

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def on_train_batch_end(self, training_step_outputs):
        return training_step_outputs

    def on_train_epoch_end(self, outputs) -> None:
        torch.stack([x["loss"] for x in outputs]).mean()

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"x": loss}

    def on_validation_epoch_end(self, outputs) -> None:
        torch.stack([x["x"] for x in outputs]).mean()

    def test_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"y": loss}

    def on_test_epoch_end(self, outputs) -> None:
        torch.stack([x["y"] for x in outputs]).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def test_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def predict_dataloader(self):
        return DataLoader(RandomDataset(32, 64))


class ManualOptimBoringModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        output = self(batch)
        loss = self.loss(batch, output)
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        return loss
