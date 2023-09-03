import unittest

from omegaconf.dictconfig import DictConfig

from eztorch.optimizers.optimizer_factory import optimizer_factory
from tests.helpers.models import BoringModel, LargeBoringModel


class TestOptimizerFactory(unittest.TestCase):
    def setUp(self):
        self.base_model = BoringModel()
        self.large_model = LargeBoringModel()
        self.sgd_config = DictConfig(
            {
                "name": "sgd",
                "initial_lr": 2.0,
                "batch_size": None,
                "num_steps_per_epoch": None,
                "exclude_wd_norm": False,
                "exclude_wd_bias": False,
                "scaler": None,
                "params": {},
                "scheduler": None,
            }
        )

    def test_no_exclude_optimizer_factory(self):
        optimizer, _ = optimizer_factory(
            **self.sgd_config, keys_without_decay=[], model=self.base_model
        )
        assert len(optimizer.param_groups) == 1
        # one linear layer with bias
        assert len(optimizer.param_groups[0]["params"]) == 2

        optimizer, _ = optimizer_factory(
            **self.sgd_config, keys_without_decay=[], model=self.large_model
        )
        assert len(optimizer.param_groups) == 1
        # 3 linear layers without biases, 1 linear layer with bias and 3 batch norms
        assert len(optimizer.param_groups[0]["params"]) == 11

    def test_no_exclude_optimizer_factory_layer_decay(self):
        optimizer, _ = optimizer_factory(
            **self.sgd_config,
            keys_without_decay=[],
            model=self.base_model,
            layer_decay_lr=0.7,
        )

        print(optimizer.param_groups)
        assert len(optimizer.param_groups) == 1
        # one linear layer with bias
        assert len(optimizer.param_groups[0]["params"]) == 2

        optimizer, _ = optimizer_factory(
            **self.sgd_config,
            keys_without_decay=[],
            model=self.large_model,
            layer_decay_lr=0.7,
        )
        assert len(optimizer.param_groups) == 4
        # 3 linear layers without biases, 1 linear layer with bias and 3 batch norms
        assert len(optimizer.param_groups[0]["params"]) == 3
        assert len(optimizer.param_groups[1]["params"]) == 3
        assert len(optimizer.param_groups[2]["params"]) == 3
        assert len(optimizer.param_groups[3]["params"]) == 2

    def test_exclude_bias_optimizer_factory(self):
        self.sgd_config.exclude_wd_bias = True

        optimizer, _ = optimizer_factory(
            **self.sgd_config, keys_without_decay=[], model=self.base_model
        )
        assert len(optimizer.param_groups) == 2
        # one linear layer with bias
        assert len(optimizer.param_groups[0]["params"]) == 1
        # num biases
        assert len(optimizer.param_groups[1]["params"]) == 1

        optimizer, _ = optimizer_factory(**self.sgd_config, model=self.large_model)

        assert len(optimizer.param_groups) == 2
        # 3 linear layers without biases, 1 linear layer with bias and 3 batch norms
        assert len(optimizer.param_groups[0]["params"]) == 7
        # num biases
        assert len(optimizer.param_groups[1]["params"]) == 4

    def test_exclude_bias_optimizer_factory_layer_decay(self):
        self.sgd_config.exclude_wd_bias = True

        optimizer, _ = optimizer_factory(
            **self.sgd_config,
            keys_without_decay=[],
            model=self.base_model,
            layer_decay_lr=0.7,
        )
        assert len(optimizer.param_groups) == 2
        # one linear layer with bias
        assert len(optimizer.param_groups[0]["params"]) == 1
        # num biases
        assert len(optimizer.param_groups[1]["params"]) == 1

        optimizer, _ = optimizer_factory(
            **self.sgd_config, model=self.large_model, layer_decay_lr=0.7
        )

        assert len(optimizer.param_groups) == 8
        # 3 linear layers without biases, 1 linear layer with bias and 3 batch norms
        assert len(optimizer.param_groups[0]["params"]) == 2
        assert len(optimizer.param_groups[1]["params"]) == 2
        assert len(optimizer.param_groups[2]["params"]) == 2
        assert len(optimizer.param_groups[3]["params"]) == 1
        # num biases
        assert len(optimizer.param_groups[4]["params"]) == 1
        assert len(optimizer.param_groups[5]["params"]) == 1
        assert len(optimizer.param_groups[6]["params"]) == 1
        assert len(optimizer.param_groups[7]["params"]) == 1

    def test_exclude_norm_optimizer_factory(self):
        self.sgd_config.exclude_wd_norm = True

        optimizer, _ = optimizer_factory(
            **self.sgd_config, keys_without_decay=[], model=self.base_model
        )
        assert len(optimizer.param_groups) == 1
        # one linear layer with bias
        assert len(optimizer.param_groups[0]["params"]) == 2

        optimizer, _ = optimizer_factory(
            **self.sgd_config, keys_without_decay=[], model=self.large_model
        )
        assert len(optimizer.param_groups) == 2
        # 3 linear layers without biases, 1 linear layer with bias and 3 batch norms
        assert len(optimizer.param_groups[0]["params"]) == 5
        # num norm parameters
        assert len(optimizer.param_groups[1]["params"]) == 6

    def test_exclude_norm_optimizer_factory_layer_decay(self):
        self.sgd_config.exclude_wd_norm = True

        optimizer, _ = optimizer_factory(
            **self.sgd_config,
            keys_without_decay=[],
            model=self.base_model,
            layer_decay_lr=0.7,
        )
        assert len(optimizer.param_groups) == 1
        # one linear layer with bias
        assert len(optimizer.param_groups[0]["params"]) == 2

        optimizer, _ = optimizer_factory(
            **self.sgd_config,
            keys_without_decay=[],
            model=self.large_model,
            layer_decay_lr=0.7,
        )
        assert len(optimizer.param_groups) == 7
        # 3 linear layers without biases, 1 linear layer with bias and 3 batch norms
        assert len(optimizer.param_groups[0]["params"]) == 1
        assert len(optimizer.param_groups[1]["params"]) == 1
        assert len(optimizer.param_groups[2]["params"]) == 1
        assert len(optimizer.param_groups[3]["params"]) == 2
        # num norm parameters
        assert len(optimizer.param_groups[4]["params"]) == 2
        assert len(optimizer.param_groups[5]["params"]) == 2
        assert len(optimizer.param_groups[6]["params"]) == 2

    def test_exclude_bias_and_norm_optimizer_factory(self):
        self.sgd_config.exclude_wd_norm = True
        self.sgd_config.exclude_wd_bias = True

        optimizer, _ = optimizer_factory(
            **self.sgd_config, keys_without_decay=[], model=self.base_model
        )
        assert len(optimizer.param_groups) == 2
        # one linear layer with bias
        assert len(optimizer.param_groups[0]["params"]) == 1
        # num biases
        assert len(optimizer.param_groups[1]["params"]) == 1

        optimizer, _ = optimizer_factory(
            **self.sgd_config, keys_without_decay=[], model=self.large_model
        )
        assert len(optimizer.param_groups) == 2
        # 3 linear layers without biases, 1 linear layer with bias and 3 batch norms
        assert len(optimizer.param_groups[0]["params"]) == 4
        # num norm + biases parameters
        assert len(optimizer.param_groups[1]["params"]) == 7

    def test_exclude_bias_and_norm_optimizer_factory_layer_decay(self):
        self.sgd_config.exclude_wd_norm = True
        self.sgd_config.exclude_wd_bias = True

        optimizer, _ = optimizer_factory(
            **self.sgd_config,
            keys_without_decay=[],
            model=self.base_model,
            layer_decay_lr=0.7,
        )
        assert len(optimizer.param_groups) == 2
        # one linear layer with bias
        assert len(optimizer.param_groups[0]["params"]) == 1
        # num biases
        assert len(optimizer.param_groups[1]["params"]) == 1

        optimizer, _ = optimizer_factory(
            **self.sgd_config,
            keys_without_decay=[],
            model=self.large_model,
            layer_decay_lr=0.7,
        )
        assert len(optimizer.param_groups) == 8
        # 3 linear layers without biases, 1 linear layer with bias and 3 batch norms
        assert len(optimizer.param_groups[0]["params"]) == 1
        assert len(optimizer.param_groups[1]["params"]) == 1
        assert len(optimizer.param_groups[2]["params"]) == 1
        assert len(optimizer.param_groups[3]["params"]) == 1
        # num norm + biases parameters
        assert len(optimizer.param_groups[4]["params"]) == 2
        assert len(optimizer.param_groups[5]["params"]) == 2
        assert len(optimizer.param_groups[6]["params"]) == 2
        assert len(optimizer.param_groups[7]["params"]) == 1

    def test_exclude_bias_via_key_and_norm_optimizer_factory(self):
        self.sgd_config.exclude_wd_norm = True
        self.sgd_config.exclude_wd_bias = False

        optimizer, _ = optimizer_factory(
            **self.sgd_config, keys_without_decay=["bias"], model=self.base_model
        )
        assert len(optimizer.param_groups) == 2
        # one linear layer with bias
        assert len(optimizer.param_groups[0]["params"]) == 1
        # num biases
        assert len(optimizer.param_groups[1]["params"]) == 1

        optimizer, _ = optimizer_factory(
            **self.sgd_config, keys_without_decay=["bias"], model=self.large_model
        )
        assert len(optimizer.param_groups) == 2
        # 3 linear layers without biases, 1 linear layer with bias and 3 batch norms
        assert len(optimizer.param_groups[0]["params"]) == 4
        # num norm + biases parameters
        assert len(optimizer.param_groups[1]["params"]) == 7

    def test_exclude_bias_via_key_and_norm_optimizer_factory_layer_decay(self):
        self.sgd_config.exclude_wd_norm = True
        self.sgd_config.exclude_wd_bias = False

        optimizer, _ = optimizer_factory(
            **self.sgd_config,
            keys_without_decay=["bias"],
            model=self.base_model,
            layer_decay_lr=0.7,
        )
        assert len(optimizer.param_groups) == 2
        # one linear layer with bias
        assert len(optimizer.param_groups[0]["params"]) == 1
        # num biases
        assert len(optimizer.param_groups[1]["params"]) == 1

        optimizer, _ = optimizer_factory(
            **self.sgd_config,
            keys_without_decay=["bias"],
            model=self.large_model,
            layer_decay_lr=0.7,
        )
        assert len(optimizer.param_groups) == 8
        # 3 linear layers without biases, 1 linear layer with bias and 3 batch norms
        assert len(optimizer.param_groups[0]["params"]) == 1
        assert len(optimizer.param_groups[1]["params"]) == 1
        assert len(optimizer.param_groups[2]["params"]) == 1
        assert len(optimizer.param_groups[3]["params"]) == 1
        # num norm + biases parameters
        assert len(optimizer.param_groups[4]["params"]) == 2
        assert len(optimizer.param_groups[5]["params"]) == 2
        assert len(optimizer.param_groups[6]["params"]) == 2
        assert len(optimizer.param_groups[7]["params"]) == 1


if __name__ == "__main__":
    unittest.main()
