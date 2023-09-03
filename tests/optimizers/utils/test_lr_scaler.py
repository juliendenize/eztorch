import math
import unittest

from eztorch.optimizers.utils import scale_learning_rate


class TestLrScaler(unittest.TestCase):
    def setUp(self) -> None:
        self.initial_lr = 2.0
        self.batch_size = 16

    def test_none_scaler(self) -> None:
        lr = scale_learning_rate(self.initial_lr, None, self.batch_size)
        assert lr == self.initial_lr
        lr = scale_learning_rate(self.initial_lr, "none", self.batch_size)
        assert lr == self.initial_lr

    def test_linear_scaler(self) -> None:
        lr = scale_learning_rate(self.initial_lr, "linear", self.batch_size)
        assert lr == self.initial_lr * self.batch_size / 256

    def test_sqrt_scaler(self) -> None:
        lr = scale_learning_rate(self.initial_lr, "sqrt", self.batch_size)
        assert lr == self.initial_lr * math.sqrt(self.batch_size)


if __name__ == "__main__":
    unittest.main()
