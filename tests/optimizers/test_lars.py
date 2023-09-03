import unittest

from torch.optim import SGD

from eztorch.optimizers.lars import LARS
from tests.helpers.models import BoringModel


class TestOptimizerFactory(unittest.TestCase):
    def test_init_lars(self):
        model = BoringModel()
        LARS(model.parameters(), lr=0.1)
