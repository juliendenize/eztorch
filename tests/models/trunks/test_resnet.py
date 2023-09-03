import unittest

import torch
from torch import nn

from eztorch.models.trunks.resnet import _ResNets, create_resnet


class TestResnet(unittest.TestCase):
    def test_all_resnets(self):
        for resnet in _ResNets:
            create_resnet(resnet)

    def test_resnet_small_input_with_fc(self):
        resnet = create_resnet("resnet18", small_input=True)
        assert isinstance(resnet.fc, nn.Linear)
        assert resnet.conv1.kernel_size == (3, 3)
        assert resnet.conv1.stride == (1, 1)
        assert resnet.conv1.padding == (1, 1)
        assert isinstance(resnet.maxpool, nn.Identity)

    def test_resnet_small_input_without_fc(self):
        resnet = create_resnet("resnet18", small_input=True, num_classes=0)
        assert isinstance(resnet.fc, nn.Identity)
        assert resnet.conv1.kernel_size == (3, 3)
        assert resnet.conv1.stride == (1, 1)
        assert resnet.conv1.padding == (1, 1)
        assert isinstance(resnet.maxpool, nn.Identity)

    def test_resnet_large_input_with_fc(self):
        resnet = create_resnet("resnet18", small_input=False)
        assert isinstance(resnet.fc, nn.Linear)
        assert resnet.conv1.kernel_size == (7, 7)
        assert resnet.conv1.stride == (2, 2)
        assert resnet.conv1.padding == (3, 3)
        assert isinstance(resnet.maxpool, nn.MaxPool2d)

    def test_resnet_large_input_without_fc(self):
        resnet = create_resnet("resnet18", small_input=False, num_classes=0)
        assert isinstance(resnet.fc, nn.Identity)
        assert resnet.conv1.kernel_size == (7, 7)
        assert resnet.conv1.stride == (2, 2)
        assert resnet.conv1.padding == (3, 3)
        assert isinstance(resnet.maxpool, nn.MaxPool2d)

    def test_resnet_forward(self):
        resnet = create_resnet("resnet18")
        x = torch.rand((1, 3, 224, 224))
        resnet(x)


if __name__ == "__main__":
    unittest.main()
