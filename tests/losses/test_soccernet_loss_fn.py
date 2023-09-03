import unittest

import torch

from eztorch.losses.spot_loss import (compute_spot_loss_fn,
                                      get_spot_bce_weights,
                                      get_spot_softmax_weights)

SOCCERNET_CLASS_WEIGHTS = torch.tensor([0.3, 0.7])


class TestSpotLossFn(unittest.TestCase):
    def test_get_spot_bce_weights(self):
        bce_target = torch.tensor(
            [
                [[0.3, 0.5], [0.7, 0.5]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ]
        )

        bce_has_label = torch.Tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ]
        ).bool()

        class_weights = torch.tensor([[1.0, 3], [2, 4]])

        ignore_class = torch.zeros(bce_target.shape, dtype=torch.bool)

        bce_weights, num_bce = get_spot_bce_weights(
            class_target=bce_target,
            has_class_target=bce_has_label,
            class_weights=class_weights,
            ignore_class=ignore_class,
        )

        expected_bce_weights = class_weights[1:2, :] * torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ]
        ) + class_weights[0:1, :] * torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 1.0], [0.0, 1.0]],
                [[1.0, 1.0], [1.0, 0.0]],
            ]
        )

        assert torch.allclose(bce_weights, expected_bce_weights)
        assert num_bce == 12

    def test_get_spot_bce_weights_with_ignore(self):
        bce_target = torch.tensor(
            [
                [[0.3, 0.5], [0.7, 0.5]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ]
        )

        bce_has_label = torch.Tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ]
        ).bool()

        class_weights = torch.tensor([[1.0, 3], [2, 4]])

        ignore_class = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 1.0]],
            ],
            dtype=torch.bool,
        )

        bce_weights, num_bce = get_spot_bce_weights(
            class_target=bce_target,
            has_class_target=bce_has_label,
            class_weights=class_weights,
            ignore_class=ignore_class,
        )

        expected_bce_weights = class_weights[1:2, :] * torch.tensor(
            [
                [[0.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ) + class_weights[0:1, :] * torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 1.0], [0.0, 1.0]],
                [[1.0, 0.0], [1.0, 0.0]],
            ]
        )

        assert torch.allclose(bce_weights, expected_bce_weights)
        assert num_bce == 9

    def test_get_spot_bce_weights_none_class_weights(self):
        bce_target = torch.tensor(
            [
                [[0.3, 0.5], [0.7, 0.5]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ]
        )

        bce_has_label = torch.Tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ]
        ).bool()

        bce_weights, num_bce = get_spot_bce_weights(
            class_target=bce_target,
            has_class_target=bce_has_label,
            class_weights=None,
            ignore_class=torch.zeros(bce_target.shape, dtype=torch.bool),
        )

        expected_bce_weights = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[1.0, 1.0], [1.0, 1.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ]
        )

        assert torch.allclose(bce_weights, expected_bce_weights)
        assert num_bce == 12

    def test_get_spot_bce_weights_none_class_weights_with_ignore(self):
        bce_target = torch.tensor(
            [
                [[0.3, 0.5], [0.7, 0.5]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ]
        )

        bce_has_label = torch.Tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ]
        ).bool()

        ignore_class = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 1.0]],
            ],
            dtype=torch.bool,
        )

        bce_weights, num_bce = get_spot_bce_weights(
            class_target=bce_target,
            has_class_target=bce_has_label,
            class_weights=None,
            ignore_class=ignore_class,
        )

        expected_bce_weights = torch.tensor(
            [
                [[0.0, 1.0], [1.0, 1.0]],
                [[1.0, 1.0], [1.0, 1.0]],
                [[1.0, 0.0], [1.0, 0.0]],
            ]
        )

        assert torch.allclose(bce_weights, expected_bce_weights)
        assert num_bce == 9

    def test_compute_spot_loss_fn(self):
        bce_target = torch.tensor(
            [
                [[0.3, 0.5], [0.7, 0.5]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ]
        )

        bce_has_label = torch.Tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ]
        ).bool()

        b, t, c = bce_target.shape

        bce_preds = torch.randn((b, t, c))
        bce_target = torch.cat((bce_target, bce_target))
        bce_has_label = torch.cat((bce_has_label, bce_has_label))

        ignore_class = torch.randint(
            0,
            2,
            (b * 2, t, c),
        )

        weights = torch.tensor(
            [
                [[0.3]],
                [[0.0]],
                [[1.0]],
                [[0.7]],
                [[1.0]],
                [[0.0]],
            ]
        )

        compute_spot_loss_fn(
            bce_preds,
            bce_target,
            bce_has_label,
            ignore_class,
            SOCCERNET_CLASS_WEIGHTS[:2].reshape(1, -1).repeat(2, 1),
            weights,
        )

    def test_compute_spot_loss_fn_with_mixup_weights(self):
        bce_target = torch.tensor(
            [
                [[0.3, 0.5], [0.7, 0.5]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ]
        )

        bce_has_label = torch.Tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ]
        ).bool()

        b, t, c = bce_target.shape

        bce_preds = torch.randn((b, t, c))
        bce_target = torch.cat((bce_target, bce_target))
        bce_has_label = torch.cat((bce_has_label, bce_has_label))

        ignore_class = torch.randint(
            0,
            2,
            (b * 2, t, c),
        )
        mixup_weights = torch.tensor(
            [
                [[0.3]],
                [[0.0]],
                [[1.0]],
                [[0.7]],
                [[1.0]],
                [[0.0]],
            ]
        )

        compute_spot_loss_fn(
            bce_preds,
            bce_target,
            bce_has_label,
            ignore_class,
            None,
            mixup_weights,
        )

    def test_compute_spot_loss_fn_with_class_weights(self):
        bce_target = torch.tensor(
            [
                [[0.3, 0.5], [0.7, 0.5]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ]
        )

        bce_has_label = torch.Tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ]
        ).bool()

        bce_preds = torch.randn(bce_target.shape)

        ignore_class = torch.randint(
            0,
            2,
            bce_target.shape,
        )

        compute_spot_loss_fn(
            bce_preds,
            bce_target,
            bce_has_label,
            ignore_class,
            SOCCERNET_CLASS_WEIGHTS[:2].reshape(1, -1).repeat(2, 1),
            None,
        )

    def test_compute_spot_loss_fn_without_class_weights_without_all(self):
        bce_target = torch.tensor(
            [
                [[0.3, 0.5], [0.7, 0.5]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ]
        )

        bce_has_label = torch.Tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ]
        ).bool()

        bce_preds = torch.randn(bce_target.shape)

        ignore_class = torch.randint(
            0,
            2,
            bce_target.shape,
        )

        compute_spot_loss_fn(
            bce_preds,
            bce_target,
            bce_has_label,
            ignore_class,
            None,
            None,
        )
