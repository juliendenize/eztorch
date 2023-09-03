import unittest

import torch

from eztorch.transforms.video.spot.spotting_mixup import (SpottingMixup,
                                                          mix_spotting)


class TestActionSpottingMixup(unittest.TestCase):
    def test_mix_action_spotting(self):
        x = torch.tensor(
            [
                [[0.5, 0.8], [0.4, 0.6]],
                [[0.2, 0.8], [0.4, 0.6]],
                [[0.5, 0.8], [0.4, 0.5]],
                [[0.4, 0.2], [0.2, 0.5]],
            ]
        )

        mix_value = torch.tensor(
            [
                [[0.5]],
                [[0.3]],
                [[0.7]],
                [[0.2]],
            ]
        )

        permutation = torch.tensor([1, 0, 2, 3])

        labels = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 0.0]],
                [[1.0, 1.0], [0.0, 1.0]],
            ]
        )

        ignore_class = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ],
        )

        (
            x_mixed,
            labels_mixed,
            has_label_mixed,
            ignore_class_mixed,
            mixed_weights,
        ) = mix_spotting(
            x,
            mix_value,
            permutation,
            labels,
            labels.bool(),
            ignore_class,
        )

        expected_x_mixed = mix_value * x + (1 - mix_value) * x[permutation]

        expected_labels_mixed = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 0.0]],
                [[1.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 0.0]],
                [[1.0, 1.0], [0.0, 1.0]],
            ]
        )

        expected_has_label_mixed = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 0.0]],
                [[1.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 0.0]],
                [[1.0, 1.0], [0.0, 1.0]],
            ]
        ).bool()

        expected_ignore_class_mixed = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ]
        )

        expected_mixed_weights = torch.tensor(
            [
                [[0.5]],
                [[0.3]],
                [[0.7]],
                [[0.2]],
                [[0.5]],
                [[0.7]],
                [[0.3]],
                [[0.8]],
            ]
        )

        assert torch.allclose(x_mixed, expected_x_mixed)
        assert torch.allclose(labels_mixed, expected_labels_mixed)
        assert torch.allclose(has_label_mixed, expected_has_label_mixed)
        assert torch.allclose(ignore_class_mixed, expected_ignore_class_mixed)
        assert torch.allclose(mixed_weights, expected_mixed_weights)

    def test_action_spotting_mixup(self):

        spotting_mixup = SpottingMixup(alpha=0.5)
        x = torch.tensor(
            [
                [[0.5, 0.8], [0.4, 0.6]],
                [[0.2, 0.8], [0.4, 0.6]],
                [[0.5, 0.8], [0.4, 0.5]],
                [[0.4, 0.2], [0.2, 0.5]],
            ]
        )

        labels = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 0.0]],
                [[1.0, 1.0], [0.0, 1.0]],
            ]
        )

        ignore_class = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ]
        )

        spotting_mixup(
            batch={
                "input": x,
                "labels": labels,
                "has_label": labels.bool(),
                "ignore_class": ignore_class,
            }
        )
