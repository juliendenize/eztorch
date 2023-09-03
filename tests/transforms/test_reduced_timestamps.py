import unittest

import torch

from eztorch.transforms.video.soccernet.reduce_timestamps import (
    BatchReduceTimestamps, ReduceTimestamps)


class TestReducedTimestamps(unittest.TestCase):
    def test_batch_reduced_timestamps(self):
        x = torch.tensor(
            [
                [[0.5, 0.8], [0.4, 0.6]],
                [[0.5, 0.8], [0.4, 0.6]],
                [[0.2, 0.8], [0.4, 0.6]],
                [[0.2, 0.8], [0.4, 0.6]],
                [[0.2, 0.8], [0.4, 0.6]],
                [[0.2, 0.8], [0.4, 0.6]],
            ]
        )

        labels = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
            ]
        )

        has_label = labels.bool()

        ignore_class = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )

        timestamps = torch.tensor(
            [
                [0.0, 0.48, 1.0, 1.48],
                [0.0, 0.5, 1.0, 1.5],
                [2.0, 2.5, 3.0, 3.5],
                [4.0, 4.5, 5.0, 5.5],
                [6.0, 6.48, 7.0, 7.5],
                [6.0, 6.5, 7.0, 7.5],
            ]
        )

        (reduced_batch) = BatchReduceTimestamps()(
            {
                "input": x,
                "labels": labels,
                "has_label": has_label,
                "ignore_class": ignore_class,
                "timestamps": timestamps,
            }
        )

        expected_reduced_batch = {
            "input": x,
            "labels": torch.tensor(
                [
                    [[1.0, 0.0], [1.0, 1.0]],
                    [[1.0, 0.0], [1.0, 1.0]],
                    [[0.0, 1.0], [1.0, 1.0]],
                    [[0.0, 1.0], [1.0, 1.0]],
                    [[0.0, 1.0], [0.0, 1.0]],
                    [[0.0, 1.0], [0.0, 1.0]],
                ]
            ),
            "has_label": torch.tensor(
                [
                    [[1.0, 0.0], [1.0, 1.0]],
                    [[1.0, 0.0], [1.0, 1.0]],
                    [[0.0, 1.0], [1.0, 1.0]],
                    [[0.0, 1.0], [1.0, 1.0]],
                    [[0.0, 1.0], [0.0, 1.0]],
                    [[0.0, 1.0], [0.0, 1.0]],
                ]
            ).bool(),
            "ignore_class": torch.tensor(
                [
                    [[0, 0], [0, 0]],
                    [[0, 0], [0, 0]],
                    [[0, 0], [0, 0]],
                    [[0, 1], [0, 0]],
                    [[0, 0], [0, 0]],
                    [[0, 0], [0, 0]],
                ]
            ).to(torch.bool),
            "timestamps": torch.tensor(
                [
                    [0.24, 1.24],
                    [0.25, 1.25],
                    [2.25, 3.25],
                    [4.25, 5.25],
                    [6.24, 7.25],
                    [6.25, 7.25],
                ]
            ),
        }

        for key, value in expected_reduced_batch.items():
            assert torch.allclose(reduced_batch[key], value, rtol=0.001, atol=0.001)

    def test_reduced_timestamps(self):
        x = torch.tensor(
            [
                [[0.5, 0.8], [0.4, 0.6]],
                [[0.2, 0.8], [0.4, 0.6]],
                [[0.2, 0.8], [0.4, 0.6]],
                [[0.2, 0.8], [0.4, 0.6]],
            ]
        )

        labels = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
            ]
        )

        has_label = labels.bool()

        ignore_class = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )

        timestamps = torch.tensor(
            [
                [0.0, 0.5, 1.0, 1.5],
                [2.0, 2.5, 3.0, 3.5],
                [4.0, 4.5, 5.0, 5.5],
                [6.0, 6.5, 7.0, 7.5],
            ]
        )

        expected_reduced_batch = {
            "input": x,
            "labels": torch.tensor(
                [
                    [[1.0, 0.0], [1.0, 1.0]],
                    [[0.0, 1.0], [1.0, 1.0]],
                    [[0.0, 1.0], [1.0, 1.0]],
                    [[0.0, 1.0], [0.0, 1.0]],
                ]
            ),
            "has_label": torch.tensor(
                [
                    [[1.0, 0.0], [1.0, 1.0]],
                    [[0.0, 1.0], [1.0, 1.0]],
                    [[0.0, 1.0], [1.0, 1.0]],
                    [[0.0, 1.0], [0.0, 1.0]],
                ]
            ).bool(),
            "timestamps": torch.tensor(
                [
                    [0.25, 1.25],
                    [2.25, 3.25],
                    [4.25, 5.25],
                    [6.25, 7.25],
                ]
            ),
        }

        for i in range(len(x)):
            (reduced) = ReduceTimestamps()(
                {
                    "input": x[i],
                    "labels": labels[i],
                    "has_label": has_label[i],
                    "ignore_class": ignore_class[i],
                    "timestamps": timestamps[i],
                }
            )

            for key, value in expected_reduced_batch.items():
                assert torch.allclose(reduced[key], value[i], rtol=0.001, atol=0.001)
