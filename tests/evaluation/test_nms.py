import unittest

import torch

from eztorch.evaluation.nms import (perform_all_classes_NMS, perform_hard_NMS,
                                    perform_soft_NMS)


class TestSCETokenMasks(unittest.TestCase):
    def test_perform_hard_NMS(self):
        values = torch.tensor(
            [
                0.5,
                1.0,
                0.2,
                0.3,
                0.1,
                0.2,
                0.6,
            ]
        )

        window = 3
        threshold = 0.49

        keep_indexes = perform_hard_NMS(values, window, threshold)

        expected_keep_indexes = torch.tensor(
            [False, True, False, False, False, False, True]
        )

        assert torch.allclose(keep_indexes, expected_keep_indexes)

    def test_perform_hard_NMS_longer_window(self):
        values = torch.tensor(
            [
                0.5,
                1.0,
                0.2,
                0.3,
                0.1,
                0.2,
                0.6,
            ]
        )

        window = 14
        threshold = 0.49

        keep_indexes = perform_hard_NMS(values, window, threshold)

        expected_keep_indexes = torch.tensor(
            [False, True, False, False, False, False, False]
        )

        assert torch.allclose(keep_indexes, expected_keep_indexes)

    def test_perform_all_classes_hard_NMS(self):
        values = torch.tensor(
            [
                [0.5, 0.3],
                [1.0, 0.6],
                [0.2, 0.8],
                [0.3, 0.2],
                [0.1, 0.1],
                [0.2, 0.9],
                [0.6, 0.8],
            ]
        )

        window = 3
        threshold = 0.49
        step_timestamp = 1.0

        kept_values, kept_timestamps_per_class = perform_all_classes_NMS(
            values, step_timestamp, window, threshold, nms_type="hard"
        )

        expected_kept_values = [
            torch.tensor(
                [
                    1.0,
                    0.6,
                ]
            ),
            torch.tensor([0.8, 0.9]),
        ]

        expected_kept_timestamps_per_class = [
            torch.tensor([1.0, 6.0]),
            torch.tensor([2.0, 5.0]),
        ]

        assert all(
            [
                torch.allclose(kept_value, expected_kept_value)
                for kept_value, expected_kept_value in zip(
                    kept_values, expected_kept_values
                )
            ]
        )
        assert all(
            [
                torch.allclose(timestamp_per_class, expected_timestamp_per_class)
                for timestamp_per_class, expected_timestamp_per_class in zip(
                    kept_timestamps_per_class, expected_kept_timestamps_per_class
                )
            ]
        )

    def test_perform_all_classes_hard_NMS_step_timestamp(self):
        values = torch.tensor(
            [
                [0.5, 0.3],
                [1.0, 0.6],
                [0.2, 0.8],
                [0.3, 0.2],
                [0.1, 0.1],
                [0.2, 0.9],
                [0.6, 0.8],
            ]
        )

        window = 3
        threshold = 0.49
        step_timestamp = 0.5

        kept_values, kept_timestamps_per_class = perform_all_classes_NMS(
            values, step_timestamp, window, threshold, nms_type="hard"
        )

        expected_kept_values = [
            torch.tensor(
                [
                    1.0,
                    0.6,
                ]
            ),
            torch.tensor([0.8, 0.9]),
        ]

        expected_kept_timestamps_per_class = [
            torch.tensor([0.5, 3.0]),
            torch.tensor([1.0, 2.5]),
        ]

        assert all(
            [
                torch.allclose(kept_value, expected_kept_value)
                for kept_value, expected_kept_value in zip(
                    kept_values, expected_kept_values
                )
            ]
        )
        assert all(
            [
                torch.allclose(timestamp_per_class, expected_timestamp_per_class)
                for timestamp_per_class, expected_timestamp_per_class in zip(
                    kept_timestamps_per_class, expected_kept_timestamps_per_class
                )
            ]
        )

    def test_perform_soft_NMS(self):
        values = torch.tensor(
            [
                0.0001,
                1.0,
                0.2,
                0.3,
                0.1,
                0.2,
                0.6,
            ]
        )

        window = 3
        threshold = 0.001

        decayed_values = perform_soft_NMS(values, window, threshold)
        print(decayed_values)

        # TODO: Test output value

    def test_perform_all_classes_soft_NMS_step_timestamp(self):
        values = torch.tensor(
            [
                [0.5, 0.3],
                [1.0, 0.6],
                [0.2, 0.8],
                [0.3, 0.2],
                [0.1, 0.1],
                [0.2, 0.9],
                [0.6, 0.8],
            ]
        )

        window = 3
        threshold = 0.2
        step_timestamp = 0.5

        kept_values, kept_timestamps_per_class = perform_all_classes_NMS(
            values, step_timestamp, window, threshold, nms_type="soft"
        )

        print(kept_values, kept_timestamps_per_class)

        # TODO: Test output value
