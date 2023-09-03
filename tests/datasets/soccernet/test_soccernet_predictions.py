import math
import unittest

import torch

from eztorch.datasets.soccernet_utils.parse_utils import \
    REVERSE_ACTION_SPOTTING_LABELS
from eztorch.datasets.soccernet_utils.predictions import (
    aggregate_predictions, postprocess_spotting_half_predictions)


class TestSoccerNetPredictions(unittest.TestCase):
    def test_aggregate_predictions(self):

        predictions = torch.tensor(
            [
                [0.5, 0.6],
                [0.4, 0.7],
                [0.7, 0.3],
                [0.3, 0.5],
                [0.8, 0.2],
                [0.9, 0.9],
            ]
        )

        timestamps = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [8.0, 2.0],
                [8.0, 8.0],
            ]
        )

        expected_predictions = torch.tensor(
            [
                [0.3, 0.6],
                [0.5, 0.7],
                [0.7, 0.0],
                [0.0, 0.0],
                [0.8, 0.0],
                [0.0, 0.5],
                [0.0, 0.0],
                [0.9, 0.9],
            ]
        )

        expected_timestamps = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

        new_predictions, new_timestamps = aggregate_predictions(
            predictions, timestamps, 8, 1.0
        )

        assert torch.allclose(expected_predictions, new_predictions)
        assert torch.allclose(expected_timestamps, new_timestamps)

    def test_aggregate_predictions_with_ignore(self):

        predictions = torch.tensor(
            [
                [0.5, 0.6],
                [0.4, 0.7],
                [0.7, 0.3],
                [0.3, 0.5],
                [0.8, 0.2],
                [0.9, 0.9],
            ]
        )

        timestamps = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [8.0, 2.0],
                [8.0, 8.0],
            ]
        )

        expected_predictions = torch.tensor(
            [
                [0.0, 0.6],
                [0.5, 0.7],
                [0.7, 0.0],
                [0.0, 0.0],
                [0.8, 0.0],
                [0.0, 0.5],
                [0.0, 0.0],
                [0.9, 0.0],
                [0.0, 0.0],
            ]
        )

        expected_timestamps = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])

        new_predictions, new_timestamps = aggregate_predictions(
            predictions, timestamps, 8, 1.0, True
        )

        assert torch.allclose(expected_predictions, new_predictions)
        assert torch.allclose(expected_timestamps, new_timestamps)

    def test_postprocess_spotting_half_predictions(self):
        predictions = torch.tensor(
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

        NMS_args = {
            "window": 3,
            "threshold": 0.49,
        }

        half_id = 0

        half_predictions = postprocess_spotting_half_predictions(
            predictions,
            half_id,
            15,
            NMS_args,
        )

        expected_half_predictions = [
            {
                "gameTime": f"{half_id} - {int(15 / 60):02d}:{int(15 % 60)}",
                "label": REVERSE_ACTION_SPOTTING_LABELS[0],
                "position": f"{int(15000)}",
                "half": str(half_id),
                "confidence": float(1.0),
            },
            {
                "gameTime": f"{half_id} - {int(90 / 60):02d}:{int(90 % 60)}",
                "label": REVERSE_ACTION_SPOTTING_LABELS[0],
                "position": f"{int(90000)}",
                "half": str(half_id),
                "confidence": float(0.6),
            },
            {
                "gameTime": f"{half_id} - {int(30 / 60):02d}:{int(30 % 60)}",
                "label": REVERSE_ACTION_SPOTTING_LABELS[1],
                "position": f"{int(30000)}",
                "half": str(half_id),
                "confidence": float(0.8),
            },
            {
                "gameTime": f"{half_id} - {int(75 / 60):02d}:{int(75 % 60)}",
                "label": REVERSE_ACTION_SPOTTING_LABELS[1],
                "position": f"{int(75000)}",
                "half": str(half_id),
                "confidence": float(0.9),
            },
        ]

        for expected, actual in zip(expected_half_predictions, half_predictions):
            for key in expected:
                if type(expected[key]) is float:
                    assert math.isclose(
                        expected[key], actual[key], rel_tol=1e-6, abs_tol=0.0
                    )
                else:
                    assert expected[key] == actual[key]
