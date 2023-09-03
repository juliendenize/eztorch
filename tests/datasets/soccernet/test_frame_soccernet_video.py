import unittest
from pathlib import Path

import torch

from eztorch.datasets.decoders.frame_soccernet_video import FrameSoccerNetVideo
from eztorch.datasets.utils_fn import get_video_to_frame_path_fn


class TestFrameSoccerNetVideo(unittest.TestCase):
    def setUp(self) -> None:
        self.default_args = {
            "video_path": Path("/video/"),
            "half_path": Path("/video/half1"),
            "transform": None,
            "video_frame_to_path_fn": get_video_to_frame_path_fn(zeros=8),
            "num_threads_io": 0,
        }

    def test_same_fps_get_timestamps_indices(self) -> None:
        video = FrameSoccerNetVideo(
            duration=2700, fps_video=2, fps=2, num_frames=5400, **self.default_args
        )
        (
            timestamps,
            frame_indices,
            fps_video_frame_indices,
        ) = video.get_timestamps_and_frame_indices(0.75, 15.02)

        expected_timestamps = torch.arange(0.5, 15, 0.5)
        expected_frame_indices = torch.arange(1, 15 * 2)
        expected_fps_video_frame_indices = expected_frame_indices

        assert torch.allclose(timestamps, expected_timestamps)
        assert torch.allclose(frame_indices, expected_frame_indices)
        assert torch.allclose(fps_video_frame_indices, expected_fps_video_frame_indices)

        video = FrameSoccerNetVideo(
            duration=2700, fps_video=2, fps=2, num_frames=5400, **self.default_args
        )
        (
            timestamps,
            frame_indices,
            fps_video_frame_indices,
        ) = video.get_timestamps_and_frame_indices(0.75, 15.52)

        expected_timestamps = torch.arange(0.5, 15.5, 0.5)
        expected_frame_indices = torch.arange(1, 15 * 2 + 1)
        expected_fps_video_frame_indices = expected_frame_indices

        assert torch.allclose(timestamps, expected_timestamps)
        assert torch.allclose(frame_indices, expected_frame_indices)
        assert torch.allclose(fps_video_frame_indices, expected_fps_video_frame_indices)

        video = FrameSoccerNetVideo(
            duration=2699.5, fps_video=2, fps=2, num_frames=5399, **self.default_args
        )
        (
            timestamps,
            frame_indices,
            fps_video_frame_indices,
        ) = video.get_timestamps_and_frame_indices(0.0, 2699.5)

        expected_timestamps = torch.arange(0.0, 2699.5, 0.5)
        expected_frame_indices = torch.arange(0, 2699 * 2 + 1)
        expected_fps_video_frame_indices = expected_frame_indices

        assert torch.allclose(timestamps, expected_timestamps)
        assert torch.allclose(frame_indices, expected_frame_indices)
        assert torch.allclose(fps_video_frame_indices, expected_fps_video_frame_indices)

    def test_different_fps_get_timestamps_indices(self) -> None:
        video = FrameSoccerNetVideo(
            duration=2700, fps_video=25, fps=2, num_frames=67500, **self.default_args
        )
        (
            timestamps,
            frame_indices,
            fps_video_frame_indices,
        ) = video.get_timestamps_and_frame_indices(0.75, 15.02)

        expected_timestamps = torch.arange(0.5, 15, 0.5)
        expected_frame_indices = torch.arange(1, 15 * 2)
        expected_fps_video_frame_indices = torch.floor(
            expected_frame_indices / 2 * 25
        ).to(dtype=torch.long)

        assert torch.allclose(timestamps, expected_timestamps)
        assert torch.allclose(frame_indices, expected_frame_indices)
        assert torch.allclose(fps_video_frame_indices, expected_fps_video_frame_indices)

        video = FrameSoccerNetVideo(
            duration=2700, fps_video=25, fps=2, num_frames=67500, **self.default_args
        )
        (
            timestamps,
            frame_indices,
            fps_video_frame_indices,
        ) = video.get_timestamps_and_frame_indices(0.75, 15.52)

        expected_timestamps = torch.arange(0.5, 15.5, 0.5)
        expected_frame_indices = torch.arange(1, 15 * 2 + 1)
        expected_fps_video_frame_indices = torch.floor(
            expected_frame_indices / 2 * 25
        ).to(dtype=torch.long)

        assert torch.allclose(timestamps, expected_timestamps)
        assert torch.allclose(frame_indices, expected_frame_indices)
        assert torch.allclose(fps_video_frame_indices, expected_fps_video_frame_indices)

        video = FrameSoccerNetVideo(
            duration=2699.96, fps_video=25, fps=2, num_frames=67499, **self.default_args
        )
        (
            timestamps,
            frame_indices,
            fps_video_frame_indices,
        ) = video.get_timestamps_and_frame_indices(0.0, 2699.96)

        expected_timestamps = torch.arange(0.0, 2699.5, 0.5)
        expected_frame_indices = torch.arange(0, 2699 * 2 + 1)
        expected_fps_video_frame_indices = torch.floor(
            expected_frame_indices / 2 * 25
        ).to(dtype=torch.long)

        assert torch.allclose(timestamps, expected_timestamps)
        assert torch.allclose(frame_indices, expected_frame_indices)
        assert torch.allclose(fps_video_frame_indices, expected_fps_video_frame_indices)

        video = FrameSoccerNetVideo(
            duration=2700, fps_video=25, fps=4, num_frames=67500, **self.default_args
        )
        (
            timestamps,
            frame_indices,
            fps_video_frame_indices,
        ) = video.get_timestamps_and_frame_indices(0.76, 15.02)

        expected_timestamps = torch.arange(0.75, 15, 0.25)
        expected_frame_indices = torch.arange(3, 15 * 4)
        expected_fps_video_frame_indices = torch.tensor(
            [
                19,
                25,
                31,
                37,
                44,
                50,
                56,
                62,
                69,
                75,
                81,
                87,
                94,
                100,
                106,
                112,
                119,
                125,
                131,
                137,
                144,
                150,
                156,
                162,
                169,
                175,
                181,
                187,
                194,
                200,
                206,
                212,
                219,
                225,
                231,
                237,
                244,
                250,
                256,
                262,
                269,
                275,
                281,
                287,
                294,
                300,
                306,
                312,
                319,
                325,
                331,
                337,
                344,
                350,
                356,
                362,
                369,
            ]
        )

        assert torch.allclose(timestamps, expected_timestamps)
        assert torch.allclose(frame_indices, expected_frame_indices)
        assert torch.allclose(fps_video_frame_indices, expected_fps_video_frame_indices)
