from __future__ import annotations

import logging
import random
from fractions import Fraction
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.utils.data

from eztorch.datasets.decoders.frame_video import GeneralFrameVideo
from eztorch.datasets.utils_fn import (get_time_difference_indices,
                                       get_video_to_frame_path_fn)
from eztorch.transforms.video.temporal_difference_transform import \
    temporal_difference
from eztorch.utils.mask import mask_tube_in_sequence

logger = logging.getLogger(__name__)


class FrameSoccerNetVideo(GeneralFrameVideo):
    """FrameSoccerNetVideo is an abstractions for accessing clips based on their start and end time for a video
    where each frame is stored as an image.

    Args:
        video_path: The path of the video.
        half_path: The path of the half.
        duration: The duration of the video in seconds.
        fps_video: The fps of the video.
        fps: The target fps for the video. This is needed to link the frames
            to a second timestamp in the video.
        num_frames: The number of frames of the video.
        transform: The transform to apply to the frames.
        video_frame_to_path_fn: A function that maps from the video path and a frame
            index integer to the file path where the frame is located.
        time_difference_prob: Probability to apply time difference.
        num_threads_io: Controls whether parallelizable io operations are
            performed across multiple threads.
        num_threads_decode: Controls whether parallelizable decode operations are
            performed across multiple threads.
        num_decode: Number of decode to perform. If > 1, the videos decoded are stored in a list.
        num_decode_time_diff: Number of time difference decode to perform. If > 1, the videos decoded are stored in a list.
        mask_ratio: Masking ratio for the video.
        mask_ratio: Sequence tube size for masking the video.
        min_clip_duration: The minimum duration of a clip.
        decode_float: Whether to decode the clip as float.
    """

    def __init__(
        self,
        video_path: str | Path,
        half_path: str | Path,
        duration: float,
        fps_video: int,
        fps: int,
        num_frames: int,
        transform: Callable | None = None,
        video_frame_to_path_fn: Callable[[str, int], int] = get_video_to_frame_path_fn(
            zeros=8
        ),
        time_difference_prob: float = 0.0,
        num_threads_io: int = 0,
        num_threads_decode: int = 0,
        num_decode: int = 1,
        num_decode_time_diff: int = 2,
        mask_ratio: float = 0.0,
        mask_tube: int = 2,
        min_clip_duration: float = 0,
        decode_float: bool = False,
    ) -> None:

        super().__init__(
            num_threads_io=num_threads_io,
            num_threads_decode=num_threads_decode,
            transform=transform,
        )

        self._decode_float = decode_float
        self._duration = duration
        self._fps_video = fps_video
        self._fps = fps

        self._different_fps = self._fps_video != self._fps

        assert self._fps_video >= self._fps

        self._num_frames = num_frames
        self._time_difference_prob = time_difference_prob

        self._video_frame_to_path_fn = video_frame_to_path_fn

        self._video_path = video_path
        self._half_path = half_path
        self._name = Path(Path(self._video_path).name) / Path(self._half_path.name)

        self._num_decode = num_decode
        self._num_decode_time_diff = num_decode_time_diff
        self._mask_ratio = mask_ratio
        self._mask_tube = mask_tube
        self._min_clip_duration = min_clip_duration

    @property
    def name(self) -> str:
        """The name of the video."""
        return self._name

    @property
    def duration(self) -> float:
        """The video's duration/end-time in seconds."""
        return self._duration

    def _get_frame_index_for_time(self, time_sec: float, fps: int) -> int:
        return round(fps * time_sec)

    def get_timestamps_and_frame_indices(
        self,
        start_sec: float,
        end_sec: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieves timestamps and frame indices from the stored video at the specified start and end times in
        seconds.

        Args:
            start_sec: The clip start time in seconds.
            end_sec: The clip end time in seconds.

        Returns:
            The timestamps and the frame indices for video and decoded FPS.
        """
        if start_sec < 0 or start_sec > self._duration or end_sec > self._duration:
            logger.warning(
                f"No frames found within {start_sec} and {end_sec} seconds. Video starts"
                f"at time 0 and ends at {self._duration}."
            )
            return None

        frac_fps = Fraction(self._fps)
        frac_fps_video = Fraction(self._fps_video)
        over_frac_fps_video = Fraction(1, frac_fps_video)

        # Round the clip start_sec and end_sec to the fps possible values
        start_sec = Fraction(Fraction(int(start_sec * self._fps)), frac_fps)
        end_sec = Fraction(Fraction(int(end_sec * self._fps)), frac_fps)

        if self._different_fps:
            # Round the clip start_sec and end_sec to the fps video possible values
            if start_sec % self._fps_video != 0:
                start_sec = Fraction(
                    Fraction(int(start_sec * self._fps_video)), frac_fps_video
                )
            if end_sec % over_frac_fps_video != 0:
                end_sec = Fraction(
                    Fraction(int(end_sec * self._fps_video)), frac_fps_video
                )

        start_frame_index = self._get_frame_index_for_time(start_sec, self._fps_video)
        end_frame_index = self._get_frame_index_for_time(end_sec, self._fps_video)
        fps_video_frame_indices = torch.arange(start_frame_index, end_frame_index)

        timestamps = fps_video_frame_indices * float(over_frac_fps_video)

        if self._different_fps:
            keep_indices = torch.tensor(
                [
                    i
                    for i in range(0, self._fps_video)
                    for j in range(0, self._fps)
                    if round(Fraction(frac_fps_video, frac_fps) * j) - i == 0
                ]
            )
            keep_timestamp = torch.isin(
                fps_video_frame_indices % self._fps_video, keep_indices
            )

            fps_video_frame_indices = fps_video_frame_indices[keep_timestamp]
            timestamps = timestamps[keep_timestamp]

            # Round the timestamps back to the fps possible values.
            timestamps = (timestamps * self._fps).round() / self._fps

            frame_indices = (
                torch.arange(timestamps[0] * self._fps, timestamps[-1] * self._fps + 1)
                .round()
                .to(dtype=torch.long)
            )

        else:
            frame_indices = fps_video_frame_indices

        if (
            self._min_clip_duration > 0
            and len(frame_indices) < self._min_clip_duration * self._fps
        ):
            num_lacking_frames = self._min_clip_duration * self._fps - len(
                frame_indices
            )
            if start_frame_index == 0:
                fps_video_frame_indices = torch.cat(
                    [
                        torch.zeros(
                            num_lacking_frames, dtype=fps_video_frame_indices.dtype
                        ),
                        fps_video_frame_indices,
                    ]
                )
                frame_indices = torch.cat(
                    [
                        torch.zeros(num_lacking_frames, dtype=frame_indices.dtype),
                        frame_indices,
                    ]
                )
                timestamps = torch.cat(
                    [
                        torch.zeros(num_lacking_frames, dtype=timestamps.dtype),
                        timestamps,
                    ]
                )
            else:
                fps_video_frame_indices = torch.cat(
                    [
                        fps_video_frame_indices,
                        torch.tensor(
                            [
                                fps_video_frame_indices[-1]
                                for _ in range(num_lacking_frames)
                            ],
                            dtype=fps_video_frame_indices.dtype,
                        ),
                    ]
                )
                frame_indices = torch.cat(
                    [
                        frame_indices,
                        torch.tensor(
                            [frame_indices[-1] for _ in range(num_lacking_frames)],
                            dtype=frame_indices.dtype,
                        ),
                    ]
                )
                timestamps = torch.cat(
                    [
                        timestamps,
                        torch.tensor(
                            [timestamps[-1] for _ in range(num_lacking_frames)],
                            dtype=timestamps.dtype,
                        ),
                    ]
                )

        return timestamps, frame_indices, fps_video_frame_indices

    def get_clip(
        self,
        start_sec: float,
        end_sec: float,
    ) -> dict[str, torch.Tensor | None | list[torch.Tensor]]:
        """Retrieves frames from the stored video at the specified start and end times in seconds.

        Args:
            start_sec: The clip start time in seconds
            end_sec: The clip end time in seconds

        Returns:
            A dictionary containing the clip data and information.
        """

        (
            timestamps,
            frame_indices,
            fps_video_frame_indices,
        ) = self.get_timestamps_and_frame_indices(start_sec, end_sec)

        if self._mask_ratio > 0:
            t = timestamps.shape[0]

            (
                _,
                indices_kept,
                inversed_temporal_masked_indices,
                _,
            ) = mask_tube_in_sequence(self._mask_ratio, self._mask_tube, t, "cpu")

            frame_indices_to_decode = frame_indices[indices_kept]
            fps_video_frame_indices_to_decode = fps_video_frame_indices[indices_kept]

        else:
            frame_indices_to_decode = frame_indices
            fps_video_frame_indices_to_decode = fps_video_frame_indices

        if self._time_difference_prob > 0.0:
            decode_time_difference = [
                i < self._num_decode_time_diff
                and random.random() < self._time_difference_prob
                for i in range(self._num_decode)
            ]
            do_time_difference = any(decode_time_difference)
        else:
            decode_time_difference = [False for _ in range(self._num_decode)]
            do_time_difference = False

        if do_time_difference:
            (
                fps_video_frame_indices_to_decode,
                keep_frames,
            ) = get_time_difference_indices(
                np.array(fps_video_frame_indices_to_decode),
                True,
                self._num_frames - 1,
            )

            clip_paths = [
                self._video_frame_to_path(i) for i in fps_video_frame_indices_to_decode
            ]
            clip_frames = self._load_images_with_retries(
                clip_paths,
            )
        else:
            clip_paths = [
                self._video_frame_to_path(i) for i in fps_video_frame_indices_to_decode
            ]
            clip_frames = self._load_images_with_retries(
                clip_paths,
            )

        clip_frames = clip_frames.permute(1, 0, 2, 3)

        if do_time_difference:
            time_clip_frames = temporal_difference(
                clip_frames.to(dtype=torch.float32),
                use_grayscale=True,
                absolute=False,
            )[:, keep_frames, :, :]
            clip_frames = clip_frames[:, keep_frames, :, :]
            fps_video_frame_indices_to_decode = fps_video_frame_indices_to_decode[
                keep_frames
            ]

        if self._decode_float:
            clip_frames = clip_frames.to(torch.float32)
            if do_time_difference:
                time_clip_frames = time_clip_frames.to(torch.float32)

        if self._num_decode > 1:
            videos = [
                time_clip_frames if has_time_difference else clip_frames
                for has_time_difference in decode_time_difference
            ]
        else:
            videos = time_clip_frames if do_time_difference else clip_frames

        out = {
            "video": videos,
            "clip_start": timestamps[0].item(),
            "clip_end": timestamps[-1].item(),
            "clip_duration": (timestamps[-1] - timestamps[0]).item(),
            "frame_indices": frame_indices,
            "fps_video_frame_indices": fps_video_frame_indices,
            "time_difference": do_time_difference,
            "timestamps": timestamps,
        }

        if self._mask_ratio > 0.0:
            out["inversed_temporal_masked_indices"] = inversed_temporal_masked_indices

        return out

    def _video_frame_to_path(self, frame_index: int) -> str:
        return self._video_frame_to_path_fn(self._half_path, frame_index)
