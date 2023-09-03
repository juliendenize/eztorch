from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path

import torch
import torch.utils.data
from pytorchvideo.data.video import Video

logger = logging.getLogger(__name__)


class DumbSoccerNetVideo(Video):
    """DumbSoccerNetVideo is an abstractions for accessing clips based on their start and end time for a video
    where each frame is randomly generated.

    Args:
        video_path: The path of the video.
        half_path: The path of the half.
        duration: The duration of the video in seconds.
        fps_video: The fps of the video.
        fps: The target fps for the video. This is needed to link the frames
            to a second timestamp in the video.
        num_frames: The number of frames of the video.
        min_clip_duration: The minimum duration of a clip.
        num_decode: Number of duplicate output clip.
    """

    def __init__(
        self,
        video_path: str | Path,
        half_path: str | Path,
        duration: float,
        fps_video: int,
        fps: int,
        num_frames: int,
        min_clip_duration: float,
        num_decode: int,
        **kwargs,
    ) -> None:

        self._duration = duration
        self._fps_video = fps_video
        self._fps = fps

        self._different_fps = self._fps_video != self._fps

        assert self._fps_video >= self._fps

        self._num_frames = num_frames

        self._video_path = video_path
        self._half_path = half_path
        self._name = Path(Path(self._video_path).name) / Path(self._half_path.name)
        self._min_clip_duration = min_clip_duration
        self._num_decode = num_decode

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieves timestamps and frame indices from the stored video at the specified start and end times in
        seconds.

        Args:
            start_sec: The clip start time in seconds
            end_sec: The clip end time in seconds

        Returns:
            The timestamps and the frame indices.
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
        """Retrieves frames from the stored video at the specified start and end times in seconds (the video always
        starts at 0 seconds). Returned frames will be in [start_sec, end_sec). Given that PathManager may be
        fetching the frames from network storage, to handle transient errors, frame reading is retried N times.
        Note that as end_sec is exclusive, so you may need to use `get_clip(start_sec, duration + EPS)` to get the
        last frame.

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

        videos = torch.randn((3, timestamps.shape[0], 224, 224), dtype=torch.float32)

        if self._num_decode > 1:
            videos = [videos for _ in range(self._num_decode)]

        return {
            "video": videos,
            "clip_start": timestamps[0].item(),
            "clip_end": timestamps[-1].item(),
            "clip_duration": (timestamps[-1] - timestamps[0]).item(),
            "frame_indices": frame_indices,
            "fps_video_frame_indices": fps_video_frame_indices,
            "timestamps": timestamps,
        }
