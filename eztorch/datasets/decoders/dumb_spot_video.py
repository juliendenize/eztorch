from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.utils.data
from pytorchvideo.data.video import Video

logger = logging.getLogger(__name__)


class DumbSpotVideo(Video):
    """DumbSpotVideo is an abstractions for accessing clips based on their start and end time for a video where
    each frame is randomly generated.

    Args:
        video_path: The path of the video.
        fps: The target fps for the video. This is needed to link the frames
            to a second timestamp in the video.
        num_frames: The number of frames of the video.
        min_clip_duration: The minimum duration of a clip.
        num_decode: Number of duplicate output clip.
    """

    def __init__(
        self,
        video_path: str | Path,
        fps: int,
        num_frames: int,
        num_decode: int = 1,
        min_clip_duration: float = 0,
        **kwargs,
    ) -> None:
        self._fps = fps
        self._num_frames = num_frames

        self._video_path = video_path
        self._name = Path(Path(self._video_path).name) / Path(self._half_path.name)

        self._num_decode = num_decode
        self._min_clip_duration = min_clip_duration

    @property
    def name(self) -> str:
        """The name of the video."""
        return self._name

    def get_frame_indices(
        self,
        start_frame: float,
        end_frame: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieves frame indices from the stored video at the specified start and end frames.

        Args:
            start_frame: The clip start frame
            end_frame: The clip end frame

        Returns:
            The frame indices.
        """
        if (
            start_frame < 0
            or start_frame >= self._num_frames
            or end_frame >= self._num_frames
        ):
            logger.warning(
                f"No frames found within {start_frame} and {end_frame} frames. Video starts"
                f"at frame 0 and ends at {self._num_frames}."
            )
            return None

        video_frame_indices = torch.arange(start_frame, end_frame + 1)

        if (
            self._min_clip_duration > 0
            and len(video_frame_indices) < self._min_clip_duration
        ):
            num_lacking_frames = self._min_clip_duration * self._fps - len(
                video_frame_indices
            )
            if start_frame == 0:
                video_frame_indices = torch.cat(
                    [
                        torch.zeros(
                            num_lacking_frames, dtype=video_frame_indices.dtype
                        ),
                        video_frame_indices,
                    ]
                )
            else:
                video_frame_indices = torch.cat(
                    [
                        video_frame_indices,
                        torch.tensor(
                            [
                                video_frame_indices[-1]
                                for _ in range(num_lacking_frames)
                            ],
                            dtype=video_frame_indices.dtype,
                        ),
                    ]
                )

        return video_frame_indices

    def get_clip(
        self,
        start_frame: int,
        end_frame: int,
    ) -> dict[str, torch.Tensor | None | list[torch.Tensor]]:
        """Retrieves frames from the stored video at the specified starting and ending frames.

        Args:
            start_frame: The clip start frame
            end_frame: The clip end frame

        Returns:
            A dictionary containing the clip data and information.
        """

        frame_indices = self.get_frame_indices(start_frame, end_frame)

        videos = torch.randn((3, frame_indices.shape[0], 224, 224), dtype=torch.float32)

        if self._num_decode > 1:
            videos = [videos for _ in range(self._num_decode)]

        return {
            "video": videos,
            "clip_start": frame_indices[0].item(),
            "clip_end": frame_indices[-1].item(),
            "frame_indices": frame_indices,
        }
