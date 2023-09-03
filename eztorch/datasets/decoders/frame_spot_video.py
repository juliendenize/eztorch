from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import torch
import torch.utils.data

from eztorch.datasets.decoders.frame_video import GeneralFrameVideo
from eztorch.datasets.utils_fn import get_video_to_frame_path_fn
from eztorch.utils.mask import mask_tube_in_sequence

logger = logging.getLogger(__name__)


class FrameSpotVideo(GeneralFrameVideo):
    """FrameSpotVideo is an abstractions for accessing clips based on their start and end time for a video where
    each frame is stored as an image.

    Args:
        video_path: The path of the video.
        num_frames: The number of frames of the video.
        transform: The transform to apply to the frames.
        video_frame_to_path_fn: A function that maps from the video path and a frame
            index integer to the file path where the frame is located.
        num_threads_io: Controls whether parallelizable io operations are
            performed across multiple threads.
        num_threads_decode: Controls whether parallelizable decode operations are
            performed across multiple threads.
        num_decode: Number of decode to perform. If > 1, the videos decoded are stored in a list.
        mask_ratio: Masking ratio for the video.
        mask_ratio: Sequence tube size for masking the video.
        min_clip_duration: The minimum duration of a clip.
        decode_float: Whether to decode the clip as float.
    """

    def __init__(
        self,
        video_path: str | Path,
        num_frames: int,
        transform: Callable | None = None,
        video_frame_to_path_fn: Callable[[str, int], int] = get_video_to_frame_path_fn(
            zeros=6, incr=0
        ),
        num_threads_io: int = 0,
        num_threads_decode: int = 0,
        num_decode: int = 1,
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

        self._num_frames = num_frames
        self._decode_float = decode_float

        self._video_frame_to_path_fn = video_frame_to_path_fn

        self._video_path = video_path
        self._name: Path = Path(Path(self._video_path).name) / Path(
            self._video_path.name
        )

        self._num_decode = num_decode
        self._mask_ratio = mask_ratio
        self._mask_tube = mask_tube
        self._min_clip_duration = min_clip_duration

    @property
    def name(self) -> str:
        """The name of the video."""
        return self._name

    @property
    def duration(self) -> int:
        return self._num_frames

    def get_frame_indices(
        self,
        start_frame: int,
        end_frame: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieves frame indices from the stored video at the specified starting frame and end frame.

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
                f"No frames found within {start_frame} and {end_frame} seconds. Video starts "
                f"at frame 0 and ends at {self._num_frames}."
            )
            return None

        video_frame_indices = torch.arange(start_frame, end_frame + 1)

        if (
            self._min_clip_duration > 0
            and len(video_frame_indices) < self._min_clip_duration
        ):
            num_lacking_frames = self._min_clip_duration - len(video_frame_indices)
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
        start_frame: float,
        end_frame: float,
    ) -> dict[str, torch.Tensor | None | list[torch.Tensor]]:
        """Retrieves frames from the stored video at the specified starting and ending frames.

        Args:
            start_frame: The clip start frame
            end_frame: The clip end frame

        Returns:
            A dictionary containing the clip data and information.
        """
        frame_indices = self.get_frame_indices(start_frame, end_frame)

        if self._mask_ratio > 0:
            t = frame_indices.shape[0]

            (
                _,
                indices_kept,
                inversed_temporal_masked_indices,
                _,
            ) = mask_tube_in_sequence(self._mask_ratio, self._mask_tube, t, "cpu")

            frame_indices_to_decode = frame_indices[indices_kept]

        else:
            frame_indices_to_decode = frame_indices

        clip_paths = [self._video_frame_to_path(i) for i in frame_indices_to_decode]
        clip_frames = self._load_images_with_retries(
            clip_paths,
        )

        clip_frames = clip_frames.permute(1, 0, 2, 3)
        if self._decode_float:
            clip_frames = clip_frames.to(torch.float32)

        if self._num_decode > 1:
            videos = [clip_frames for _ in range(self._num_decode)]
        else:
            videos = clip_frames

        out = {
            "video": videos,
            "frame_start": frame_indices[0].item(),
            "frame_end": frame_indices[-1].item(),
            "frame_indices": frame_indices,
        }

        if self._mask_ratio > 0.0:
            out["inversed_temporal_masked_indices"] = inversed_temporal_masked_indices

        return out

    def _video_frame_to_path(self, frame_index: int) -> str:
        return self._video_frame_to_path_fn(self._video_path, frame_index)
