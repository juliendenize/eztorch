from __future__ import annotations

import logging
import math
import os
import pathlib
import random
import re
from abc import ABC
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from typing import Callable

import numpy as np
import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr
from numpy.typing import NDArray
from pytorchvideo.data.video import Video
from torchvision.io import decode_image, read_file, read_image

from eztorch.datasets.utils_fn import get_video_to_frame_path_fn
from eztorch.transforms.video.temporal_difference_transform import \
    temporal_difference

logger = logging.getLogger(__name__)


class GeneralFrameVideo(Video, ABC):
    """GeneralFrameVideo is an abstract class for accessing clips stored as frames.

    Args:
        num_threads_io: Controls whether parallelizable io operations are
            performed across multiple threads.
        num_threads_decode: Controls whether parallelizable decode operations are
            performed across multiple threads.
        transform: The transform to apply to the frames.
    """

    def __init__(
        self,
        num_threads_io: int = 0,
        num_threads_decode: int = 0,
        transform: Callable | None = None,
    ) -> None:
        self._num_threads_io = num_threads_io
        self._num_threads_decode = num_threads_decode
        self._transform = transform

    def _load_images_with_retries(
        self,
        image_paths: list[str],
    ) -> torch.Tensor:
        """Loads the given image paths decodes them and returns them as a stacked tensors.

        Args:
            image_paths: A list of paths to images.

        Returns:
            A tensor of the clip's RGB frames with shape:
            (time, height, width, channel). The frames are of type ``torch.uint8`` and
            in the range :math:`[0,255]`.

        Raises:
            Exception: If unable to load images.
        """

        def fetch_img(image_path: str) -> torch.Tensor:
            img_byte = read_file(image_path)
            return img_byte

        def decode_img(fetch_idx: int, img_bytes: torch.Tensor) -> None:
            img = decode_image(img_bytes)
            if self._transform is not None:
                img = self._transform(img)
            imgs[fetch_idx] = img
            return

        if self._num_threads_io > 0:
            imgs = [None for i in range(len(image_paths))]

            work_queue_size = []
            with ThreadPoolExecutor(max_workers=self._num_threads_io) as read_pool:
                imgs_bytes = read_pool.map(fetch_img, image_paths)
                with ThreadPoolExecutor(
                    max_workers=max(self._num_threads_decode, 1)
                ) as work_pool:
                    for _ in work_pool.map(
                        decode_img, range(len(image_paths)), imgs_bytes
                    ):
                        work_queue_size.append(work_pool._work_queue.qsize())

        else:
            imgs = [read_image(image_path) for image_path in image_paths]

        imgs = torch.stack(imgs)

        return imgs


class FrameVideo(GeneralFrameVideo):
    """FrameVideo is an abstractions for accessing clips based on their start and end time for a video where each
    frame is stored as an image.

    Args:
        video_path: The path of the video.
        duration: The duration of the video in seconds.
        fps: The target fps for the video. This is needed to link the frames
            to a second timestamp in the video.
        frame_filter:
            Function to subsample frames in a clip before loading.
            If ``None``, no subsampling is performed
        video_frame_to_path_fn: A function that maps from the video path and a frame
            index integer to the file path where the frame is located.
        video_frame_paths: List of frame paths for each index of a video.
        num_threads_io: Controls whether parallelizable io operations are
            performed across multiple threads.
        num_threads_decode: Controls whether parallelizable decode operations are
            performed across multiple threads.
        transform: The transform to apply to the frames.
        decode_float: Whether to decode the clip as float.
    """

    def __init__(
        self,
        video_path: str,
        duration: float,
        fps: int,
        frame_filter: Callable[[list[int]], tuple[NDArray, NDArray]] | None = None,
        time_difference_prob: float = 0.0,
        video_frame_to_path_fn: None
        | (Callable[[str, int], int]) = get_video_to_frame_path_fn(),
        video_frame_paths: list[str] | None = None,
        num_threads_io: int = 0,
        num_threads_decode: int = 0,
        transform: Callable | None = None,
        decode_float: bool = False,
    ) -> None:
        super().__init__(
            num_threads_io=num_threads_io,
            num_threads_decode=num_threads_decode,
            transform=transform,
        )
        self._duration = duration
        self._fps = fps
        self._time_difference_prob = time_difference_prob
        self._decode_float = decode_float

        self._frame_filter = frame_filter

        assert (video_frame_to_path_fn is None) != (
            video_frame_paths is None
        ), "Only one of video_frame_to_path_fn or video_frame_paths can be provided"
        self._video_frame_to_path_fn = video_frame_to_path_fn
        self._video_frame_paths = video_frame_paths

        # Set the pathname to the parent directory of the first frame.
        self._video_path = video_path
        self._name = pathlib.Path(video_path).name

    @classmethod
    def from_directory(
        cls,
        video_path: str,
        fps: int,
        num_frames: int | None = None,
        path_order_cache: dict[str, list[str]] | None = None,
        frame_path_fn: Callable[[str, int], str] | None = None,
        **kwargs,
    ):
        """
        Args:
            video_path: Path to frame video directory.
            fps: The target fps for the video. This is needed to link the frames
                to a second timestamp in the video.
            num_frames: If not ``None``, number of frames in the video.
            path_order_cache: An optional mapping from directory-path to list
                of frames in the directory in numerical order. Used for speedup by
                caching the frame paths.
            frame_path_fn: Function to retrieve frame path from the video directory path and the frame index.
        """
        if path_order_cache is not None and video_path in path_order_cache:
            return cls.from_frame_paths(
                video_path, fps, path_order_cache[video_path], num_frames, **kwargs
            )

        assert g_pathmgr.isdir(video_path), f"{video_path} is not a directory"

        if num_frames is None:
            rel_frame_paths = g_pathmgr.ls(video_path)

            def natural_keys(text):
                return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)]

            rel_frame_paths.sort(key=natural_keys)
            frame_paths = [os.path.join(video_path, f) for f in rel_frame_paths]
        elif frame_path_fn is not None:
            frame_paths = [frame_path_fn(video_path, i) for i in range(num_frames)]
        else:
            frame_paths = None

        if path_order_cache is not None and frame_paths is not None:
            path_order_cache[video_path] = frame_paths
        return cls.from_frame_paths(video_path, fps, frame_paths, num_frames, **kwargs)

    @classmethod
    def from_frame_paths(
        cls,
        video_path: str,
        fps: int,
        video_frame_paths: list[str] | None = None,
        num_frames: int | None = None,
        **kwargs,
    ):
        """
        Args:
            video_path: Path to the video directory.
            fps: The target fps for the video. This is needed to link the frames
                to a second timestamp in the video.
            video_frame_paths: A list of paths to each frames in the video.
            num_frames: If not ``None``, number of frames in the video.
        """
        assert (
            video_frame_paths is not None or num_frames is not None
        ), "video_frame_paths is empty or num_frames should be specified"

        duration = Fraction(num_frames, fps) or Fraction(len(video_frame_paths), fps)
        return cls(
            video_path, duration, fps, video_frame_paths=video_frame_paths, **kwargs
        )

    @property
    def name(self) -> str:
        """The name of the video."""
        return self._name

    @property
    def duration(self) -> float:
        """The video's duration/end-time in seconds."""
        return self._duration

    def _get_frame_index_for_time(self, time_sec: float) -> int:
        return math.ceil(self._fps * time_sec)

    def get_clip(
        self,
        start_sec: float,
        end_sec: float,
    ) -> dict[str, torch.Tensor | None]:
        """Retrieves frames from the stored video at the specified start and end times in seconds (the video always
        starts at 0 seconds). Returned frames will be in [start_sec, end_sec). Given that PathManager may be
        fetching the frames from network storage, to handle transient errors, frame reading is retried N times.
        Note that as end_sec is exclusive, so you may need to use `get_clip(start_sec, duration + EPS)` to get the
        last frame.

        Args:
            start_sec: The clip start time in seconds
            end_sec: The clip end time in seconds

        Returns:
            A dictionary constraining the clip data and information.
        """
        if start_sec < 0 or start_sec > self._duration:
            logger.warning(
                f"No frames found within {start_sec} and {end_sec} seconds. Video starts"
                f"at time 0 and ends at {self._duration}."
            )
            return None

        end_sec = min(end_sec, self._duration)

        start_frame_index = self._get_frame_index_for_time(start_sec)
        end_frame_index = self._get_frame_index_for_time(end_sec)
        frame_indices = list(range(start_frame_index, end_frame_index))

        if (
            self._time_difference_prob > 0.0
            and self._time_difference_prob > random.random()
        ):
            do_time_difference = True
        else:
            do_time_difference = False

        # Frame filter function to allow for subsampling before loading
        if self._frame_filter:
            frame_indices, keep_frames = self._frame_filter(
                frame_indices, time_difference=do_time_difference
            )
        else:
            frame_indices = np.array(frame_indices)
            keep_frames = np.array([True for i in range(len(frame_indices))])

        unique, unique_indices = np.unique(frame_indices, return_inverse=True)

        clip_paths = [self._video_frame_to_path(i) for i in unique]

        clip_frames = self._load_images_with_retries(clip_paths)

        clip_frames = clip_frames[unique_indices]

        clip_frames = clip_frames.permute(1, 0, 2, 3)

        if do_time_difference:
            clip_frames = temporal_difference(
                clip_frames, use_grayscale=True, absolute=False
            )[:, keep_frames, :, :]

            frame_indices = frame_indices[keep_frames]

        if self._decode_float:
            clip_frames = clip_frames.to(torch.float32)

        return {
            "video": clip_frames,
            "frame_indices": frame_indices,
            "audio": None,
            "time_difference": do_time_difference,
        }

    def _video_frame_to_path(self, frame_index: int) -> str:
        if self._video_frame_to_path_fn:
            return self._video_frame_to_path_fn(self._video_path, frame_index)
        elif self._video_frame_paths:
            return self._video_frame_paths[frame_index]
        else:
            raise Exception(
                "One of _video_frame_to_path_fn or _video_frame_paths must be set"
            )
