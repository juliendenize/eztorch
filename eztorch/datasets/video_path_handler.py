import io
import pathlib
from typing import Optional

from iopath.common.file_io import g_pathmgr
from pytorchvideo.data.video import Video

from eztorch.datasets.decoders import DecoderType


class VideoPathHandler:
    """Utility class that handles all deciphering and caching of video paths for encoded and frame videos."""

    def __init__(self) -> None:
        # Pathmanager isn't guaranteed to be in correct order,
        # sorting is expensive, so we cache paths in case of frame video and reuse.
        # TODO Find a way to specify when caching is OK.
        self.path_order_cache = {}

    def video_from_path(
        self,
        filepath: str,
        decode_audio=False,
        decoder="pyav",
        num_frames: Optional[int] = None,
        **kwargs,
    ) -> Video:
        """Retrieve a video from the specified path.

        Args:
            filepath: The path to the video.
            decode_audio: If True, decode audio.
            decoder: The decoder to use. Options are:

                * ``'decord'``
                * ``'frame'``
                * ``'lmdb'``
                * ``'pyav'``
                * ``'torchvision'``

            num_frames: If not ``None, number of frames in the video for frame decoder.
            **kwargs: additional parameters given to the decoders.

        Raises:
            NotImplementedError: If lmdb is not installed and decoder is ``'lmdb'``.
        """

        if DecoderType(decoder) == DecoderType.PYAV:
            from pytorchvideo.data.encoded_video_pyav import EncodedVideoPyAV

            with g_pathmgr.open(filepath, "rb") as fh:
                video_file = io.BytesIO(fh.read())

            return EncodedVideoPyAV(
                video_file, pathlib.Path(filepath).name, decode_audio, **kwargs
            )

        elif DecoderType(decoder) == DecoderType.TORCHVISION:
            from pytorchvideo.data.encoded_video_torchvision import \
                EncodedVideoTorchVision

            with g_pathmgr.open(filepath, "rb") as fh:
                video_file = io.BytesIO(fh.read())

            return EncodedVideoTorchVision(
                video_file, pathlib.Path(filepath).name, decode_audio, **kwargs
            )

        elif DecoderType(decoder) == DecoderType.FRAME:
            from eztorch.datasets.decoders.frame_video import FrameVideo

            assert not decode_audio, "decode_audio must be False when using FrameVideo"

            return FrameVideo.from_directory(
                filepath,
                path_order_cache=self.path_order_cache,
                num_frames=num_frames,
                **kwargs,
            )

        else:
            raise NotImplementedError(f"Unknown decoder type {decoder}")
