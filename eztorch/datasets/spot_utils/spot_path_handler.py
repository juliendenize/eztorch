from pytorchvideo.data.video import Video

from eztorch.datasets.decoders import DecoderType


class SpotPathHandler:
    """Utility class that handles all deciphering and caching of video paths for encoded and frame videos."""

    def __init__(self) -> None:
        return

    def video_from_path(
        self,
        decoder: DecoderType,
        video_path: str,
        num_frames: int,
        **kwargs,
    ) -> Video:
        """Retrieve a video from the specified path.

        Args:
            decoder: The decoder for the video.
            video_path: The path to the video.
            num_frames: The number of frames of the video.

        Returns:
            The video to decode.
        """

        if DecoderType(decoder) == DecoderType.FRAME:
            from eztorch.datasets.decoders.frame_spot_video import \
                FrameSpotVideo

            return FrameSpotVideo(
                video_path=video_path,
                num_frames=num_frames,
                **kwargs,
            )

        elif DecoderType(decoder) == DecoderType.DUMB:
            from eztorch.datasets.decoders.dumb_spot_video import DumbSpotVideo

            return DumbSpotVideo(
                video_path=video_path,
                num_frames=num_frames,
                **kwargs,
            )
        else:
            raise NotImplementedError
