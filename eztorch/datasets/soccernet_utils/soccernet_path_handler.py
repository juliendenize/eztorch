from pytorchvideo.data.video import Video

from eztorch.datasets.decoders import DecoderType


class SoccerNetPathHandler:
    """Utility class that handles all deciphering and caching of video paths for encoded and frame videos."""

    def __init__(self) -> None:
        return

    def video_from_path(
        self,
        decoder: DecoderType,
        video_path: str,
        half_path: str,
        duration: float,
        fps_video: int,
        fps: int,
        num_frames: int,
        **kwargs,
    ) -> Video:
        """Retrieve a video from the specified path.

        Args:
            decoder: The decoder for the video.
            video_path: The path to the video.
            half_path: The path to the half.
            duration: The duration of the video.
            fps_video: The fps of the video.
            fps: The fps to extract frames.
            num_frames: The number of frames of the video.

        Returns:
            The video to decode.
        """

        if DecoderType(decoder) == DecoderType.FRAME:
            from eztorch.datasets.decoders.frame_soccernet_video import \
                FrameSoccerNetVideo

            return FrameSoccerNetVideo(
                video_path=video_path,
                half_path=half_path,
                duration=duration,
                fps_video=fps_video,
                fps=fps,
                num_frames=num_frames,
                **kwargs,
            )

        elif DecoderType(decoder) == DecoderType.DUMB:
            from eztorch.datasets.decoders.dumb_soccernet_video import \
                DumbSoccerNetVideo

            return DumbSoccerNetVideo(
                video_path=video_path,
                half_path=half_path,
                duration=duration,
                fps_video=fps_video,
                fps=fps,
                num_frames=num_frames,
                **kwargs,
            )
        else:
            raise NotImplementedError
