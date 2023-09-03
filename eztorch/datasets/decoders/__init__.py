from enum import Enum

from eztorch.datasets.decoders.dumb_soccernet_video import DumbSoccerNetVideo
from eztorch.datasets.decoders.dumb_spot_video import DumbSpotVideo
from eztorch.datasets.decoders.frame_soccernet_video import FrameSoccerNetVideo
from eztorch.datasets.decoders.frame_spot_video import FrameSpotVideo
from eztorch.datasets.decoders.frame_video import FrameVideo, GeneralFrameVideo


class DecoderType(Enum):
    PYAV = "pyav"
    TORCHVISION = "torchvision"
    FRAME = "frame"
    DUMB = "dumb"
