from eztorch.models.heads.linear import LinearHead
from eztorch.models.heads.mlp import MLPHead

try:
    import pytorchvideo
except ImportError:
    pass
else:
    from eztorch.models.heads.linear3d import (Linear3DHead,
                                               create_linear3d_head)
    from eztorch.models.heads.video_resnet import (
        VideoResNetHead, VideoResNetTemporalHead, create_video_resnet_head,
        create_video_resnet_temporal_head)
