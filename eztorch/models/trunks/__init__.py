try:
    import timm
except ImportError:
    pass
else:
    from timm import create_model as create_model_timm

    from eztorch.models.trunks.swin import (create_swin, create_swin_base,
                                            create_swin_small,
                                            create_swin_tiny)
    from eztorch.models.trunks.viswin import (create_viswin,
                                              create_viswin_base,
                                              create_viswin_small,
                                              create_viswin_tiny)
    from eztorch.models.trunks.vit import (create_vit, create_vit_base,
                                           create_vit_small, create_vit_tiny)
    from eztorch.models.trunks.vivit import (create_vivit, create_vivit_base,
                                             create_vivit_small,
                                             create_vivit_tiny)
try:
    import pytorchvideo
except ImportError:
    pass
else:
    from pytorchvideo.models.resnet import _MODEL_STAGE_DEPTH

    from eztorch.models.trunks.r2plus1d import create_r2plus1d
    from eztorch.models.trunks.r2plus1d_18 import create_r2plus1d_18
    from eztorch.models.trunks.resnet3d_basic import create_resnet3d_basic
    from eztorch.models.trunks.s3d import create_s3d
    from eztorch.models.trunks.video_model import create_video_head_model
    from eztorch.models.trunks.x3d import create_x3d

    _MODEL_STAGE_DEPTH[18] = (2, 2, 2, 2)


from eztorch.models.trunks.resnet import create_resnet
from eztorch.models.trunks.transformer_token_handler import (
    create_transformer_token_handler_model,
    create_vitransformer_token_handler_model)
