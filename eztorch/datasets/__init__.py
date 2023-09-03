from eztorch.datasets import collate_fn
from eztorch.datasets.dict_dataset import (DictCIFAR10, DictCIFAR100,
                                           DictDataset)
from eztorch.datasets.dumb_dataset import DumbDataset
from eztorch.datasets.folder_dataset import DatasetFolder

try:
    import pytorchvideo
except ImportError:
    pass
else:
    import eztorch.datasets.decoders
    import eztorch.datasets.soccernet_utils
    from eztorch.datasets import clip_samplers
    from eztorch.datasets.hmdb51 import Hmdb51
    from eztorch.datasets.kinetics import Kinetics
    from eztorch.datasets.labeled_video_dataset import (
        LabeledVideoDataset, LabeledVideoPaths,
        create_frames_files_from_folder, create_video_files_from_folder,
        labeled_video_dataset)
    from eztorch.datasets.soccernet import (ImageSoccerNet, SoccerNet,
                                            image_soccernet_dataset,
                                            soccernet_dataset)
    from eztorch.datasets.spot import Spot, spot_dataset
    from eztorch.datasets.ucf101 import Ucf101
    from eztorch.datasets.utils_fn import (get_subsample_fn,
                                           get_time_difference_indices,
                                           get_video_to_frame_path_fn,
                                           random_subsample, uniform_subsample)
