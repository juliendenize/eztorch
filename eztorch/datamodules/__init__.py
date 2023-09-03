import torchvision

from eztorch.datamodules.base import BaseDataModule
from eztorch.datamodules.cifar import (CIFAR10DataModule, CIFAR100DataModule,
                                       CIFARDataModule)
from eztorch.datamodules.dumb import DumbDataModule
from eztorch.datamodules.folder import FolderDataModule
from eztorch.datamodules.imagenet import (Imagenet100DataModule,
                                          ImagenetDataModule)
from eztorch.datamodules.stl10 import STL10DataModule
from eztorch.datamodules.tiny_imagenet import TinyImagenetDataModule

try:
    import pytorchvideo
except ImportError:
    pass
else:
    from eztorch.datamodules.hmdb51 import Hmdb51DataModule
    from eztorch.datamodules.kinetics import (Kinetics200DataModule,
                                              Kinetics400DataModule,
                                              Kinetics600DataModule,
                                              Kinetics700DataModule,
                                              KineticsDataModule)
    from eztorch.datamodules.soccernet import (ImageSoccerNetDataModule,
                                               SoccerNetDataModule)
    from eztorch.datamodules.spot import SpotDataModule
    from eztorch.datamodules.ucf101 import Ucf101DataModule
    from eztorch.datamodules.video import VideoBaseDataModule
