Datamodules
============

.. automodule:: eztorch.datamodules

Datamodules are tools from Pytorch-lightning that allows to wrap all the logic to download datasets, verify its integrity, make transform, load the datasets, and make the dataloaders.

Eztorch contains basic wrapper to contain the logic of Datamodules to work with Hydra.

Several Datamodules already exist to handle various datasets.


Base Datamodules
----------------

Base
^^^^
.. autoclass:: BaseDataModule

.. autoclass:: FolderDataModule

.. autoclass:: DumbDataModule

Video
^^^^^
.. autoclass:: VideoBaseDataModule


Image Datamodules
-------------------


CIFAR
^^^^^
.. autoclass:: CIFAR10DataModule

.. autoclass:: CIFAR100DataModule


ImageNet
^^^^^^^^
.. autoclass:: ImagenetDataModule

.. autoclass:: Imagenet100DataModule


STL10
^^^^^
.. autoclass:: STL10DataModule


Tiny-ImageNet
^^^^^^^^^^^^^
.. autoclass:: TinyImagenetDataModule


Video Datamodules
-----------------


HMDB51
^^^^^^
.. autoclass:: Hmdb51DataModule

Kinetics
^^^^^^^^

.. autoclass:: Kinetics200DataModule

.. autoclass:: Kinetics400DataModule

.. autoclass:: Kinetics600DataModule

.. autoclass:: Kinetics700DataModule

SoccerNet
^^^^^^^^^

.. autoclass:: SoccerNetDataModule
.. autoclass:: ImageSoccerNetDataModule

Spot
^^^^
.. autoclass:: SpotDataModule

UCF101
^^^^^^
.. autoclass:: Ucf101DataModule
