Datasets
========

.. automodule:: eztorch.datasets

Datasets permits to iterate over data from a source of data. In Eztorch, to properly use the datasets for training, validating or evaluating, one should pass through datamodules.

Eztorch defines datasets and various tools to properly use datasets from Torchvision and custom ones.


Dataset Wrapper
----------------

General
^^^^^^^

.. autoclass:: DictDataset

.. autoclass:: DatasetFolder

.. autoclass:: DumbDataset

Image
^^^^^

.. autoclass:: DictCIFAR10

.. autoclass:: DictCIFAR100


Video
^^^^^
.. autoclass:: LabeledVideoDataset

.. autoclass:: Hmdb51

.. autoclass:: Kinetics

.. autofunction:: soccernet_dataset

.. autoclass:: SoccerNet

.. autofunction:: spot_dataset

.. autoclass:: Spot

.. autoclass:: Ucf101


Clip samplers
-------------

Clip samplers are used in video datasets to correctly sample clips when iteration over videos following a rule.

.. toctree::
   :maxdepth: 4

   eztorch.datasets.clip_samplers

Decoders
--------

Decoders are used in video datasets to decode the clips.

.. toctree::
   :maxdepth: 4

   eztorch.datasets.decoders


Collate functions
------------------

Used to collate samples for the dataloader.

.. toctree::
   :maxdepth: 4

   eztorch.datasets.collate_fn
