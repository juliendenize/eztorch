Video transforms
================

Eztorch supports transforms from Torchaug and from Pytorchvideo.

On top of that, several video transforms have been defined.

.. automodule:: eztorch.transforms.video

Video transforms
----------------

.. autoclass:: RemoveTimeDim

.. autoclass:: RandomResizedCrop

.. autoclass:: RandomTemporalDifference

.. autoclass:: TemporalDifference

SoccerNet
---------

.. automodule:: eztorch.transforms.video.soccernet

.. autoclass:: BatchReduceTimestamps

.. autoclass:: BatchMiddleTimestamps

.. autoclass:: ReduceTimestamps

.. autoclass:: MiddleTimestamps

Spot
----

.. automodule:: eztorch.transforms.video.spot

.. autoclass:: BatchMiddleFrames

.. autoclass:: MiddleFrames

.. autoclass:: SpottingMixup
