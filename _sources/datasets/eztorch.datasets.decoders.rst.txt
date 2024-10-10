Video Decoders
==============

.. automodule:: eztorch.datasets.decoders

Decoders type
^^^^^^^^^^^^^

.. autoclass:: DecoderType

Frame decoders
^^^^^^^^^^^^^^

Used to decode videos stored as frames.

.. autoclass:: GeneralFrameVideo

.. autoclass:: FrameVideo

.. autoclass:: FrameSoccerNetVideo

.. autoclass:: FrameSpotVideo

Dumb decoders
^^^^^^^^^^^^^

Used to simulate video decoding, return randomly generated tensors.

.. autoclass:: DumbSoccerNetVideo

.. autoclass:: DumbSpotVideo
