Clip Samplers
=============

.. automodule:: eztorch.datasets.clip_samplers

Used to properly sample clips in videos.

Random Clip Samplers
--------------------

.. autoclass:: RandomClipSampler

.. autoclass:: RandomMultiClipSampler

.. autoclass:: RandomCVRLSampler


Deterministic Clip Samplers
---------------------------

.. autoclass:: ConstantClipsPerVideoSampler

.. autoclass:: ConstantClipsWithHalfOverlapPerVideoClipSampler

.. autoclass:: MinimumFullCoverageClipSampler

.. autoclass:: UniformClipSampler

SoccerNet Clip Samplers
-----------------------

.. automodule:: eztorch.datasets.clip_samplers.soccernet

.. autoclass:: SoccerNetClipSampler

.. autoclass:: SoccerNetClipSamplerDistributedSamplerWrapper

.. autoclass:: ActionWindowSoccerNetClipSampler

.. autoclass:: FeatureExtractionSoccerNetClipSampler

.. autoclass:: ImagesSoccerNetClipSampler

.. autoclass:: RandomWindowSoccerNetClipSampler

.. autoclass:: SlidingWindowSoccerNetClipSampler

.. autoclass:: UniformWindowSoccerNetClipSampler

.. autoclass:: UniformWindowWithoutOverlapSoccerNetClipSampler

Spot Clip Samplers
------------------

.. automodule:: eztorch.datasets.clip_samplers.spot

.. autoclass:: SpotClipSampler

.. autoclass:: SpotClipSamplerDistributedSamplerWrapper

.. autoclass:: FeatureExtractionSpotClipSampler

.. autoclass:: ImagesSpotClipSampler

.. autoclass:: SlidingWindowSpotClipSampler

.. autoclass:: UniformWindowWithoutOverlapSpotClipSampler
