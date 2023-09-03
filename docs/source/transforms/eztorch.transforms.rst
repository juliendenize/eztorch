Transforms
==========

.. automodule:: eztorch.transforms

Eztorch supports transforms from Torchaug and from Torchvision.

On top of that, several video transforms have been defined.

Video transforms
----------------

.. toctree::
   :maxdepth: 4

   eztorch.transforms.video

Apply key
---------

.. autoclass:: ApplyTransformToKey

.. autoclass:: ApplyTransformToKeyOnList

.. autoclass:: ApplySameTransformInputKeyOnList

.. autoclass:: ApplySameTransformToKeyOnList

.. autoclass:: ApplyTransformAudioKey

.. autoclass:: ApplyTransformAudioKeyOnList

.. autoclass:: ApplyTransformInputKey

.. autoclass:: ApplyTransformInputKeyOnList

.. autoclass:: ApplyTransformOnDict

Apply transform on list
-----------------------

.. autoclass:: ApplyTransformOnList

.. autoclass:: ApplyTransformsOnList


Dict keep keys
--------------

.. autoclass:: DictKeepKeys

.. autoclass:: DictKeepInputLabel

.. autoclass:: DictKeepInputLabelIdx

Div 255
-------

.. autoclass:: Div255Input

Multi-crop
----------

.. autoclass:: MultiCropTransform


Only input transform
---------------------

.. autoclass:: OnlyInputTransform

.. autoclass:: OnlyInputListTransform

.. autoclass:: OnlyInputListSameTransform

.. autoclass:: OnlyInputTransformWithDictTransform

.. autoclass:: OnlyInputListTransformWithDictTransform

Random Resized Crop
-------------------
.. autoclass:: RandomResizedCrop

Remove key
----------
.. autoclass:: RemoveKey

.. autoclass:: RemoveAudioKey

.. autoclass:: RemoveInputKey
