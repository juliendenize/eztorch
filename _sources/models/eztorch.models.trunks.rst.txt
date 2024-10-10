Trunks
======

.. automodule:: eztorch.models.trunks


Image
-----

ResNet and ResNext
^^^^^^^^^^^^^^^^^^
.. autofunction:: create_resnet


Timm
^^^^

Timm models are accessible through Eztorch to retrieve VITs, Efficient-Net, ...

.. autofunction:: create_model_timm


Video
-----

Pytorchvideo
^^^^^^^^^^^^

Pytorchvideo models are accessible if the library has been installed and it is possible to use them to retrieve their models.

Video model and Head wrapper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: create_video_head_model


ResNet 3D with basic blocks
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: create_resnet3d_basic


R2+1D
^^^^^

General R2+1D

.. autofunction:: create_r2plus1d


R2+1D18 often used in papers

.. autofunction:: create_r2plus1d_18


S3D
^^^

.. autofunction:: create_s3d


X3D
^^^

.. autofunction:: create_x3d
