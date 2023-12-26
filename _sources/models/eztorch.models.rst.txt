Overview
========

Eztorch allows you to instantiate several kind of models as well as different parts of models. We will detail each of the different kind of customization you can have below.


Trunks
-----------
.. toctree::
   :maxdepth: 3

   eztorch.models.trunks

Trunks, also called encoders, are the main parts of your model to learn representations. Eztorch allows you to instantiate trunks for Image or Video representation.

Heads
-----------

.. toctree::
   :maxdepth: 3

   eztorch.models.heads

Heads are fixed on top of trunks to specialize to a specific task. Among different heads we can cite:
   - Linear heads to perform classification.
   - MLP heads to make projectors in Self-Supervised Learning.


Siamese models
---------------------------

.. toctree::
   :maxdepth: 3

   eztorch.models.siamese

Siamese models are models based on a siamese architecture. It is generally used for Self-Supervised or/and Knowledge-distillation training. It means that a model is used twice in a pipeline. Sometimes the model is duplicated to have a self-distillation paradigm through a momentum update of the duplication of the model.


Supervised model
----------------

.. toctree::
   :maxdepth: 3

   eztorch.models.supervised

The supervised model works based on a trunk and a head.


Finetuning model
----------------

.. toctree::
   :maxdepth: 3

   eztorch.models.finetuning

The finetuning model works based on a pretrained trunk and a head and has several parameters trick implemented for finetuning.


Dummy model
-----------

.. toctree::
   :maxdepth: 1

   eztorch.models.dummy


The dummy model is a model implemented just to make some quick pipeline tests.
