Evaluation pipeline
===================

.. automodule:: eztorch.evaluation


Testing time augmentation
-------------------------

Get aggregation function
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: get_test_time_augmentation_fn


Aggregation functions
^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: average_group

.. autofunction:: average_same_num_aug

.. autofunction:: max_same_num_aug


Linear Classifier
-----------------

.. autoclass:: LinearClassifierEvaluation

Feature Extractor
-----------------

.. autoclass:: FeatureExtractor

.. autoclass:: SoccerNetFeatureExtractor

.. autoclass:: SpotFeatureExtractor

NMS
---

.. autofunction:: perform_all_classes_NMS

.. autofunction:: perform_soft_NMS

.. autofunction:: perform_hard_NMS

.. autofunction:: perform_hard_NMS_per_class
