from eztorch.evaluation.feature_extractor import (FeatureExtractor,
                                                  SoccerNetFeatureExtractor,
                                                  SpotFeatureExtractor)
from eztorch.evaluation.linear_classifier import LinearClassifierEvaluation
from eztorch.evaluation.nms import (perform_all_classes_NMS, perform_hard_NMS,
                                    perform_hard_NMS_per_class,
                                    perform_soft_NMS)
from eztorch.evaluation.test_time_augmentation_fn import (
    average_group, average_same_num_aug, get_test_time_augmentation_fn,
    max_same_num_aug)
