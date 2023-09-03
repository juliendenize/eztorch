from eztorch.datasets.soccernet_utils.parse_utils import (
    extract_frames_from_annotated_videos_ffmpeg, make_annotations_ffmpeg)
from eztorch.datasets.soccernet_utils.predictions import (
    add_clip_prediction, add_clips_predictions, aggregate_and_filter_clips,
    get_rounded_timestamps, get_timestamps_indexes, merge_predictions,
    postprocess_spotting_half_predictions, save_spotting_predictions)
from eztorch.datasets.soccernet_utils.soccernet_path_handler import \
    SoccerNetPathHandler
from eztorch.datasets.soccernet_utils.soccernet_paths import SoccerNetPaths
