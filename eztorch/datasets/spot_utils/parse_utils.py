from enum import Enum


class SpotDatasets(Enum):
    TENNIS = "tennis"
    FS_COMP = "fs_comp"
    FS_PERF = "fs_perf"


TENNIS_CLASSES = {
    "far_court_bounce": 0,
    "far_court_swing": 1,
    "far_court_serve": 2,
    "near_court_bounce": 3,
    "near_court_swing": 4,
    "near_court_serve": 5,
}

REVERSE_TENNIS_CLASSES = {
    0: "far_court_bounce",
    1: "far_court_swing",
    2: "far_court_serve",
    3: "near_court_bounce",
    4: "near_court_swing",
    5: "near_court_serve",
}

FS_COMP_CLASSES = {
    "jump_landing": 0,
    "jump_takeoff": 1,
    "spin_landing": 2,
    "spin_takeoff": 3,
}

REVERSE_FS_COMP_CLASSES = {
    0: "jump_landing",
    1: "jump_takeoff",
    2: "spin_landing",
    3: "spin_takeoff",
}

FS_PERF_CLASSES = {
    "jump_landing": 0,
    "jump_takeoff": 1,
    "spin_landing": 2,
    "spin_takeoff": 3,
}

REVERSE_FS_PERF_CLASSES = {
    0: "jump_landing",
    1: "jump_takeoff",
    2: "spin_landing",
    3: "spin_takeoff",
}


LABELS_SPOT_DATASETS = {
    SpotDatasets.TENNIS: TENNIS_CLASSES,
    SpotDatasets.FS_COMP: FS_COMP_CLASSES,
    SpotDatasets.FS_PERF: FS_PERF_CLASSES,
}


REVERSE_LABELS_SPOT_DATASETS = {
    SpotDatasets.TENNIS: REVERSE_TENNIS_CLASSES,
    SpotDatasets.FS_COMP: REVERSE_FS_COMP_CLASSES,
    SpotDatasets.FS_PERF: REVERSE_FS_PERF_CLASSES,
}
