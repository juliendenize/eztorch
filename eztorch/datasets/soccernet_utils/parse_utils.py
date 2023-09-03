from __future__ import annotations

import copy
import json
import os
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch.utilities import rank_zero_info, rank_zero_warn
from torch.utils.data import Dataset

ACTION_SPOTTING_LABELS_FILENAME = "Labels-v2.json"
BALL_SPOTTING_LABELS_FILENAME = "Labels-ball.json"
ALL_SPOTTING_LABELS_FILENAME = [
    ACTION_SPOTTING_LABELS_FILENAME,
    BALL_SPOTTING_LABELS_FILENAME,
]
VIDEOS_EXTENSION = ".mkv"

ACTION_SPOTTING_LABELS = {
    "Penalty": 0,
    "Kick-off": 1,
    "Goal": 2,
    "Substitution": 3,
    "Offside": 4,
    "Shots on target": 5,
    "Shots off target": 6,
    "Clearance": 7,
    "Ball out of play": 8,
    "Throw-in": 9,
    "Foul": 10,
    "Indirect free-kick": 11,
    "Direct free-kick": 12,
    "Corner": 13,
    "Yellow card": 14,
    "Red card": 15,
    "Yellow->red card": 16,
    "No action": 17,
}

BALL_SPOTTING_LABELS = {"PASS": 0, "DRIVE": 1, "NO ACTION": 2}

ALL_SPOTTING_LABELS = {
    "Penalty": 0,
    "Kick-off": 1,
    "Goal": 2,
    "Substitution": 3,
    "Offside": 4,
    "Shots on target": 5,
    "Shots off target": 6,
    "Clearance": 7,
    "Ball out of play": 8,
    "Throw-in": 9,
    "Foul": 10,
    "Indirect free-kick": 11,
    "Direct free-kick": 12,
    "Corner": 13,
    "Yellow card": 14,
    "Red card": 15,
    "Yellow->red card": 16,
    "PASS": 17,
    "DRIVE": 18,
    "No action": 19,
}

REVERSE_ACTION_SPOTTING_LABELS = {
    0: "Penalty",
    1: "Kick-off",
    2: "Goal",
    3: "Substitution",
    4: "Offside",
    5: "Shots on target",
    6: "Shots off target",
    7: "Clearance",
    8: "Ball out of play",
    9: "Throw-in",
    10: "Foul",
    11: "Indirect free-kick",
    12: "Direct free-kick",
    13: "Corner",
    14: "Yellow card",
    15: "Red card",
    16: "Yellow->red card",
    17: "No action",
}

REVERSE_BALL_SPOTTING_LABELS = {0: "PASS", 1: "DRIVE", 2: "NO ACTION"}

REVERSE_ALL_SPOTTING_LABELS = {
    0: "Penalty",
    1: "Kick-off",
    2: "Goal",
    3: "Substitution",
    4: "Offside",
    5: "Shots on target",
    6: "Shots off target",
    7: "Clearance",
    8: "Ball out of play",
    9: "Throw-in",
    10: "Foul",
    11: "Indirect free-kick",
    12: "Direct free-kick",
    13: "Corner",
    14: "Yellow card",
    15: "Red card",
    16: "Yellow->red card",
    17: "PASS",
    18: "DRIVE",
    19: "No action",
}

TEAMS = {"home": 0, "away": 1, "not applicable": 2}

VISIBILITY = {"visible": 1, "not shown": 0}


class SoccerNetTask(Enum):
    ACTION = "action"
    BALL = "ball"
    ALL = "all"


def get_soccernet_weights(
    dataset: Dataset,
    type: str | None = "positives_per_class_soccernet",
    device: Any = "cpu",
    **kwargs,
):
    """Retrieve the soccernet weights type from dataset.

    Args:
        dataset: The SoccerNet dataset
        type: The type of weights to retrieve.
        device: The device on which to put the weights.

    Returns:
        The weights.
    """
    if type == "positives_per_class_soccernet":
        return dataset.get_class_weights().to(device=device)
    elif type == "weighted_positives_per_class_soccernet":
        positives_per_class = dataset.get_class_weights()
        positives_propertion_per_class = dataset.get_class_proportion_weights()
        weighted_positives_per_class = torch.ones_like(positives_per_class)
        weighted_positives_per_class[1, :] = positives_propertion_per_class
        weighted_positives_per_class[0, :] = (
            positives_per_class[0, :]
            * positives_propertion_per_class
            / positives_per_class[1, :]
        )
        return weighted_positives_per_class.to(device=device)
    elif type == "positives_propertion_class_soccernet":
        return dataset.get_class_proportion_weights().to(device=device)
    elif type == "positive_weights_counts":
        label_counts: torch.Tensor = dataset.get_label_class_counts().to(device=device)
        pos_weights = kwargs.get("positive_weights", 0.03)
        weights = torch.tensor([[1 - pos_weights], [pos_weights]], device=device)
        tot_counts = label_counts.sum(0, keepdim=True)
        weights = tot_counts * weights / label_counts
        weights[0, :] = weights[0, :].clamp(0, 1)
        weights[1, :] = weights[1, :].clamp(1, 1e9)
        return weights
    elif type == "positive_weights":
        pos_weights = kwargs.get("positive_weights", 5)
        weights = torch.tensor([[pos_weights, 1]], device=device).expand(
            2, dataset.num_classes
        )
    elif type == "none" or type is None:
        return None
    elif type == "card_weights":
        label_counts: torch.Tensor = dataset.get_label_class_counts().to(device=device)
        if dataset.deleted_yellow_to_red:
            card_counts = label_counts[1, -2:]
            n_samples = card_counts.sum()
            card_weights = n_samples / (2 * card_counts)
        else:
            card_counts = label_counts[1, -3:]
            n_samples = card_counts.sum()
            card_weights = n_samples / (3 * card_counts)
        return card_weights

    else:
        raise NotImplementedError(
            f"get_soccernet_weights does not support value: {type}."
        )


def recursive_files_in_dir_search(
    dir: Path, relative_dir: Path, task: SoccerNetTask = SoccerNetTask.ACTION
) -> dict[str, dict[str, str]]:
    """Find all videos and labels json in the SoccerNet raw dataset.

    Args:
        dir: Directory to find subdirs or the files. Called recursively.
        relative_dir: Root directory to start the search and used to only keep relative path.
        task: Task of spotting.

    Returns:
        Dictionary with keys being the directory and values dictionary with keys being labels or halves and values the name of the file.
    """
    subdirs = [f for f in dir.iterdir() if f.is_dir()]

    if len(subdirs) > 0:
        recursive_subdirs = {}
        for subdir in subdirs:
            recursive_subdirs.update(
                recursive_files_in_dir_search(subdir, relative_dir)
            )
        return recursive_subdirs

    kept_dir = os.path.relpath(dir, relative_dir)

    output_dict = {kept_dir: {}}

    if SoccerNetTask(task) == SoccerNetTask.ACTION:
        LABELS_FILENAME = [ACTION_SPOTTING_LABELS_FILENAME]
    elif SoccerNetTask(task) == SoccerNetTask.BALL:
        LABELS_FILENAME = [BALL_SPOTTING_LABELS_FILENAME]
    elif SoccerNetTask(task) == SoccerNetTask.ALL:
        LABELS_FILENAME = ALL_SPOTTING_LABELS_FILENAME

    for f in dir.iterdir():
        if f.name in LABELS_FILENAME:
            output_dict[kept_dir]["labels"] = f.name
        elif f.suffix == VIDEOS_EXTENSION:
            output_dict[kept_dir][f.stem] = f.name

    return output_dict


def process_annotation(
    annotation: dict[str, Any], task: SoccerNetTask = SoccerNetTask.ACTION
):
    """Process annotation from SoccerNet dictionary, meaning using ids instead of strings for actions, teams and
    visibility aswell as converting in seconds the position of the action.

    Args:
        annotation: The annotation to process.
        task: The SoccerNet task.

    Returns:
        The processed annotation.
    """
    new_annotation = {}
    for element, value in annotation.items():
        if element == "label":
            if SoccerNetTask(task) == SoccerNetTask.ACTION:
                new_annotation[element] = ACTION_SPOTTING_LABELS[value]
            elif SoccerNetTask(task) == SoccerNetTask.BALL:
                new_annotation[element] = BALL_SPOTTING_LABELS[value]
            elif SoccerNetTask(task) == SoccerNetTask.ALL:
                new_annotation[element] = ALL_SPOTTING_LABELS[value]
        elif element == "position":
            new_annotation[element] = int(value) / 1000
        elif element == "team":
            new_annotation[element] = TEAMS[value]
        elif element == "visibility":
            new_annotation[element] = VISIBILITY[value]
        else:
            new_annotation[element] = value
    return new_annotation


def get_video_duration_ffmpeg(filename: str | Path) -> float:
    """Retrieve the video duration of the file using ffprobe.

    Args:
        filename: The video path.

    Returns:
        The duration.
    """
    import subprocess

    result = subprocess.run(
        ["ffprobe", "-i", filename, "-show_format", "-v", "quiet"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    result = subprocess.run(
        ["sed", "-n", "s/duration=//p"],
        input=result.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return float(result.stdout)


def make_annotations_ffmpeg(
    dir_path: str | Path,
    out_file: str | Path,
    task: SoccerNetTask = SoccerNetTask.ACTION,
    resolution: str = "224p",
    is_challenge_split: bool = False,
) -> None:
    """Use FFmpeg to create annotations from a SoccerNet raw dataset.

    Args:
        dir_path: Path to the dataset.
        out_file: Path to the dumped annotations.
        task: The SoccerNetTask.
        resolution: Resolution of the videos.
        is_challenge_split: Whether it is the challenge split in which case no labels file are available.
    """
    dir_path = Path(dir_path)
    out_file = Path(out_file)

    dir_files = recursive_files_in_dir_search(dir_path, dir_path)
    global_annotations = {}

    if SoccerNetTask(task) == SoccerNetTask.ACTION:
        LABELS_FILENAME = ACTION_SPOTTING_LABELS_FILENAME
    elif SoccerNetTask(task) == SoccerNetTask.BALL:
        LABELS_FILENAME = BALL_SPOTTING_LABELS_FILENAME

    for dir in dir_files:
        if not is_challenge_split:
            video_annotation: dict = json.load(open(dir_path / dir / LABELS_FILENAME))
        else:
            video_annotation = {"UrlLocal": str(dir)}
        halves_annotation = {"halves": {}}
        for file_stem in dir_files[dir]:
            if (
                dir_files[dir][file_stem] == LABELS_FILENAME
                or resolution not in dir_files[dir][file_stem]
            ):
                continue

            if not is_challenge_split:
                half_annotation, copied_content = {}, copy.copy(video_annotation)
                half_annotations = [
                    annotation
                    for annotation in copied_content["annotations"]
                    if annotation["gameTime"][0] == file_stem[0]
                ]
                half_annotation["annotations"] = half_annotations
            else:
                half_annotations = {"annotations": []}

            half_annotation["UrlLocal"] = str(
                os.path.join(dir, dir_files[dir][file_stem])
            )
            half_annotation["duration"] = get_video_duration_ffmpeg(
                str(dir_path / dir / dir_files[dir][file_stem])
            )

            halves_annotation["halves"][file_stem[0]] = half_annotation

        if not is_challenge_split:
            _ = video_annotation.pop("annotations")

        global_annotations[dir] = {**video_annotation, **halves_annotation}

    with open(out_file, "w+") as out:
        json.dump(global_annotations, out, indent=4)

    return


def extract_frames_from_video_ffmpeg(
    video_path: str | Path,
    frame_dir: str | Path,
    FPS: int = 2,
) -> tuple[int | float]:
    """Create a list of frames from a video using FFmpeg.

    Args:
        video_path: The path of the video.
        frame_dir: The path to output the frales.
        FPS: The desired FPS for the frames.
    """
    import re
    import subprocess

    frames_path = Path(frame_dir) / "%08d.jpg"
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostats",
            "-i",
            video_path,
            "-vf",
            f"fps={FPS}",
            "-q:v",
            "1",
            frames_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    result_str = result.stdout.decode("utf-8")

    num_frames_regex = r"frame=\s*([\d]+)"
    duration_regex = r"time=\s*(\d{2}:\d{2}:\d{2}.\d{2})"

    num_frames_matches = re.finditer(num_frames_regex, result_str, re.MULTILINE)
    for num_frames_match in num_frames_matches:
        num_frames = int(num_frames_match.group(1))

    duration_matches = re.finditer(duration_regex, result_str, re.MULTILINE)
    for duration_match in duration_matches:
        duration = duration_match.group(1)

    video_read = (
        int(duration[3:5]) * 60 + int(duration[6:8]) + (0.01 * int(duration[6:8]) * FPS)
    )
    duration = int(duration[3:5]) * 60 + int(duration[6:8])

    if num_frames == (duration * FPS):
        rank_zero_info(f"{video_path}, Video read properly")
    elif num_frames > (duration * FPS) and num_frames == video_read * FPS:
        if num_frames == video_read * FPS:
            old_duration = duration
            duration = num_frames / FPS
            rank_zero_warn(
                f"{video_path}, Video not read properly: More frames extracted {num_frames} for an expected of {old_duration * FPS} frames because of rounding error of FFMPEG. Changed old duration {duration} to {duration}."
            )
    else:
        old_duration = duration
        duration = num_frames / FPS
        rank_zero_warn(
            f"{video_path}, Video not read properly: {num_frames} frames extracted and an expected of {old_duration * FPS} frames for a duration of {old_duration} and FPS of {FPS}. Duration changed to {duration}."
        )

    return num_frames, duration


def extract_frames_from_annotated_videos_ffmpeg(
    json_file: str | Path,
    video_dir: str | Path,
    frame_dir: str | Path,
    save_annotation_dir: str | Path,
    FPS: int = 2,
    print_freq: int = 1,
) -> None:
    """Create a SoccerNet dataset of frames from a dataset annotation and the raw dataset using FFmpeg.

    Args:
        json_file: The annotation of the dataset.
        video_dir: The path to the raw dataset containing the videos.
        frame_dir: The path to output the frames.
        save_annotation_dir: The path to save the annotations.
        FPS: The desired FPS for the frames.
        print_freq: The frequence to print that videos have been processed.
    """
    json_content = json.load(open(json_file))
    json_file = Path(json_file)
    video_dir = Path(video_dir)
    frame_dir = Path(frame_dir)
    save_annotation_dir = Path(save_annotation_dir)

    new_json_content = copy.copy(json_content)

    for i, (video, video_annotation) in enumerate(json_content.items()):
        if i % print_freq == 0 or i == 0:
            print(f"Processing video {i+1}/{len(json_content)}")
        video_path = Path(video_annotation["UrlLocal"])
        for half, half_annotation in video_annotation["halves"].items():
            video_half_path = Path(half_annotation["UrlLocal"])
            extracted_video_half_path = frame_dir / video_path / video_half_path.stem
            extracted_video_half_path.mkdir(exist_ok=True, parents=True)

            num_frames, video_duration = extract_frames_from_video_ffmpeg(
                str(video_dir / video_half_path), str(extracted_video_half_path), FPS
            )

            if not video_duration == half_annotation["duration"]:
                rank_zero_info(
                    f"Duration changed from {half_annotation['duration']} to {video_duration}."
                )

            new_json_content[video]["halves"][half]["UrlLocal"] = str(
                video_path / video_half_path.stem
            )
            new_json_content[video]["halves"][half]["fps"] = FPS
            new_json_content[video]["halves"][half]["num_frames"] = num_frames
            new_json_content[video]["halves"][half]["duration"] = video_duration

    with open(save_annotation_dir / json_file.name, "w+") as out:
        json.dump(new_json_content, out, indent=4)

    return
