from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from iopath.common.file_io import g_pathmgr
from omegaconf import DictConfig
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from torch import Tensor
from torch.utils.data import Dataset

from eztorch.datasets.decoders import DecoderType
from eztorch.datasets.decoders.frame_soccernet_video import FrameSoccerNetVideo
from eztorch.datasets.soccernet_utils.features import load_features
from eztorch.datasets.soccernet_utils.parse_utils import (
    ACTION_SPOTTING_LABELS, ALL_SPOTTING_LABELS, BALL_SPOTTING_LABELS,
    REVERSE_ACTION_SPOTTING_LABELS, REVERSE_ALL_SPOTTING_LABELS,
    REVERSE_BALL_SPOTTING_LABELS, SoccerNetTask)
from eztorch.datasets.soccernet_utils.soccernet_path_handler import \
    SoccerNetPathHandler
from eztorch.datasets.soccernet_utils.soccernet_paths import SoccerNetPaths
from eztorch.datasets.utils_fn import get_video_to_frame_path_fn
from eztorch.utils.utils import get_global_rank, get_world_size

log = logging.getLogger(__name__)


class SoccerNet(Dataset):
    """SoccerNet handles the storage, loading, decoding and clip sampling for a soccernet dataset. It assumes each
    video is stored as either an encoded video (e.g. mp4, avi) or a frame video (e.g. a folder of jpg, or png)

    Args:
        annotated_videos: List containing video annotations.
        transform: This callable is evaluated on the clip output before
            the clip is returned. It can be used for user defined preprocessing and
            augmentations on the clips.
        decoder: Defines what type of decoder used to decode a video.
        decoder_args: Arguments to configure the decoder.
    """

    def __init__(
        self,
        annotated_videos: SoccerNetPaths,
        transform: Callable[[dict], Any] | None = None,
        decoder: str = "frame",
        decoder_args: DictConfig = {},
        label_args: DictConfig | None = None,
        features_args: DictConfig | None = None,
        task: SoccerNetTask = SoccerNetTask.ACTION,
    ) -> None:
        if DecoderType(decoder) not in [DecoderType.FRAME, DecoderType.DUMB]:
            raise NotImplementedError

        self._transform = transform
        self._annotated_videos = annotated_videos
        self._decoder = decoder
        self._decoder_args = decoder_args
        self._label_args = label_args
        self._features_args: DictConfig | None = features_args
        self._task = SoccerNetTask(task)

        self._annotated_videos.set_fps(self._decoder_args["fps"])

        self._video_path_handler = SoccerNetPathHandler()
        self.deleted_yellow_to_red = False

        self._precompute_labels()

        if self.labels is not None:
            if label_args.get("has_no_action", False):
                not_has_class = torch.logical_not(
                    torch.any(self.has_label, -1, keepdim=True)
                )

                self.labels = torch.cat(
                    (self.labels, not_has_class.type_as(self.labels), -1)
                )
                self.has_label = torch.cat(
                    (self.has_label, not_has_class.type_as(self.has_label), -1)
                )

            self.ignore_class = torch.zeros_like(self.has_label)

        if self._features_args is not None:
            features_path = Path(features_args.get("dir", ""))
            filename = features_args["filename"]
            video_zip_prefix = features_args.get("video_zip_prefix", "")

            self._video_features = load_features(
                features_path,
                [path.decode() for path in self._annotated_videos._video_paths],
                filename,
                video_zip_prefix=video_zip_prefix,
                as_tensor=True,
            )

    def __len__(self):
        """
        Returns:
            Number of videos in dataset. There are 2 times more halves.
        """
        return self._annotated_videos.num_videos

    def __getitem__(self, index: tuple[Any]) -> dict:
        """Retrieves the next clip based on the clip sampling strategy and video sampler.

        Args:
            index: Tuple containing the video index, the half index (0 or 1) and the start and end of the clip to decode.

        Returns:
            A dictionary with the following format.

            .. code-block:: text

                {
                    "input": <The frames>,
                    "video_index": <Video index>,
                    "half_index": <Half index>,
                    "half": <Half value>,
                    "clip_start": <Actual starting second of the clip>,
                    "clip_end": <Actual ending second of the clip>,
                    "clip_duration": <Duration of the clip>,
                    "timestamps": <The different timestamps of the frames>,
                    "video_name": <Name of the video>,
                    "half_name": <Name of the half>,
                    "labels": <Labels of the frames>,
                }
        """

        video_index, half_index, clip_start, clip_end = index

        annotated_video = self._annotated_videos[(video_index, half_index)]

        if clip_end <= 0:
            clip_end = annotated_video["duration"]

        video_path: Path = annotated_video["video_path"]
        half_path: Path = annotated_video["half_path"]

        video = self._video_path_handler.video_from_path(
            decoder=self._decoder,
            video_path=video_path,
            half_path=half_path,
            num_frames=annotated_video["num_frames"],
            duration=annotated_video["duration"],
            fps_video=self._annotated_videos.fps_videos,
            **self._decoder_args,
        )

        loaded_clip = video.get_clip(clip_start, clip_end)

        frames = loaded_clip["video"]

        sample_dict = {
            "input": frames,
            "video_index": video_index,
            "half_index": half_index,
            "half": annotated_video["half_id"],
            "clip_start": loaded_clip["clip_start"],
            "clip_end": loaded_clip["clip_end"],
            "clip_duration": loaded_clip["clip_duration"],
            "timestamps": loaded_clip["timestamps"],
            "frame_indices": loaded_clip["frame_indices"],
            "video_name": video_path.name,
            "half_name": half_path.name,
            "half_duration": annotated_video["duration"],
        }

        if self._label_args is not None:
            video_num_timestamps = self.num_timestamps_per_half[
                annotated_video["half_idx"]
            ]
            start_timestamps = (
                self.cum_num_timestamps_per_half[annotated_video["half_idx"] - 1]
                if annotated_video["half_idx"] > 0
                else 0
            )

            labels = self.labels[start_timestamps + loaded_clip["frame_indices"], :]

            has_label = self.has_label[
                start_timestamps + loaded_clip["frame_indices"], :
            ]

            ignore_class = self.ignore_class[
                start_timestamps + loaded_clip["frame_indices"], :
            ]

            sample_dict.update(
                {
                    "labels": labels,
                    "has_label": has_label,
                    "ignore_class": ignore_class,
                    "video_num_timestamps": video_num_timestamps,
                }
            )

        if "inversed_temporal_masked_indices" in loaded_clip:
            sample_dict["inversed_temporal_masked_indices"] = loaded_clip[
                "inversed_temporal_masked_indices"
            ]

        if self._features_args is not None:
            fps_features = int(self._features_args["fps"])
            fps_video = int(self._decoder_args["fps"])
            features_indices = loaded_clip["frame_indices"]

            fps_out_features = self._features_args.get("fps_out", fps_features)

            if fps_features < fps_video:
                features_indices = torch.div(
                    features_indices[:: fps_video // fps_features],
                    fps_video // fps_features,
                    rounding_mode="floor",
                ).long()
            elif fps_features > fps_video:
                features_indices = features_indices * (fps_features // fps_video)
                tube_size = self._features_args["tube_size"]
                features_indices = features_indices[::tube_size]

            if fps_out_features < fps_features:
                features_indices = torch.div(
                    features_indices[:: fps_features // fps_out_features],
                    fps_video // fps_features,
                    rounding_mode="floor",
                ).long()
            elif fps_out_features > fps_features:
                raise NotImplementedError(
                    f"fps_out_features should be superior or equal to fps_features."
                    f" Got {fps_out_features} and {fps_features}."
                )
            sample_dict["features"] = self._video_features[video_index][
                int(annotated_video["half_id"])
            ][features_indices]

        if self._transform is not None:
            sample_dict = self._transform(sample_dict)

        video.close()

        return sample_dict

    @property
    def num_halves(self):
        """Number of halves."""
        return self._annotated_videos.num_halves

    @property
    def task(self):
        return self._task

    @property
    def num_videos(self):
        """Number of videos."""
        return self._annotated_videos.num_videos

    @property
    def num_classes(self):
        """Number of classes."""
        if self._task == SoccerNetTask.ACTION:
            return 17 if not self.deleted_yellow_to_red else 16
        elif self._task == SoccerNetTask.BALL:
            return 2
        elif self.task == SoccerNetTask.ALL:
            return 19
        else:
            raise NotImplementedError(f"{self._task} is not supported.")

    @property
    def LABELS(self):
        if self._task == SoccerNetTask.ACTION:
            return ACTION_SPOTTING_LABELS
        elif self._task == SoccerNetTask.BALL:
            return BALL_SPOTTING_LABELS
        elif self._task == SoccerNetTask.ALL:
            return ALL_SPOTTING_LABELS

    @property
    def REVERSE_LABELS(self):
        if self._task == SoccerNetTask.ACTION:
            return REVERSE_ACTION_SPOTTING_LABELS
        elif self._task == SoccerNetTask.BALL:
            return REVERSE_BALL_SPOTTING_LABELS
        elif self._task == SoccerNetTask.ALL:
            return REVERSE_ALL_SPOTTING_LABELS

    @property
    def global_rank(self):
        return get_global_rank()

    @property
    def world_size(self):
        return get_world_size()

    def get_half_metadata(self, video_index: int, half_index: int):
        """Get the metadata of the specified half.

        Args:
            video_index: The video index.
            half_index: The half index.

        Returns:
            The metadata of the half.
        """
        return self._annotated_videos.get_half_metadata(video_index, half_index)

    def get_video_metadata(self, video_idx: int):
        """Get the metadata of the specified video.

        Args:
            video_index: The video index.

        Returns:
            The metadata of the video.
        """
        return self._annotated_videos.get_video_metadata(video_idx)

    def get_class_weights(self) -> Tensor | None:
        """Get the weights of negatives and positives for each class.

        Returns:
            The weights in shape (2, C).
        """

        if self.labels is None:
            return None

        labels = np.asarray(self.labels)
        classes = np.unique(labels)
        class_weights = torch.cat(
            [
                torch.tensor(
                    compute_class_weight(
                        "balanced", y=labels[:, i], classes=classes
                    ).reshape(2, 1)
                )
                for i in range(self.num_classes)
            ],
            1,
        )
        return class_weights

    def get_label_class_counts(self) -> Tensor | None:
        """Get the number of positives and negatives for each class.

        Returns:
            The counts of labels.
        """

        if self.labels is None:
            return None

        pos_counts = self.labels.sum(0, keepdim=True)
        neg_counts = self.labels.shape[0] - pos_counts

        counts = torch.cat((neg_counts, pos_counts))

        return counts

    def get_class_proportion_weights(self) -> Tensor | None:
        """Get the weights of positives for each class proportionally.

        Returns:
            The weights.
        """

        if self.labels is None:
            return None

        class_weights = torch.tensor(
            compute_class_weight(
                "balanced",
                y=np.asarray(self._annotated_videos._label_annotation_per_half),
                classes=np.asarray(range(self.num_classes)),
            )
        )
        return class_weights

    def get_half_duration(self, video_index, half_index):
        """Get the half duration.

        Args:
            video_index: The video index.
            half_index: The half index.

        Returns:
            The duration of the half.
        """
        return self._annotated_videos._duration_per_half[
            video_index * 2 + half_index
        ].item()

    def _precompute_labels(self):
        """Precompute the labels of all the halves."""

        precompute_labels = False
        make_cache = False

        if self._label_args is not None:
            precompute_labels = True

            cache_dir = self._label_args.pop("cache_dir", None)
            if cache_dir is not None:
                cache_dir = Path(cache_dir)
                labels_file = cache_dir / "labels.pt"
                num_timestamps_per_half_file = cache_dir / "num_timestamps_per_half.pt"

                if cache_dir.exists():
                    precompute_labels = False

                    log.info(
                        f"{self.global_rank} rank: Dataset labels already computed and stored in {cache_dir}."
                    )

                    self.num_timestamps_per_half = torch.load(
                        num_timestamps_per_half_file
                    )
                    self.cum_num_timestamps_per_half = (
                        self.num_timestamps_per_half.cumsum(0)
                    )

                    kept_values = torch.cat(
                        [
                            torch.arange(
                                self.cum_num_timestamps_per_half[i - 1] if i > 0 else 0,
                                end,
                            )
                            for i, end in enumerate(self.cum_num_timestamps_per_half)
                        ]
                    )

                    self.labels = torch.load(labels_file)[kept_values][
                        :, : self.num_classes
                    ]
                    self.has_label = self.labels.bool()

                    return

                else:
                    make_cache = True
                    cache_dir.mkdir(parents=True, exist_ok=False)

        if precompute_labels:
            labels_tot = [None for i in range(self.num_halves)]

        if self._label_args is not None:
            num_timestamps_per_half = [None for i in range(self.num_halves)]

            i = 0
            for video_idx in range(len(self)):
                if (
                    (video_idx + 1) % 10 == 0
                    or video_idx == 0
                    or video_idx == self._annotated_videos.num_videos - 1
                ):
                    log.info(
                        f"{self.global_rank} rank: Computing num timestamps and/or label video {video_idx+1}/{len(self)}."
                    )
                for half_idx in range(2):
                    half_metadata = self._annotated_videos.get_half_metadata(
                        video_idx, half_idx
                    )

                    clip_end = half_metadata["duration"]

                    video_path: Path = half_metadata["video_path"]
                    half_path: Path = half_metadata["half_path"]

                    video: FrameSoccerNetVideo = (
                        self._video_path_handler.video_from_path(
                            decoder=self._decoder,
                            video_path=video_path,
                            half_path=half_path,
                            num_frames=half_metadata["num_frames"],
                            duration=half_metadata["duration"],
                            fps_video=self._annotated_videos.fps_videos,
                            **self._decoder_args,
                        )
                    )

                    timestamps, _, _ = video.get_timestamps_and_frame_indices(
                        0, clip_end
                    )

                    if precompute_labels:
                        (labels,) = _get_labels(
                            annotations=half_metadata["annotations"],
                            start_position=0,
                            end_position=clip_end,
                            half_path=half_path,
                            timestamps=timestamps,
                            **self._label_args,
                            LABELS=self.LABELS,
                            REVERSE_LABELS=self.REVERSE_LABELS,
                        )

                        labels_tot[i] = labels

                    num_timestamps_per_half[i] = timestamps.shape[0]

                    i += 1

                if (
                    (video_idx + 1) % 10 == 0
                    or video_idx == 0
                    or video_idx == self._annotated_videos.num_videos - 1
                ):
                    log.info(
                        f"{self.global_rank} rank: Done computing num timestamps and/or label video {video_idx+1}/{len(self)}."
                    )

            self.num_timestamps_per_half = torch.tensor(num_timestamps_per_half)
            self.cum_num_timestamps_per_half = self.num_timestamps_per_half.cumsum(0)

        else:
            self.num_timestamps_per_half = None
            self.cum_num_timestamps_per_half = None

        if self._label_args is not None and precompute_labels:
            self.labels = torch.cat(labels_tot)
            self.has_label = self.labels.bool()
            if make_cache:
                log.info(f"{self.global_rank} rank: Caching labels in {cache_dir}.")
                torch.save(self.labels, labels_file)
                torch.save(self.num_timestamps_per_half, num_timestamps_per_half_file)

        elif self._label_args is None:
            self.labels = None
            self.has_label = None

        return

    def __repr__(self) -> str:
        return f"{__class__.__name__}(num_videos={self.num_videos}, num_halves={self.num_halves}, transform={self._transform}, decoder={self._decoder}, decoder_args={self._decoder_args})"


def _get_labels(
    annotations: dict[str, Any],
    start_position: float,
    end_position: float,
    half_path: Path | str,
    timestamps: torch.Tensor,
    radius_label: int,
    LABELS=ACTION_SPOTTING_LABELS,
    REVERSE_LABELS=REVERSE_BALL_SPOTTING_LABELS,
    **kwargs,
):
    positions = annotations["position"]
    good_position = torch.where(
        torch.logical_and(positions < end_position, positions >= start_position)
    )

    kept_annotations = {
        key: annotation[good_position] for key, annotation in annotations.items()
    }
    num_labels = len(LABELS)

    labels = torch.zeros(num_labels, len(timestamps))

    for idx_timestamp, timestamp in enumerate(timestamps):
        timestamp_label_annotations = {}
        for idx_position, position in enumerate(kept_annotations["position"]):
            label = kept_annotations["label"][idx_position].item()
            displacement = position - timestamp
            abs_round_displacement = abs(displacement)
            if abs_round_displacement <= radius_label:
                if label in timestamp_label_annotations:
                    warnings.warn(
                        f"{half_path}: Label {REVERSE_LABELS[label]} class found for timestamp {timestamp} for position {position}, consider using a smaller label radius or changing annotation."
                    )
                timestamp_label_annotations[label] = 1
        for label in range(num_labels):
            if label in timestamp_label_annotations:
                labels[label, idx_timestamp] = 1
        return labels.permute(1, 0)


class ImageSoccerNet(SoccerNet):
    def __init__(
        self,
        annotated_videos: SoccerNetPaths,
        transform: Callable[[dict], Any] | None = None,
        decoder: str = "frame",
        decoder_args: DictConfig = {},
    ) -> None:
        super().__init__(annotated_videos, transform, decoder, decoder_args, None)

        self._decoder_args["time_difference_prob"] = self._decoder_args.get(
            "time_difference_prob", 0.0
        )
        self._path_fn = get_video_to_frame_path_fn(zeros=8)

    def _precompute_labels(self):
        pass

    def __getitem__(self, index: tuple[Any]) -> dict:
        video_index, half_index, num_frame = index

        annotated_video = self._annotated_videos[(video_index, half_index)]

        half_path: Path = annotated_video["half_path"]

        assert (
            self._decoder_args["time_difference_prob"] <= 0
        ), "Time difference for images not supported."

        image_path = (self._path_fn(half_path, num_frame),)
        with g_pathmgr.open(image_path, "rb") as f:
            img = Image.open(f)
            img_rgb = img.convert("RGB")
        img_rgb = np.array(img_rgb, dtype=np.uint8)
        image = torch.as_tensor(img_rgb)

        image = image.permute(2, 0, 1).to(torch.float32)

        sample = {"input": image}

        if self._transform is not None:
            sample = self._transform(sample)

        return sample

    def __len__(self):
        return self._annotated_videos.number_of_frames_fps


def soccernet_dataset(
    data_path: str,
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    video_path_prefix: str = "",
    decoder: str = "frame",
    decoder_args: DictConfig = {},
    label_args: DictConfig | None = None,
    features_args: DictConfig | None = None,
    task: SoccerNetTask = SoccerNetTask.ACTION,
) -> SoccerNet:
    """A helper function to create ``SoccerNet`` object.

    Args:
        data_path: Path to the data.
        transform: This callable is evaluated on the clip output before
                the clip is returned.
        video_path_prefix: Path to root directory with the videos that are
                loaded in ``SoccerNet``. All the video paths before loading
                are prefixed with this path.
        decoder: Defines what type of decoder used to decode a video.
        decoder_args: Arguments to configure the decoder.
        label_args: Arguments to configure the labels.
        features_args: Arguments to configure the extracted features.
        task: The task of action spotting, `action` and `ball` supported.

    Returns:
        The dataset instantiated.
    """

    video_paths = SoccerNetPaths.from_path(
        data_path,
        video_path_prefix,
        task,
    )

    dataset = SoccerNet(
        video_paths,
        transform,
        decoder=decoder,
        decoder_args=decoder_args,
        label_args=label_args,
        features_args=features_args,
        task=task,
    )
    return dataset


def image_soccernet_dataset(
    data_path: str,
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    video_path_prefix: str = "",
    decoder: str = "frame",
    decoder_args: DictConfig = {},
    features_args: DictConfig | None = None,
    task: SoccerNetTask = SoccerNetTask.ACTION,
) -> ImageSoccerNet:
    """A helper function to create ``ImageSoccerNet`` object.

    Args:
        data_path: Path to the data.
        transform: This callable is evaluated on the image output before
                the image is returned.
        video_path_prefix: Path to root directory with the videos that are
                loaded in ``SoccerNet``. All the video paths before loading
                are prefixed with this path.
        decoder: Defines what type of decoder used to decode a video.
        decoder_args: Arguments to configure the decoder.

    Returns:
        The dataset instantiated.
    """

    video_paths = SoccerNetPaths.from_path(
        data_path,
        video_path_prefix,
        task,
    )

    dataset = ImageSoccerNet(
        video_paths,
        transform,
        decoder=decoder,
        decoder_args=decoder_args,
        features_args=features_args,
    )
    return dataset
