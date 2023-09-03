from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset

from eztorch.datasets.decoders import DecoderType
from eztorch.datasets.decoders.frame_spot_video import FrameSpotVideo
from eztorch.datasets.spot_utils.features import load_features
from eztorch.datasets.spot_utils.parse_utils import (
    LABELS_SPOT_DATASETS, REVERSE_LABELS_SPOT_DATASETS, SpotDatasets)
from eztorch.datasets.spot_utils.spot_path_handler import SpotPathHandler
from eztorch.datasets.spot_utils.spot_paths import SpotPaths
from eztorch.utils.utils import get_global_rank, get_local_rank, get_world_size

log = logging.getLogger(__name__)


class Spot(Dataset):
    """Spot handles the storage, loading, decoding and clip sampling for a spot dataset. It assumes each video is
    stored as a frame video (e.g. a folder of jpg, or png)

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
        annotated_videos: SpotPaths,
        transform: Callable[[dict], Any] | None = None,
        decoder: str = "frame",
        decoder_args: DictConfig = {},
        label_args: DictConfig | None = None,
        features_args: DictConfig | None = None,
        dataset: SpotDatasets = SpotDatasets.TENNIS,
    ) -> None:
        if DecoderType(decoder) not in [DecoderType.FRAME, DecoderType.DUMB]:
            raise NotImplementedError

        self._transform = transform
        self._annotated_videos = annotated_videos
        self._decoder = decoder
        self._decoder_args = decoder_args
        self._label_args = label_args
        self._features_args: DictConfig | None = features_args
        self._dataset = SpotDatasets(dataset)

        self._video_path_handler = SpotPathHandler()

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
                    "clip_start": <Actual starting frame of the clip>,
                    "clip_end": <Actual ending frame of the clip>,
                    "video_name": <Name of the video>,
                    "labels": <Labels of the frames>,
                }
        """

        video_index, clip_start, clip_end = index

        annotated_video = self._annotated_videos[video_index]

        if clip_end < 0:
            clip_end = annotated_video["num_frames"] - 1

        video_path: Path = annotated_video["video_path"]

        video: FrameSpotVideo = self._video_path_handler.video_from_path(
            decoder=self._decoder,
            video_path=video_path,
            num_frames=annotated_video["num_frames"],
            **self._decoder_args,
        )

        loaded_clip = video.get_clip(clip_start, clip_end)

        frames = loaded_clip["video"]

        sample_dict = {
            "input": frames,
            "video_index": video_index,
            "frame_start": loaded_clip["frame_start"],
            "frame_end": loaded_clip["frame_end"],
            "frame_indices": loaded_clip["frame_indices"],
            "video_name": video_path.name,
            "num_frames": annotated_video["num_frames"],
        }

        if self._label_args is not None:
            start_frames = (
                self._annotated_videos.cum_num_frames_per_video[video_index - 1]
                if video_index > 0
                else 0
            )

            labels = self.labels[start_frames + loaded_clip["frame_indices"], :]

            has_label = self.has_label[start_frames + loaded_clip["frame_indices"], :]

            ignore_class = self.ignore_class[
                start_frames + loaded_clip["frame_indices"], :
            ]
            sample_dict.update(
                {
                    "labels": labels,
                    "has_label": has_label,
                    "ignore_class": ignore_class,
                }
            )

        if "inversed_temporal_masked_indices" in loaded_clip:
            sample_dict["inversed_temporal_masked_indices"] = loaded_clip[
                "inversed_temporal_masked_indices"
            ]

        if self._features_args is not None:
            features_indices = loaded_clip["frame_indices"]
            sample_dict["features"] = self._video_features[video_index][
                features_indices
            ]

        if self._transform is not None:
            sample_dict = self._transform(sample_dict)

        video.close()

        return sample_dict

    @property
    def dataset(self):
        return self._dataset

    @property
    def num_videos(self):
        """Number of videos."""
        return self._annotated_videos.num_videos

    @property
    def num_classes(self):
        """Number of classes."""
        return len(LABELS_SPOT_DATASETS[self.dataset])

    @property
    def LABELS(self):
        LABELS_SPOT_DATASETS[self.dataset]

    @property
    def REVERSE_LABELS(self):
        REVERSE_LABELS_SPOT_DATASETS[self.dataset]

    @property
    def global_rank(self):
        return get_global_rank()

    @property
    def local_rank(self):
        return get_local_rank()

    @property
    def world_size(self):
        return get_world_size()

    def get_video_metadata(self, video_idx: int):
        """Get the metadata of the specified video.

        Args:
            video_index: The video index.

        Returns:
            The metadata of the video.
        """
        return self._annotated_videos[video_idx]

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

                if cache_dir.exists():
                    precompute_labels = False

                    log.info(
                        f"{self.global_rank} rank: Dataset labels already computed and stored in {cache_dir}."
                    )

                    self.labels = torch.load(labels_file)[:, : self.num_classes]
                    self.has_label = self.labels.bool()

                    return

                else:
                    make_cache = True
                    cache_dir.mkdir(parents=True, exist_ok=False)

        if precompute_labels:
            labels_tot = [None for i in range(self.num_videos)]

        if self._label_args is not None:
            i = 0
            for video_idx in range(len(self)):
                if (
                    (video_idx + 1) % 10 == 0
                    or video_idx == 0
                    or video_idx == self._annotated_videos.num_videos - 1
                ):
                    log.info(
                        f"{self.global_rank} rank: Computing labels video {video_idx+1}/{len(self)}."
                    )
                video_metadata = self._annotated_videos[video_idx]

                if precompute_labels:
                    (labels) = _get_labels(
                        annotations=video_metadata,
                        num_frames=video_metadata["num_frames"],
                        num_labels=self.num_classes,
                        **self._label_args,
                    )

                    labels_tot[i] = labels

                i += 1

            if (
                (video_idx + 1) % 10 == 0
                or video_idx == 0
                or video_idx == self._annotated_videos.num_videos - 1
            ):
                log.info(
                    f"{self.global_rank} rank: Done computing num timestamps and/or label video {video_idx+1}/{len(self)}."
                )

        if self._label_args is not None and precompute_labels:
            self.labels = torch.cat(labels_tot)
            self.has_label = self.labels.bool()
            if make_cache:
                log.info(f"{self.global_rank} rank: Caching labels in {cache_dir}.")
                torch.save(self.labels, labels_file)

        elif self._label_args is None:
            self.labels = None
            self.has_label = None

        return

    def __repr__(self) -> str:
        return (
            f"{__class__.__name__}(num_videos={self.num_videos}, "
            f"transform={self._transform}, decoder={self._decoder}, "
            f"decoder_args={self._decoder_args})"
        )


def _get_labels(
    annotations: dict[str, Any],
    num_frames: int,
    num_labels: int,
    radius_label: int = 0,
    **kwargs,
):
    events = annotations["events"]
    labels = torch.zeros(num_frames, num_labels)

    for idx_frame_event, frame_event in enumerate(events["frame"]):
        label = events["label"][idx_frame_event].item()
        idx_change = torch.arange(
            frame_event - radius_label, frame_event + radius_label + 1
        )
        idx_change = torch.maximum(idx_change, torch.tensor(0))
        idx_change = torch.minimum(idx_change, torch.tensor(num_frames - 1))
        labels[idx_change, label] = 1

    return labels


def spot_dataset(
    data_path: str,
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    video_path_prefix: str = "",
    decoder: str = "frame",
    decoder_args: DictConfig = {},
    label_args: DictConfig | None = None,
    features_args: DictConfig | None = None,
    dataset: SpotDatasets = SpotDatasets.TENNIS,
) -> Spot:
    """A helper function to create ``Spot`` object.

    Args:
        data_path: Path to the data.
        transform: This callable is evaluated on the clip output before
                the clip is returned.
        video_path_prefix: Path to root directory with the videos that are
                loaded. All the video paths before loading
                are prefixed with this path.
        decoder: Defines what type of decoder used to decode a video.
        decoder_args: Arguments to configure the decoder.
        label_args: Arguments to configure the labels.
        features_args: Arguments to configure the extracted features.
        dataset: The spotting dataset.

    Returns:
        The dataset instantiated.
    """

    video_paths = SpotPaths.from_path(
        data_path,
        video_path_prefix,
        dataset,
    )

    spot = Spot(
        video_paths,
        transform,
        decoder=decoder,
        decoder_args=decoder_args,
        label_args=label_args,
        features_args=features_args,
        dataset=dataset,
    )
    return spot
