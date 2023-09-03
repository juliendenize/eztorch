from pytorchvideo.data.clip_sampling import UniformClipSampler

from eztorch.datasets.clip_samplers.constant_clips_per_video_sampler import \
    ConstantClipsPerVideoSampler
from eztorch.datasets.clip_samplers.constant_clips_with_half_overlap_per_video_clip_sampler import \
    ConstantClipsWithHalfOverlapPerVideoClipSampler
from eztorch.datasets.clip_samplers.minimum_full_coverage_clip_sampler import \
    MinimumFullCoverageClipSampler
from eztorch.datasets.clip_samplers.random_clip_sampler import (
    RandomClipSampler, RandomCVRLSampler, RandomMultiClipSampler)
