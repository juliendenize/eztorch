from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from eztorch.schedulers.linear_warmup_cosine_annealing_lr import \
    LinearWarmupCosineAnnealingLR

_SCHEDULERS = {
    "cosine_annealing_lr": CosineAnnealingLR,
    "linear_warmup_cosine_annealing_lr": LinearWarmupCosineAnnealingLR,
    "multi_step_lr": MultiStepLR,
}
