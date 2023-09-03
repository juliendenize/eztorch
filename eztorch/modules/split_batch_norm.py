import torch
from torch import Tensor, nn
from torch.nn import BatchNorm2d, Module


class SplitBatchNorm2D(BatchNorm2d):
    """Split batch normalization in several pieces to simulate several devices.

    Args:
        num_features: :math:`C` from an expected input of size :math:`(N, C, H, W)`.
        num_splits: Number of devices to simulate.
    """

    def __init__(self, num_features: int, num_splits: int, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W),
                running_mean_split,
                running_var_split,
                self.weight.repeat(self.num_splits),
                self.bias.repeat(self.num_splits),
                True,
                self.momentum,
                self.eps,
            ).view(N, C, H, W)
            self.running_mean.data.copy_(
                running_mean_split.view(self.num_splits, C).mean(dim=0)
            )
            self.running_var.data.copy_(
                running_var_split.view(self.num_splits, C).mean(dim=0)
            )
            return outcome
        else:
            return nn.functional.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )

    def __repr__(self):
        return (
            "{}({num_features}, num_splits={num_splits}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats})".format(
                __class__.__name__, **self.__dict__
            )
        )


def convert_to_split_batchnorm(module: Module, num_splits: int) -> Module:
    """Convert BatchNorm layers to SplitBatchNorm layers in module.

    Args:
        module: Module to convert.
        num_splits: Number of splits for the :class:`SplitBatchNorm2D` layers.

    Returns:
        The converted module.
    """
    module_output = module
    if isinstance(module, BatchNorm2d):
        module_output = SplitBatchNorm2D(
            num_features=module.num_features,
            num_splits=num_splits,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            track_running_stats=module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, convert_to_split_batchnorm(child, num_splits))
    del module
    return module_output
