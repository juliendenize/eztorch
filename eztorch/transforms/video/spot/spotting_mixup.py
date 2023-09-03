from typing import Any, Dict

import torch
from torch import Tensor
from torch.nn import Module


def mix_spotting(
    x: Tensor,
    mix_value: Tensor,
    permutation: Tensor,
    labels: Tensor,
    has_label: Tensor,
    ignore_class: Tensor,
):
    """Make mixup of the batch for action spotting.

    Args:
        x: The batch values to mix.
        mix_value: Value coefficients for mixing.
        permutation: Permutation to perform mix.
        labels: Labels of the timestamps in the batch.
        has_label: Whether timestamps have label.
        ignore_class: Whether class in the batch should be ignored.

    Returns:
        Tuple containing:
        - The mixed input.
        - The mixed class labels.
        - The `ignore_class` of the mixed elements.
        - The concatenated `mix_value` of the mixed elements.
    """
    x_permuted = x[permutation]
    labels_permuted = labels[permutation]
    has_label_permuted = has_label[permutation]
    ignore_class_permuted = ignore_class[permutation]

    labels_cat = torch.cat((labels, labels_permuted))
    has_label_cat = torch.cat((has_label, has_label_permuted))
    ignore_class_cat = torch.cat((ignore_class, ignore_class_permuted))

    mix_value_x = mix_value.view([-1, *([1] * (x.ndim - 1))])
    one_minus_mix_value_x = 1 - mix_value_x
    mix_value_label = mix_value.view([-1, *([1] * (labels.ndim - 1))])
    one_minus_mix_value_label = 1 - mix_value_label

    x_mixed = mix_value_x * x + one_minus_mix_value_x * x_permuted
    mixed_weights = torch.cat((mix_value_label, one_minus_mix_value_label))

    return (
        x_mixed,
        labels_cat,
        has_label_cat,
        ignore_class_cat,
        mixed_weights,
    )


class SpottingMixup(Module):
    """Make mixup for spotting for labels.

    Args:
        alpha: Alpha value for the beta distribution of mixup.
    """

    def __init__(
        self,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.mix_sampler = torch.distributions.Beta(alpha, alpha)

    def forward(self, batch: Dict[str, Any]):
        (x, labels, has_label, ignore_class,) = (
            batch["input"],
            batch["labels"],
            batch["has_label"],
            batch["ignore_class"],
        )

        device, dtype = x.device, x.dtype
        batch_size = x.shape[0]

        with torch.inference_mode():
            mix_value = self.mix_sampler.sample((batch_size,)).to(
                device=device, dtype=dtype
            )
        mix_value = mix_value.clone()

        permutation = torch.randperm(batch_size, device=device)

        (
            x_mixed,
            labels_after_mix,
            has_label_after_mix,
            ignore_class_after_mix,
            mixed_weights,
        ) = mix_spotting(
            x=x,
            mix_value=mix_value,
            permutation=permutation,
            labels=labels,
            has_label=has_label,
            ignore_class=ignore_class,
        )

        new_batch = {
            "input": x_mixed,
            "labels": labels_after_mix,
            "ignore_class": ignore_class_after_mix,
            "has_label": has_label_after_mix,
            "mixup_weights": mixed_weights,
        }

        return new_batch

    def __repr__(self):
        return f"{__class__.__name__}(alpha={self.alpha})"
