from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


class LossType(Enum):
    BCE = "bce"
    FOCAL = "focal"
    SOFTMAX = "softmax"


def compute_spot_loss_class(
    class_preds: Tensor,
    class_target: Tensor,
    has_label: Tensor,
    ignore_class: Tensor,
    class_weights: Tensor,
    mixup_weights: Tensor | None,
    loss_type: LossType = LossType.BCE,
    alpha: float = 0.25,
    gamma: float = 2,
):
    if LossType(loss_type) in [LossType.BCE, LossType.FOCAL]:
        bce_weights, num_bce = get_spot_bce_weights(
            class_target=class_target,
            has_class_target=has_label,
            class_weights=class_weights,
            ignore_class=ignore_class,
        )

        bce_loss = F.binary_cross_entropy_with_logits(
            class_preds,
            class_target,
            reduction="none",
        )

        if LossType(loss_type) == LossType.FOCAL:
            p = torch.sigmoid(class_preds)
            p_t = p * class_target + (1 - p) * (1 - class_target)
            bce_loss = bce_loss * ((1 - p_t) ** gamma)

            if alpha >= 0:
                alpha_t = alpha * class_target + (1 - alpha) * (1 - class_target)
                bce_loss = alpha_t * bce_loss

        if mixup_weights is not None:
            bce_loss *= bce_weights.type_as(bce_loss)
            bce_loss = bce_loss.sum((-1, -2))
            bce_loss *= mixup_weights.squeeze()
            bce_loss = bce_loss.sum() / torch.maximum(
                num_bce, torch.tensor(1, device=bce_loss.device)
            )

        else:
            bce_loss *= bce_weights.type_as(bce_loss)
            bce_loss = bce_loss.sum()
            bce_loss /= torch.maximum(num_bce, torch.tensor(1, device=bce_loss.device))

        return bce_loss

    elif LossType(loss_type) == LossType.SOFTMAX:
        b, t, c = class_preds.shape

        class_weights, num_softmax = get_spot_softmax_weights(
            has_class=has_label,
            ignore_class=ignore_class,
        )

        class_preds = class_preds.reshape(b * t, c)
        class_target = class_target.reshape(b * t, c)

        softmax_loss: Tensor = F.cross_entropy(
            class_preds, class_target, reduction="none", weight=class_weights
        )

        softmax_loss = softmax_loss.reshape(b, t)

        if mixup_weights is not None:
            softmax_loss *= class_weights.type_as(softmax_loss)
            softmax_loss = softmax_loss.sum(-1)
            softmax_loss *= mixup_weights.squeeze()
            softmax_loss /= torch.maximum(
                num_softmax, torch.tensor(1, device=softmax_loss.device)
            )

        else:
            softmax_loss *= class_weights.type_as(softmax_loss)
            softmax_loss = softmax_loss.sum()
            softmax_loss /= torch.maximum(
                num_softmax, torch.tensor(1, device=softmax_loss.device)
            )

        return softmax_loss
    else:
        raise NotImplementedError(f"{type} does not exist.")


def get_spot_bce_weights(
    class_target: Tensor,
    has_class_target: Tensor,
    class_weights: Optional[Tensor],
    ignore_class: Tensor,
) -> Tensor:
    """Get the soccernet weights for binary cross entropy loss.

    Args:
        class_target: The one-hot encoder of target classes.
        has_class_target: Whether there is a label.
        class_weights: Vector of weights for each class.
        ignore_class: Whether to ignore class.

    Returns:
        The weights.
    """

    has_class_target = has_class_target.bool()
    ignore_class = ignore_class.bool()

    if class_weights is not None:
        not_has_class_target = torch.logical_not(has_class_target)
        bce_weights = (
            has_class_target.type_as(class_weights) * class_weights[1:2, :]
            + not_has_class_target.type_as(class_weights) * class_weights[0:1, :]
        )
    else:
        bce_weights = torch.ones(class_target.shape, device=class_target.device)

    bce_weights *= 1 - ignore_class.type_as(bce_weights)

    return bce_weights, torch.logical_not(ignore_class).sum()


def get_spot_softmax_weights(
    has_class: Tensor,
    ignore_class: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Get the soccernet weights for card loss.

    Args:
        has_class: Whether each element in each sample in the batch has the class.
        ignore_class: Whether to ignore for each element in each sample in the batch the class

    Returns:
        The softmax weights.
    """

    has_class = has_class.bool()
    has_class = has_class.any(dim=-1)

    ignore_class = ignore_class.bool()
    ignore_class = ignore_class.any(dim=-1)

    use_softmax_criterion = torch.logical_and(
        has_class,
        torch.logical_not(ignore_class),
    )

    softmax_weights = torch.ones(has_class.shape, device=has_class.device)
    softmax_weights *= use_softmax_criterion.type_as(softmax_weights)

    return softmax_weights, use_softmax_criterion.sum()


def compute_spot_loss_fn(
    class_preds: Tensor,
    class_target: Tensor,
    has_label: Tensor,
    ignore_class: Tensor,
    class_weights: Optional[Tensor] = None,
    mixup_weights: Optional[Tensor] = None,
    class_loss_type: LossType = LossType.BCE,
    alpha: float = 0.25,
    gamma: float = 2,
) -> Tensor:
    """Compute the soccernet loss function which is a classification loss.

    Args:
        class_preds: Predictions for the classes. Expected shape: (B, T, C).
        class_target: Multi-label encoded class. Can be continuous label in case of mixup. Expected shape: (B, T, C).
        has_label: Whether there is a label. Expected shape: (B', T, C).
        ignore_class: Whether to ignore class. Expected shape: (B', T, C).
        class_weights: Weights of negatives and positives for BCE loss. Expected shape: (2, C).
        mixup_weights: Weights for mixup for loss. Expected shape: (B', T, C).
        class_loss_type: Type of loss to use. Can be BCE, softmax or Focal.
        alpha: For focal loss.
        gamma: For focal loss.

    Returns:
        The reduced sum of the classification losses.
    """
    if class_weights is not None:
        class_weights = class_weights.to(device=class_target.device)

    if mixup_weights is not None:
        mixup_weights_1, mixup_weights_2 = mixup_weights.chunk(2)

        class_target_1, class_target_2 = class_target.chunk(2)
        ignore_class_1, ignore_class_2 = ignore_class.chunk(2)
        has_label_1, has_label_2 = has_label.chunk(2)

        bce_loss_1 = compute_spot_loss_class(
            class_preds=class_preds,
            class_target=class_target_1,
            has_label=has_label_1,
            ignore_class=ignore_class_1,
            class_weights=class_weights,
            mixup_weights=mixup_weights_1,
            loss_type=class_loss_type,
            alpha=alpha,
            gamma=gamma,
        )

        bce_loss_2 = compute_spot_loss_class(
            class_preds=class_preds,
            class_target=class_target_2,
            has_label=has_label_2,
            ignore_class=ignore_class_2,
            class_weights=class_weights,
            mixup_weights=mixup_weights_2,
            loss_type=class_loss_type,
            alpha=alpha,
            gamma=gamma,
        )

        bce_loss = bce_loss_1 + bce_loss_2

    else:
        bce_loss = compute_spot_loss_class(
            class_preds=class_preds,
            class_target=class_target,
            has_label=has_label,
            ignore_class=ignore_class,
            class_weights=class_weights,
            mixup_weights=mixup_weights,
            loss_type=class_loss_type,
            alpha=alpha,
            gamma=gamma,
        )

    return bce_loss


def compute_soccernet_softmax_loss_fn(
    class_preds: Tensor,
    class_target: Tensor,
    has_target: Tensor,
    ignore_class: Tensor,
    class_weights: Tensor | None = None,
    mixup_weights: Optional[Tensor] = None,
) -> Tensor:
    b, t, c = class_preds.shape

    if mixup_weights is None:

        softmax_weights, num_softmax = get_spot_softmax_weights(
            has_class=has_target,
            ignore_class=ignore_class,
        )

        class_preds = class_preds.reshape(b * t, c)
        class_target = class_target.reshape(b * t, c)

        class_loss: Tensor = F.cross_entropy(
            class_preds, class_target, reduction="none", weight=class_weights
        )

        class_loss = class_loss.reshape(b, t)

        class_loss *= softmax_weights.type_as(class_loss)
        class_loss = class_loss.sum()
        class_loss /= torch.maximum(
            num_softmax, torch.tensor(1, device=class_loss.device)
        )
    else:
        mixup_weights_1, mixup_weights_2 = mixup_weights.chunk(2)
        class_target_1, class_target_2 = class_target.chunk(2)
        has_target_1, has_target_2 = has_target.chunk(2)
        ignore_class_1, ignore_class_2 = ignore_class.chunk(2)

        class_preds = class_preds.reshape(b * t, c)

        softmax_weights_1, num_softmax_1 = get_spot_softmax_weights(
            has_class=has_target_1,
            ignore_class=ignore_class_1,
        )

        softmax_weights_2, num_softmax_2 = get_spot_softmax_weights(
            has_class=has_target_2,
            ignore_class=ignore_class_2,
        )

        class_target_1 = class_target_1.reshape(b * t, c)
        class_target_2 = class_target_2.reshape(b * t, c)

        class_loss_1: Tensor = F.cross_entropy(
            class_preds, class_target_1, reduction="none", weight=class_weights
        )
        class_loss_2: Tensor = F.cross_entropy(
            class_preds, class_target_2, reduction="none", weight=class_weights
        )

        class_loss_1 = class_loss_1.reshape(b, t)
        class_loss_2 = class_loss_2.reshape(b, t)

        class_loss_1 *= softmax_weights_1.type_as(class_loss_1)
        class_loss_1 = class_loss_1.sum(-1)
        class_loss_1 *= mixup_weights_1.squeeze()
        class_loss_1 = class_loss_1.sum() / torch.maximum(
            num_softmax_1, torch.tensor(1, device=class_loss_1.device)
        )

        class_loss_2 *= softmax_weights_2.type_as(class_loss_2)
        class_loss_2 = class_loss_2.sum(-1)
        class_loss_2 *= mixup_weights_2.squeeze()
        class_loss_2 = class_loss_2.sum() / torch.maximum(
            num_softmax_2, torch.tensor(1, device=class_loss_2.device)
        )

        class_loss = class_loss_1 + class_loss_2

    return class_loss
