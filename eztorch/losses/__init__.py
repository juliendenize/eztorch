from eztorch.losses.moco_loss import compute_moco_loss
from eztorch.losses.mocov3_loss import compute_mocov3_loss
from eztorch.losses.ressl_loss import compute_ressl_loss, compute_ressl_mask
from eztorch.losses.sce_loss import compute_sce_loss, compute_sce_mask
from eztorch.losses.sce_token_loss import (compute_sce_token_loss,
                                           compute_sce_token_masks)
from eztorch.losses.simclr_loss import (compute_simclr_loss,
                                        compute_simclr_masks)
from eztorch.losses.spot_loss import compute_spot_loss_fn
