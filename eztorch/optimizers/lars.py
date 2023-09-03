from typing import Callable, Iterable, Optional

import torch
from torch.nn import Parameter
from torch.optim import Optimizer


class LARS(torch.optim.Optimizer):
    """LARS optimizer, no rate scaling or weight decay for parameters <= 1D.

    References LARS:
        - https://arxiv.org/pdf/1708.03888.pdf

    Args:
        params: Parameters to optimize.
        lr: Learning rate of the optimizer.
        weight_decay: Weight decay to apply.
        momentum: Momentum for optimization.
        trust_coefficient: LARS trust coefficient.
    """

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 0,
        weight_decay: float = 0,
        momentum: float = 0.9,
        trust_coefficient: float = 0.001,
    ) -> None:
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            trust_coefficient=trust_coefficient,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1:  # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g["weight_decay"])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0,
                            (g["trust_coefficient"] * param_norm / update_norm),
                            one,
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)
                p.add_(mu, alpha=-g["lr"])

        return loss
