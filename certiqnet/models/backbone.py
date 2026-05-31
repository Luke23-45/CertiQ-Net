"""Analytic backbone policy."""

import torch
import torch.nn as nn
from torch import Tensor

from certiqnet.math.backbone_params import ConstrainedBackboneParams


class AnalyticBackbone(nn.Module):
    """Compute backbone logits and routing distribution."""

    def __init__(self, params: ConstrainedBackboneParams) -> None:
        super().__init__()
        self.params = params

    def logits(self, Q: Tensor, mu: Tensor) -> Tensor:
        """Return ``u_i = gamma log(mu_i) - alpha (Q_i + c) / mu_i^beta``."""
        assert Q.shape[-1] == mu.shape[-1], "Q and mu last dimensions must match."
        assert (Q >= 0).all(), "Queue lengths must be non-negative."
        assert (mu > 0).all(), "Service rates must be strictly positive."
        log_mu = torch.log(mu.clamp(min=1e-8))
        y = (Q + self.params.c) / mu.pow(self.params.beta)
        return self.params.gamma * log_mu - self.params.alpha * y

    def forward(self, Q: Tensor, mu: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(p_base, u_base)``."""
        u = self.logits(Q, mu)
        p = torch.softmax(u, dim=-1)
        p = p.clamp_min(torch.finfo(p.dtype).tiny)
        p = p / p.sum(dim=-1, keepdim=True)
        return p, u
