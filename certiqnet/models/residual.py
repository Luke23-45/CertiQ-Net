"""Bounded residual logit head."""

import torch
import torch.nn as nn
from torch import Tensor


class BoundedResidualHead(nn.Module):
    """Project per-resource embeddings to hard-bounded scalar residual logits."""

    def __init__(self, d_local: int, R_max: float) -> None:
        super().__init__()
        assert R_max > 0, "R_max must be strictly positive."
        self.R_max = R_max
        self.proj = nn.Linear(d_local, 1)

    def forward(self, z1: Tensor) -> Tensor:
        """Return residual logits in ``[-R_max, R_max]`` with shape ``(B, N)``."""
        return self.R_max * torch.tanh(self.proj(z1).squeeze(-1))
