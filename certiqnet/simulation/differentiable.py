"""Differentiable surrogate rollout utilities."""

import torch
from torch import Tensor


def surrogate_queue_cost(Q: Tensor, pi: Tensor, lam: float, mu: Tensor, dt: float = 1.0) -> Tensor:
    """One-step differentiable queue surrogate used for gradient-based training."""
    arrivals = lam * pi * dt
    service = mu.unsqueeze(0) * torch.sigmoid(Q) * dt
    Q_next = (Q + arrivals - service).clamp(min=0.0)
    return Q_next.sum(dim=-1).mean()
