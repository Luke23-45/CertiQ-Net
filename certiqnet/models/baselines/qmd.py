"""Greedy route minimizing quadratic drift ``(2 * Q_i + 1) / mu_i``."""

import torch
import torch.nn as nn
from torch import Tensor

from certiqnet.dispatcher.certiq.certificate import normalize_policy
from certiqnet.models.baselines._base import baseline_device, expand_mu, make_lagrangian_diagnostics


class QuadraticMinDrift(nn.Module):
    def __init__(self, N: int, beta: float = 1.0, C: float = float("inf")) -> None:
        super().__init__()
        self.N = N
        self.beta = beta
        self.C = C

    def forward(
        self, Q: Tensor, mu: Tensor, xi: Tensor | None = None, training_mode: bool = False
    ) -> tuple[Tensor, DispatcherDiagnostics]:
        del xi, training_mode
        device = baseline_device(self, Q)
        Q = Q.to(device=device)
        mu = mu.to(device=device, dtype=Q.dtype)
        mu_b = expand_mu(Q, mu)
        idx = ((2.0 * Q + 1.0) / mu_b).argmin(dim=-1)
        pi = torch.zeros_like(Q)
        pi.scatter_(1, idx.unsqueeze(-1), 1.0)
        pi = normalize_policy(pi)
        return pi, make_lagrangian_diagnostics(pi, Q, mu_b, self.beta, self.C)
