"""Stochastic route with softmax over quadratic-drift indices."""

import torch
import torch.nn as nn
from torch import Tensor

from certiqnet.dispatcher.certiq.certificate import normalize_policy
from certiqnet.dispatcher.types import DispatcherDiagnostics
from certiqnet.models.baselines._base import baseline_device, expand_mu, make_lagrangian_diagnostics


class SoftQuadraticMinDrift(nn.Module):
    def __init__(
        self, N: int, tau: float = 1.0, beta: float = 1.0, C: float = float("inf")
    ) -> None:
        super().__init__()
        self.N = N
        self.tau = tau
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
        logits = -((2.0 * Q + 1.0) / mu_b) / self.tau
        pi = torch.softmax(logits, dim=-1)
        pi = normalize_policy(pi)
        return pi, make_lagrangian_diagnostics(pi, Q, mu_b, self.beta, self.C)
