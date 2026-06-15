"""Pure analytic backbone baseline (CertifiedGeometry wrapper)."""

import torch
import torch.nn as nn
from torch import Tensor

from certiqnet.dispatcher.certiq.certificate import normalize_policy
from certiqnet.dispatcher.certiq.geometry import CertifiedGeometry
from certiqnet.models.baselines._base import baseline_device, expand_mu, make_lagrangian_diagnostics


class AnalyticBackbonePolicy(nn.Module):
    def __init__(self, N: int, beta: float = 1.0, C: float = float("inf")) -> None:
        super().__init__()
        self.N = N
        self.beta = beta
        self.C = C
        self.geometry = CertifiedGeometry(
            alpha_min=1e-3,
            beta_min=1e-3,
            gamma_max=2.0,
            alpha_init=1.0,
            beta_init=beta,
            gamma_init=0.0,
            c_init=0.0,
            C=0.0 if C == float("inf") else C,
        )

    def forward(
        self, Q: Tensor, mu: Tensor, xi: Tensor | None = None, training_mode: bool = False
    ) -> tuple[Tensor, DispatcherDiagnostics]:
        del xi, training_mode
        device = baseline_device(self, Q)
        Q = Q.to(device=device)
        mu = mu.to(device=device, dtype=Q.dtype)
        mu_b = expand_mu(Q, mu)
        pi, _ = self.geometry.policy(Q, mu_b)
        pi = normalize_policy(pi)
        return pi, make_lagrangian_diagnostics(pi, Q, mu_b, self.beta, self.C)
