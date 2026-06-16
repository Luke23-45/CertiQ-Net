"""Maximum Pressure — network-aware dispatch via routing matrix ``P``.

When ``P`` is ``None`` (single-hop networks), this reduces to
:class:`MaxWeight <certiqnet.models.baselines.max_weight.MaxWeight>`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from certiqnet.dispatcher.certiq.certificate import normalize_policy
from certiqnet.dispatcher.types import DispatcherDiagnostics
from certiqnet.models.baselines._base import baseline_device, expand_mu, make_lagrangian_diagnostics


class MaximumPressure(nn.Module):
    def __init__(
        self,
        N: int,
        beta: float = 1.0,
        C: float = float("inf"),
        P: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.N = N
        self.beta = beta
        self.C = C
        if P is not None:
            self.register_buffer("_P", P.clone())
        else:
            self._P: Tensor | None = None

    def forward(
        self, Q: Tensor, mu: Tensor, xi: Tensor | None = None, training_mode: bool = False
    ) -> tuple[Tensor, DispatcherDiagnostics]:
        del xi, training_mode
        device = baseline_device(self, Q)
        Q = Q.to(device=device)
        mu = mu.to(device=device, dtype=Q.dtype)
        mu_b = expand_mu(Q, mu)

        if self._P is None:
            idx = (-Q * mu_b.pow(self.beta)).argmin(dim=-1)
        else:
            P = self._P.to(device=device, dtype=Q.dtype)
            downstream = Q @ P.T
            pressure = mu_b.pow(self.beta) * (Q - downstream)
            idx = (-pressure).argmin(dim=-1)

        pi = torch.zeros_like(Q)
        pi.scatter_(1, idx.unsqueeze(-1), 1.0)
        pi = normalize_policy(pi)
        return pi, make_lagrangian_diagnostics(pi, Q, mu_b, self.beta, self.C)
