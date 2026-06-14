"""Greedy route minimizing quadratic drift ``(2 * Q_i + 1) / mu_i``."""

import torch
import torch.nn as nn
from torch import Tensor

from certiqnet.dispatcher.certiq.certificate import arrival_coordinate, policy_entropy, normalize_policy
from certiqnet.models.baselines._base import baseline_device, expand_mu
from certiqnet.dispatcher.types import DispatcherDiagnostics

def make_diagnostics(
    pi: Tensor,
    p_cert: Tensor,
    Q: Tensor,
    mu: Tensor,
    beta: float,
    C: float,
    usage: Tensor | None = None,
) -> DispatcherDiagnostics:
    nan = torch.full((Q.shape[0],), float("nan"), device=Q.device)
    usage_v = torch.zeros_like(nan) if usage is None else usage
    y = Q / mu.pow(beta).clamp_min(torch.finfo(mu.dtype).tiny)
    A_cert = arrival_coordinate(p_cert, y)
    A_final = arrival_coordinate(pi, y)
    m_q = y.min(dim=-1).values
    B_q = m_q + C
    return DispatcherDiagnostics(
        A_cert=A_cert,
        A_proposal=A_final,
        A_final=A_final,
        m_Q=m_q,
        B_Q=B_q,
        certificate_slack=B_q - A_final,
        usage_raw=usage_v,
        usage_final=usage_v,
        usage_cap=nan,
        fallback_active=torch.zeros(Q.shape[0], dtype=torch.bool, device=Q.device),
        correction_magnitude=torch.zeros_like(nan),
        policy_entropy=policy_entropy(pi),
        selected_resource=pi.argmax(dim=-1),
        pressure_mean=torch.zeros_like(nan),
        pressure_max=torch.zeros_like(nan),
        pressure_update_norm=torch.zeros_like(nan),
        projection_nu=torch.zeros_like(nan),
        projection_active=torch.zeros(Q.shape[0], dtype=torch.bool, device=Q.device),
        proposal_slack=B_q - A_final,
        solver_status=torch.zeros(Q.shape[0], dtype=torch.long, device=Q.device),
    )


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
        return pi, make_diagnostics(pi, pi, Q, mu_b, self.beta, self.C)
