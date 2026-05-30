"""Baseline policies with CertiQ-Net-compatible diagnostics."""

import torch
import torch.nn as nn
from torch import Tensor

from certiqnet.math.backbone_params import ConstrainedBackboneParams
from certiqnet.math.certificate import (
    CertificateDiagnostics,
    arrival_envelope_A,
    policy_entropy,
    project_probability_vector,
    validate_probability_vector,
)
from certiqnet.math.lyapunov import min_coord
from certiqnet.models.backbone import AnalyticBackbone
from certiqnet.models.certiqnet_s import CertiQNetS, _expand_mu
from certiqnet.utils.config_schemas import CertiQNetSConfig


def _diag(
    pi: Tensor,
    p_base: Tensor,
    Q: Tensor,
    mu: Tensor,
    beta: float,
    C_B: float,
    eta: Tensor | None = None,
) -> CertificateDiagnostics:
    nan = torch.full((Q.shape[0],), float("nan"), device=Q.device)
    eta_v = torch.zeros_like(nan) if eta is None else eta
    A_base = arrival_envelope_A(p_base, Q, mu, beta)
    A_pi = arrival_envelope_A(pi, Q, mu, beta)
    m_q = min_coord(Q, mu, beta)
    B_q = m_q + C_B
    return CertificateDiagnostics(
        A_base=A_base,
        A_nn=A_pi,
        A_pi=A_pi,
        m_Q=m_q,
        B_Q=B_q,
        drift_slack=B_q - A_pi,
        eta_raw=eta_v,
        eta_final=eta_v,
        eta_safe=nan,
        fallback_active=torch.zeros(Q.shape[0], dtype=torch.bool, device=Q.device),
        residual_norm=torch.zeros_like(nan),
        policy_entropy=policy_entropy(pi),
        selected_resource=pi.argmax(dim=-1),
    )


class AnalyticBackbonePolicy(nn.Module):
    """Pure analytic backbone baseline."""

    def __init__(self, N: int, beta: float = 1.0, C_B: float = float("inf")) -> None:
        super().__init__()
        self.N = N
        self.beta = beta
        self.C_B = C_B
        self.backbone = AnalyticBackbone(ConstrainedBackboneParams(beta_init=beta))

    def forward(
        self, Q: Tensor, mu: Tensor, xi: Tensor | None = None, training_mode: bool = False
    ) -> tuple[Tensor, CertificateDiagnostics]:
        del xi, training_mode
        mu_b = _expand_mu(Q, mu)
        pi, _ = self.backbone(Q, mu_b)
        pi = project_probability_vector(pi)
        validate_probability_vector(pi)
        return pi, _diag(pi, pi, Q, mu_b, self.beta, self.C_B)


class RandomPolicy(nn.Module):
    """Uniform random routing baseline."""

    def __init__(self, N: int, beta: float = 1.0, C_B: float = float("inf")) -> None:
        super().__init__()
        self.N = N
        self.beta = beta
        self.C_B = C_B

    def forward(
        self, Q: Tensor, mu: Tensor, xi: Tensor | None = None, training_mode: bool = False
    ) -> tuple[Tensor, CertificateDiagnostics]:
        del xi, training_mode
        mu_b = _expand_mu(Q, mu)
        pi = torch.full_like(Q, 1.0 / self.N)
        pi = project_probability_vector(pi)
        return pi, _diag(pi, pi, Q, mu_b, self.beta, self.C_B)


class JoinShortestWeightedQueue(nn.Module):
    """Greedy route to the shortest weighted queue ``Q_i / mu_i``."""

    def __init__(self, N: int, beta: float = 1.0, C_B: float = float("inf")) -> None:
        super().__init__()
        self.N = N
        self.beta = beta
        self.C_B = C_B

    def forward(
        self, Q: Tensor, mu: Tensor, xi: Tensor | None = None, training_mode: bool = False
    ) -> tuple[Tensor, CertificateDiagnostics]:
        del xi, training_mode
        mu_b = _expand_mu(Q, mu)
        idx = (Q / mu_b.pow(self.beta)).argmin(dim=-1)
        pi = torch.zeros_like(Q)
        pi.scatter_(1, idx.unsqueeze(-1), 1.0)
        pi = project_probability_vector(pi)
        return pi, _diag(pi, pi, Q, mu_b, self.beta, self.C_B)


class CertiQNetXAblation(CertiQNetS):
    """NOT CERTIFIED: neural residual mixture with gate forced to one."""

    def forward(
        self, Q: Tensor, mu: Tensor, xi: Tensor | None = None, training_mode: bool = False
    ) -> tuple[Tensor, CertificateDiagnostics]:
        pi, diag = super().forward(Q, mu, xi, training_mode=True)
        return pi, diag


class CertiQNetSAblation_NoGate(CertiQNetS):
    """NOT CERTIFIED: residual active with hard gate disabled."""

    def forward(
        self, Q: Tensor, mu: Tensor, xi: Tensor | None = None, training_mode: bool = False
    ) -> tuple[Tensor, CertificateDiagnostics]:
        del training_mode
        return super().forward(Q, mu, xi, training_mode=True)


class CertiQNetSAblation_LearnedPhiOnly(AnalyticBackbonePolicy):
    """Backbone with learned phi and no neural residual."""


def build_default_s_ablation(N: int) -> CertiQNetS:
    """Construct a default S model for ablation launchers."""
    return CertiQNetS(CertiQNetSConfig(R_cert=50.0, beta=1.0), N=N)
