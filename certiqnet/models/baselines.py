"""Simple comparison policies with z3 dispatcher diagnostics."""

import torch
import torch.nn as nn
from torch import Tensor

from certiqnet.dispatcher.certificate import arrival_coordinate, normalize_policy, policy_entropy
from certiqnet.dispatcher.geometry import CertifiedGeometry
from certiqnet.dispatcher.types import DispatcherDiagnostics


def _expand_mu(Q: Tensor, mu: Tensor) -> Tensor:
    if mu.dim() == 1:
        return mu.unsqueeze(0).expand(Q.shape[0], -1)
    return mu


def _baseline_device(module: nn.Module, reference: Tensor) -> torch.device:
    param = next(module.parameters(), None)
    return reference.device if param is None else param.device


def _diag(
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
        projection_slack=B_q - A_final,
    )


class AnalyticBackbonePolicy(nn.Module):
    """Pure analytic backbone baseline."""

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
        device = _baseline_device(self, Q)
        Q = Q.to(device=device)
        mu = mu.to(device=device, dtype=Q.dtype)
        mu_b = _expand_mu(Q, mu)
        pi, _ = self.geometry.policy(Q, mu_b)
        pi = normalize_policy(pi)
        return pi, _diag(pi, pi, Q, mu_b, self.beta, self.C)


class RandomPolicy(nn.Module):
    """Uniform random routing baseline."""

    def __init__(self, N: int, beta: float = 1.0, C: float = float("inf")) -> None:
        super().__init__()
        self.N = N
        self.beta = beta
        self.C = C

    def forward(
        self, Q: Tensor, mu: Tensor, xi: Tensor | None = None, training_mode: bool = False
    ) -> tuple[Tensor, DispatcherDiagnostics]:
        del xi, training_mode
        device = _baseline_device(self, Q)
        Q = Q.to(device=device)
        mu = mu.to(device=device, dtype=Q.dtype)
        mu_b = _expand_mu(Q, mu)
        pi = torch.full_like(Q, 1.0 / self.N)
        pi = normalize_policy(pi)
        return pi, _diag(pi, pi, Q, mu_b, self.beta, self.C)


class JoinShortestWeightedQueue(nn.Module):
    """Greedy route to the shortest weighted queue ``Q_i / mu_i``."""

    def __init__(self, N: int, beta: float = 1.0, C: float = float("inf")) -> None:
        super().__init__()
        self.N = N
        self.beta = beta
        self.C = C

    def forward(
        self, Q: Tensor, mu: Tensor, xi: Tensor | None = None, training_mode: bool = False
    ) -> tuple[Tensor, DispatcherDiagnostics]:
        del xi, training_mode
        device = _baseline_device(self, Q)
        Q = Q.to(device=device)
        mu = mu.to(device=device, dtype=Q.dtype)
        mu_b = _expand_mu(Q, mu)
        idx = (Q / mu_b.pow(self.beta)).argmin(dim=-1)
        pi = torch.zeros_like(Q)
        pi.scatter_(1, idx.unsqueeze(-1), 1.0)
        pi = normalize_policy(pi)
        return pi, _diag(pi, pi, Q, mu_b, self.beta, self.C)


class ShortestExpectedDelay(nn.Module):
    """Greedy route to the shortest expected delay ``(Q_i + 1) / mu_i``."""

    def __init__(self, N: int, beta: float = 1.0, C: float = float("inf")) -> None:
        super().__init__()
        self.N = N
        self.beta = beta
        self.C = C

    def forward(
        self, Q: Tensor, mu: Tensor, xi: Tensor | None = None, training_mode: bool = False
    ) -> tuple[Tensor, DispatcherDiagnostics]:
        del xi, training_mode
        device = _baseline_device(self, Q)
        Q = Q.to(device=device)
        mu = mu.to(device=device, dtype=Q.dtype)
        mu_b = _expand_mu(Q, mu)
        idx = ((Q + 1.0) / mu_b).argmin(dim=-1)
        pi = torch.zeros_like(Q)
        pi.scatter_(1, idx.unsqueeze(-1), 1.0)
        pi = normalize_policy(pi)
        return pi, _diag(pi, pi, Q, mu_b, self.beta, self.C)


class QuadraticMinDrift(nn.Module):
    """Greedy route minimizing quadratic drift ``(2 * Q_i + 1) / mu_i``."""

    def __init__(self, N: int, beta: float = 1.0, C: float = float("inf")) -> None:
        super().__init__()
        self.N = N
        self.beta = beta
        self.C = C

    def forward(
        self, Q: Tensor, mu: Tensor, xi: Tensor | None = None, training_mode: bool = False
    ) -> tuple[Tensor, DispatcherDiagnostics]:
        del xi, training_mode
        device = _baseline_device(self, Q)
        Q = Q.to(device=device)
        mu = mu.to(device=device, dtype=Q.dtype)
        mu_b = _expand_mu(Q, mu)
        idx = ((2.0 * Q + 1.0) / mu_b).argmin(dim=-1)
        pi = torch.zeros_like(Q)
        pi.scatter_(1, idx.unsqueeze(-1), 1.0)
        pi = normalize_policy(pi)
        return pi, _diag(pi, pi, Q, mu_b, self.beta, self.C)


class SoftSED(nn.Module):
    """Stochastic route with softmax over expected-delay indices."""

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
        device = _baseline_device(self, Q)
        Q = Q.to(device=device)
        mu = mu.to(device=device, dtype=Q.dtype)
        mu_b = _expand_mu(Q, mu)
        logits = -((Q + 1.0) / mu_b) / self.tau
        pi = torch.softmax(logits, dim=-1)
        pi = normalize_policy(pi)
        return pi, _diag(pi, pi, Q, mu_b, self.beta, self.C)


class SoftQuadraticMinDrift(nn.Module):
    """Stochastic route with softmax over quadratic-drift indices."""

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
        device = _baseline_device(self, Q)
        Q = Q.to(device=device)
        mu = mu.to(device=device, dtype=Q.dtype)
        mu_b = _expand_mu(Q, mu)
        logits = -((2.0 * Q + 1.0) / mu_b) / self.tau
        pi = torch.softmax(logits, dim=-1)
        pi = normalize_policy(pi)
        return pi, _diag(pi, pi, Q, mu_b, self.beta, self.C)
