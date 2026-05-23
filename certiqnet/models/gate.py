"""Gate mechanisms for CertiQ-Net."""

import torch
import torch.nn as nn
from torch import Tensor

from certiqnet.math.certificate import arrival_envelope_A
from certiqnet.math.lyapunov import min_coord, tail_size


class RawGate(nn.Module):
    """Permutation-invariant scalar raw gate ``eta_raw``."""

    def __init__(self, d_global: int, eta_max: float = 1.0) -> None:
        super().__init__()
        assert 0 < eta_max <= 1.0, "eta_max must be in (0, 1]."
        self.eta_max = eta_max
        self.linear = nn.Linear(d_global, 1)

    def forward(self, g: Tensor) -> Tensor:
        """Return ``eta_raw`` with shape ``(B,)``."""
        return self.eta_max * torch.sigmoid(self.linear(g).squeeze(-1))


def hard_tail_fallback(
    eta_raw: Tensor, Q: Tensor, mu: Tensor, beta: float, R_cert: float
) -> tuple[Tensor, Tensor]:
    """Apply exact Route-A hard fallback gate."""
    assert R_cert >= 0, "R_cert must be non-negative."
    inside = tail_size(Q, mu, beta) <= R_cert
    eta = torch.where(inside, eta_raw, torch.zeros_like(eta_raw))
    return eta, ~inside


def smooth_tail_gate(
    eta_raw: Tensor,
    Q: Tensor,
    mu: Tensor,
    beta: float,
    R_cert: float,
    tau: float = 10.0,
) -> Tensor:
    """Return differentiable training surrogate for Route A only."""
    return eta_raw * torch.sigmoid(tau * (R_cert - tail_size(Q, mu, beta)))


def drift_envelope_projection(
    eta_raw: Tensor,
    Q: Tensor,
    mu: Tensor,
    beta: float,
    C_B: float,
    p_base: Tensor,
    p_nn: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply exact Route-B drift-envelope projection."""
    assert C_B < float("inf"), "C_B must be finite for projection."
    A_base = arrival_envelope_A(p_base, Q, mu, beta)
    A_nn = arrival_envelope_A(p_nn, Q, mu, beta)
    B = min_coord(Q, mu, beta) + C_B
    denom = (A_nn - A_base).clamp(min=1e-8)
    eta_safe_raw = (B - A_base) / denom
    eta_safe = torch.where(
        A_nn <= A_base,
        torch.ones_like(eta_raw),
        eta_safe_raw.clamp(0.0, 1.0),
    )
    eta_proj = torch.minimum(eta_raw, eta_safe)
    return eta_proj, eta_safe
