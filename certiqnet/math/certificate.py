"""Certificate quantities for CertiQ-Net z2."""

from dataclasses import dataclass

import torch
from torch import Tensor

from certiqnet.math.lyapunov import arrival_envelope_A as _arrival_envelope_A
from certiqnet.math.lyapunov import min_coord


def arrival_envelope_A(pi: Tensor, Q: Tensor, mu: Tensor, beta: float) -> Tensor:
    """Return ``A_pi(Q) = sum_i pi_i Q_i / mu_i^beta``."""
    return _arrival_envelope_A(pi, Q, mu, beta)


def certified_envelope_B(Q: Tensor, mu: Tensor, beta: float, C_B: float) -> Tensor:
    """Return certified arrival envelope ``B(Q) = m(Q) + C_B``."""
    assert C_B >= 0, "C_B must be non-negative."
    return min_coord(Q, mu, beta) + C_B


def drift_slack(pi: Tensor, Q: Tensor, mu: Tensor, beta: float, C_B: float) -> Tensor:
    """Return ``B(Q) - A_pi(Q)``. Non-negative means the certificate is satisfied."""
    return certified_envelope_B(Q, mu, beta, C_B) - arrival_envelope_A(pi, Q, mu, beta)


@dataclass(frozen=True)
class CertificateDiagnostics:
    """All 13 mandatory certificate diagnostics from the z2 specification."""

    A_base: Tensor
    A_nn: Tensor
    A_pi: Tensor
    m_Q: Tensor
    B_Q: Tensor
    drift_slack: Tensor
    eta_raw: Tensor
    eta_final: Tensor
    eta_safe: Tensor
    fallback_active: Tensor
    residual_norm: Tensor
    policy_entropy: Tensor
    selected_resource: Tensor


def policy_entropy(pi: Tensor) -> Tensor:
    """Return categorical entropy for each batch element."""
    assert (pi >= 0).all(), "Routing probabilities must be non-negative."
    return -(pi * pi.clamp_min(1e-9).log()).sum(dim=-1)


def project_probability_vector(pi: Tensor) -> Tensor:
    """Project a possibly invalid probability tensor back onto the simplex.

    This is a defensive repair step for training-time instability.  It clips
    negative and non-finite values to zero, then renormalizes rows.  Rows that
    collapse to all zeros are replaced with a uniform distribution.
    """
    pi = torch.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    row_sum = pi.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(pi, 1.0 / pi.shape[-1])
    pi = torch.where(row_sum > 0, pi / row_sum.clamp_min(1e-12), uniform)
    return pi


def validate_probability_vector(pi: Tensor, atol: float = 1e-5) -> None:
    """Assert non-negativity and normalization of a probability tensor."""
    min_pi = float(pi.min().item())
    assert (pi >= -atol).all(), f"Negative routing probability detected (min={min_pi:.3e})."
    assert torch.allclose(
        pi.sum(dim=-1), torch.ones_like(pi.sum(dim=-1)), atol=atol
    ), "Routing probabilities must sum to one."
