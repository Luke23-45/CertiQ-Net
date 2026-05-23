"""Lyapunov geometry for CertiQ-Net z2."""

import torch
from torch import Tensor


def _assert_same_last_dim(*tensors: Tensor) -> None:
    n = tensors[0].shape[-1]
    assert all(t.shape[-1] == n for t in tensors), "Last dimensions must match."


def weighted_coord(Q: Tensor, mu: Tensor, beta: float) -> Tensor:
    """Return weighted coordinates ``y_i(Q) = Q_i / mu_i^beta``."""
    _assert_same_last_dim(Q, mu)
    assert beta > 0, "Lyapunov beta must be strictly positive."
    assert (Q >= 0).all(), "Queue lengths must be non-negative."
    assert (mu > 0).all(), "Service rates must be strictly positive."
    return Q / mu.pow(beta)


def lyapunov_V(Q: Tensor, mu: Tensor, beta: float) -> Tensor:
    """Return ``V(Q) = 0.5 * sum_i Q_i^2 / mu_i^beta`` with shape ``(*)``."""
    y = weighted_coord(Q, mu, beta)
    return 0.5 * (Q * y).sum(dim=-1)


def min_coord(Q: Tensor, mu: Tensor, beta: float) -> Tensor:
    """Return ``m(Q) = min_i y_i(Q)`` with shape ``(*)``."""
    return weighted_coord(Q, mu, beta).min(dim=-1).values


def tail_size(Q: Tensor, mu: Tensor, beta: float) -> Tensor:
    """Return ``S_beta(Q) = sum_i Q_i / mu_i^beta`` with shape ``(*)``."""
    return weighted_coord(Q, mu, beta).sum(dim=-1)


def R0_bound(lam: float, mu: Tensor, beta: float, pi: Tensor) -> Tensor:
    """Return the exact boundary term ``R_0^pi(Q)`` without the indicator loss."""
    assert lam > 0, "Arrival rate lambda must be strictly positive."
    _assert_same_last_dim(mu, pi)
    assert (mu > 0).all(), "Service rates must be strictly positive."
    assert (pi >= 0).all(), "Routing probabilities must be non-negative."
    assert torch.allclose(
        pi.sum(dim=-1), torch.ones_like(pi.sum(dim=-1)), atol=1e-5
    ), "Routing probabilities must sum to one."
    term1 = (lam / 2.0) * (pi / mu.pow(beta)).sum(dim=-1)
    term2 = (0.5 * mu.pow(1.0 - beta)).sum(dim=-1)
    return term1 + term2


def C0_bound(lam: float, mu: Tensor, beta: float) -> float:
    """Return uniform upper bound ``C_0`` on ``R_0^pi(Q)``."""
    assert lam > 0, "Arrival rate lambda must be strictly positive."
    assert beta > 0, "Lyapunov beta must be strictly positive."
    assert (mu > 0).all(), "Service rates must be strictly positive."
    term1 = (lam / 2.0) * mu.pow(-beta).max().item()
    term2 = (0.5 * mu.pow(1.0 - beta)).sum().item()
    return float(term1 + term2)


def arrival_envelope_A(pi: Tensor, Q: Tensor, mu: Tensor, beta: float) -> Tensor:
    """Return ``A_pi(Q) = sum_i pi_i Q_i / mu_i^beta``."""
    _assert_same_last_dim(pi, Q, mu)
    assert (pi >= 0).all(), "Routing probabilities must be non-negative."
    assert torch.allclose(
        pi.sum(dim=-1), torch.ones_like(pi.sum(dim=-1)), atol=1e-5
    ), "Routing probabilities must sum to one."
    return (pi * weighted_coord(Q, mu, beta)).sum(dim=-1)


def generator_drift_V(Q: Tensor, mu: Tensor, pi: Tensor, lam: float, beta: float) -> Tensor:
    """Return exact generator drift ``(L_pi V)(Q)`` from the CTMC identity."""
    A = arrival_envelope_A(pi, Q, mu, beta)
    service_term = (mu.pow(1.0 - beta) * Q).sum(dim=-1)
    r0 = R0_bound(lam, mu, beta, pi)
    return lam * A - service_term + r0
