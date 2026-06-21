"""Lyapunov geometry for CertiQ-Net z2."""

from torch import Tensor


def weighted_coord(Q: Tensor, mu: Tensor, beta: float) -> Tensor:
    """η_i = (1 / μ_i)^β."""
    return (1.0 / mu).pow(beta) * Q


def lyapunov_V(Q: Tensor, mu: Tensor, beta: float) -> Tensor:
    """Lyapunov function V(Q) = sum_i η_i = sum_i Q_i / μ_i^β."""
    return weighted_coord(Q, mu, beta).sum(dim=-1)


def min_coord(Q: Tensor, mu: Tensor, beta: float) -> Tensor:
    """Smallest coordinate weighting by (1/μ_i)^β."""
    return weighted_coord(Q, mu, beta).min(dim=-1).values


def tail_size(Q: Tensor, mu: Tensor, beta: float) -> Tensor:
    """Tail size: V(Q) / min_i η_i."""
    V = lyapunov_V(Q, mu, beta)
    m = min_coord(Q, mu, beta).clamp_min(1e-12)
    return V / m


def R0_bound(lam: float, mu: Tensor, beta: float, pi: Tensor) -> Tensor:
    """Radius bound R0 based on arrival rate and stationary distribution."""
    mu_beta = mu.pow(-beta)
    return (lam * (mu_beta * pi).sum()).sqrt()


def C0_bound(lam: float, mu: Tensor, beta: float) -> float:
    """Scalar bound C0 based on arrival rate and service rates."""
    mu_inv = 1.0 / mu
    return (lam * mu_inv.max()).sqrt().item()


def arrival_envelope_A(pi: Tensor, Q: Tensor, mu: Tensor, beta: float) -> Tensor:
    """Arrival envelope A(π) at current state."""
    eta = weighted_coord(Q, mu, beta)
    return (pi * eta).sum(dim=-1)


def generator_drift_V(Q: Tensor, mu: Tensor, pi: Tensor, lam: float, beta: float) -> Tensor:
    """Generator drift of Lyapunov function at state Q."""
    eta = weighted_coord(Q, mu, beta)
    return lam * (pi * eta).sum(dim=-1) - (mu * eta).sum(dim=-1)
