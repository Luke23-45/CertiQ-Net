"""Certificate operators for z3 dispatch."""

import torch
from torch import Tensor

from certiqnet.dispatcher.types import CertificateMode


def arrival_coordinate(pi: Tensor, y: Tensor) -> Tensor:
    """Return ``sum_i pi_i y_i``."""
    assert pi.shape == y.shape, "pi and y must have identical shape."
    return (pi * y).sum(dim=-1)


def policy_entropy(pi: Tensor) -> Tensor:
    """Return categorical entropy."""
    return -(pi * pi.clamp_min(1e-9).log()).sum(dim=-1)


def normalize_policy(pi: Tensor) -> Tensor:
    """Return a finite strictly positive simplex row for each batch item."""
    eps = torch.finfo(pi.dtype).tiny
    pi = torch.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    row_sum = pi.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(pi, 1.0 / pi.shape[-1])
    pi = torch.where(row_sum > 0, pi / row_sum.clamp_min(eps), uniform)
    pi = pi.clamp_min(eps)
    return pi / pi.sum(dim=-1, keepdim=True)


def certify_usage(
    *,
    mode: CertificateMode,
    usage_raw: Tensor,
    y: Tensor,
    B: Tensor,
    p_cert: Tensor,
    p_proposal: Tensor,
    fallback_radius: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(usage_final, usage_cap, fallback_active)``."""
    if mode == "uncertified":
        cap = torch.ones_like(usage_raw)
        return usage_raw, cap, torch.zeros_like(usage_raw, dtype=torch.bool)

    if mode == "fallback":
        tail = y.sum(dim=-1)
        inside = tail <= fallback_radius
        cap = inside.to(dtype=usage_raw.dtype)
        return usage_raw * cap, cap, ~inside

    if mode != "projection":
        raise ValueError(f"Unsupported certificate mode: {mode}")

    A_cert = arrival_coordinate(p_cert, y)
    A_prop = arrival_coordinate(p_proposal, y)
    denom = (A_prop - A_cert).clamp_min(torch.finfo(A_prop.dtype).eps)
    raw_cap = ((B - A_cert) / denom).clamp(0.0, 1.0)
    cap = torch.where(A_prop <= A_cert, torch.ones_like(usage_raw), raw_cap)
    return torch.minimum(usage_raw, cap), cap, torch.zeros_like(usage_raw, dtype=torch.bool)

