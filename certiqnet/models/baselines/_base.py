"""Shared helpers used across all baseline policies."""

import torch
import torch.nn as nn
from torch import Tensor

from certiqnet.dispatcher.types import DispatcherDiagnostics


def expand_mu(Q: Tensor, mu: Tensor) -> Tensor:
    if mu.dim() == 1:
        return mu.unsqueeze(0).expand(Q.shape[0], -1)
    return mu


def baseline_device(module: nn.Module, reference: Tensor) -> torch.device:
    param = next(module.parameters(), None)
    return reference.device if param is None else param.device


def make_lagrangian_diagnostics(
    pi: Tensor,
    Q: Tensor,
    mu: Tensor,
    beta: float,
    C: float,
    cost: Tensor | None = None,
) -> DispatcherDiagnostics:
    """Build DispatcherDiagnostics for a baseline policy (PPO-Lagrangian)."""
    batch = Q.shape[0]
    device = Q.device
    y = cost if cost is not None else Q / mu.pow(beta).clamp_min(torch.finfo(mu.dtype).tiny)
    a_final = (pi * y).sum(dim=-1)
    m_q = y.min(dim=-1).values
    B_q = m_q + C
    constraint_violation = (a_final - B_q).clamp(min=0.0)
    nan = torch.full((batch,), float("nan"), device=device)
    return DispatcherDiagnostics(
        A_proposal=a_final,
        A_final=a_final,
        m_Q=m_q,
        B_Q=B_q,
        certificate_slack=B_q - a_final,
        constraint_violation=constraint_violation,
        usage_raw=nan,
        usage_final=nan,
        usage_cap=nan,
        policy_entropy=-(pi * pi.clamp_min(torch.finfo(pi.dtype).eps).log()).sum(dim=-1),
        selected_resource=pi.argmax(dim=-1),
        pressure_mean=torch.zeros_like(nan),
        pressure_max=torch.zeros_like(nan),
        pressure_update_norm=torch.zeros_like(nan),
        projection_multiplier=torch.zeros_like(nan),
        projection_active=torch.zeros_like(nan),
        solver_status=torch.zeros_like(nan),
        fallback_flag=torch.zeros_like(nan),
        correction_magnitude=torch.zeros_like(nan),
    )



