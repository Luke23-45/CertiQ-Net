"""Shared helpers used across all baseline policies."""

import torch
import torch.nn as nn
from torch import Tensor

from certiqnet.dispatcher.certiq.certificate import arrival_coordinate, policy_entropy
from certiqnet.dispatcher.types import DispatcherDiagnostics


def expand_mu(Q: Tensor, mu: Tensor) -> Tensor:
    if mu.dim() == 1:
        return mu.unsqueeze(0).expand(Q.shape[0], -1)
    return mu


def baseline_device(module: nn.Module, reference: Tensor) -> torch.device:
    param = next(module.parameters(), None)
    return reference.device if param is None else param.device


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
