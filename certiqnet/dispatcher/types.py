"""Typed public objects for the z3 CertiQ Dispatcher."""

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class DispatcherDiagnostics:
    """Auditable certificate and proposal diagnostics for one forward pass."""

    A_cert: Tensor
    A_proposal: Tensor
    A_final: Tensor
    m_Q: Tensor
    B_Q: Tensor
    certificate_slack: Tensor
    usage_raw: Tensor
    usage_final: Tensor
    usage_cap: Tensor
    fallback_active: Tensor
    correction_magnitude: Tensor
    policy_entropy: Tensor
    selected_resource: Tensor
    pressure_mean: Tensor
    pressure_max: Tensor
    pressure_update_norm: Tensor
    projection_nu: Tensor
    projection_active: Tensor
    projection_slack: Tensor


@dataclass(frozen=True)
class DispatcherForward:
    """Full dispatcher output used by training and evaluation."""

    pi: Tensor
    diagnostics: DispatcherDiagnostics
    value: Tensor
    p_cert: Tensor
    p_proposal: Tensor
    usage_raw: Tensor
    usage_final: Tensor
    proposal_logits: Tensor
    index_values: Tensor | None = None
