"""Typed public objects for the z3 CertiQ Dispatcher."""

from dataclasses import dataclass, field

import torch

from torch import Tensor


@dataclass(frozen=True)
class DispatcherDiagnostics:
    """Diagnostics for one forward pass (constraint-violation fields are
    for logging / monitoring only — the Lagrangian dual variable is
    managed by the training module, not produced by the model)."""

    A_proposal: Tensor
    A_final: Tensor
    m_Q: Tensor
    B_Q: Tensor
    certificate_slack: Tensor
    constraint_violation: Tensor
    usage_raw: Tensor
    usage_final: Tensor
    usage_cap: Tensor
    policy_entropy: Tensor
    selected_resource: Tensor
    pressure_mean: Tensor
    pressure_max: Tensor
    pressure_update_norm: Tensor
    projection_multiplier: Tensor = field(
        default_factory=lambda: torch.tensor(float("nan"))
    )
    projection_active: Tensor = field(
        default_factory=lambda: torch.tensor(float("nan"))
    )
    solver_status: Tensor = field(default_factory=lambda: torch.tensor(float("nan")))
    fallback_flag: Tensor = field(default_factory=lambda: torch.tensor(float("nan")))
    correction_magnitude: Tensor = field(
        default_factory=lambda: torch.tensor(float("nan"))
    )


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
