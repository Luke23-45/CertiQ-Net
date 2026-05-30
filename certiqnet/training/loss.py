"""Training loss components for CertiQ-Net."""

import torch
import torch.nn as nn
from torch import Tensor

from certiqnet.math.certificate import CertificateDiagnostics
from certiqnet.utils.config_schemas import LossConfig


class CertiQNetLoss(nn.Module):
    """Total loss with individually logged components."""

    def __init__(self, loss_cfg: LossConfig) -> None:
        super().__init__()
        self.cfg = loss_cfg

    def rollout_cost(self, Q: Tensor, mu: Tensor, pi: Tensor, diag: CertificateDiagnostics) -> Tensor:
        """Arrival envelope ``A_pi(Q)`` as the training objective.

        Unlike the static sum-of-queues cost, ``A_pi = sum_i pi_i * Q_i / mu_i^beta``
        depends on the policy pi, giving a gradient signal that teaches the residual
        to route toward low-weighted-queue resources.
        """
        del Q, mu, pi
        return diag.A_pi.float().mean()

    def bc_loss(self, pi: Tensor, p_target: Tensor | None = None) -> Tensor:
        """KL imitation loss, zero when no curriculum target is provided."""
        if p_target is None:
            return torch.zeros((), device=pi.device, dtype=pi.dtype)
        return torch.nn.functional.kl_div(pi.clamp_min(1e-9).log(), p_target, reduction="batchmean")

    def gate_penalty(self, eta: Tensor) -> Tensor:
        """Quadratic penalty pushing gate toward open (eta=1)."""
        return (1.0 - eta).square().mean()

    def drift_penalty(self, diag: CertificateDiagnostics) -> Tensor:
        """Squared positive certificate violation penalty."""
        return (diag.A_pi - diag.B_Q).clamp(min=0.0).square().mean()

    def residual_size_penalty(self, diag: CertificateDiagnostics) -> Tensor:
        """Residual-size proxy from logged residual infinity norm."""
        return diag.residual_norm.square().mean()

    def entropy_term(self, pi: Tensor) -> Tensor:
        """Categorical entropy term."""
        return -(pi * pi.clamp_min(1e-8).log()).sum(dim=-1).mean()

    def forward(
        self,
        Q: Tensor,
        mu: Tensor,
        pi: Tensor,
        diag: CertificateDiagnostics,
        cfg: LossConfig,
    ) -> dict[str, Tensor]:
        """Return all scalar loss components and total."""
        J = self.rollout_cost(Q, mu, pi, diag)
        L_bc = self.bc_loss(pi)
        L_gate = self.gate_penalty(diag.eta_final)
        L_drift = self.drift_penalty(diag)
        L_res = self.residual_size_penalty(diag)
        L_ent = self.entropy_term(pi)
        total = (
            cfg.rollout_weight * J
            + cfg.omega_bc * L_bc
            + cfg.omega_gate * L_gate
            + cfg.omega_drift * L_drift
            + cfg.omega_res * L_res
            + cfg.omega_ent * L_ent
        )
        return {
            "total": total,
            "rollout": J,
            "bc": L_bc,
            "gate": L_gate,
            "drift": L_drift,
            "residual": L_res,
            "entropy": L_ent,
        }
