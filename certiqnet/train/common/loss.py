"""Training loss components for CertiQ‑Net (formal 3-term objective)."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CertiQNetLoss(nn.Module):
    """Total loss matching the formal definition (3 terms).

    .. math::
        \\mathcal L(\\Theta) =
        \\omega_{\\mathrm{roll}} \\mathcal L_{\\mathrm{roll}}
        - \\omega_{\\mathrm{ent}} \\mathcal L_{\\mathrm{ent}}
        + \\omega_{\\mathrm{kl}} \\mathcal L_{\\mathrm{kl}}
    """

    def __init__(
        self,
        omega_roll: float = 1.0,
        omega_ent: float = 0.001,
        omega_kl: float = 0.05,
    ) -> None:
        super().__init__()
        self.omega_roll = omega_roll
        self.omega_ent = omega_ent
        self.omega_kl = omega_kl

    def rollout_cost(self, cost_trace: Tensor, dt_trace: Tensor) -> Tensor:
        """Time-weighted average backlog (evaluation metric)."""
        weights = dt_trace.clamp_min(1e-9)
        total_time = weights.sum().clamp_min(1e-9)
        return (cost_trace * weights).sum() / total_time

    def entropy_term(self, pi: Tensor) -> Tensor:
        """Policy entropy :math:`\\mathcal L_{\\mathrm{ent}} = H(\\pi)`."""
        return -(pi * pi.clamp_min(1e-8).log()).sum(dim=-1).mean()

    def proposal_kl(self, proposal_logits: Tensor, p_cert: Tensor) -> Tensor:
        """KL divergence from proposal to certified policy
        :math:`\\mathcal L_{\\mathrm{kl}} = \\mathrm{KL}(q_\\Theta \\| \\pi^\\star)`,
        with the certified side **detached** (stop-gradient)."""
        q = torch.softmax(proposal_logits, dim=-1)
        p = p_cert.detach()
        return F.kl_div(p.clamp_min(1e-9).log(), q, reduction="batchmean")

    def forward(
        self,
        proposal_logits: Tensor,
        p_cert: Tensor,
        rollout_log_probs: Tensor | None = None,
        rollout_returns: Tensor | None = None,
    ) -> dict[str, Tensor]:
        q = torch.softmax(proposal_logits, dim=-1)
        L_ent = self.entropy_term(q)
        L_kl = self.proposal_kl(proposal_logits, p_cert)

        L_roll = torch.zeros((), device=proposal_logits.device, dtype=proposal_logits.dtype)
        if rollout_log_probs is not None and rollout_returns is not None:
            L_roll = -(rollout_log_probs * rollout_returns.detach()).mean()

        total = (
            self.omega_roll * L_roll
            - self.omega_ent * L_ent
            + self.omega_kl * L_kl
        )
        return {
            "total": total,
            "entropy": L_ent,
            "kl": L_kl,
            "roll": L_roll,
        }
