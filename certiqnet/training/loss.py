"""Training loss components for CertiQ-Net."""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from certiqnet.dispatcher.types import DispatcherDiagnostics
from certiqnet.utils.config_schemas import LossConfig


class CertiQNetLoss(nn.Module):
    """Total loss with individually logged z3 dispatcher components."""

    def __init__(self, loss_cfg: LossConfig) -> None:
        super().__init__()
        self.cfg = loss_cfg

    def rollout_cost(self, cost_trace: Tensor, dt_trace: Tensor) -> Tensor:
        """Return a time-weighted average queueing cost."""
        weights = dt_trace.clamp_min(1e-9)
        total_time = weights.sum().clamp_min(1e-9)
        return (cost_trace * weights).sum() / total_time

    def bc_loss(self, pi: Tensor, p_target: Tensor | None = None) -> Tensor:
        """KL imitation loss, zero when no curriculum target is provided."""
        if p_target is None:
            return torch.zeros((), device=pi.device, dtype=pi.dtype)
        p_target = p_target.to(device=pi.device, dtype=pi.dtype)
        return torch.nn.functional.kl_div(pi.clamp_min(1e-9).log(), p_target, reduction="batchmean")

    def action_loss(self, logits: Tensor, target: Tensor | None = None) -> Tensor:
        """Cross-entropy against a heuristic action label."""
        if target is None:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(logits, target.to(device=logits.device, dtype=torch.long))

    def margin_loss(self, logits: Tensor, target: Tensor | None = None) -> Tensor:
        """Encourage the target action logit to exceed the runner-up."""
        if target is None:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        target = target.to(device=logits.device, dtype=torch.long)
        target_logit = logits.gather(1, target.unsqueeze(-1)).squeeze(-1)
        masked = logits.clone()
        masked.scatter_(1, target.unsqueeze(-1), float("-inf"))
        runner_up = masked.max(dim=-1).values
        return F.relu(1.0 - (target_logit - runner_up)).mean()

    def usage_penalty(self, usage: Tensor) -> Tensor:
        """Quadratic penalty encouraging useful proposal usage."""
        return (1.0 - usage).square().mean()

    def certificate_penalty(self, diag: DispatcherDiagnostics) -> Tensor:
        """Squared positive certificate violation penalty."""
        return (diag.A_final - diag.B_Q).clamp(min=0.0).square().mean()

    def correction_size_penalty(self, diag: DispatcherDiagnostics) -> Tensor:
        """Proposal correction size penalty."""
        return diag.correction_magnitude.square().mean()

    def entropy_term(self, pi: Tensor) -> Tensor:
        """Categorical entropy term."""
        return -(pi * pi.clamp_min(1e-8).log()).sum(dim=-1).mean()

    def policy_kl(self, pi: Tensor, ref_pi: Tensor) -> Tensor:
        """KL divergence from the reference policy."""
        return torch.nn.functional.kl_div(pi.clamp_min(1e-9).log(), ref_pi, reduction="batchmean")

    def actor_loss(self, log_prob: Tensor, advantage: Tensor) -> Tensor:
        """Simple actor loss for on-policy advantage estimates."""
        return -(log_prob * advantage.detach()).mean()

    def critic_loss(self, value: Tensor, target_return: Tensor) -> Tensor:
        """Squared-error critic loss."""
        return 0.5 * (value - target_return.detach()).square().mean()

    def forward(
        self,
        pi: Tensor,
        diag: DispatcherDiagnostics,
        cfg: LossConfig,
        *,
        imitation_target: Tensor | None = None,
        action_target: Tensor | None = None,
        ref_pi: Tensor | None = None,
        actor_log_prob: Tensor | None = None,
        advantage: Tensor | None = None,
        value: Tensor | None = None,
        target_return: Tensor | None = None,
        policy_logits: Tensor | None = None,
        entropy_weight: float | None = None,
    ) -> dict[str, Tensor]:
        """Return all scalar loss components and total."""
        del cfg
        L_bc = self.bc_loss(pi, imitation_target)
        L_action = torch.zeros((), device=pi.device, dtype=pi.dtype)
        L_margin = torch.zeros((), device=pi.device, dtype=pi.dtype)
        if policy_logits is not None:
            L_action = self.action_loss(policy_logits, action_target)
            L_margin = self.margin_loss(policy_logits, action_target)
        L_usage = self.usage_penalty(diag.usage_final)
        L_certificate = self.certificate_penalty(diag)
        L_correction = self.correction_size_penalty(diag)
        L_ent = self.entropy_term(pi)
        L_kl = torch.zeros((), device=pi.device, dtype=pi.dtype)
        if ref_pi is not None:
            L_kl = self.policy_kl(pi, ref_pi)
        L_actor = torch.zeros((), device=pi.device, dtype=pi.dtype)
        if actor_log_prob is not None and advantage is not None:
            L_actor = self.actor_loss(actor_log_prob, advantage)
        L_critic = torch.zeros((), device=pi.device, dtype=pi.dtype)
        if value is not None and target_return is not None:
            L_critic = self.critic_loss(value, target_return)
        ent_w = float(self.cfg.entropy_weight if entropy_weight is None else entropy_weight)
        total = (
            self.cfg.rollout_weight * L_actor
            + self.cfg.omega_bc * L_bc
            + self.cfg.omega_action * L_action
            + self.cfg.omega_margin * L_margin
            + self.cfg.omega_usage * L_usage
            + self.cfg.omega_certificate * L_certificate
            + self.cfg.omega_correction * L_correction
            - ent_w * L_ent
            + self.cfg.policy_kl_weight * L_kl
            + self.cfg.value_weight * L_critic
        )
        return {
            "total": total,
            "actor": L_actor,
            "critic": L_critic,
            "bc": L_bc,
            "action": L_action,
            "margin": L_margin,
            "usage": L_usage,
            "certificate": L_certificate,
            "correction": L_correction,
            "kl": L_kl,
            "entropy": L_ent,
        }
