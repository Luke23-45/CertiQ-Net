"""Training loss components for CertiQ‑Net."""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from certiqnet.dispatcher.types import DispatcherDiagnostics


class CertiQNetLoss(nn.Module):
    """Total loss with individually logged z3 dispatcher components.

    Parameters are exposed as individual ``__init__`` arguments so that
    ``LightningCLI`` can inject them via ``class_path`` / ``init_args``.
    """

    def __init__(
        self,
        omega_bc: float = 1.0,
        omega_action: float = 1.5,
        omega_margin: float = 0.1,
        omega_usage: float = 0.1,
        omega_certificate: float = 5.0,
        omega_correction: float = 0.01,
        rollout_weight: float = 1.0,
        policy_kl_weight: float = 0.05,
        value_weight: float = 1.0,
        entropy_weight: float = 0.001,
    ) -> None:
        super().__init__()
        self.omega_bc = omega_bc
        self.omega_action = omega_action
        self.omega_margin = omega_margin
        self.omega_usage = omega_usage
        self.omega_certificate = omega_certificate
        self.omega_correction = omega_correction
        self.rollout_weight = rollout_weight
        self.policy_kl_weight = policy_kl_weight
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight

    def rollout_cost(self, cost_trace: Tensor, dt_trace: Tensor) -> Tensor:
        weights = dt_trace.clamp_min(1e-9)
        total_time = weights.sum().clamp_min(1e-9)
        return (cost_trace * weights).sum() / total_time

    def bc_loss(self, pi: Tensor, p_target: Tensor | None = None) -> Tensor:
        if p_target is None:
            return torch.zeros((), device=pi.device, dtype=pi.dtype)
        p_target = p_target.to(device=pi.device, dtype=pi.dtype)
        return F.kl_div(pi.clamp_min(1e-9).log(), p_target, reduction="batchmean")

    def action_loss(self, logits: Tensor, target: Tensor | None = None) -> Tensor:
        if target is None:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(logits, target.to(device=logits.device, dtype=torch.long))

    def margin_loss(self, logits: Tensor, target: Tensor | None = None) -> Tensor:
        if target is None:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        target = target.to(device=logits.device, dtype=torch.long)
        target_logit = logits.gather(1, target.unsqueeze(-1)).squeeze(-1)
        masked = logits.clone()
        masked.scatter_(1, target.unsqueeze(-1), float("-inf"))
        runner_up = masked.max(dim=-1).values
        return F.relu(1.0 - (target_logit - runner_up)).mean()

    def usage_penalty(self, usage: Tensor) -> Tensor:
        return (1.0 - usage).square().mean()

    def certificate_penalty(self, diag: DispatcherDiagnostics) -> Tensor:
        return (diag.A_proposal - diag.B_Q).clamp(min=0.0).square().mean()

    def correction_size_penalty(self, diag: DispatcherDiagnostics) -> Tensor:
        return diag.correction_magnitude.square().mean()

    def entropy_term(self, pi: Tensor) -> Tensor:
        return -(pi * pi.clamp_min(1e-8).log()).sum(dim=-1).mean()

    def policy_kl(self, pi: Tensor, ref_pi: Tensor) -> Tensor:
        return F.kl_div(pi.clamp_min(1e-9).log(), ref_pi, reduction="batchmean")

    def actor_loss(self, log_prob: Tensor, advantage: Tensor) -> Tensor:
        return -(log_prob * advantage.detach()).mean()

    def ppo_actor_loss(
        self,
        log_prob: Tensor,
        old_log_prob: Tensor,
        advantage: Tensor,
        clip_epsilon: float = 0.2,
    ) -> Tensor:
        ratio = (log_prob - old_log_prob).exp()
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
        return -torch.min(surr1, surr2).mean()

    def critic_loss(self, value: Tensor, target_return: Tensor) -> Tensor:
        return 0.5 * (value - target_return.detach()).square().mean()

    def ppo_critic_loss(
        self,
        value: Tensor,
        old_value: Tensor,
        target_return: Tensor,
        clip_epsilon: float = 0.2,
    ) -> Tensor:
        value_clipped = old_value + torch.clamp(value - old_value, -clip_epsilon, clip_epsilon)
        loss_unclipped = 0.5 * (value - target_return.detach()).square()
        loss_clipped = 0.5 * (value_clipped - target_return.detach()).square()
        return torch.max(loss_unclipped, loss_clipped).mean()

    def forward(
        self,
        pi: Tensor,
        diag: DispatcherDiagnostics,
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
        old_log_prob: Tensor | None = None,
        old_value: Tensor | None = None,
        use_ppo: bool = False,
        ppo_clip_epsilon: float = 0.2,
    ) -> dict[str, Tensor]:
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
            if use_ppo and old_log_prob is not None:
                L_actor = self.ppo_actor_loss(
                    actor_log_prob, old_log_prob, advantage, ppo_clip_epsilon
                )
            else:
                L_actor = self.actor_loss(actor_log_prob, advantage)
        L_critic = torch.zeros((), device=pi.device, dtype=pi.dtype)
        if value is not None and target_return is not None:
            if use_ppo and old_value is not None:
                L_critic = self.ppo_critic_loss(
                    value, old_value, target_return, ppo_clip_epsilon
                )
            else:
                L_critic = self.critic_loss(value, target_return)
        ent_w = float(self.entropy_weight if entropy_weight is None else entropy_weight)
        total = (
            self.rollout_weight * L_actor
            + self.omega_bc * L_bc
            + self.omega_action * L_action
            + self.omega_margin * L_margin
            + self.omega_usage * L_usage
            + self.omega_certificate * L_certificate
            + self.omega_correction * L_correction
            - ent_w * L_ent
            + self.policy_kl_weight * L_kl
            + self.value_weight * L_critic
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
