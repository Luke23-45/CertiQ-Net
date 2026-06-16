"""Base LightningModule for CertiQ‑Net training (REINFORCE + PPO)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical
from torch import nn

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    pl = None

from certiqnet.dispatcher.types import DispatcherDiagnostics
from certiqnet.utils.ctmc import CTMCEnvironment
from certiqnet.train.common.loss import CertiQNetLoss
from certiqnet.dispatcher.delay_geometry import sed_index, quadratic_drift_index


def validation_selection_score(violation: Tensor, avg_cost: Tensor, p95_backlog: Tensor) -> Tensor:
    return violation * 1e9 + avg_cost * 1e3 + p95_backlog


def _stack_mean(diags: list[DispatcherDiagnostics]) -> DispatcherDiagnostics:
    fields = DispatcherDiagnostics.__dataclass_fields__.keys()
    values: dict[str, Tensor] = {}
    for name in fields:
        tensors = [getattr(diag, name) for diag in diags]
        if tensors[0].dtype == torch.bool:
            values[name] = torch.stack(tensors, dim=0).any(dim=0)
        elif tensors[0].dtype in {torch.int32, torch.int64, torch.long}:
            values[name] = tensors[-1]
        else:
            values[name] = torch.stack(tensors, dim=0).mean(dim=0).detach()
    return DispatcherDiagnostics(**values)


class BaseCertiQLightningModule(pl.LightningModule if pl is not None else nn.Module):
    """Base Lightning wrapper for on-policy rollout training.

    Subclasses override ``_make_observation`` to inject domain-specific
    observation transforms (e.g. from an adapter).
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: CertiQNetLoss = None,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        rollout_horizon: int = 16,
        use_ppo: bool = True,
        ppo_epochs: int = 4,
        ppo_clip_epsilon: float = 0.2,
        ppo_manual_clip_val: float = 1.0,
        entropy_warmup_epochs: int = 20,
        imitation_warmup_epochs: int = 0,
        expert_mode: str = "sed",
        critic_bootstrap_epochs: int = 3,
        imitation_decay_rate: float = 0.96,
        target_kl_cert: float = 0.01,
        initial_policy_kl_weight: float = 0.05,
        entropy_weight: float = 0.01,
        lam: float = 1.0,
        dual_lambda_lr: float = 0.01,
        dual_lambda_init: float = 0.0,
        dual_lambda_momentum: float = 0.9,
        dual_lr_warmup_steps: int = 10,
        dual_lambda_max: float = 10.0,
        dual_lr_decay: float = 1.0,
        # ── RL return estimation ──────────────────────────────────────
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        # ── Validation rollout ────────────────────────────────────────
        val_horizon_max: int = 8,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn if loss_fn is not None else CertiQNetLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.rollout_horizon = int(rollout_horizon)
        self.entropy_warmup_epochs = int(entropy_warmup_epochs)
        self.imitation_warmup_epochs = imitation_warmup_epochs
        self.critic_bootstrap_epochs = critic_bootstrap_epochs
        self.imitation_decay_rate = imitation_decay_rate
        self.target_kl_cert = target_kl_cert
        self.current_kl_weight = initial_policy_kl_weight
        self.epoch_kl_records = []
        self.use_ppo = bool(use_ppo)
        self.expert_mode = expert_mode
        self.ppo_epochs = int(ppo_epochs)
        self.ppo_clip_epsilon = float(ppo_clip_epsilon)
        self.entropy_weight = float(entropy_weight)
        self.lam = float(lam)
        self.dual_lambda_lr = float(dual_lambda_lr)
        self.dual_lambda_momentum = float(dual_lambda_momentum)
        self.dual_lr_warmup_steps = int(dual_lr_warmup_steps)
        self.dual_lambda_max = float(dual_lambda_max)
        self.dual_lr_decay = float(dual_lr_decay)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.val_horizon_max = int(val_horizon_max)
        self.register_buffer("dual_lambda", torch.tensor(float(dual_lambda_init)))
        self.register_buffer("_smoothed_residual", torch.tensor(0.0))
        self.register_buffer("_dual_update_count", torch.tensor(0, dtype=torch.long))
        if self.use_ppo:
            self.automatic_optimization = False
            self._gradient_clip_val = float(ppo_manual_clip_val)
        if hasattr(self, "save_hyperparameters"):
            self.save_hyperparameters(ignore=["model", "loss_fn"])

    def on_train_epoch_start(self) -> None:
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "resample_train_data"):
            dm.resample_train_data()

    # ── Scheduling helpers ──────────────────────────────────────────

    def _entropy_weight(self) -> float:
        epoch = int(getattr(self.trainer, "current_epoch", 0))
        rl_start = self.imitation_warmup_epochs + self.critic_bootstrap_epochs
        if epoch < rl_start:
            return 0.0  # Zero entropy during imitation + bootstrap
        return self.entropy_weight  # Constant entropy during RL exploration

    def _imitation_weight(self) -> float:
        epoch = int(getattr(self.trainer, "current_epoch", 0))
        base = float(self.loss_fn.omega_bc)
        if epoch >= self.imitation_warmup_epochs:
            return 0.0
        return base

    def _supervised_weight(self) -> float:
        epoch = int(getattr(self.trainer, "current_epoch", 0))
        if epoch >= self.imitation_warmup_epochs:
            return 0.0
        return 1.0

    def _collect_expert_actions(self, Q: Tensor, mu: Tensor) -> Tensor:
        if mu.dim() == 1:
            mu = mu.unsqueeze(0).expand(Q.shape[0], -1)
        mu_safe = mu.clamp_min(mu.new_tensor(1e-12))
        if self.expert_mode == "sed":
            return sed_index(Q, mu_safe).argmin(dim=-1)
        elif self.expert_mode == "qmd":
            return quadratic_drift_index(Q, mu_safe).argmin(dim=-1)
        else:
            raise ValueError(f"Unknown expert_mode: {self.expert_mode}")

    # ── Domain hook ─────────────────────────────────────────────────

    def _make_observation(self, Q: Tensor, mu: Tensor, xi: Tensor | None) -> tuple[Tensor, Tensor, Tensor | None]:
        """Override in domain subclasses to apply adapter transforms."""
        dm = getattr(self.trainer, "datamodule", None)
        adapter = getattr(dm, "adapter", None) if dm is not None else None
        if adapter is not None:
            return adapter.make_observation(Q, mu)
        return Q, mu, xi

    # ── Training step ───────────────────────────────────────────────

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor | None:
        del batch_idx
        Q0, mu0, xi0 = batch["Q"], batch["mu"], batch.get("xi")
        sed_action = batch.get("sed_action")
        dm = getattr(self.trainer, "datamodule", None)
        if hasattr(self.model, "reset_dispatch_state"):
            self.model.reset_dispatch_state()
        expert_action = self._collect_expert_actions(Q0, mu0)
        init_out = self.model.forward_full(Q0, mu0, xi0, training_mode=True)
        expert_pi = F.one_hot(expert_action, num_classes=int(Q0.shape[-1])).to(
            device=init_out.pi.device,
            dtype=init_out.pi.dtype,
        )
        imitation_weight = self._imitation_weight()
        entropy_weight = self._entropy_weight()
        supervised_weight = self._supervised_weight()

        env = CTMCEnvironment(N=int(Q0.shape[-1]), lam=self.lam, mu=mu0[0], B=Q0.shape[0])
        env.reset(Q0.detach().clone())

        visited_states: list[Tensor] = []
        policy_diagnostics: list[DispatcherDiagnostics] = []
        log_probs: list[Tensor] = []
        old_log_probs: list[Tensor] = []
        values: list[Tensor] = []
        old_values: list[Tensor] = []
        rewards: list[Tensor] = []
        entropies: list[Tensor] = []
        kl_terms: list[Tensor] = []
        actions_list: list[Tensor] = []

        for _ in range(self.rollout_horizon):
            Q_obs = env.Q.clone()
            mu_obs = mu0
            xi_obs = xi0
            Q_obs, mu_obs, xi_obs = self._make_observation(Q_obs, mu_obs, xi_obs)
            out = self.model.forward_full(Q_obs, mu_obs, xi_obs, training_mode=True)
            dist = Categorical(probs=out.pi)
            action_idx = dist.sample()
            actions_list.append(action_idx)
            action_pi = F.one_hot(action_idx, num_classes=int(Q0.shape[-1])).float()
            step = env.step(action_pi)
            reward = -(step["cost"].float() * step["dt"].float())

            lp = dist.log_prob(action_idx)
            visited_states.append(Q_obs.detach().cpu())
            policy_diagnostics.append(out.diagnostics)
            log_probs.append(lp)
            old_log_probs.append(lp.detach())
            values.append(out.value)
            old_values.append(out.value.detach())
            rewards.append(reward)
            entropies.append(dist.entropy())
            kl_terms.append(self.loss_fn.policy_kl(out.pi.detach(), out.p_proposal))

        rewards_t = torch.stack(rewards, dim=0)
        values_t = torch.stack(values, dim=0)
        old_values_t = torch.stack(old_values, dim=0)
        log_probs_t = torch.stack(log_probs, dim=0)
        old_log_probs_t = torch.stack(old_log_probs, dim=0)
        entropies_t = torch.stack(entropies, dim=0)
        actions_t = torch.stack(actions_list, dim=0)
        kl_loss = torch.stack(kl_terms, dim=0).mean()

        gamma = self.gamma
        gae_lambda = self.gae_lambda
        with torch.no_grad():
            final_Q = env.Q.clone()
            final_out = self.model.forward_full(final_Q, mu0, xi0, training_mode=True)
            final_val = final_out.value.detach()
        advantages = torch.zeros_like(rewards_t)
        gae = torch.zeros_like(rewards_t[0])
        for t in range(self.rollout_horizon - 1, -1, -1):
            if t == self.rollout_horizon - 1:
                next_value = final_val
            else:
                next_value = old_values_t[t + 1]
            delta = rewards_t[t] + gamma * next_value - old_values_t[t]
            gae = delta + gamma * gae_lambda * gae
            advantages[t] = gae
        returns = advantages + old_values_t
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        rollout_diag = _stack_mean(policy_diagnostics)

        # Dual variable update (adaptive: momentum + warmup + decay + max cap)
        residual = rollout_diag.A_final.mean() - rollout_diag.B_Q.mean()
        epoch = int(getattr(self.trainer, "current_epoch", 0))
        rl_start = self.imitation_warmup_epochs + self.critic_bootstrap_epochs
        if epoch >= rl_start:
            with torch.no_grad():
                # Effective LR with warmup
                warmup_progress = self._dual_update_count.float() / max(1, self.dual_lr_warmup_steps)
                effective_lr = self.dual_lambda_lr * min(1.0, warmup_progress)
                # Multiplicative decay after warmup
                effective_lr *= self.dual_lr_decay ** self._dual_update_count.float()
                # Optional EMA smoothing of the residual
                if self.dual_lambda_momentum > 0:
                    self._smoothed_residual.mul_(self.dual_lambda_momentum)
                    self._smoothed_residual.add_(residual, alpha=1.0 - self.dual_lambda_momentum)
                    update_residual = self._smoothed_residual
                else:
                    update_residual = residual
                # Dual step
                self.dual_lambda.add_(effective_lr * update_residual)
                self.dual_lambda.clamp_(min=0.0, max=self.dual_lambda_max)
                self._dual_update_count.add_(1)

        # Log constraint metrics (signed residual, positive violation, dual)
        violation_amount = rollout_diag.constraint_violation.mean()
        violation_rate = (rollout_diag.constraint_violation > 0).float().mean()
        self.log("constraint_residual", residual, on_step=True, on_epoch=True)
        self.log("constraint_violation", violation_amount, on_step=True, on_epoch=True)
        self.log("constraint_violation_rate", violation_rate, on_step=True, on_epoch=True)
        self.log("dual_lambda", self.dual_lambda, on_step=True, on_epoch=True)

        if self.use_ppo:
            opt = self.optimizers()
            ppo_epochs = self.ppo_epochs
            Q_stack = torch.stack(visited_states, dim=0).to(self.device)
            flat_Q = Q_stack.reshape(-1, Q_stack.shape[-1])
            target_action = sed_action if sed_action is not None else expert_action

            for _ in range(ppo_epochs):
                opt.zero_grad()
                flat_mu = mu0.repeat(self.rollout_horizon, 1)
                flat_xi = xi0.repeat(self.rollout_horizon, 1) if xi0 is not None else None
                out_eval = self.model.forward_full(flat_Q, flat_mu, flat_xi, training_mode=True)
                dist_eval = Categorical(probs=out_eval.pi)
                new_log_probs = dist_eval.log_prob(actions_t.reshape(-1))
                new_values = out_eval.value.reshape(-1)
                actor_loss = self.loss_fn.ppo_actor_loss(
                    new_log_probs,
                    old_log_probs_t.reshape(-1),
                    advantages.reshape(-1),
                    self.ppo_clip_epsilon,
                )
                critic_loss = self.loss_fn.ppo_critic_loss(
                    new_values,
                    old_values_t.reshape(-1),
                    returns.reshape(-1),
                    self.ppo_clip_epsilon,
                )
                # Compute imitation targets on diverse rollout states (not Q0)
                flat_expert_action = self._collect_expert_actions(flat_Q, flat_mu)
                flat_expert_pi = F.one_hot(flat_expert_action, num_classes=int(flat_Q.shape[-1])).to(
                    device=out_eval.pi.device, dtype=out_eval.pi.dtype,
                )
                bc_loss = self.loss_fn.bc_loss(out_eval.pi, flat_expert_pi)
                action_loss = self.loss_fn.action_loss(out_eval.proposal_logits, flat_expert_action)
                margin_loss = self.loss_fn.margin_loss(out_eval.proposal_logits, flat_expert_action)
                residual = out_eval.diagnostics.A_final - out_eval.diagnostics.B_Q
                constraint_loss = self.loss_fn.constraint_loss(
                    residual,
                    self.dual_lambda,
                ) if epoch >= rl_start else torch.zeros((), device=out_eval.pi.device)
                entropy_loss = dist_eval.entropy().mean()
                kl_loss_dynamic = self.loss_fn.policy_kl(out_eval.pi.detach(), out_eval.p_proposal)
                self.epoch_kl_records.append(kl_loss_dynamic.detach().cpu().item())

                is_actor_active = (epoch >= rl_start)
                effective_actor_weight = self.loss_fn.rollout_weight if is_actor_active else 0.0

                total_loss = (
                    effective_actor_weight * actor_loss
                    + self.loss_fn.value_weight * critic_loss
                    + imitation_weight * bc_loss
                    + supervised_weight * self.loss_fn.omega_action * action_loss
                    + supervised_weight * self.loss_fn.omega_margin * margin_loss
                    + constraint_loss
                    + self.current_kl_weight * kl_loss_dynamic
                    - entropy_weight * entropy_loss
                )
                self.manual_backward(total_loss)
                if self._gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._gradient_clip_val)
                opt.step()

            self.log("actor", actor_loss, on_step=True, on_epoch=True)
            self.log("critic", critic_loss, on_step=True, on_epoch=True)
            self.log("bc", bc_loss, on_step=True, on_epoch=True)
            self.log("constraint_loss", constraint_loss, on_step=True, on_epoch=True)
            self.log("total", total_loss, on_step=True, on_epoch=True, prog_bar=True)
            self._log_diagnostics(rollout_diag, "train")

        else:
            actor_loss = self.loss_fn.actor_loss(log_probs_t.reshape(-1), advantages.reshape(-1))
            critic_loss = self.loss_fn.critic_loss(values_t.reshape(-1), returns.reshape(-1))
            bc_loss = self.loss_fn.bc_loss(init_out.pi, expert_pi)
            target_action = sed_action
            if target_action is None:
                target_action = expert_action
            action_loss = self.loss_fn.action_loss(init_out.proposal_logits, target_action)
            margin_loss = self.loss_fn.margin_loss(init_out.proposal_logits, target_action)
            c_residual = init_out.diagnostics.A_final - init_out.diagnostics.B_Q
            constraint_loss = self.loss_fn.constraint_loss(
                c_residual,
                self.dual_lambda,
            ) if epoch >= rl_start else torch.zeros((), device=init_out.pi.device)
            entropy_loss = entropies_t.mean()

            losses = {
                "actor": actor_loss,
                "critic": critic_loss,
                "bc": bc_loss,
                "action": action_loss,
                "margin": margin_loss,
                "constraint": constraint_loss,
                "kl": kl_loss,
                "entropy": entropy_loss,
            }
            
            self.epoch_kl_records.append(kl_loss.detach().cpu().item())
            is_actor_active = (epoch >= rl_start)
            effective_actor_weight = self.loss_fn.rollout_weight if is_actor_active else 0.0

            losses["total"] = (
                effective_actor_weight * actor_loss
                + self.loss_fn.value_weight * critic_loss
                + imitation_weight * bc_loss
                + supervised_weight * self.loss_fn.omega_action * action_loss
                + supervised_weight * self.loss_fn.omega_margin * margin_loss
                + constraint_loss
                + self.current_kl_weight * kl_loss
                - entropy_weight * entropy_loss
            )

            for key, value in losses.items():
                self.log(key, value, on_step=True, on_epoch=True, prog_bar=(key == "total"))
            self._log_diagnostics(rollout_diag, "train")

            if dm is not None:
                if hasattr(dm, "record_policy_states"):
                    dm.record_policy_states(torch.cat(visited_states, dim=0).to("cpu", non_blocking=True))
                if hasattr(dm, "record_teacher_states"):
                    dm.record_teacher_states(Q0.detach().to("cpu", non_blocking=True))

            return losses["total"]

        if dm is not None:
            if hasattr(dm, "record_policy_states"):
                dm.record_policy_states(torch.cat(visited_states, dim=0).to("cpu", non_blocking=True))
            if hasattr(dm, "record_teacher_states"):
                dm.record_teacher_states(Q0.detach().to("cpu", non_blocking=True))
        return None

    def on_train_epoch_end(self) -> None:
        epoch = getattr(self.trainer, "current_epoch", 0)
        if epoch >= self.imitation_warmup_epochs and self.epoch_kl_records:
            mean_kl = sum(self.epoch_kl_records) / len(self.epoch_kl_records)
            if mean_kl > 1.5 * self.target_kl_cert:
                self.current_kl_weight = min(2.0, self.current_kl_weight * 2.0)
            elif mean_kl < self.target_kl_cert / 1.5:
                self.current_kl_weight = max(1e-4, self.current_kl_weight * 0.5)
        self.epoch_kl_records.clear()
        self.log("train/current_kl_weight", self.current_kl_weight)

    # ── Validation step ─────────────────────────────────────────────

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        del batch_idx
        Q, mu, xi = batch["Q"], batch["mu"], batch.get("xi")
        if hasattr(self.model, "reset_dispatch_state"):
            self.model.reset_dispatch_state()
        env = CTMCEnvironment(N=int(Q.shape[-1]), lam=self.lam, mu=mu[0], B=Q.shape[0])
        env.reset(Q.detach().clone())
        dm = getattr(self.trainer, "datamodule", None)
        policy_diagnostics: list[DispatcherDiagnostics] = []
        queue_trace: list[Tensor] = []
        cost_trace: list[Tensor] = []
        dt_trace: list[Tensor] = []
        val_horizon = max(4, min(self.rollout_horizon, self.val_horizon_max))
        with torch.no_grad():
            for _ in range(val_horizon):
                Q_obs = env.Q.clone()
                mu_obs = mu
                xi_obs = xi
                Q_obs, mu_obs, xi_obs = self._make_observation(Q_obs, mu_obs, xi_obs)
                out = self.model.forward_full(Q_obs, mu_obs, xi_obs, training_mode=False)
                action_idx = out.pi.argmax(dim=-1)
                action_pi = F.one_hot(action_idx, num_classes=int(Q.shape[-1])).float()
                step = env.step(action_pi)
                policy_diagnostics.append(out.diagnostics)
                queue_trace.append(step["Q"].detach())
                cost_trace.append(step["cost"].detach())
                dt_trace.append(step["dt"].detach())
        diag = _stack_mean(policy_diagnostics)
        backlog = torch.cat(queue_trace, dim=0).sum(dim=-1).float()
        cost_t = torch.cat(cost_trace, dim=0).float()
        dt_t = torch.cat(dt_trace, dim=0).float()
        avg_cost = self.loss_fn.rollout_cost(cost_t, dt_t)
        p95_backlog = backlog.quantile(0.95)
        violation = (diag.constraint_violation.mean() > 1e-4).float()
        selection_score = validation_selection_score(violation, avg_cost, p95_backlog)
        self.log("val/CONSTRAINT_VIOLATION", violation, prog_bar=True)
        self.log("val/avg_cost", avg_cost, prog_bar=True)
        self.log("val/p95_backlog", p95_backlog, prog_bar=False)
        self.log("val/selection_score", selection_score, prog_bar=False)
        self._log_diagnostics(diag, "val")

    # ── Diagnostics logging ────────────────────────────────────────

    def _log_diagnostics(self, diag: DispatcherDiagnostics, stage: str) -> None:
        self.log(f"{stage}/A_final", diag.A_final.mean())
        self.log(f"{stage}/m_Q", diag.m_Q.mean())
        self.log(f"{stage}/B_Q", diag.B_Q.mean())
        self.log(f"{stage}/certificate_slack_min", diag.certificate_slack.min())
        self.log(f"{stage}/certificate_slack_mean", diag.certificate_slack.mean())
        self.log(f"{stage}/constraint_violation", diag.constraint_violation.mean())
        self.log(f"{stage}/usage_raw", diag.usage_raw.nanmean())
        self.log(f"{stage}/usage_final", diag.usage_final.nanmean())
        self.log(f"{stage}/usage_open_rate", (diag.usage_final > 0.1).float().mean())
        self.log(f"{stage}/usage_cap", diag.usage_cap.nanmean())
        self.log(f"{stage}/policy_entropy", diag.policy_entropy.mean())
        self.log(f"{stage}/pressure_mean", diag.pressure_mean.mean())
        self.log(f"{stage}/pressure_max", diag.pressure_max.mean())
        self.log(f"{stage}/pressure_update_norm", diag.pressure_update_norm.mean())

    # ── Optimizer ───────────────────────────────────────────────────

    def configure_optimizers(self) -> dict[str, object]:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        max_epochs = getattr(self.trainer, "max_epochs", 200) if getattr(self, "trainer", None) else 200
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
