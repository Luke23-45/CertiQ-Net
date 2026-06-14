"""PyTorch Lightning module for CertiQ-Net."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import Categorical

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    pl = None

from certiqnet.dispatcher.types import DispatcherDiagnostics
from certiqnet.simulation.ctmc import CTMCEnvironment
from certiqnet.training.loss import CertiQNetLoss
from certiqnet.dispatcher.delay_geometry import sed_index


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
            if name == "solver_status":
                values[name] = torch.stack(tensors, dim=0).max(dim=0).values
            else:
                values[name] = tensors[-1]
        else:
            values[name] = torch.stack(tensors, dim=0).mean(dim=0)
    return DispatcherDiagnostics(**values)


class CertiQNetLightningModule(pl.LightningModule if pl is not None else torch.nn.Module):
    """Lightning wrapper that trains with on-policy queueing rollouts.

    Supports both REINFORCE (default) and PPO (when ``use_ppo`` is ``True``
    in the config).  PPO uses a clipped surrogate objective for more stable
    long-horizon training.
    """

    def __init__(self, model: torch.nn.Module, cfg: DictConfig) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.loss_fn = CertiQNetLoss(cfg.loss)
        self.rollout_horizon = int(cfg.trainer.rollout_horizon)
        self.entropy_warmup_epochs = int(cfg.trainer.entropy_warmup_epochs)
        self.imitation_warmup_epochs = int(cfg.trainer.imitation_warmup_epochs)
        self.use_ppo = bool(getattr(cfg.trainer, "use_ppo", False))
        self.ppo_clip_epsilon = float(getattr(cfg.trainer, "ppo_clip_epsilon", 0.2))
        if self.use_ppo:
            self.automatic_optimization = False
        if hasattr(self, "save_hyperparameters"):
            self.save_hyperparameters(ignore=["model"])

    def on_train_epoch_start(self) -> None:
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "resample_train_data"):
            dm.resample_train_data()

    def _entropy_weight(self) -> float:
        epoch = int(getattr(self.trainer, "current_epoch", 0))
        base = float(self.loss_fn.cfg.entropy_weight)
        if epoch >= self.entropy_warmup_epochs:
            return 0.0
        frac = 1.0 - (epoch / max(self.entropy_warmup_epochs, 1))
        return base * frac

    def _imitation_weight(self) -> float:
        epoch = int(getattr(self.trainer, "current_epoch", 0))
        base = float(self.loss_fn.cfg.omega_bc)
        if epoch >= self.imitation_warmup_epochs:
            post_warmup = epoch - self.imitation_warmup_epochs
            decay_epochs = 30
            if post_warmup >= decay_epochs:
                return 0.0
            return base * 0.75 * (1.0 - post_warmup / decay_epochs)
        frac = 1.0 - (epoch / max(self.imitation_warmup_epochs, 1))
        return base * (0.5 + 0.5 * frac)

    def _supervised_weight(self) -> float:
        epoch = int(getattr(self.trainer, "current_epoch", 0))
        if epoch >= self.imitation_warmup_epochs:
            post_warmup = epoch - self.imitation_warmup_epochs
            decay_epochs = 40
            return max(0.0, 1.0 - post_warmup / decay_epochs)
        return 1.0

    def _collect_expert_actions(self, Q: Tensor, mu: Tensor) -> Tensor:
        if mu.dim() == 1:
            mu = mu.unsqueeze(0).expand(Q.shape[0], -1)
        mu_safe = mu.clamp_min(torch.finfo(Q.dtype).tiny)
        return sed_index(Q, mu_safe).argmin(dim=-1)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        Q0, mu0, xi0 = batch["Q"], batch["mu"], batch.get("xi")
        sed_action = batch.get("sed_action")
        dm = getattr(self.trainer, "datamodule", None)
        if hasattr(self.model, "reset_dispatch_state"):
            self.model.reset_dispatch_state()
        expert_action = self._collect_expert_actions(Q0, mu0)
        init_out = self.model.forward_full(Q0, mu0, xi0, certify=True, training_mode=True)
        expert_pi = F.one_hot(expert_action, num_classes=int(Q0.shape[-1])).to(
            device=init_out.pi.device,
            dtype=init_out.pi.dtype,
        )
        imitation_weight = self._imitation_weight()
        entropy_weight = self._entropy_weight()
        supervised_weight = self._supervised_weight()

        env = CTMCEnvironment(N=int(Q0.shape[-1]), lam=float(self.cfg.env.lam), mu=mu0[0], B=Q0.shape[0])
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
            if dm is not None and hasattr(dm, "adapter"):
                Q_obs, mu_obs, xi_obs = dm.adapter.make_observation(env.Q, mu0)
            out = self.model.forward_full(Q_obs, mu_obs, xi_obs, certify=True, training_mode=True)
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
            kl_terms.append(self.loss_fn.policy_kl(out.pi, out.p_cert))

        rewards_t = torch.stack(rewards, dim=0)
        values_t = torch.stack(values, dim=0)
        old_values_t = torch.stack(old_values, dim=0)
        log_probs_t = torch.stack(log_probs, dim=0)
        old_log_probs_t = torch.stack(old_log_probs, dim=0)
        entropies_t = torch.stack(entropies, dim=0)
        actions_t = torch.stack(actions_list, dim=0)
        kl_loss = torch.stack(kl_terms, dim=0).mean()

        gamma = 0.99
        gae_lambda = 0.95
        with torch.no_grad():
            final_Q = env.Q.clone()
            final_out = self.model.forward_full(final_Q, mu0, xi0, certify=True, training_mode=True)
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
        rollout_diag = DispatcherDiagnostics(
            A_cert=rollout_diag.A_cert,
            A_proposal=rollout_diag.A_proposal,
            A_final=rollout_diag.A_final,
            m_Q=rollout_diag.m_Q,
            B_Q=rollout_diag.B_Q,
            certificate_slack=rollout_diag.certificate_slack,
            usage_raw=rollout_diag.usage_raw,
            usage_final=rollout_diag.usage_final,
            usage_cap=rollout_diag.usage_cap,
            fallback_active=rollout_diag.fallback_active,
            correction_magnitude=rollout_diag.correction_magnitude,
            policy_entropy=rollout_diag.policy_entropy,
            selected_resource=rollout_diag.selected_resource,
            pressure_mean=rollout_diag.pressure_mean,
            pressure_max=rollout_diag.pressure_max,
            pressure_update_norm=rollout_diag.pressure_update_norm,
            projection_nu=rollout_diag.projection_nu,
            projection_active=rollout_diag.projection_active,
            proposal_slack=rollout_diag.proposal_slack,
            solver_status=rollout_diag.solver_status,
        )

        if self.use_ppo:
            opt = self.optimizers()
            ppo_epochs = int(getattr(self.cfg.trainer, "ppo_epochs", 4))
            Q_stack = torch.stack(visited_states, dim=0).to(self.device)
            flat_Q = Q_stack.reshape(-1, Q_stack.shape[-1])
            target_action = sed_action if sed_action is not None else expert_action

            for _ in range(ppo_epochs):
                opt.zero_grad()
                flat_mu = mu0.repeat(self.rollout_horizon, 1)
                flat_xi = xi0.repeat(self.rollout_horizon, 1) if xi0 is not None else None
                out_eval = self.model.forward_full(flat_Q, flat_mu, flat_xi, certify=True, training_mode=True)
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
                curr_init_out = self.model.forward_full(Q0, mu0, xi0, certify=True, training_mode=True)
                bc_loss = self.loss_fn.bc_loss(curr_init_out.pi, expert_pi)
                action_loss = self.loss_fn.action_loss(curr_init_out.proposal_logits, target_action)
                margin_loss = self.loss_fn.margin_loss(curr_init_out.proposal_logits, target_action)
                usage_loss = self.loss_fn.usage_penalty(rollout_diag.usage_final)
                certificate_loss = self.loss_fn.certificate_penalty(rollout_diag)
                correction_loss = self.loss_fn.correction_size_penalty(rollout_diag)
                entropy_loss = entropies_t.mean()

                total_loss = (
                    self.loss_fn.cfg.rollout_weight * actor_loss
                    + self.loss_fn.cfg.value_weight * critic_loss
                    + imitation_weight * bc_loss
                    + supervised_weight * getattr(self.loss_fn.cfg, "omega_action", 0.0) * action_loss
                    + supervised_weight * getattr(self.loss_fn.cfg, "omega_margin", 0.0) * margin_loss
                    + self.loss_fn.cfg.omega_usage * usage_loss
                    + self.loss_fn.cfg.omega_certificate * certificate_loss
                    + self.loss_fn.cfg.omega_correction * correction_loss
                    + self.loss_fn.cfg.policy_kl_weight * kl_loss
                    - entropy_weight * entropy_loss
                )
                self.manual_backward(total_loss)
                clip_val = float(getattr(self.cfg.trainer, "gradient_clip_val", 1.0))
                if clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)
                opt.step()

            self.log("actor", actor_loss, on_step=True, on_epoch=True)
            self.log("critic", critic_loss, on_step=True, on_epoch=True)
            self.log("bc", bc_loss, on_step=True, on_epoch=True)
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
            usage_loss = self.loss_fn.usage_penalty(rollout_diag.usage_final)
            certificate_loss = self.loss_fn.certificate_penalty(rollout_diag)
            correction_loss = self.loss_fn.correction_size_penalty(rollout_diag)
            entropy_loss = entropies_t.mean()

            losses = {
                "actor": actor_loss,
                "critic": critic_loss,
                "bc": bc_loss,
                "action": action_loss,
                "margin": margin_loss,
                "usage": usage_loss,
                "certificate": certificate_loss,
                "correction": correction_loss,
                "kl": kl_loss,
                "entropy": entropy_loss,
            }
            losses["total"] = (
                self.loss_fn.cfg.rollout_weight * actor_loss
                + self.loss_fn.cfg.value_weight * critic_loss
                + imitation_weight * bc_loss
                + supervised_weight * getattr(self.loss_fn.cfg, "omega_action", 0.0) * action_loss
                + supervised_weight * getattr(self.loss_fn.cfg, "omega_margin", 0.0) * margin_loss
                + self.loss_fn.cfg.omega_usage * usage_loss
                + self.loss_fn.cfg.omega_certificate * certificate_loss
                + self.loss_fn.cfg.omega_correction * correction_loss
                + self.loss_fn.cfg.policy_kl_weight * kl_loss
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

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        del batch_idx
        Q, mu, xi = batch["Q"], batch["mu"], batch.get("xi")
        if hasattr(self.model, "reset_dispatch_state"):
            self.model.reset_dispatch_state()
        env = CTMCEnvironment(N=int(Q.shape[-1]), lam=float(self.cfg.env.lam), mu=mu[0], B=Q.shape[0])
        env.reset(Q.detach().clone())
        dm = getattr(self.trainer, "datamodule", None)
        policy_diagnostics: list[DispatcherDiagnostics] = []
        queue_trace: list[Tensor] = []
        cost_trace: list[Tensor] = []
        dt_trace: list[Tensor] = []
        rollout_horizon = max(4, min(self.rollout_horizon, 8))
        with torch.no_grad():
            for _ in range(rollout_horizon):
                Q_obs = env.Q.clone()
                mu_obs = mu
                xi_obs = xi
                if dm is not None and hasattr(dm, "adapter"):
                    Q_obs, mu_obs, xi_obs = dm.adapter.make_observation(env.Q, mu)
                out = self.model.forward_full(Q_obs, mu_obs, xi_obs, certify=True, training_mode=False)
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
        violation = (diag.certificate_slack.min() < -1e-4).float()
        selection_score = validation_selection_score(violation, avg_cost, p95_backlog)
        self.log("val/CERTIFICATE_VIOLATION", violation, prog_bar=True)
        self.log("val/avg_cost", avg_cost, prog_bar=True)
        self.log("val/p95_backlog", p95_backlog, prog_bar=False)
        self.log("val/selection_score", selection_score, prog_bar=False)
        self._log_diagnostics(diag, "val")

    def _log_diagnostics(self, diag: DispatcherDiagnostics, stage: str) -> None:
        self.log(f"{stage}/A_cert", diag.A_cert.mean())
        self.log(f"{stage}/A_proposal", diag.A_proposal.mean())
        self.log(f"{stage}/A_final", diag.A_final.mean())
        self.log(f"{stage}/m_Q", diag.m_Q.mean())
        self.log(f"{stage}/B_Q", diag.B_Q.mean())
        self.log(f"{stage}/certificate_slack_min", diag.certificate_slack.min())
        self.log(f"{stage}/certificate_slack_mean", diag.certificate_slack.mean())
        self.log(f"{stage}/usage_raw", diag.usage_raw.nanmean())
        self.log(f"{stage}/usage_final", diag.usage_final.nanmean())
        self.log(f"{stage}/usage_open_rate", (diag.usage_final > 0.1).float().mean())
        self.log(f"{stage}/usage_cap", diag.usage_cap.nanmean())
        self.log(f"{stage}/fallback_rate", diag.fallback_active.float().mean())
        self.log(f"{stage}/correction_magnitude", diag.correction_magnitude.mean())
        self.log(f"{stage}/policy_entropy", diag.policy_entropy.mean())
        self.log(f"{stage}/pressure_mean", diag.pressure_mean.mean())
        self.log(f"{stage}/pressure_max", diag.pressure_max.mean())
        self.log(f"{stage}/pressure_update_norm", diag.pressure_update_norm.mean())

    def configure_optimizers(self) -> dict[str, object]:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.trainer.lr,
            weight_decay=self.cfg.trainer.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
