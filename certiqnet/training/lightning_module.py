"""PyTorch Lightning module for CertiQ-Net."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import Categorical

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:  # pragma: no cover
    pl = None

from certiqnet.math.certificate import CertificateDiagnostics
from certiqnet.simulation.ctmc import CTMCEnvironment
from certiqnet.training.loss import CertiQNetLoss


def validation_selection_score(violation: Tensor, avg_cost: Tensor, p95_backlog: Tensor) -> Tensor:
    """Lexicographic checkpoint score: violation, then cost, then tail backlog."""
    return violation * 1e9 + avg_cost * 1e3 + p95_backlog


def _stack_mean(diags: list[CertificateDiagnostics]) -> CertificateDiagnostics:
    """Aggregate a rollout of diagnostics into one mean diagnostic record."""
    fields = CertificateDiagnostics.__dataclass_fields__.keys()
    values: dict[str, Tensor] = {}
    for name in fields:
        tensors = [getattr(diag, name) for diag in diags]
        if tensors[0].dtype == torch.bool:
            values[name] = torch.stack(tensors, dim=0).any(dim=0)
        elif tensors[0].dtype in {torch.int32, torch.int64, torch.long}:
            values[name] = tensors[-1]
        else:
            values[name] = torch.stack(tensors, dim=0).mean(dim=0)
    return CertificateDiagnostics(**values)


class CertiQNetLightningModule(pl.LightningModule if pl is not None else torch.nn.Module):
    """Lightning wrapper that trains with on-policy queueing rollouts."""

    def __init__(self, model: torch.nn.Module, cfg: DictConfig) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.loss_fn = CertiQNetLoss(cfg.loss)
        self.rollout_horizon = int(cfg.trainer.rollout_horizon)
        self.entropy_warmup_epochs = int(cfg.trainer.entropy_warmup_epochs)
        self.imitation_warmup_epochs = int(cfg.trainer.imitation_warmup_epochs)
        if hasattr(self, "save_hyperparameters"):
            self.save_hyperparameters(ignore=["model"])

    def on_train_epoch_start(self) -> None:
        """Refresh the synthetic buffer and fold in newly observed policy states."""
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
            return 0.25 * base
        frac = 1.0 - (epoch / max(self.imitation_warmup_epochs, 1))
        return base * (0.5 + 0.5 * frac)

    def _collect_expert_actions(self, Q: Tensor, mu: Tensor) -> Tensor:
        from certiqnet.models.baselines import JoinShortestWeightedQueue

        expert = JoinShortestWeightedQueue(N=int(Q.shape[-1]), beta=float(self.cfg.model.beta))
        with torch.no_grad():
            pi, _ = expert(Q, mu, training_mode=False)
        return pi.argmax(dim=-1)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Run an on-policy rollout plus warm-start imitation and certificate regularization."""
        del batch_idx
        Q0, mu0, xi0 = batch["Q"], batch["mu"], batch.get("xi")
        dm = getattr(self.trainer, "datamodule", None)
        expert_action = self._collect_expert_actions(Q0, mu0)
        expert_pi = F.one_hot(expert_action, num_classes=int(Q0.shape[-1])).float()

        init_out = self.model.forward_with_aux(Q0, mu0, xi0, training_mode=True)
        imitation_weight = self._imitation_weight()
        entropy_weight = self._entropy_weight()

        env = CTMCEnvironment(N=int(Q0.shape[-1]), lam=float(self.cfg.env.lam), mu=mu0[0], B=Q0.shape[0])
        env.reset(Q0.detach().clone())

        visited_states: list[Tensor] = []
        policy_diagnostics: list[CertificateDiagnostics] = []
        log_probs: list[Tensor] = []
        values: list[Tensor] = []
        rewards: list[Tensor] = []
        entropies: list[Tensor] = []
        ref_policies: list[Tensor] = []
        pi_rollouts: list[Tensor] = []

        for _ in range(self.rollout_horizon):
            Q_obs = env.Q.clone()
            mu_obs = mu0
            xi_obs = xi0
            if dm is not None and hasattr(dm, "adapter"):
                Q_obs, mu_obs, xi_obs = dm.adapter.make_observation(env.Q, mu0)
            out = self.model.forward_with_aux(Q_obs, mu_obs, xi_obs, training_mode=True)
            dist = Categorical(probs=out.pi)
            action_idx = dist.sample()
            action_pi = F.one_hot(action_idx, num_classes=int(Q0.shape[-1])).float()
            step = env.step(action_pi)
            reward = -(step["cost"].float() * step["dt"].float())

            visited_states.append(Q_obs.detach().cpu())
            policy_diagnostics.append(out.diag)
            log_probs.append(dist.log_prob(action_idx))
            values.append(out.value)
            rewards.append(reward)
            entropies.append(dist.entropy())
            ref_policies.append(out.p_base)
            pi_rollouts.append(out.pi)

        rewards_t = torch.stack(rewards, dim=0)
        values_t = torch.stack(values, dim=0)
        log_probs_t = torch.stack(log_probs, dim=0)
        entropies_t = torch.stack(entropies, dim=0)
        ref_policies_t = torch.stack(ref_policies, dim=0)
        pi_rollouts_t = torch.stack(pi_rollouts, dim=0)

        returns = torch.zeros_like(rewards_t)
        running = torch.zeros_like(rewards_t[-1])
        for t in range(self.rollout_horizon - 1, -1, -1):
            running = rewards_t[t] + running
            returns[t] = running
        advantages = returns - values_t
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        rollout_diag = _stack_mean(policy_diagnostics)
        rollout_diag = CertificateDiagnostics(
            A_base=rollout_diag.A_base,
            A_nn=rollout_diag.A_nn,
            A_pi=rollout_diag.A_pi,
            m_Q=rollout_diag.m_Q,
            B_Q=rollout_diag.B_Q,
            drift_slack=rollout_diag.drift_slack,
            eta_raw=rollout_diag.eta_raw,
            eta_final=rollout_diag.eta_final,
            eta_safe=rollout_diag.eta_safe,
            fallback_active=rollout_diag.fallback_active,
            residual_norm=rollout_diag.residual_norm,
            policy_entropy=rollout_diag.policy_entropy,
            selected_resource=rollout_diag.selected_resource,
        )

        actor_loss = self.loss_fn.actor_loss(log_probs_t.reshape(-1), advantages.reshape(-1))
        critic_loss = self.loss_fn.critic_loss(values_t.reshape(-1), returns.reshape(-1))
        bc_loss = self.loss_fn.bc_loss(init_out.pi, expert_pi)
        gate_loss = self.loss_fn.gate_penalty(rollout_diag.eta_final)
        drift_loss = self.loss_fn.drift_penalty(rollout_diag)
        residual_loss = self.loss_fn.residual_size_penalty(rollout_diag)
        kl_loss = self.loss_fn.policy_kl(
            pi_rollouts_t.reshape(-1, pi_rollouts_t.shape[-1]),
            ref_policies_t.reshape(-1, ref_policies_t.shape[-1]),
        )
        entropy_loss = entropies_t.mean()
        losses = {
            "actor": actor_loss,
            "critic": critic_loss,
            "bc": bc_loss,
            "gate": gate_loss,
            "drift": drift_loss,
            "residual": residual_loss,
            "kl": kl_loss,
            "entropy": entropy_loss,
        }
        losses["total"] = (
            self.loss_fn.cfg.rollout_weight * actor_loss
            + self.loss_fn.cfg.value_weight * critic_loss
            + self.loss_fn.cfg.omega_bc * bc_loss
            + self.loss_fn.cfg.omega_gate * gate_loss
            + self.loss_fn.cfg.omega_drift * drift_loss
            + self.loss_fn.cfg.omega_res * residual_loss
            + self.loss_fn.cfg.policy_kl_weight * kl_loss
            - entropy_weight * entropy_loss
        )

        for key, value in losses.items():
            self.log(f"train/{key}", value, on_step=True, on_epoch=True, prog_bar=(key == "total"))
        self._log_diagnostics(rollout_diag, "train")

        if dm is not None:
            if hasattr(dm, "record_policy_states"):
                dm.record_policy_states(torch.cat(visited_states, dim=0))
            if hasattr(dm, "record_teacher_states"):
                dm.record_teacher_states(Q0.detach().cpu())

        return losses["total"]

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """Run exact-gate validation and expose queueing metrics for checkpointing."""
        del batch_idx
        Q, mu, xi = batch["Q"], batch["mu"], batch.get("xi")
        env = CTMCEnvironment(N=int(Q.shape[-1]), lam=float(self.cfg.env.lam), mu=mu[0], B=Q.shape[0])
        env.reset(Q.detach().clone())
        dm = getattr(self.trainer, "datamodule", None)
        policy_diagnostics: list[CertificateDiagnostics] = []
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
                out = self.model.forward_with_aux(Q_obs, mu_obs, xi_obs, training_mode=False)
                action_idx = out.pi.argmax(dim=-1)
                action_pi = F.one_hot(action_idx, num_classes=int(Q.shape[-1])).float()
                step = env.step(action_pi)
                policy_diagnostics.append(out.diag)
                queue_trace.append(step["Q"].detach())
                cost_trace.append(step["cost"].detach())
                dt_trace.append(step["dt"].detach())
        diag = _stack_mean(policy_diagnostics)
        backlog = torch.cat(queue_trace, dim=0).sum(dim=-1).float()
        cost_t = torch.cat(cost_trace, dim=0).float()
        dt_t = torch.cat(dt_trace, dim=0).float()
        avg_cost = self.loss_fn.rollout_cost(cost_t, dt_t)
        p95_backlog = backlog.quantile(0.95)
        violation = (diag.drift_slack.min() < -1e-4).float()
        selection_score = validation_selection_score(violation, avg_cost, p95_backlog)
        self.log("val/CERTIFICATE_VIOLATION", violation, prog_bar=True)
        self.log("val/avg_cost", avg_cost, prog_bar=True)
        self.log("val/p95_backlog", p95_backlog, prog_bar=False)
        self.log("val/selection_score", selection_score, prog_bar=False)
        self._log_diagnostics(diag, "val")

    def _log_diagnostics(self, diag: CertificateDiagnostics, stage: str) -> None:
        """Log all diagnostic fields in aggregate form."""
        self.log(f"{stage}/A_base", diag.A_base.mean())
        self.log(f"{stage}/A_nn", diag.A_nn.mean())
        self.log(f"{stage}/A_pi", diag.A_pi.mean())
        self.log(f"{stage}/m_Q", diag.m_Q.mean())
        self.log(f"{stage}/B_Q", diag.B_Q.mean())
        self.log(f"{stage}/drift_slack_min", diag.drift_slack.min())
        self.log(f"{stage}/drift_slack_mean", diag.drift_slack.mean())
        self.log(f"{stage}/eta_raw", diag.eta_raw.nanmean())
        self.log(f"{stage}/eta_final", diag.eta_final.nanmean())
        self.log(f"{stage}/eta_final_mean", diag.eta_final.nanmean())
        self.log(f"{stage}/eta_final_open_rate", (diag.eta_final > 0.1).float().mean())
        self.log(f"{stage}/eta_safe", diag.eta_safe.nanmean())
        self.log(f"{stage}/fallback_rate", diag.fallback_active.float().mean())
        self.log(f"{stage}/residual_norm", diag.residual_norm.mean())
        self.log(f"{stage}/policy_entropy", diag.policy_entropy.mean())

    def configure_optimizers(self) -> dict[str, object]:
        """Configure AdamW with cosine annealing."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.trainer.lr,
            weight_decay=self.cfg.trainer.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
