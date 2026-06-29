"""Base LightningModule for CertiQ‑Net training (formal 5-term objective)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from collections import deque
from torch import Tensor, nn
from torch.distributions import Categorical

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    pl = None

from certiqnet.dispatcher.types import DispatcherDiagnostics
from certiqnet.train.common.loss import CertiQNetLoss
from certiqnet.data.qgym.context import build_qgym_context
from certiqnet.utils.ctmc import CTMCEnvironment


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
    """Base Lightning wrapper with the formal 3-term objective.

    The certificate layer is part of every forward pass (it defines the
    policy).  No value critic, no PPO, no Lagrangian dual.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: CertiQNetLoss = None,
        input_normalization: str = "none",
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        rollout_horizon: int = 16,
        context_dim: int = 0,
        lam: float = 1.0,
        gamma: float = 0.99,
        val_horizon_max: int = 64,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn if loss_fn is not None else CertiQNetLoss()
        self.input_normalization = str(input_normalization)
        self.lr = lr
        self.weight_decay = weight_decay
        self.rollout_horizon = int(rollout_horizon)
        self.context_dim = int(context_dim)
        self.lam = float(lam)
        self.gamma = float(gamma)
        self.val_horizon_max = int(val_horizon_max)
        if hasattr(self, "save_hyperparameters"):
            self.save_hyperparameters(ignore=["model", "loss_fn"])

    def on_train_epoch_start(self) -> None:
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "resample_train_data"):
            dm.resample_train_data()

    # ── Domain hook ─────────────────────────────────────────────────

    def _make_observation(self, Q: Tensor, mu: Tensor, xi: Tensor | None) -> tuple[Tensor, Tensor, Tensor | None]:
        dm = getattr(self.trainer, "datamodule", None)
        adapter = getattr(dm, "adapter", None) if dm is not None else None
        if adapter is not None:
            Q, mu, adapter_xi = adapter.make_observation(Q, mu)
            if xi is None:
                xi = adapter_xi
        if mu.dim() == 1:
            mu = mu.unsqueeze(0).expand(Q.shape[0], -1)
        if self.input_normalization == "per_sample":
            col_max = Q.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
            Q = Q / col_max
        if self.context_dim > 0:
            if xi is None:
                xi = build_qgym_context(Q, mu, context_dim=self.context_dim)
            elif xi.shape[-1] != self.context_dim:
                if xi.shape[-1] > self.context_dim:
                    xi = xi[..., : self.context_dim]
                else:
                    pad = torch.zeros(*xi.shape[:-1], self.context_dim - xi.shape[-1], device=xi.device, dtype=xi.dtype)
                    xi = torch.cat([xi, pad], dim=-1)
        return Q, mu, xi

    def _rollout_context(
        self,
        Q: Tensor,
        mu: Tensor,
        *,
        history_Q: deque[Tensor],
        history_action: deque[Tensor],
        history_dt: deque[Tensor],
    ) -> Tensor | None:
        if self.context_dim <= 0:
            return None
        ctx_dim = self.context_dim
        return build_qgym_context(
            Q,
            mu,
            history_Q=list(history_Q),
            history_action=list(history_action),
            history_dt=list(history_dt),
            context_dim=ctx_dim,
        )

    # ── Training step ───────────────────────────────────────────────

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor | None:
        del batch_idx
        Q0, mu0, xi0 = self._make_observation(batch["Q"], batch["mu"], batch.get("xi"))
        dm = getattr(self.trainer, "datamodule", None)
        if hasattr(self.model, "reset_dispatch_state"):
            self.model.reset_dispatch_state()

        init_out = self.model.forward_full(Q0, mu0, xi0, training_mode=True)
        self._log_diagnostics(init_out.diagnostics, "train")

        # Rollout-cost optimization via REINFORCE
        N = int(Q0.shape[-1])
        env = CTMCEnvironment(N=N, lam=self.lam, mu=mu0[0], B=Q0.shape[0])
        env.reset(Q0.detach().clone())
        history_Q: deque[Tensor] = deque([Q0.detach().clone()], maxlen=max(2, min(4, self.rollout_horizon)))
        history_action: deque[Tensor] = deque(maxlen=max(2, min(4, self.rollout_horizon)))
        history_dt: deque[Tensor] = deque(maxlen=max(2, min(4, self.rollout_horizon)))

        policy_diagnostics: list[DispatcherDiagnostics] = []
        log_probs: list[Tensor] = []
        rewards: list[Tensor] = []

        for _ in range(self.rollout_horizon):
            Q_obs = env.Q.clone()
            mu_obs = mu0
            xi_obs = self._rollout_context(
                Q_obs,
                mu_obs,
                history_Q=history_Q,
                history_action=history_action,
                history_dt=history_dt,
            )
            Q_obs, mu_obs, xi_obs = self._make_observation(Q_obs, mu_obs, xi_obs)
            out = self.model.forward_full(Q_obs, mu_obs, xi_obs, training_mode=True)
            dist = Categorical(probs=out.pi)
            action_idx = dist.sample()
            action_pi = F.one_hot(action_idx, num_classes=N).float()
            step = env.step(action_pi)
            reward = -(step["cost"].float() * step["dt"].float())
            policy_diagnostics.append(out.diagnostics)
            log_probs.append(dist.log_prob(action_idx))
            rewards.append(reward)
            history_action.append(action_pi.detach())
            history_dt.append(step["dt"].detach())
            history_Q.append(step["Q"].detach())

        rewards_t = torch.stack(rewards, dim=0)
        log_probs_t = torch.stack(log_probs, dim=0)

        # Simple discounted returns (no GAE, no value baseline)
        returns = torch.zeros_like(rewards_t)
        running = torch.zeros_like(rewards_t[0])
        for t in reversed(range(self.rollout_horizon)):
            running = rewards_t[t] + self.gamma * running
            returns[t] = running
        returns = (returns - returns.mean()) / returns.std(unbiased=False).clamp_min(1e-6)

        rollout_diag = _stack_mean(policy_diagnostics)
        self._log_diagnostics(rollout_diag, "train_rollout")

        losses = self.loss_fn(
            proposal_logits=init_out.proposal_logits,
            p_cert=init_out.p_cert,
            rollout_log_probs=log_probs_t.reshape(-1),
            rollout_returns=returns.reshape(-1),
        )

        for key, value in losses.items():
            self.log(key, value, on_step=True, on_epoch=True, prog_bar=(key == "total"))

        if dm is not None:
            if hasattr(dm, "record_policy_states"):
                dm.record_policy_states(Q0.detach().to("cpu", non_blocking=True))

        return losses["total"]

    # ── Validation step ─────────────────────────────────────────────

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        del batch_idx
        Q, mu, xi = self._make_observation(batch["Q"], batch["mu"], batch.get("xi"))
        val_horizon = max(self.rollout_horizon, self.val_horizon_max)

        def _run_rollout(
            init_Q: Tensor,
            init_mu: Tensor,
            init_xi: Tensor | None,
            horizon: int,
        ) -> tuple[DispatcherDiagnostics, Tensor, Tensor, Tensor, Tensor]:
            if hasattr(self.model, "reset_dispatch_state"):
                self.model.reset_dispatch_state()
            env = CTMCEnvironment(
                N=int(init_Q.shape[-1]),
                lam=self.lam,
                mu=init_mu[0],
                B=init_Q.shape[0],
            )
            env.reset(init_Q.detach().clone())
            history_Q: deque[Tensor] = deque([init_Q.detach().clone()], maxlen=max(2, min(4, horizon)))
            history_action: deque[Tensor] = deque(maxlen=max(2, min(4, horizon)))
            history_dt: deque[Tensor] = deque(maxlen=max(2, min(4, horizon)))
            policy_diagnostics: list[DispatcherDiagnostics] = []
            queue_trace: list[Tensor] = []
            cost_trace: list[Tensor] = []
            dt_trace: list[Tensor] = []
            with torch.no_grad():
                for _ in range(horizon):
                    Q_obs = env.Q.clone()
                    mu_obs = init_mu
                    xi_obs = self._rollout_context(
                        Q_obs,
                        mu_obs,
                        history_Q=history_Q,
                        history_action=history_action,
                        history_dt=history_dt,
                    )
                    Q_obs, mu_obs, xi_obs = self._make_observation(Q_obs, mu_obs, xi_obs)
                    out = self.model.forward_full(Q_obs, mu_obs, xi_obs, training_mode=False)
                    action_idx = out.pi.argmax(dim=-1)
                    action_pi = F.one_hot(action_idx, num_classes=int(init_Q.shape[-1])).float()
                    step = env.step(action_pi)
                    policy_diagnostics.append(out.diagnostics)
                    queue_trace.append(step["Q"].detach())
                    cost_trace.append(step["cost"].detach())
                    dt_trace.append(step["dt"].detach())
                    history_action.append(action_pi.detach())
                    history_dt.append(step["dt"].detach())
                    history_Q.append(step["Q"].detach())
            diag = _stack_mean(policy_diagnostics)
            backlog = torch.cat(queue_trace, dim=0).sum(dim=-1).float()
            cost_t = torch.cat(cost_trace, dim=0).float()
            dt_t = torch.cat(dt_trace, dim=0).float()
            avg_cost = self.loss_fn.rollout_cost(cost_t, dt_t)
            p95_backlog = backlog.quantile(0.95)
            violation = diag.constraint_violation.mean()
            violation_rate = (diag.constraint_violation > 1e-4).float().mean()
            return diag, avg_cost, p95_backlog, violation, violation_rate

        zero_Q = torch.zeros_like(Q)
        primary_diag, avg_cost, p95_backlog, violation, violation_rate = _run_rollout(
            zero_Q, mu, xi, val_horizon
        )
        selection_score = validation_selection_score(violation, avg_cost, p95_backlog)
        self.log("val/CONSTRAINT_VIOLATION", violation, prog_bar=True)
        self.log("val/CONSTRAINT_VIOLATION_RATE", violation_rate, prog_bar=False)
        self.log("val/avg_cost", avg_cost, prog_bar=True)
        self.log("val/p95_backlog", p95_backlog, prog_bar=False)
        self.log("val/selection_score", selection_score, prog_bar=False)
        self._log_diagnostics(primary_diag, "val")

        if Q.abs().sum().item() > 0:
            aux_diag, aux_avg_cost, aux_p95_backlog, aux_violation, aux_violation_rate = _run_rollout(
                Q, mu, xi, val_horizon
            )
            self.log("val/dataset_start_CONSTRAINT_VIOLATION", aux_violation, prog_bar=False)
            self.log("val/dataset_start_CONSTRAINT_VIOLATION_RATE", aux_violation_rate, prog_bar=False)
            self.log("val/dataset_start_avg_cost", aux_avg_cost, prog_bar=False)
            self.log("val/dataset_start_p95_backlog", aux_p95_backlog, prog_bar=False)
            self._log_diagnostics(aux_diag, "val_dataset_start")

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
        eta_min = max(self.lr * 0.1, 1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=eta_min
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
