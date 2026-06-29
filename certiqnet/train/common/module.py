"""Base LightningModule for CertiQ‑Net training (formal 5-term objective)."""

from __future__ import annotations

from contextlib import nullcontext
import torch
import torch.nn.functional as F
from collections import deque
from pathlib import Path
from torch import Tensor, nn
from torch.distributions import Categorical

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    pl = None

from certiqnet.dispatcher.types import DispatcherDiagnostics
from certiqnet.train.common.loss import CertiQNetLoss
from certiqnet.data.qgym.context import build_qgym_context
from certiqnet.utils.qgym_rollout import (
    as_batch_tensor,
    build_qgym_rollout_env,
    extract_qgym_queues,
    qgym_action_from_pi,
)


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
        qgym_env_config: str | Path | dict | None = None,
        qgym_seed: int = 0,
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
        self.qgym_env_config = qgym_env_config
        self.qgym_seed = int(qgym_seed)
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
        network: Tensor | None = None,
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
            network=network,
            context_dim=ctx_dim,
        )

    def _build_qgym_env(self, batch_size: int, device: torch.device) -> object:
        if self.qgym_env_config is None:
            raise ValueError("qgym_env_config is required for QGym rollouts.")
        return build_qgym_rollout_env(
            self.qgym_env_config,
            batch=batch_size,
            seed=self.qgym_seed,
            device=device,
        )

    # ── Training step ───────────────────────────────────────────────

    def _run_qgym_rollout_single(
        self,
        init_Q: Tensor,
        init_mu: Tensor,
        init_xi: Tensor | None,
        horizon: int,
        *,
        training_mode: bool,
        sample_actions: bool,
    ) -> tuple[
        DispatcherDiagnostics,
        Tensor,
        Tensor,
        list[Tensor],
        list[Tensor],
    ]:
        """Run one QGym rollout for a single sample."""
        if hasattr(self.model, "reset_dispatch_state"):
            self.model.reset_dispatch_state()

        env = self._build_qgym_env(batch_size=1, device=init_Q.device)
        env.reset(init_queues=init_Q.detach().clone().unsqueeze(0))

        history_Q: deque[Tensor] = deque([init_Q.detach().clone().unsqueeze(0)], maxlen=max(2, min(4, horizon)))
        history_action: deque[Tensor] = deque(maxlen=max(2, min(4, horizon)))
        history_dt: deque[Tensor] = deque(maxlen=max(2, min(4, horizon)))

        policy_diagnostics: list[DispatcherDiagnostics] = []
        log_probs: list[Tensor] = []
        rewards: list[Tensor] = []
        queue_trace: list[Tensor] = []
        cost_trace: list[Tensor] = []
        dt_trace: list[Tensor] = []

        context = nullcontext() if training_mode else torch.no_grad()
        with context:
            for _ in range(horizon):
                current_q = env.obs.queues if hasattr(env, "obs") else env.env_state.queues
                Q_raw = as_batch_tensor(current_q, device=init_Q.device)
                mu_obs = as_batch_tensor(init_mu, device=init_Q.device)
                xi_obs = self._rollout_context(
                    Q_raw,
                    mu_obs,
                    network=getattr(env, "network", None),
                    history_Q=history_Q,
                    history_action=history_action,
                    history_dt=history_dt,
                )
                Q_obs, mu_obs, xi_obs = self._make_observation(Q_raw, mu_obs, xi_obs)
                out = self.model.forward_full(Q_obs, mu_obs, xi_obs, training_mode=training_mode)

                if sample_actions:
                    dist = Categorical(probs=out.pi)
                    action_idx = dist.sample()
                    log_probs.append(dist.log_prob(action_idx))
                else:
                    action_idx = out.pi.argmax(dim=-1)

                action_pi = F.one_hot(action_idx, num_classes=int(init_Q.shape[-1])).float()
                action = qgym_action_from_pi(action_pi, env.network, Q_raw, sample=False)
                step_obs, _reward, _done, _truncated, info = env.step(action)
                step_Q = extract_qgym_queues(step_obs, info, device=init_Q.device)
                step_cost = as_batch_tensor(info.get("cost", 0.0), device=init_Q.device).reshape(-1)
                step_dt = as_batch_tensor(info.get("event_time", 1.0), device=init_Q.device).reshape(-1)

                policy_diagnostics.append(out.diagnostics)
                rewards.append((-step_cost).detach())
                queue_trace.append(Q_raw.detach())
                cost_trace.append(step_cost.detach())
                dt_trace.append(step_dt.detach())
                history_action.append(action.detach())
                history_dt.append(step_dt.detach())
                history_Q.append(step_Q.detach())

        diag = _stack_mean(policy_diagnostics)
        backlog = torch.cat(queue_trace, dim=0).sum(dim=-1).float()
        cost_t = torch.cat(cost_trace, dim=0).float()
        dt_t = torch.cat(dt_trace, dim=0).float()
        avg_cost = self.loss_fn.rollout_cost(cost_t, dt_t)
        p95_backlog = backlog.quantile(0.95)
        return diag, avg_cost, p95_backlog, log_probs, rewards

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor | None:
        del batch_idx
        Q0, mu0, xi0 = self._make_observation(batch["Q"], batch["mu"], batch.get("xi"))
        dm = getattr(self.trainer, "datamodule", None)
        if hasattr(self.model, "reset_dispatch_state"):
            self.model.reset_dispatch_state()

        init_out = self.model.forward_full(Q0, mu0, xi0, training_mode=True)
        self._log_diagnostics(init_out.diagnostics, "train")

        # Rollout-cost optimization via QGym REINFORCE.
        rollout_diagnostics: list[DispatcherDiagnostics] = []
        rollout_log_probs: list[Tensor] = []
        rollout_returns: list[Tensor] = []

        for sample_idx in range(Q0.shape[0]):
            sample_Q = Q0[sample_idx]
            sample_mu = mu0[sample_idx]
            sample_xi = xi0[sample_idx] if xi0 is not None else None
            diag, _avg_cost, _p95_backlog, log_probs, rewards = self._run_qgym_rollout_single(
                sample_Q,
                sample_mu,
                sample_xi,
                self.rollout_horizon,
                training_mode=True,
                sample_actions=True,
            )
            rollout_diagnostics.append(diag)
            sample_rewards = torch.stack(rewards, dim=0)
            sample_log_probs = torch.stack(log_probs, dim=0)
            sample_returns = torch.zeros_like(sample_rewards)
            running = torch.zeros_like(sample_rewards[0])
            for t in reversed(range(self.rollout_horizon)):
                running = sample_rewards[t] + self.gamma * running
                sample_returns[t] = running
            rollout_log_probs.append(sample_log_probs)
            rollout_returns.append(sample_returns)

        log_probs_t = torch.cat(rollout_log_probs, dim=0)
        returns = torch.cat(rollout_returns, dim=0)
        rollout_diag = _stack_mean(rollout_diagnostics)
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
            diags: list[DispatcherDiagnostics] = []
            avg_costs: list[Tensor] = []
            p95_backlogs: list[Tensor] = []
            violations: list[Tensor] = []
            violation_rates: list[Tensor] = []

            for sample_idx in range(init_Q.shape[0]):
                sample_Q = init_Q[sample_idx]
                sample_mu = init_mu[sample_idx]
                sample_xi = init_xi[sample_idx] if init_xi is not None else None
                diag, avg_cost, p95_backlog, _log_probs, _rewards = self._run_qgym_rollout_single(
                    sample_Q,
                    sample_mu,
                    sample_xi,
                    horizon,
                    training_mode=False,
                    sample_actions=False,
                )
                diags.append(diag)
                avg_costs.append(avg_cost)
                p95_backlogs.append(p95_backlog)
                violations.append(diag.constraint_violation.mean())
                violation_rates.append((diag.constraint_violation > 1e-4).float().mean())

            return (
                _stack_mean(diags),
                torch.stack(avg_costs).mean(),
                torch.stack(p95_backlogs).mean(),
                torch.stack(violations).mean(),
                torch.stack(violation_rates).mean(),
            )

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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
