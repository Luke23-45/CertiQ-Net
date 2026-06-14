"""LightningModule — moved to ``certiqnet.train.common.module``.

A backward-compat ``CertiQNetLightningModule`` alias is provided that
accepts the old ``(model, cfg)`` signature.
"""

from __future__ import annotations

import warnings

import torch
from omegaconf import DictConfig

from certiqnet.train.common.module import BaseCertiQLightningModule, validation_selection_score  # noqa: F401
from certiqnet.train.common.loss import CertiQNetLoss


class CertiQNetLightningModule(BaseCertiQLightningModule):
    """Backward-compat wrapper that unpacks a ``DictConfig``.

    .. deprecated::
        Use ``BaseCertiQLightningModule`` or a domain-specific subclass
        (e.g. ``QueueingLightningModule``) with explicit params.
    """

    def __init__(self, model: torch.nn.Module, cfg: DictConfig) -> None:
        warnings.warn(
            "CertiQNetLightningModule(model, cfg) is deprecated. "
            "Use BaseCertiQLightningModule(model, loss_fn, lr=..., ...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        loss_fn = CertiQNetLoss(
            omega_bc=float(cfg.loss.omega_bc),
            omega_action=float(cfg.loss.get("omega_action", 1.5)),
            omega_margin=float(cfg.loss.get("omega_margin", 0.1)),
            omega_usage=float(cfg.loss.omega_usage),
            omega_certificate=float(cfg.loss.omega_certificate),
            omega_correction=float(cfg.loss.omega_correction),
            rollout_weight=float(cfg.loss.rollout_weight),
            policy_kl_weight=float(cfg.loss.policy_kl_weight),
            value_weight=float(cfg.loss.value_weight),
            entropy_weight=float(cfg.loss.entropy_weight),
        )
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            lr=float(cfg.trainer.lr),
            weight_decay=float(cfg.trainer.weight_decay),
            rollout_horizon=int(cfg.trainer.rollout_horizon),
            use_ppo=bool(getattr(cfg.trainer, "use_ppo", False)),
            ppo_epochs=int(getattr(cfg.trainer, "ppo_epochs", 4)),
            ppo_clip_epsilon=float(getattr(cfg.trainer, "ppo_clip_epsilon", 0.2)),
            ppo_manual_clip_val=float(getattr(cfg.trainer, "ppo_manual_clip_val", 1.0)),
            entropy_warmup_epochs=int(cfg.trainer.entropy_warmup_epochs),
            imitation_warmup_epochs=int(cfg.trainer.imitation_warmup_epochs),
            entropy_weight=float(cfg.loss.entropy_weight),
            lam=float(cfg.env.lam),
        )
