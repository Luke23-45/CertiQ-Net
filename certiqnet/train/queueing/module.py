"""Queueing-domain LightningModule."""

from __future__ import annotations

from torch import Tensor

from certiqnet.train.common.module import BaseCertiQLightningModule


class QueueingLightningModule(BaseCertiQLightningModule):
    """LightningModule for the queueing domain.

    This is the primary domain — uses ``QueueingAdapter``-style
    observation transforms (default: identity) and is compatible with
    both ``CertiQNetDataModule`` (synthetic) and ``QGymDataModule``.
    """

    def _make_observation(self, Q: Tensor, mu: Tensor, xi: Tensor | None) -> tuple[Tensor, Tensor, Tensor | None]:
        dm = getattr(self.trainer, "datamodule", None)
        adapter = getattr(dm, "adapter", None) if dm is not None else None
        if adapter is not None:
            Q_obs, mu_obs, xi_obs = adapter.make_observation(Q, mu)
            mu_obs = mu_obs.to(device=Q.device, dtype=Q.dtype) if mu_obs is not None else mu
            return Q_obs, mu_obs, xi_obs
        if mu.dim() == 1:
            mu = mu.unsqueeze(0).expand(Q.shape[0], -1)
        return Q, mu, xi

