"""Queueing-domain LightningModule."""

from __future__ import annotations

from torch import Tensor

from certiqnet.train.common.module import BaseCertiQLightningModule


class QueueingLightningModule(BaseCertiQLightningModule):
    """LightningModule for the queueing domain.

    This is the primary domain — uses ``QueueingAdapter``-style
    observation transforms (default: identity) and is compatible with
    ``QGymDataModule``.
    """

    def _make_observation(self, Q: Tensor, mu: Tensor, xi: Tensor | None) -> tuple[Tensor, Tensor, Tensor | None]:
        Q, mu, xi = super()._make_observation(Q, mu, xi)
        # Move mu to the correct device/dtype after adapter transforms
        if mu is not None:
            mu = mu.to(device=Q.device, dtype=Q.dtype)
        return Q, mu, xi

