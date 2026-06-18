"""Channel-domain LightningModule."""

from __future__ import annotations

from torch import Tensor

from certiqnet.train.common.module import BaseCertiQLightningModule


class ChannelLightningModule(BaseCertiQLightningModule):
    """LightningModule for the wireless-channel domain.

    Uses ``ChannelAdapter``-style observation transforms (SNR-based
    context vectors).
    """

    def _make_observation(self, Q: Tensor, mu: Tensor, xi: Tensor | None) -> tuple[Tensor, Tensor, Tensor | None]:
        Q, mu, xi = super()._make_observation(Q, mu, xi)
        return Q, mu, xi
