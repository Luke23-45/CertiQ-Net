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
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "adapter"):
            return dm.adapter.make_observation(Q, mu)
        return Q, mu, xi
