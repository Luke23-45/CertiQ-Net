"""Mixture-of-Experts-domain LightningModule."""

from __future__ import annotations

from torch import Tensor

from certiqnet.train.common.module import BaseCertiQLightningModule


class MoELightningModule(BaseCertiQLightningModule):
    """LightningModule for the mixture-of-experts domain.

    Uses ``MoEAdapter``-style observation transforms (expert
    throughput estimates).
    """

    def _make_observation(self, Q: Tensor, mu: Tensor, xi: Tensor | None) -> tuple[Tensor, Tensor, Tensor | None]:
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "adapter"):
            return dm.adapter.make_observation(Q, mu)
        return Q, mu, xi
