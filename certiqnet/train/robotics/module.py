"""Robotics-domain LightningModule."""

from __future__ import annotations

from torch import Tensor

from certiqnet.train.common.module import BaseCertiQLightningModule


class RoboticsLightningModule(BaseCertiQLightningModule):
    """LightningModule for the robotics spatial-task allocation domain.

    Uses ``RoboticsAdapter``-style observation transforms (travel-time
    context).
    """

    def _make_observation(self, Q: Tensor, mu: Tensor, xi: Tensor | None) -> tuple[Tensor, Tensor, Tensor | None]:
        dm = getattr(self.trainer, "datamodule", None)
        adapter = getattr(dm, "adapter", None) if dm is not None else None
        if adapter is not None:
            return adapter.make_observation(Q, mu)
        return Q, mu, xi
