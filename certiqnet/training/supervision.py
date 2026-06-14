"""Heuristic supervision utilities for CertiQ-Net."""

from __future__ import annotations

from torch import Tensor

from certiqnet.dispatcher.delay_geometry import sed_index
from certiqnet.dispatcher.delay_geometry import quadratic_drift_index


def heuristic_actions(Q: Tensor, mu: Tensor) -> tuple[Tensor, Tensor]:
    """Return SED (primary) and QMD (evaluation baseline) labels."""
    mu_safe = mu.clamp_min(mu.new_tensor(1e-12))
    sed = sed_index(Q, mu_safe).argmin(dim=-1)
    qmd = quadratic_drift_index(Q, mu_safe).argmin(dim=-1)
    return sed, qmd
