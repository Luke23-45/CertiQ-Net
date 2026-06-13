"""Heuristic supervision utilities for CertiQ-Net."""

from __future__ import annotations

from torch import Tensor


def heuristic_actions(Q: Tensor, mu: Tensor) -> tuple[Tensor, Tensor]:
    """Return heuristic SED and QMD labels for a batch of states."""
    mu_safe = mu.clamp_min(mu.new_tensor(1e-12))
    sed = ((Q + 1.0) / mu_safe).argmin(dim=-1)
    qmd = ((2.0 * Q + 1.0) / mu_safe).argmin(dim=-1)
    return sed, qmd
