"""Permutation-equivariant learned proposal for z3 dispatch."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from certiqnet.dispatcher.interaction import (
    DispatchInteractionEncoder,
    legacy_token_features,
)


def mlp(in_dim: int, hidden_dim: int, out_dim: int, layers: int) -> nn.Sequential:
    """Build a compact GELU MLP."""
    assert layers >= 1, "layers must be positive."
    dims = [in_dim] + [hidden_dim] * (layers - 1) + [out_dim]
    modules: list[nn.Module] = []
    for idx in range(len(dims) - 1):
        modules.append(nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            modules.append(nn.GELU())
    return nn.Sequential(*modules)


class SetProposal(nn.Module):
    """Shared-local, interaction-aware, equivariant proposal operator."""

    def __init__(
        self,
        *,
        d_xi: int,
        d_local: int,
        d_global: int,
        hidden_dim: int,
        local_layers: int,
        update_layers: int,
        pooling: str,
        correction_bound: float,
        usage_max: float,
        encoder_layers: int = 2,
        num_heads: int = 4,
        num_inducing_points: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        del local_layers
        assert d_xi >= 0, "d_xi must be non-negative."
        assert pooling in {"attention", "mean"}, "pooling must be attention or mean."
        assert correction_bound > 0, "correction_bound must be positive."
        assert 0 < usage_max <= 1, "usage_max must be in (0, 1]."
        self.d_xi = int(d_xi)
        self.pooling = pooling
        self.correction_bound = float(correction_bound)
        self.usage_max = float(usage_max)
        self.encoder = DispatchInteractionEncoder(
            feature_dim=10 + self.d_xi,
            d_model=d_local,
            d_global=d_global,
            num_layers=encoder_layers,
            num_heads=num_heads,
            num_inducing_points=num_inducing_points,
            dropout=dropout,
            global_feature_dim=10,
        )
        self.update = mlp(d_local + d_global, hidden_dim, d_local, update_layers)
        self.correction = nn.Linear(d_local, 1)
        self.usage = nn.Linear(d_global, 1)
        self.value = nn.Linear(d_global, 1)

    def forward(
        self,
        Q: Tensor,
        mu: Tensor,
        y: Tensor,
        u_cert: Tensor,
        pressure: Tensor | None = None,
        xi: Tensor | None = None,
        pressure_scale: float = 0.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return ``(p_proposal, correction, usage_raw, value)``."""
        if pressure is None:
            pressure = torch.zeros_like(Q)
        token_features, global_features = legacy_token_features(
            Q, mu, y, pressure, u_cert, xi, d_xi=self.d_xi
        )
        z_tokens, z_global = self.encoder(token_features, global_features)
        z1 = self.update(torch.cat([z_tokens, z_global.unsqueeze(1).expand(-1, Q.shape[1], -1)], dim=-1))
        correction = self.correction_bound * torch.tanh(self.correction(z1).squeeze(-1))
        logits = u_cert + correction - float(pressure_scale) * pressure
        p = torch.softmax(logits, dim=-1)
        p = p.clamp_min(torch.finfo(p.dtype).tiny)
        p = p / p.sum(dim=-1, keepdim=True)
        usage_raw = self.usage_max * torch.sigmoid(self.usage(z_global).squeeze(-1))
        value = self.value(z_global).squeeze(-1)
        return p, correction, usage_raw, value
