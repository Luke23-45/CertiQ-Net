"""Permutation-equivariant learned proposal for z3 dispatch."""

import torch
import torch.nn as nn
from torch import Tensor


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
    """Shared-local, invariant-global, equivariant proposal operator."""

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
    ) -> None:
        super().__init__()
        assert d_xi >= 0, "d_xi must be non-negative."
        assert pooling in {"attention", "mean"}, "pooling must be attention or mean."
        assert correction_bound > 0, "correction_bound must be positive."
        assert 0 < usage_max <= 1, "usage_max must be in (0, 1]."
        self.d_xi = int(d_xi)
        self.pooling = pooling
        self.correction_bound = float(correction_bound)
        self.usage_max = float(usage_max)
        self.local = mlp(4 + d_xi, hidden_dim, d_local, local_layers)
        self.attn = nn.Linear(d_local, 1, bias=False) if pooling == "attention" else None
        self.global_out = nn.Linear(d_local, d_global)
        self.update = mlp(d_local + d_global, hidden_dim, d_local, update_layers)
        self.correction = nn.Linear(d_local, 1)
        self.usage = nn.Linear(d_global, 1)
        self.value = nn.Linear(d_global, 1)

    def _features(self, Q: Tensor, mu: Tensor, y: Tensor, xi: Tensor | None) -> Tensor:
        batch, n = Q.shape
        lam_total = mu.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(mu.dtype).tiny)
        features = [
            torch.log1p(Q).unsqueeze(-1),
            torch.log(mu.clamp_min(torch.finfo(mu.dtype).tiny)).unsqueeze(-1),
            y.unsqueeze(-1),
            (mu / lam_total).unsqueeze(-1),
        ]
        if xi is None:
            if self.d_xi > 0:
                features.append(torch.zeros(batch, n, self.d_xi, device=Q.device, dtype=Q.dtype))
        else:
            assert xi.shape == (batch, n, self.d_xi), "xi must have shape (B, N, d_xi)."
            features.append(xi)
        return torch.cat(features, dim=-1)

    def forward(
        self, Q: Tensor, mu: Tensor, y: Tensor, u_cert: Tensor, xi: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return ``(p_proposal, correction, usage_raw, value)``."""
        z0 = self.local(self._features(Q, mu, y, xi))
        if self.pooling == "attention":
            assert self.attn is not None
            weights = torch.softmax(self.attn(z0).squeeze(-1), dim=-1)
            pooled = (weights.unsqueeze(-1) * z0).sum(dim=1)
        else:
            pooled = z0.mean(dim=1)
        g = self.global_out(pooled)
        z1 = self.update(torch.cat([z0, g.unsqueeze(1).expand(-1, Q.shape[1], -1)], dim=-1))
        correction = self.correction_bound * torch.tanh(self.correction(z1).squeeze(-1))
        logits = u_cert + correction
        p = torch.softmax(logits, dim=-1)
        p = p.clamp_min(torch.finfo(p.dtype).tiny)
        p = p / p.sum(dim=-1, keepdim=True)
        usage_raw = self.usage_max * torch.sigmoid(self.usage(g).squeeze(-1))
        value = self.value(g).squeeze(-1)
        return p, correction, usage_raw, value

