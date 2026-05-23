"""Permutation-equivariant resource encoder."""

import torch
import torch.nn as nn
from torch import Tensor


def build_mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
    """Build a GELU MLP with full type annotations."""
    assert n_layers >= 1, "n_layers must be at least 1."
    dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
    layers: list[nn.Module] = []
    for idx in range(len(dims) - 1):
        layers.append(nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class ResourceEncoder(nn.Module):
    """Shared local encoder plus permutation-invariant global pooling."""

    def __init__(
        self,
        d_xi: int,
        d_local: int,
        d_global: int,
        hidden_dim: int,
        n_layers_local: int,
        n_layers_res: int,
        pooling: str = "attention",
    ) -> None:
        super().__init__()
        assert d_xi >= 0, "d_xi must be non-negative."
        assert pooling in {"attention", "mean"}, "pooling must be 'attention' or 'mean'."
        in_dim = 4 + d_xi
        self.f_loc = build_mlp(in_dim, hidden_dim, d_local, n_layers_local)
        self.attn_scorer = nn.Linear(d_local, 1, bias=False) if pooling == "attention" else None
        self.rho_out = nn.Linear(d_local, d_global)
        self.f_res = build_mlp(d_local + d_global, hidden_dim, d_local, n_layers_res)
        self.pooling = pooling
        self.d_local = d_local
        self.d_global = d_global
        self.d_xi = d_xi

    def encode_local(self, Q: Tensor, mu: Tensor, beta: float, xi: Tensor | None = None) -> Tensor:
        """Return local embeddings ``z0`` with shape ``(B, N, d_local)``."""
        assert Q.dim() == 2 and mu.dim() == 2, "Q and mu must have shape (B, N)."
        assert Q.shape == mu.shape, "Q and mu must have identical shape."
        assert (Q >= 0).all() and (mu > 0).all(), "Invalid queue lengths or service rates."
        batch, n = Q.shape
        lam_total = mu.sum(dim=-1, keepdim=True)
        scalar_parts = [
            torch.log1p(Q),
            torch.log(mu.clamp(min=1e-8)),
            Q / mu.pow(beta),
            mu / lam_total,
        ]
        features = [part.unsqueeze(-1) for part in scalar_parts]
        if xi is not None:
            assert xi.shape[:2] == (batch, n), "xi must have shape (B, N, d_xi)."
            assert xi.shape[-1] == self.d_xi, "xi last dimension must match configured d_xi."
            features.append(xi)
        elif self.d_xi > 0:
            features.append(torch.zeros(batch, n, self.d_xi, device=Q.device, dtype=Q.dtype))
        s = torch.cat(features, dim=-1)
        return self.f_loc(s)

    def pool_global(self, z0: Tensor) -> Tensor:
        """Return invariant global context with shape ``(B, d_global)``."""
        if self.pooling == "attention":
            assert self.attn_scorer is not None
            scores = self.attn_scorer(z0).squeeze(-1)
            weights = torch.softmax(scores, dim=-1)
            z_bar = (weights.unsqueeze(-1) * z0).sum(dim=1)
        else:
            z_bar = z0.mean(dim=1)
        return self.rho_out(z_bar)

    def encode_residual(self, z0: Tensor, g: Tensor) -> Tensor:
        """Return residual embeddings ``z1`` with shape ``(B, N, d_local)``."""
        batch, n, _ = z0.shape
        g_expanded = g.unsqueeze(1).expand(batch, n, -1)
        return self.f_res(torch.cat([z0, g_expanded], dim=-1))
