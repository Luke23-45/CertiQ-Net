"""CertiQ index model with a minimal residual policy and exact certification."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from certiqnet.dispatcher.certiq.certificate import DifferentiableKLProjection
from certiqnet.dispatcher.delay_geometry import quadratic_drift_index, sed_index
from certiqnet.dispatcher.types import DispatcherDiagnostics, DispatcherForward


def expand_mu(Q: Tensor, mu: Tensor) -> Tensor:
    if mu.dim() == 1:
        return mu.unsqueeze(0).expand(Q.shape[0], -1)
    return mu


def _feature_stack(Q: Tensor, mu: Tensor, xi: Tensor | None, d_xi: int) -> tuple[Tensor, Tensor]:
    mu_safe = mu.clamp_min(torch.finfo(mu.dtype).tiny)
    inv_mu = mu_safe.reciprocal()
    qmd = quadratic_drift_index(Q, mu_safe)
    sed = sed_index(Q, mu_safe)
    token_parts = [
        Q.unsqueeze(-1),
        torch.log1p(Q).unsqueeze(-1),
        mu_safe.unsqueeze(-1),
        inv_mu.unsqueeze(-1),
        (Q / mu_safe).unsqueeze(-1),
        qmd.unsqueeze(-1),
        sed.unsqueeze(-1),
    ]
    if xi is not None:
        if d_xi <= 0:
            raise ValueError("xi was provided but the model was built without context features.")
        if xi.shape[:2] != Q.shape:
            raise ValueError("xi must match the batch and resource dimensions of Q.")
        token_parts.append(xi)
    elif d_xi > 0:
        token_parts.append(torch.zeros(Q.shape[0], Q.shape[1], d_xi, device=Q.device, dtype=Q.dtype))
    token_features = torch.cat(token_parts, dim=-1)
    global_features = torch.cat(
        [
            Q.mean(dim=-1, keepdim=True),
            Q.amax(dim=-1, keepdim=True),
            Q.std(dim=-1, keepdim=True, unbiased=False),
            mu_safe.mean(dim=-1, keepdim=True),
            mu_safe.amin(dim=-1, keepdim=True),
            qmd.mean(dim=-1, keepdim=True),
            qmd.amin(dim=-1, keepdim=True),
            qmd.amax(dim=-1, keepdim=True),
            sed.mean(dim=-1, keepdim=True),
            sed.amin(dim=-1, keepdim=True),
            sed.amax(dim=-1, keepdim=True),
        ],
        dim=-1,
    )
    return token_features, global_features


class ResidualQMDScorer(nn.Module):
    """A small shared-feature scorer for residual policy learning."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        d_xi: int = 0,
        token_layers: int = 2,
        global_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_xi = int(d_xi)
        self.feature_dim = 7 + self.d_xi
        self.global_dim = 11
        self.token_encoder = self._make_mlp(self.feature_dim, hidden_dim, hidden_dim, token_layers, dropout)
        self.global_encoder = self._make_mlp(self.global_dim, hidden_dim, hidden_dim, global_layers, dropout)
        self.readout = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, layers: int, dropout: float) -> nn.Sequential:
        dims = [in_dim] + [hidden_dim] * max(layers - 1, 0) + [out_dim]
        modules: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            modules.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                modules.append(nn.GELU())
                if dropout > 0:
                    modules.append(nn.Dropout(dropout))
        return nn.Sequential(*modules)

    def forward(self, Q: Tensor, mu: Tensor, xi: Tensor | None = None) -> Tensor:
        token_features, global_features = _feature_stack(Q, mu, xi, self.d_xi)
        local = self.token_encoder(token_features)
        pooled = self.global_encoder(global_features)
        pooled_expanded = pooled.unsqueeze(1).expand(-1, Q.shape[1], -1)
        residual = self.readout(torch.cat([local, pooled_expanded], dim=-1)).squeeze(-1)
        return residual


class CertiQIndexModel(nn.Module):
    """Residual QMD policy with an explicit KL certificate projection.

    The model keeps the formal certificate boundary explicit:

    1. compute a QMD base geometry,
    2. learn a residual correction,
    3. form a raw proposal by softmaxing the corrected index,
    4. optionally certify the proposal with the exact KL projection.
    """

    def __init__(
        self,
        N: int,
        hidden_dim: int = 64,
        tau: float = 1.0,
        exploration_temperature: float = 1.0,
        C: float = 2.0,
        beta: float = 1.0,
        cost_fn: str = "qmd",
        d_xi: int = 0,
        token_layers: int = 2,
        global_layers: int = 2,
        dropout: float = 0.0,
        candidate_top_k: int | None = None,
        gate_hidden_dim: int = 32,
        certificate_mode: str | None = None,
        constraint_mode: str = "exact",
    ) -> None:
        super().__init__()
        self.N = int(N)
        self.tau = float(tau)
        self.exploration_temperature = float(exploration_temperature)
        self.C = float(C)
        self.beta = float(beta)
        self.cost_fn = str(cost_fn)
        self.d_xi = int(d_xi)
        self.constraint_mode = str(constraint_mode)
        self.candidate_top_k = None if candidate_top_k is None else max(0, int(candidate_top_k))
        if certificate_mode is None:
            self.certificate_mode = "exact" if self.constraint_mode in {"exact", "projection", "lagrangian"} else "none"
        else:
            self.certificate_mode = str(certificate_mode)
        self.scorer = ResidualQMDScorer(
            hidden_dim=hidden_dim,
            d_xi=self.d_xi,
            token_layers=token_layers,
            global_layers=global_layers,
            dropout=dropout,
        )
        self.gate_head = nn.Sequential(
            nn.Linear(11 + 4, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
        )

    def reset_dispatch_state(self) -> None:
        """Stateless model; kept for API compatibility."""

    def _base_geometry(self, Q: Tensor, mu: Tensor) -> Tensor:
        if self.cost_fn == "qmd":
            return quadratic_drift_index(Q, mu)
        if self.cost_fn == "sed":
            return sed_index(Q, mu)
        raise ValueError(f"Unknown cost_fn: {self.cost_fn}")

    def _resolve_context(self, Q: Tensor, mu: Tensor, xi: Tensor | None) -> Tensor | None:
        if self.d_xi <= 0:
            return None
        if xi is not None and xi.shape[-1] == self.d_xi and torch.count_nonzero(xi).item() != 0:
            return xi
        mu_safe = mu.clamp_min(torch.finfo(mu.dtype).tiny)
        qmd = quadratic_drift_index(Q, mu_safe)
        sed = sed_index(Q, mu_safe)
        total = Q.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        queue_share = Q / total
        qmd_rank = torch.argsort(torch.argsort(qmd, dim=-1), dim=-1).float()
        sed_rank = torch.argsort(torch.argsort(sed, dim=-1), dim=-1).float()
        denom = max(Q.shape[-1] - 1, 1)
        qmd_rank = qmd_rank / denom
        sed_rank = sed_rank / denom
        qmd_gap = (qmd - qmd.min(dim=-1, keepdim=True).values) / (qmd.abs().mean(dim=-1, keepdim=True) + 1.0)
        sed_gap = (sed - sed.min(dim=-1, keepdim=True).values) / (sed.abs().mean(dim=-1, keepdim=True) + 1.0)
        disagreement = (qmd.argmin(dim=-1) != sed.argmin(dim=-1)).float().unsqueeze(-1).expand_as(Q)
        features = [queue_share, qmd_rank, sed_rank, qmd_gap, sed_gap, disagreement]
        xi_ctx = torch.stack([f for f in features], dim=-1)
        if xi_ctx.shape[-1] > self.d_xi:
            xi_ctx = xi_ctx[..., : self.d_xi]
        elif xi_ctx.shape[-1] < self.d_xi:
            pad = torch.zeros(*xi_ctx.shape[:-1], self.d_xi - xi_ctx.shape[-1], device=xi_ctx.device, dtype=xi_ctx.dtype)
            xi_ctx = torch.cat([xi_ctx, pad], dim=-1)
        return xi_ctx

    @staticmethod
    def _min_margin(cost: Tensor) -> Tensor:
        top2 = torch.topk(cost, k=min(2, cost.shape[-1]), largest=False).values
        if top2.shape[-1] == 1:
            return torch.zeros_like(top2[:, 0])
        return top2[:, 1] - top2[:, 0]

    def _ambiguity_features(self, Q: Tensor, mu: Tensor, base_index: Tensor) -> Tensor:
        qmd_index = quadratic_drift_index(Q, mu)
        sed_index_v = sed_index(Q, mu)
        qmd_margin = self._min_margin(qmd_index)
        sed_margin = self._min_margin(sed_index_v)
        base_margin = self._min_margin(base_index)
        disagreement = (qmd_index.argmin(dim=-1) != sed_index_v.argmin(dim=-1)).float()
        return torch.stack([qmd_margin, sed_margin, base_margin, disagreement], dim=-1)

    def forward_full(
        self,
        Q: Tensor,
        mu: Tensor,
        xi: Tensor | None = None,
        *,
        training_mode: bool = False,
    ) -> DispatcherForward:
        batch, _ = Q.shape
        mu_b = expand_mu(Q, mu)
        xi_ctx = self._resolve_context(Q, mu_b, xi)
        base_index = self._base_geometry(Q, mu_b)
        _, global_features = _feature_stack(Q, mu_b, xi_ctx, self.d_xi)
        residual = self.scorer(Q, mu_b, xi_ctx)
        ambiguity = self._ambiguity_features(Q, mu_b, base_index)
        gate_input = torch.cat([global_features, ambiguity], dim=-1)
        gate = torch.sigmoid(self.gate_head(gate_input)).clamp(min=0.05, max=1.0)
        learned_index = base_index + residual * gate

        effective_tau = self.tau * (self.exploration_temperature if training_mode else 1.0)
        proposal_logits = (-learned_index / effective_tau).clamp(min=-20.0, max=20.0)
        if self.candidate_top_k is not None and 0 < self.candidate_top_k < self.N:
            candidate_k = min(self.candidate_top_k, self.N)
            candidate_idx = torch.topk(base_index, k=candidate_k, largest=False).indices
            candidate_mask = torch.full_like(proposal_logits, float("-inf"))
            candidate_mask.scatter_(1, candidate_idx, 0.0)
            proposal_logits = proposal_logits + candidate_mask
        proposal = torch.softmax(proposal_logits, dim=-1)

        cost = base_index
        m_q = cost.min(dim=-1).values
        budget = m_q + self.C

        if self.certificate_mode == "exact":
            certified, projection_multiplier, solver_status = DifferentiableKLProjection.apply(
                proposal_logits,
                cost,
                budget,
            )
            projection_active = (projection_multiplier > 1e-12).to(dtype=Q.dtype)
            fallback_flag = (solver_status != 0).to(dtype=Q.dtype)
        elif self.certificate_mode == "none":
            certified = proposal
            projection_multiplier = torch.zeros(batch, device=Q.device, dtype=Q.dtype)
            solver_status = torch.zeros(batch, device=Q.device, dtype=Q.dtype)
            projection_active = torch.zeros(batch, device=Q.device, dtype=Q.dtype)
            fallback_flag = torch.zeros(batch, device=Q.device, dtype=Q.dtype)
        else:
            raise ValueError(f"Unknown certificate_mode: {self.certificate_mode}")

        correction_magnitude = (certified - proposal).abs().sum(dim=-1)
        a_proposal = (proposal * cost).sum(dim=-1)
        a_final = (certified * cost).sum(dim=-1)
        constraint_violation = (a_final - budget).clamp(min=0.0)
        policy_entropy = -(certified * certified.clamp_min(1e-9).log()).sum(dim=-1)

        diag = DispatcherDiagnostics(
            A_proposal=a_proposal,
            A_final=a_final,
            m_Q=m_q,
            B_Q=budget,
            certificate_slack=budget - a_final,
            constraint_violation=constraint_violation,
            usage_raw=torch.full((batch,), float("nan"), device=Q.device, dtype=Q.dtype),
            usage_final=torch.full((batch,), float("nan"), device=Q.device, dtype=Q.dtype),
            usage_cap=torch.full((batch,), float("nan"), device=Q.device, dtype=Q.dtype),
            policy_entropy=policy_entropy,
            selected_resource=certified.argmax(dim=-1),
            pressure_mean=torch.zeros(batch, device=Q.device, dtype=Q.dtype),
            pressure_max=torch.zeros(batch, device=Q.device, dtype=Q.dtype),
            pressure_update_norm=correction_magnitude,
            projection_multiplier=projection_multiplier,
            projection_active=projection_active,
            solver_status=solver_status,
            fallback_flag=fallback_flag,
            correction_magnitude=correction_magnitude,
        )
        return DispatcherForward(
            pi=certified,
            diagnostics=diag,
            p_cert=certified,
            p_proposal=proposal,
            usage_raw=diag.usage_raw,
            usage_final=diag.usage_final,
            proposal_logits=proposal_logits,
            index_values=learned_index,
        )

    def forward(
        self,
        Q: Tensor,
        mu: Tensor,
        xi: Tensor | None = None,
        *,
        training_mode: bool = False,
    ) -> tuple[Tensor, DispatcherDiagnostics]:
        out = self.forward_full(Q, mu, xi, training_mode=training_mode)
        return out.pi, out.diagnostics
