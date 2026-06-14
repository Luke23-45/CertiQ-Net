"""CertiQ index model for marginal-cost dispatch with KL projection."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from certiqnet.dispatcher.certificate import (
    DifferentiableKLProjection,
    arrival_coordinate,
    normalize_policy,
    policy_entropy,
)
from certiqnet.dispatcher.delay_geometry import sed_index
from certiqnet.dispatcher.interaction import DispatchInteractionEncoder, index_token_features
from certiqnet.dispatcher.types import DispatcherDiagnostics, DispatcherForward


def expand_mu(Q: Tensor, mu: Tensor) -> Tensor:
    if mu.dim() == 1:
        return mu.unsqueeze(0).expand(Q.shape[0], -1)
    return mu


class MarginalIndexHead(nn.Module):
    """Learned marginal cost index I_i(Q, mu) per resource.

    Produces raw logits (not wrapped in softmax).  The output is a
    pure learned per-resource score — no analytic baseline is baked in.
    """

    def __init__(
        self,
        N: int,
        hidden_dim: int = 64,
        d_xi: int = 0,
        *,
        encoder_layers: int = 2,
        num_heads: int = 4,
        num_inducing_points: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.N = N
        self.d_xi = int(d_xi)
        self.encoder = DispatchInteractionEncoder(
            feature_dim=6 + self.d_xi,
            d_model=hidden_dim,
            d_global=hidden_dim,
            num_layers=encoder_layers,
            num_heads=num_heads,
            num_inducing_points=num_inducing_points,
            dropout=dropout,
            global_feature_dim=8,
        )
        self.index_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, Q: Tensor, mu: Tensor, xi: Tensor | None = None) -> tuple[Tensor, Tensor]:
        token_features, global_features = index_token_features(Q, mu, xi, d_xi=self.d_xi)
        z_local, z_global = self.encoder(token_features, global_features)
        z_global_expanded = z_global.unsqueeze(1).expand(-1, Q.shape[1], -1)
        z_combined = torch.cat([z_local, z_global_expanded], dim=-1)
        logits = self.index_head(z_combined).squeeze(-1)
        value = self.value_head(z_global).squeeze(-1)
        return logits, value


class CertiQIndexModel(nn.Module):
    """Dispatch by learned marginal cost index with KL projection.

    The forward pass:

        1. Compute a per-resource **cost** (delay proxy, default SED).
        2. Compute the certificate envelope ``budget = min_i cost_i + C``.
        3. Produce raw **logits** from the learned encoder + index head.
        4. Apply the differentiable KL projection to obtain the final
           routing distribution ``pi``, which satisfies
           ``E_pi[cost] <= budget``.

    The projection layer uses implicit differentiation so that gradients
    flow end-to-end from ``pi`` back to the encoder parameters, exactly
    as in OptNet / differentiable convex optimisation layers.
    """

    def __init__(
        self,
        N: int,
        hidden_dim: int = 64,
        tau: float = 1.0,
        C: float = 2.0,
        beta: float = 1.0,
        d_xi: int = 0,
        encoder_layers: int = 2,
        num_heads: int = 4,
        num_inducing_points: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.N = N
        self.tau = tau
        self.C = C
        self.beta = beta
        self.d_xi = int(d_xi)
        self.index_head = MarginalIndexHead(
            N,
            hidden_dim=hidden_dim,
            d_xi=self.d_xi,
            encoder_layers=encoder_layers,
            num_heads=num_heads,
            num_inducing_points=num_inducing_points,
            dropout=dropout,
        )

    def reset_dispatch_state(self) -> None:
        """Stateless model; kept for API compatibility."""

    def forward_full(
        self,
        Q: Tensor,
        mu: Tensor,
        xi: Tensor | None = None,
        *,
        certify: bool = True,
        training_mode: bool = False,
    ) -> DispatcherForward:
        batch, n = Q.shape
        mu_b = expand_mu(Q, mu)

        cost = sed_index(Q, mu_b)
        cost_min = cost.min(dim=-1).values
        budget = torch.median(cost, dim=-1).values + self.C
        p_cert = normalize_policy(torch.softmax(-cost / self.tau, dim=-1))

        learned_logits, value = self.index_head(Q, mu_b, xi)
        effective_tau = self.tau
        proposal_logits = -learned_logits / effective_tau

        if certify:
            pi, nu, solver_status = DifferentiableKLProjection.apply(
                proposal_logits, cost, budget
            )
        else:
            pi = torch.softmax(proposal_logits, dim=-1)
            nu = torch.zeros(batch, device=Q.device, dtype=Q.dtype)
            solver_status = torch.zeros(batch, device=Q.device, dtype=torch.long)

        fallback_needed = solver_status > 0
        if fallback_needed.any():
            fallback_idx = cost.argmin(dim=-1)
            pi_fallback = torch.zeros_like(pi)
            pi_fallback.scatter_(1, fallback_idx.unsqueeze(-1), 1.0)
            pi = torch.where(fallback_needed.unsqueeze(-1), pi_fallback, pi)
            nu = torch.where(fallback_needed, torch.zeros_like(nu), nu)

        q_proposal = torch.softmax(proposal_logits, dim=-1)

        projected = (pi - q_proposal).abs().sum(dim=-1) > 1e-8
        a_proposal = arrival_coordinate(q_proposal, cost)
        a_final = arrival_coordinate(pi, cost)
        correction = (pi - q_proposal).abs().sum(dim=-1)

        diag = DispatcherDiagnostics(
            A_cert=arrival_coordinate(p_cert, cost),
            A_proposal=a_proposal,
            A_final=a_final,
            m_Q=cost_min,
            B_Q=budget,
            certificate_slack=budget - a_final,
            usage_raw=torch.ones(batch, device=Q.device, dtype=Q.dtype),
            usage_final=(~projected).to(dtype=Q.dtype),
            usage_cap=torch.ones(batch, device=Q.device, dtype=Q.dtype),
            fallback_active=fallback_needed,
            correction_magnitude=correction,
            policy_entropy=policy_entropy(pi),
            selected_resource=pi.argmax(dim=-1),
            pressure_mean=torch.zeros(batch, device=Q.device, dtype=Q.dtype),
            pressure_max=torch.zeros(batch, device=Q.device, dtype=Q.dtype),
            pressure_update_norm=torch.zeros(batch, device=Q.device, dtype=Q.dtype),
            projection_nu=nu,
            projection_active=projected,
            proposal_slack=budget - a_proposal,
            solver_status=solver_status,
        )
        return DispatcherForward(
            pi=pi,
            diagnostics=diag,
            value=value,
            p_cert=p_cert,
            p_proposal=q_proposal,
            usage_raw=diag.usage_raw,
            usage_final=diag.usage_final,
            proposal_logits=proposal_logits,
            index_values=-proposal_logits * effective_tau,
        )

    def forward(
        self,
        Q: Tensor,
        mu: Tensor,
        xi: Tensor | None = None,
        *,
        certify: bool = True,
        training_mode: bool = False,
    ) -> tuple[Tensor, DispatcherDiagnostics]:
        out = self.forward_full(Q, mu, xi, certify=certify, training_mode=training_mode)
        return out.pi, out.diagnostics
