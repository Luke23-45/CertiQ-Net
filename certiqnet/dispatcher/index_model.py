"""CertiQ index model for marginal-cost dispatch with KL projection."""

import torch
import torch.nn as nn
from torch import Tensor

from certiqnet.dispatcher.certificate import (
    arrival_coordinate,
    kl_project_linear,
    normalize_policy,
    policy_entropy,
)
from certiqnet.dispatcher.delay_geometry import quadratic_drift_index
from certiqnet.dispatcher.types import DispatcherDiagnostics, DispatcherForward


def expand_mu(Q: Tensor, mu: Tensor) -> Tensor:
    if mu.dim() == 1:
        return mu.unsqueeze(0).expand(Q.shape[0], -1)
    return mu


class MarginalIndexHead(nn.Module):
    """Learned marginal cost index I_i(Q, mu) per resource."""

    def __init__(self, N: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.N = N
        self.local_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
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
        del xi
        batch, n = Q.shape
        mu_safe = mu.clamp_min(torch.finfo(mu.dtype).tiny)
        qmd_drift = (2.0 * Q + 1.0) / mu_safe
        features = torch.stack(
            [
                Q,
                mu,
                torch.log1p(Q),
                torch.log(mu_safe),
                Q / mu_safe,
                qmd_drift,
            ],
            dim=-1,
        )
        z_local = self.local_encoder(features)
        z_global = z_local.mean(dim=1)
        z_global = self.global_encoder(z_global)
        z_global_expanded = z_global.unsqueeze(1).expand(-1, n, -1)
        z_combined = torch.cat([z_local, z_global_expanded], dim=-1)
        index_values = self.index_head(z_combined).squeeze(-1)
        value = self.value_head(z_global).squeeze(-1)
        return index_values, value


class CertiQIndexModel(nn.Module):
    """Dispatch by learned marginal cost index with KL projection."""

    def __init__(
        self,
        N: int,
        hidden_dim: int = 64,
        tau: float = 1.0,
        C: float = 2.0,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.N = N
        self.tau = tau
        self.C = C
        self.beta = beta
        self.index_head = MarginalIndexHead(N, hidden_dim=hidden_dim)

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

        drift = quadratic_drift_index(Q, mu_b)
        drift_min = drift.min(dim=-1).values
        budget = drift_min + self.C
        p_cert = normalize_policy(torch.softmax(-drift / self.tau, dim=-1))

        index_values, value = self.index_head(Q, mu_b, xi)
        # Sharpen temperature at eval to approach hard dispatch
        effective_tau = self.tau if training_mode else self.tau / 4.0
        q_proposal = torch.softmax(-index_values / effective_tau, dim=-1)
        q_proposal = normalize_policy(q_proposal)
        proposal_logits = -index_values / effective_tau

        if certify:
            pi, nu = kl_project_linear(q_proposal, drift, budget)
        else:
            pi = q_proposal
            nu = torch.zeros(batch, device=Q.device, dtype=Q.dtype)

        projected = (pi - q_proposal).abs().sum(dim=-1) > 1e-8
        a_proposal = arrival_coordinate(q_proposal, drift)
        a_final = arrival_coordinate(pi, drift)
        correction = (pi - q_proposal).abs().sum(dim=-1)

        diag = DispatcherDiagnostics(
            A_cert=arrival_coordinate(p_cert, drift),
            A_proposal=a_proposal,
            A_final=a_final,
            m_Q=drift_min,
            B_Q=budget,
            certificate_slack=budget - a_final,
            usage_raw=torch.ones(batch, device=Q.device, dtype=Q.dtype),
            usage_final=(~projected).to(dtype=Q.dtype),
            usage_cap=torch.ones(batch, device=Q.device, dtype=Q.dtype),
            fallback_active=torch.zeros(batch, dtype=torch.bool, device=Q.device),
            correction_magnitude=correction,
            policy_entropy=policy_entropy(pi),
            selected_resource=pi.argmax(dim=-1),
            pressure_mean=torch.zeros(batch, device=Q.device, dtype=Q.dtype),
            pressure_max=torch.zeros(batch, device=Q.device, dtype=Q.dtype),
            pressure_update_norm=torch.zeros(batch, device=Q.device, dtype=Q.dtype),
            projection_nu=nu,
            projection_active=projected,
            projection_slack=budget - a_proposal,
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
            index_values=index_values,
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
