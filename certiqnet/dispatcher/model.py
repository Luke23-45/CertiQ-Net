"""Canonical z3 CertiQ Dispatcher implementation."""

import torch
import torch.nn as nn
from torch import Tensor

from certiqnet.dispatcher.certificate import (
    arrival_coordinate,
    certify_usage,
    normalize_policy,
    policy_entropy,
)
from certiqnet.dispatcher.config import CertiQDispatcherConfig
from certiqnet.dispatcher.geometry import CertifiedGeometry
from certiqnet.dispatcher.proposal import SetProposal
from certiqnet.dispatcher.types import DispatcherDiagnostics, DispatcherForward


def expand_mu(Q: Tensor, mu: Tensor) -> Tensor:
    """Broadcast ``mu`` to ``Q`` shape."""
    if mu.dim() == 1:
        return mu.unsqueeze(0).expand(Q.shape[0], -1)
    return mu


class CertiQDispatcher(nn.Module):
    """Certified online assignment architecture from the z3 formal definition."""

    def __init__(self, cfg: CertiQDispatcherConfig, N: int, d_xi: int = 0) -> None:
        super().__init__()
        assert N >= 1, "N must be positive."
        self.cfg = cfg
        self.N = int(N)
        self.certificate_mode = cfg.certificate.mode
        self.C = float(cfg.geometry.C)
        self.geometry = CertifiedGeometry(
            alpha_min=cfg.geometry.alpha_min,
            beta_min=cfg.geometry.beta_min,
            gamma_max=cfg.geometry.gamma_max,
            alpha_init=cfg.geometry.alpha_init,
            beta_init=cfg.geometry.beta_init,
            gamma_init=cfg.geometry.gamma_init,
            c_init=cfg.geometry.c_init,
            C=cfg.geometry.C,
        )
        self.proposal = SetProposal(
            d_xi=d_xi,
            d_local=cfg.proposal.d_local,
            d_global=cfg.proposal.d_global,
            hidden_dim=cfg.proposal.hidden_dim,
            local_layers=cfg.proposal.local_layers,
            update_layers=cfg.proposal.update_layers,
            pooling=cfg.proposal.pooling,
            correction_bound=cfg.proposal.correction_bound,
            usage_max=cfg.proposal.usage_max,
        )

    @property
    def beta(self) -> float:
        return float(self.geometry.beta.detach().cpu().item())

    def forward_full(
        self,
        Q: Tensor,
        mu: Tensor,
        xi: Tensor | None = None,
        *,
        certify: bool = True,
    ) -> DispatcherForward:
        """Return full policy, proposal, value, and diagnostics."""
        assert Q.dim() == 2, "Q must have shape (B, N)."
        assert Q.shape[-1] == self.N, "Q last dimension must match model N."
        mu_b = expand_mu(Q, mu)
        assert mu_b.shape == Q.shape, "mu must have shape (N,) or (B, N)."
        assert (Q >= 0).all(), "Queue lengths must be non-negative."
        assert (mu_b > 0).all(), "Service rates must be positive."

        y = self.geometry.weighted_workload(Q, mu_b)
        p_cert, u_cert = self.geometry.policy(Q, mu_b)
        m_q, B_q = self.geometry.envelope(Q, mu_b)
        p_proposal, correction, usage_raw, value = self.proposal(Q, mu_b, y, u_cert, xi)
        mode = self.certificate_mode if certify else "uncertified"
        usage_final, usage_cap, fallback_active = certify_usage(
            mode=mode,
            usage_raw=usage_raw,
            y=y,
            B=B_q,
            p_cert=p_cert,
            p_proposal=p_proposal,
            fallback_radius=float(self.cfg.certificate.fallback_radius),
        )
        pi = (1.0 - usage_final.unsqueeze(-1)) * p_cert + usage_final.unsqueeze(-1) * p_proposal
        pi = normalize_policy(pi)
        A_cert = arrival_coordinate(p_cert, y)
        A_proposal = arrival_coordinate(p_proposal, y)
        A_final = arrival_coordinate(pi, y)
        if mode == "projection":
            violation = (A_final - B_q).clamp_min(0.0).max().item()
            tol = float(self.cfg.certificate.projection_tolerance)
            assert violation <= tol, f"Projection certificate violation: {violation:.3e} > {tol:.3e}."
        diag = DispatcherDiagnostics(
            A_cert=A_cert,
            A_proposal=A_proposal,
            A_final=A_final,
            m_Q=m_q,
            B_Q=B_q,
            certificate_slack=B_q - A_final,
            usage_raw=usage_raw,
            usage_final=usage_final,
            usage_cap=usage_cap,
            fallback_active=fallback_active,
            correction_magnitude=correction.abs().max(dim=-1).values,
            policy_entropy=policy_entropy(pi),
            selected_resource=pi.argmax(dim=-1),
        )
        return DispatcherForward(
            pi=pi,
            diagnostics=diag,
            value=value,
            p_cert=p_cert,
            p_proposal=p_proposal,
            usage_raw=usage_raw,
            usage_final=usage_final,
            proposal_logits=u_cert + correction,
        )

    def forward(
        self,
        Q: Tensor,
        mu: Tensor,
        xi: Tensor | None = None,
        *,
        certify: bool = True,
        training_mode: bool | None = None,
    ) -> tuple[Tensor, DispatcherDiagnostics]:
        """Return final dispatch policy and diagnostics."""
        del training_mode
        out = self.forward_full(Q, mu, xi, certify=certify)
        return out.pi, out.diagnostics

