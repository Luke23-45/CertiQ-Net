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


def _module_device(module: nn.Module) -> torch.device:
    """Return the primary device for a module."""
    return next(module.parameters()).device


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
        self.register_buffer("pressure_state", torch.zeros(self.N), persistent=False)

    @property
    def beta(self) -> float:
        return float(self.geometry.beta.detach().cpu().item())

    def reset_dispatch_state(self) -> None:
        """Reset runtime controller state between independent rollouts."""
        with torch.no_grad():
            self.pressure_state.zero_()

    def _pressure_bias(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        pressure = self.pressure_state.to(device=device, dtype=dtype)
        return pressure.unsqueeze(0).expand(batch_size, -1)

    def _pressure_target(self, mu_b: Tensor) -> Tensor:
        mu_mass = mu_b.clamp_min(torch.finfo(mu_b.dtype).tiny)
        mu_mass = mu_mass / mu_mass.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(mu_b.dtype).tiny)
        return mu_mass.mean(dim=0)

    def _update_pressure(self, pi: Tensor, mu_b: Tensor) -> Tensor:
        pressure = self.pressure_state.to(device=pi.device, dtype=pi.dtype)
        dispatch_mass = pi.mean(dim=0)
        target_mass = self._pressure_target(mu_b)
        next_pressure = (1.0 - float(self.cfg.pressure.decay)) * pressure + float(
            self.cfg.pressure.step_size
        ) * (dispatch_mass - target_mass)
        return next_pressure.clamp_min(0.0)

    def forward_full(
        self,
        Q: Tensor,
        mu: Tensor,
        xi: Tensor | None = None,
        *,
        certify: bool = True,
        training_mode: bool | None = None,
    ) -> DispatcherForward:
        """Return full policy, proposal, value, and diagnostics."""
        device = _module_device(self)
        Q = Q.to(device=device)
        mu = mu.to(device=device, dtype=Q.dtype)
        xi = xi.to(device=device, dtype=Q.dtype) if xi is not None else None
        assert Q.dim() == 2, "Q must have shape (B, N)."
        assert Q.shape[-1] == self.N, "Q last dimension must match model N."
        mu_b = expand_mu(Q, mu)
        assert mu_b.shape == Q.shape, "mu must have shape (N,) or (B, N)."
        assert (Q >= 0).all(), "Queue lengths must be non-negative."
        assert (mu_b > 0).all(), "Service rates must be positive."

        y = self.geometry.weighted_workload(Q, mu_b)
        p_cert, u_cert = self.geometry.policy(Q, mu_b)
        m_q, B_q = self.geometry.envelope(Q, mu_b)
        del training_mode
        pressure_bias = self._pressure_bias(Q.shape[0], Q.device, Q.dtype)
        p_proposal, correction, usage_raw, value = self.proposal(
            Q,
            mu_b,
            y,
            u_cert,
            pressure_bias,
            xi,
            pressure_scale=float(self.cfg.pressure.rho),
        )
        proposal_logits = u_cert + correction - float(self.cfg.pressure.rho) * pressure_bias
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
        pressure_next = self._update_pressure(pi, mu_b)
        pressure_mean = pressure_bias.mean(dim=-1)
        pressure_max = pressure_bias.max(dim=-1).values
        pressure_update_norm = (pressure_next - self.pressure_state.to(device=pi.device, dtype=pi.dtype)).norm()
        with torch.no_grad():
            self.pressure_state.copy_(pressure_next.detach().to(self.pressure_state.device))
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
            pressure_mean=pressure_mean,
            pressure_max=pressure_max,
            pressure_update_norm=pressure_update_norm.expand(Q.shape[0]),
            projection_nu=torch.zeros(Q.shape[0], device=Q.device, dtype=Q.dtype),
            projection_active=(usage_final < usage_raw).to(dtype=torch.bool) | fallback_active,
            projection_slack=B_q - A_proposal,
        )
        return DispatcherForward(
            pi=pi,
            diagnostics=diag,
            value=value,
            p_cert=p_cert,
            p_proposal=p_proposal,
            usage_raw=usage_raw,
            usage_final=usage_final,
            proposal_logits=proposal_logits,
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
        out = self.forward_full(Q, mu, xi, certify=certify, training_mode=training_mode)
        return out.pi, out.diagnostics
