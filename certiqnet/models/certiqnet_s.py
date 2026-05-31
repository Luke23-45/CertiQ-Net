"""CertiQ-Net-S: hard tail fallback certified model."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from certiqnet.math.backbone_params import ConstrainedBackboneParams
from certiqnet.math.certificate import (
    CertificateDiagnostics,
    arrival_envelope_A,
    policy_entropy,
    project_probability_vector,
    validate_probability_vector,
)
from certiqnet.math.lyapunov import min_coord
from certiqnet.models.backbone import AnalyticBackbone
from certiqnet.models.encoder import ResourceEncoder
from certiqnet.models.gate import RawGate, hard_tail_fallback, smooth_tail_gate
from certiqnet.utils.config_schemas import CertiQNetSConfig


def _expand_mu(Q: Tensor, mu: Tensor) -> Tensor:
    if mu.dim() == 1:
        return mu.unsqueeze(0).expand(Q.shape[0], -1)
    return mu


@dataclass(frozen=True)
class PolicyForward:
    """Expanded forward pass with training-time auxiliaries."""

    pi: Tensor
    diag: CertificateDiagnostics
    value: Tensor
    p_base: Tensor
    p_actor: Tensor
    eta_raw: Tensor
    eta_final: Tensor
    fallback_active: Tensor
    actor_logits: Tensor


class CertiQNetS(nn.Module):
    """Hard-tail fallback variant certified by exact equality to backbone in the tail."""

    def __init__(self, cfg: CertiQNetSConfig, N: int, d_xi: int = 0) -> None:
        super().__init__()
        assert N >= 1, "N must be positive."
        assert cfg.R_cert >= 0, "R_cert must be non-negative."
        assert cfg.beta > 0, "beta must be positive."
        self.cfg = cfg
        self.N = N
        self.beta = float(cfg.beta)
        self.R_cert = float(cfg.R_cert)
        self.tau = float(cfg.tau_smooth)
        self.C_B = float(cfg.C_B)
        self.backbone_params = ConstrainedBackboneParams(
            alpha_min=cfg.backbone.alpha_min,
            beta_min=cfg.backbone.beta_min,
            beta_max=cfg.backbone.beta_max,
            gamma_max=cfg.backbone.gamma_max,
            alpha_init=cfg.backbone.alpha_init,
            beta_init=cfg.backbone.beta_init,
            gamma_init=cfg.backbone.gamma_init,
            c_init=cfg.backbone.c_init,
        )
        self.backbone = AnalyticBackbone(self.backbone_params)
        self.encoder = ResourceEncoder(
            d_xi=d_xi,
            d_local=cfg.encoder.d_local,
            d_global=cfg.encoder.d_global,
            hidden_dim=cfg.encoder.hidden_dim,
            n_layers_local=cfg.encoder.n_layers_local,
            n_layers_res=cfg.encoder.n_layers_res,
            pooling=cfg.encoder.pooling,
        )
        self.actor_head = nn.Linear(cfg.encoder.d_local, 1)
        self.value_head = nn.Linear(cfg.encoder.d_global, 1)
        self.gate = RawGate(cfg.encoder.d_global, cfg.gate.eta_max)

    def _forward_impl(
        self,
        Q: Tensor,
        mu: Tensor,
        xi: Tensor | None = None,
        training_mode: bool = False,
    ) -> PolicyForward:
        """Return policy, diagnostics, and critic output."""
        assert Q.dim() == 2, "Q must have shape (B, N)."
        assert Q.shape[-1] == self.N, "Q last dimension must match model N."
        mu_b = _expand_mu(Q, mu)
        assert mu_b.shape == Q.shape, "mu must have shape (B, N) or (N,)."

        p_base, _u_base = self.backbone(Q, mu_b)
        z0 = self.encoder.encode_local(Q, mu_b, self.beta, xi)
        g = self.encoder.pool_global(z0)
        z1 = self.encoder.encode_residual(z0, g)
        actor_logits = self.actor_head(z1).squeeze(-1)
        p_actor = torch.softmax(actor_logits, dim=-1)
        eta_raw = self.gate(g)

        if training_mode:
            eta = smooth_tail_gate(eta_raw, Q, mu_b, self.beta, self.R_cert, self.tau)
            fallback_active = torch.zeros(Q.shape[0], dtype=torch.bool, device=Q.device)
            eta_safe = torch.full_like(eta, float("nan"))
        else:
            eta, fallback_active = hard_tail_fallback(eta_raw, Q, mu_b, self.beta, self.R_cert)
            eta_safe = torch.full_like(eta, float("nan"))

        pi = (1.0 - eta.unsqueeze(-1)) * p_base + eta.unsqueeze(-1) * p_actor
        pi = project_probability_vector(pi)
        validate_probability_vector(pi)

        A_base = arrival_envelope_A(p_base, Q, mu_b, self.beta)
        A_nn = arrival_envelope_A(p_actor, Q, mu_b, self.beta)
        A_pi = arrival_envelope_A(pi, Q, mu_b, self.beta)
        m_q = min_coord(Q, mu_b, self.beta)
        B_q = m_q + self.C_B
        diag = CertificateDiagnostics(
            A_base=A_base,
            A_nn=A_nn,
            A_pi=A_pi,
            m_Q=m_q,
            B_Q=B_q,
            drift_slack=B_q - A_pi,
            eta_raw=eta_raw,
            eta_final=eta,
            eta_safe=eta_safe,
            fallback_active=fallback_active,
            residual_norm=actor_logits.abs().max(dim=-1).values,
            policy_entropy=policy_entropy(pi),
            selected_resource=pi.argmax(dim=-1),
        )
        value = self.value_head(g).squeeze(-1)
        return PolicyForward(
            pi=pi,
            diag=diag,
            value=value,
            p_base=p_base,
            p_actor=p_actor,
            eta_raw=eta_raw,
            eta_final=eta,
            fallback_active=fallback_active,
            actor_logits=actor_logits,
        )

    def forward(
        self,
        Q: Tensor,
        mu: Tensor,
        xi: Tensor | None = None,
        training_mode: bool = False,
    ) -> tuple[Tensor, CertificateDiagnostics]:
        """Return final policy and diagnostics for legacy call sites."""
        out = self._forward_impl(Q, mu, xi=xi, training_mode=training_mode)
        return out.pi, out.diag

    def forward_with_aux(
        self,
        Q: Tensor,
        mu: Tensor,
        xi: Tensor | None = None,
        training_mode: bool = False,
    ) -> PolicyForward:
        """Return policy, diagnostics, and critic features for actor-critic training."""
        return self._forward_impl(Q, mu, xi=xi, training_mode=training_mode)
