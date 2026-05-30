"""CertiQ-Net-P: exact drift-envelope projection certified model."""

import math

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
from certiqnet.models.certiqnet_s import _expand_mu
from certiqnet.models.encoder import ResourceEncoder
from certiqnet.models.gate import RawGate, drift_envelope_projection
from certiqnet.models.residual import BoundedResidualHead
from certiqnet.utils.config_schemas import CertiQNetPConfig


class CertiQNetP(nn.Module):
    """Drift-envelope projection variant enforcing ``A_pi(Q) <= m(Q) + C_B``."""

    def __init__(self, cfg: CertiQNetPConfig, N: int, d_xi: int = 0) -> None:
        super().__init__()
        assert N >= 1, "N must be positive."
        assert cfg.beta > 0, "beta must be positive."
        assert cfg.C_B < float("inf"), (
            "CertiQ-Net-P requires a finite C_B from the discharged base envelope proof."
        )
        assert cfg.C_B >= 0, "C_B must be non-negative."
        self.cfg = cfg
        self.N = N
        self.beta = float(cfg.beta)
        self.C_B = float(cfg.C_B)

        # With default gamma=0 and c≈0, alpha >= 2 log(N)/C_B gives a strict margin
        # for tests and runtime audits while remaining traceable to the exact C_B proof.
        alpha_init = cfg.backbone.alpha_init
        if self.C_B > 0:
            alpha_init = max(alpha_init, 2.0 * math.log(max(N, 2)) / self.C_B)

        self.backbone_params = ConstrainedBackboneParams(
            alpha_min=cfg.backbone.alpha_min,
            beta_min=cfg.backbone.beta_min,
            gamma_max=cfg.backbone.gamma_max,
            alpha_init=alpha_init,
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
        self.residual_head = BoundedResidualHead(cfg.encoder.d_local, cfg.residual.R_max)
        self.gate = RawGate(cfg.encoder.d_global, cfg.gate.eta_max)

    def forward(
        self,
        Q: Tensor,
        mu: Tensor,
        xi: Tensor | None = None,
        training_mode: bool = False,
    ) -> tuple[Tensor, CertificateDiagnostics]:
        """Return final projected policy and diagnostics."""
        assert Q.dim() == 2, "Q must have shape (B, N)."
        assert Q.shape[-1] == self.N, "Q last dimension must match model N."
        mu_b = _expand_mu(Q, mu)
        assert mu_b.shape == Q.shape, "mu must have shape (B, N) or (N,)."

        p_base, u_base = self.backbone(Q, mu_b)
        z0 = self.encoder.encode_local(Q, mu_b, self.beta, xi)
        g = self.encoder.pool_global(z0)
        z1 = self.encoder.encode_residual(z0, g)
        r = self.residual_head(z1)
        p_nn = torch.softmax(u_base + r, dim=-1)
        eta_raw = self.gate(g)
        eta, eta_safe = drift_envelope_projection(
            eta_raw, Q, mu_b, self.beta, self.C_B, p_base, p_nn
        )
        fallback_active = torch.zeros(Q.shape[0], dtype=torch.bool, device=Q.device)
        pi = (1.0 - eta.unsqueeze(-1)) * p_base + eta.unsqueeze(-1) * p_nn
        pi = project_probability_vector(pi)
        validate_probability_vector(pi)

        A_base = arrival_envelope_A(p_base, Q, mu_b, self.beta)
        A_nn = arrival_envelope_A(p_nn, Q, mu_b, self.beta)
        A_pi = arrival_envelope_A(pi, Q, mu_b, self.beta)
        m_q = min_coord(Q, mu_b, self.beta)
        B_q = m_q + self.C_B
        violation = (A_pi - B_q).clamp(min=0.0).max().item()
        if not training_mode:
            assert violation < 1e-5, (
                f"Projection violated certificate: max violation = {violation:.2e}."
            )
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
            residual_norm=r.abs().max(dim=-1).values,
            policy_entropy=policy_entropy(pi),
            selected_resource=pi.argmax(dim=-1),
        )
        return pi, diag
