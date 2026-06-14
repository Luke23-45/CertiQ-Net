"""Diagnose gradient flow through the z3 CertiQ Index model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from certiqnet.experiments.factory import build_model, build_mu
from certiqnet.training.loss import CertiQNetLoss


def main() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "_target_": "certiqnet.dispatcher.index_model.CertiQIndexModel",
                "hidden_dim": 128,
                "tau": 1.0,
                "C": 20.0,
                "beta": 1.0,
            },
            "env": {
                "N": 10,
                "mu_mode": "fixed",
                "mu_fixed": [1.0, 1.5, 0.8, 2.0, 0.6, 1.2, 1.8, 0.9, 1.1, 1.4],
                "lam": 8.0,
                "rho_target": None,
            },
            "loss": {
                "omega_bc": 1.0,
                "omega_usage": 0.1,
                "omega_certificate": 5.0,
                "omega_correction": 0.01,
                "omega_ent": 0.0,
                "rollout_weight": 1.0,
                "policy_kl_weight": 0.05,
                "value_weight": 1.0,
                "entropy_weight": 0.001,
            },
        }
    )
    N = 10
    mu, _ = build_mu(cfg)
    model = build_model(cfg, N=N, d_xi=0)
    loss_fn = CertiQNetLoss(cfg.loss)
    Q = torch.randint(0, 100, (64, N)).float()
    mu_b = mu.unsqueeze(0).expand(Q.shape[0], -1)

    model.train()
    out = model.forward_full(Q, mu_b, certify=True)
    dist = Categorical(probs=out.pi)
    actions = dist.sample()
    log_prob = dist.log_prob(actions)
    losses = {
        "actor": loss_fn.actor_loss(log_prob, torch.ones(Q.shape[0])),
        "critic": loss_fn.critic_loss(out.value, torch.ones(Q.shape[0])),
        "usage": loss_fn.usage_penalty(out.diagnostics.usage_final),
        "certificate": loss_fn.certificate_penalty(out.diagnostics),
        "correction": loss_fn.correction_size_penalty(out.diagnostics),
        "entropy": loss_fn.entropy_term(out.pi),
    }
    total = sum(losses.values())
    total.backward()

    print("=== CertiQ Index Gradient Diagnostic ===")
    print(f"policy entropy mean       : {out.diagnostics.policy_entropy.mean().item():.6f}")
    print(f"usage mean/open-rate      : {out.diagnostics.usage_final.mean().item():.6f} / {(out.diagnostics.usage_final > 0.1).float().mean().item():.6f}")
    print(f"certificate slack mean/min: {out.diagnostics.certificate_slack.mean().item():.6f} / {out.diagnostics.certificate_slack.min().item():.6f}")
    print(f"correction magnitude mean : {out.diagnostics.correction_magnitude.mean().item():.6f}")
    for name, value in losses.items():
        print(f"{name:12s}: {value.item():.6f}")
    grads = {
        name: param.grad.norm().item()
        for name, param in model.named_parameters()
        if param.grad is not None and param.grad.norm().item() > 0
    }
    print(f"nonzero gradient tensors  : {len(grads)}")
    for name, norm in sorted(grads.items(), key=lambda item: -item[1])[:20]:
        print(f"  {norm:.6e} {name}")


if __name__ == "__main__":
    main()

