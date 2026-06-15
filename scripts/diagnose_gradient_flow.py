"""Diagnose gradient flow through the CertiQ Index model.

Checks end-to-end gradient flow through the full model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from certiqnet.experiments.factory import build_model, build_mu
from certiqnet.train.common.loss import CertiQNetLoss


def _check_full_model_gradients():
    """Verify end-to-end gradient flow (all params receive nonzero grad)."""
    print("=" * 72)
    print("  Full Model — End-to-End Gradient Flow")
    print("=" * 72)

    cfg = OmegaConf.create(
        {
            "model": {
                "_target_": "certiqnet.dispatcher.certiq.index_model.CertiQIndexModel",
                "hidden_dim": 64,
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
    loss_fn = CertiQNetLoss(
        omega_bc=float(cfg.loss.omega_bc),
        omega_action=float(cfg.loss.get("omega_action", 1.5)),
        omega_margin=float(cfg.loss.get("omega_margin", 0.1)),
        omega_usage=float(cfg.loss.omega_usage),
        rollout_weight=float(cfg.loss.rollout_weight),
        policy_kl_weight=float(cfg.loss.policy_kl_weight),
        value_weight=float(cfg.loss.value_weight),
        entropy_weight=float(cfg.loss.entropy_weight),
    )
    Q = torch.randint(0, 100, (16, N)).float()
    mu_b = mu.unsqueeze(0).expand(Q.shape[0], -1)

    model.train()
    out = model.forward_full(Q, mu_b, training_mode=True)
    dist = Categorical(probs=out.pi)
    actions = dist.sample()
    log_prob = dist.log_prob(actions)
    advantage = torch.randn(Q.shape[0])
    losses = {
        "actor": loss_fn.actor_loss(log_prob, advantage),
        "critic": loss_fn.critic_loss(out.value, torch.ones(Q.shape[0])),
        "usage": loss_fn.usage_penalty(out.diagnostics.usage_final),
        "constraint": loss_fn.constraint_loss(out.diagnostics.constraint_violation, out.diagnostics.dual_lambda),
    }
    total = sum(losses.values())
    total.backward()

    grads = []
    zero_grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            gnorm = param.grad.norm().item()
            if gnorm > 0:
                grads.append((gnorm, name))
            else:
                zero_grads.append(name)
        else:
            zero_grads.append(name)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  total parameters                 : {total_params}")
    print(f"  parameters with nonzero grad     : {len(grads)}")
    print(f"  parameters with zero grad        : {len(zero_grads)}")

    if zero_grads:
        print("  ZERO GRADIENT PARAMETERS:")
        for name in sorted(zero_grads):
            print(f"    {name}")

    grads_sorted = sorted(grads, key=lambda x: -x[0])
    print("  top-10 gradient norms:")
    for gnorm, name in grads_sorted[:10]:
        print(f"    {gnorm:.6e}  {name}")

    diagnostics = out.diagnostics
    print(f"  policy entropy                  : {diagnostics.policy_entropy.mean().item():.6f}")
    print(f"  constraint violation mean       : {diagnostics.constraint_violation.mean().item():.6f}")
    print(f"  dual lambda                     : {diagnostics.dual_lambda.mean().item():.6f}")
    print()

    return len(grads) > 0 and len(zero_grads) == 0


def main() -> None:
    model_ok = _check_full_model_gradients()

    print("=" * 72)
    print(f"  Full model gradient flow check  : {'PASS' if model_ok else 'FAIL'}")
    print("=" * 72)

    if not model_ok:
        print("WARNING: Not all model parameters received gradients.")


if __name__ == "__main__":
    main()
