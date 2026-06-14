"""Diagnose gradient flow through the CertiQ Index model.

Uses ``torch.autograd.gradcheck`` to verify the differentiable KL
projection layer, then checks end-to-end gradient flow through the
full model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from certiqnet.dispatcher.certificate import DifferentiableKLProjection
from certiqnet.experiments.factory import build_model, build_mu
from certiqnet.training.loss import CertiQNetLoss


def _check_projection_gradients():
    """Verify the implicit backward pass via ``torch.autograd.gradcheck``."""
    print("=" * 72)
    print("  KL Projection Layer — Gradient Check")
    print("=" * 72)

    torch.manual_seed(42)
    B, N = 4, 5

    # ---- Scenario 1: easy budget (gradient flows through softmax only) ----
    print("  Scenario 1: loose budget (already feasible)")
    logits1 = torch.randn(B, N, requires_grad=True, dtype=torch.float64)
    cost1 = torch.rand(B, N, dtype=torch.float64) * 0.9 + 0.1
    cost1.requires_grad_(True)
    budget1 = cost1.detach().max(dim=-1).values + 1.0
    budget1.requires_grad_(True)

    def proj_fn(ℓ, c, b):
        p, _, _ = DifferentiableKLProjection.apply(ℓ, c, b)
        return p

    grad1 = torch.autograd.gradcheck(
        lambda ℓ, c, b: proj_fn(ℓ, c, b),
        (logits1, cost1, budget1),
        eps=1e-6, atol=1e-4, rtol=1e-4, raise_exception=False,
    )
    with torch.no_grad():
        nu1 = DifferentiableKLProjection.apply(logits1, cost1, budget1)[1]
    print(f"    active constraint rate          : {(nu1 > 1e-12).float().mean().item():.2%}")
    print(f"    result                           : {'PASS' if grad1 else 'FAIL'}")

    # ---- Scenario 2: tight budget (active constraint, gradient through ν) ----
    print("  Scenario 2: tight budget (active constraint)")
    logits2 = torch.randn(B, N, requires_grad=True, dtype=torch.float64)
    cost2 = torch.rand(B, N, dtype=torch.float64) * 0.9 + 0.1
    cost2.requires_grad_(True)
    budget2 = cost2.detach().min(dim=-1).values * 1.1 + 0.01
    budget2.requires_grad_(True)

    grad2 = torch.autograd.gradcheck(
        lambda ℓ, c, b: proj_fn(ℓ, c, b),
        (logits2, cost2, budget2),
        eps=1e-6, atol=1e-4, rtol=1e-4, raise_exception=False,
    )
    with torch.no_grad():
        nu2 = DifferentiableKLProjection.apply(logits2, cost2, budget2)[1]
    print(f"    active constraint rate          : {(nu2 > 1e-12).float().mean().item():.2%}")
    print(f"    result                           : {'PASS' if grad2 else 'FAIL'}")

    # ---- Scenario 3: very tight budget (all active, near infeasible) ----
    print("  Scenario 3: very tight budget (nu >> 0)")
    logits3 = torch.randn(B, N, requires_grad=True, dtype=torch.float64)
    cost3 = torch.rand(B, N, dtype=torch.float64) * 0.9 + 0.1
    cost3.requires_grad_(True)
    budget3 = cost3.detach().min(dim=-1).values * 1.01
    budget3.requires_grad_(True)

    grad3 = torch.autograd.gradcheck(
        lambda ℓ, c, b: proj_fn(ℓ, c, b),
        (logits3, cost3, budget3),
        eps=1e-6, atol=1e-4, rtol=1e-4, raise_exception=False,
    )
    with torch.no_grad():
        nu3 = DifferentiableKLProjection.apply(logits3, cost3, budget3)[1]
        status3 = DifferentiableKLProjection.apply(logits3, cost3, budget3)[2]
    print(f"    active constraint rate          : {(nu3 > 1e-12).float().mean().item():.2%}")
    print(f"    solver status distr             : {status3.bincount().tolist() if status3.numel() > 0 else [0]}")
    print(f"    result                           : {'PASS' if grad3 else 'FAIL'}")

    grad_ok = grad1 and grad2 and grad3
    print()

    return grad_ok


def _check_full_model_gradients():
    """Verify end-to-end gradient flow (all params receive nonzero grad)."""
    print("=" * 72)
    print("  Full Model — End-to-End Gradient Flow")
    print("=" * 72)

    cfg = OmegaConf.create(
        {
            "model": {
                "_target_": "certiqnet.dispatcher.index_model.CertiQIndexModel",
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
    Q = torch.randint(0, 100, (16, N)).float()
    mu_b = mu.unsqueeze(0).expand(Q.shape[0], -1)

    model.train()
    out = model.forward_full(Q, mu_b, certify=True)
    dist = Categorical(probs=out.pi)
    actions = dist.sample()
    log_prob = dist.log_prob(actions)
    advantage = torch.randn(Q.shape[0])
    losses = {
        "actor": loss_fn.actor_loss(log_prob, advantage),
        "critic": loss_fn.critic_loss(out.value, torch.ones(Q.shape[0])),
        "usage": loss_fn.usage_penalty(out.diagnostics.usage_final),
        "certificate": loss_fn.certificate_penalty(out.diagnostics),
        "correction": loss_fn.correction_size_penalty(out.diagnostics),
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
    print(f"  certificate slack mean          : {diagnostics.certificate_slack.mean().item():.6f}")
    print(f"  correction magnitude            : {diagnostics.correction_magnitude.mean().item():.6f}")
    print(f"  projection active rate          : {diagnostics.projection_active.float().mean().item():.6f}")
    print(f"  fallback rate                   : {diagnostics.fallback_active.float().mean().item():.6f}")
    print()

    return len(grads) > 0 and len(zero_grads) == 0


def main() -> None:
    proj_ok = _check_projection_gradients()
    model_ok = _check_full_model_gradients()

    print("=" * 72)
    print(f"  Projection layer gradient check : {'PASS' if proj_ok else 'FAIL'}")
    print(f"  Full model gradient flow check  : {'PASS' if model_ok else 'FAIL'}")
    print("=" * 72)

    if not proj_ok:
        print("WARNING: Projection layer gradients may be incorrect.")
    if not model_ok:
        print("WARNING: Not all model parameters received gradients.")


if __name__ == "__main__":
    main()
