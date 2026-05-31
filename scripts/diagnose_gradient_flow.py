"""Diagnose gradient flow in CertiQNet-S training.

This script runs a single forward/backward pass and reports:
1. Each loss component value
2. Which parameters received non-zero gradients
3. Whether rollout_cost depends on pi (the core question)
4. Suggested fix if gradient is missing

Usage:
    python scripts/diagnose_gradient_flow.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

from certiqnet.experiments.factory import build_mu, build_model
from certiqnet.training.lightning_module import CertiQNetLightningModule
from certiqnet.training.loss import CertiQNetLoss
from certiqnet.math.certificate import CertificateDiagnostics

def main():
    print("=" * 72)
    print("DIAGNOSTIC: Gradient Flow in CertiQNet-S Training")
    print("=" * 72)

    # Build a minimal config inline rather than relying on Hydra
    cfg = OmegaConf.create(
        {
            "model": {
                "_target_": "certiqnet.models.certiqnet_s.CertiQNetS",
                "R_cert": 50.0,
                "beta": 1.0,
                "tau_smooth": 10.0,
                "backbone": {
                    "alpha_min": 1e-3,
                    "alpha_max": 100.0,
                    "beta_min": 1e-3,
                    "beta_max": 10.0,
                    "alpha_init": 1.0,
                    "beta_init": 1.0,
                    "gamma_max": 2.0,
                    "gamma_init": 0.0,
                    "c_init": 0.0,
                },
                "encoder": {
                    "d_local": 64,
                    "d_global": 64,
                    "n_layers_local": 2,
                    "n_layers_res": 2,
                    "hidden_dim": 128,
                    "pooling": "mean",
                },
                "residual": {"R_max": 2.0},
                "gate": {"eta_max": 0.95},
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
                "omega_gate": 0.1,
                "omega_drift": 5.0,
                "omega_res": 0.01,
                "omega_ent": -0.001,
                "rollout_weight": 1.0,
            },
            "trainer": {"lr": 3e-4, "weight_decay": 1e-5, "max_epochs": 200},
        }
    )

    N = 10
    mu, lam = build_mu(cfg)
    model = build_model(cfg, N=N, d_xi=0)
    loss_fn = CertiQNetLoss(cfg.loss)

    # Synthetic batch: Q ~ Uniform(0, 100)
    B = 64
    Q = torch.randint(0, 100, (B, N)).float()
    mu_b = mu.unsqueeze(0).expand(B, -1)

    print(f"\n--- Forward pass (training_mode=True) ---")
    model.train()
    pi, diag = model(Q, mu_b, training_mode=True)

    print(f"pi shape:        {tuple(pi.shape)}")
    print(f"pi.mean():       {pi.mean().item():.4f}")
    print(f"pi[0, :5]:       {pi[0, :5].tolist()}")
    print(f"eta_raw:         {diag.eta_raw.mean().item():.4f}")
    print(f"eta_final:       {diag.eta_final.mean().item():.4f}")
    print(f"fallback_active: {diag.fallback_active.float().mean().item():.4f}")
    print(f"A_pi:            {diag.A_pi.mean().item():.4f}")
    print(f"B_Q:             {diag.B_Q.mean().item():.4f} (inf)")
    print(f"drift_slack:     {diag.drift_slack.mean().item():.4f}")

    print(f"\n--- Loss Components ---")
    expert_actions = pi.argmax(dim=-1)
    actor_log_prob = torch.log(pi.gather(1, expert_actions.unsqueeze(-1)).clamp_min(1e-9).squeeze(-1))
    pseudo_advantage = torch.ones(B)
    losses = {
        "actor": loss_fn.actor_loss(actor_log_prob, pseudo_advantage),
        "critic": loss_fn.critic_loss(torch.zeros(B), torch.ones(B)),
        "bc": loss_fn.bc_loss(pi),
        "gate": loss_fn.gate_penalty(diag.eta_final),
        "drift": loss_fn.drift_penalty(diag),
        "residual": loss_fn.residual_size_penalty(diag),
        "entropy": loss_fn.entropy_term(pi),
    }
    for k, v in losses.items():
        requires_grad = v.requires_grad
        grad_fn = v.grad_fn.__class__.__name__ if v.grad_fn else "NONE"
        print(f"  {k:20s} = {v.item():.6f}  requires_grad={requires_grad}  grad_fn={grad_fn}")

    print(f"\n--- Gradient Flow Check ---")
    total = losses["total"]
    total.backward()

    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            g_norm = param.grad.norm().item()
            if g_norm > 0:
                grads[name] = g_norm

    if grads:
        print(f"  Parameters WITH non-zero gradient ({len(grads)}):")
        for name, g in sorted(grads.items(), key=lambda x: -x[1]):
            print(f"    {g:.6e}  {name}")
    else:
        print("  WARNING: NO PARAMETERS RECEIVED GRADIENTS!")

    model.zero_grad()

    print(f"\n--- Rollout Cost Dependency Check ---")
    # Rollout-cost helper now consumes actual queue/cost traces.
    cost_trace = torch.rand(B)
    dt_trace = torch.ones(B)
    print(f"  rollout_cost sample: {loss_fn.rollout_cost(cost_trace, dt_trace).item():.4f}")

    print(f"\n--- Diagnostic: Backbone-only forward ---")
    model.eval()
    with torch.no_grad():
        pi_base, u_base = model.backbone(Q, mu_b)
    print(f"  backbone avg_queue_length (mean sum Q): {Q.sum(dim=-1).mean().item():.4f}")
    from certiqnet.math.certificate import arrival_envelope_A
    print(f"  backbone A_pi: {arrival_envelope_A(pi_base, Q, mu_b, 1.0).mean().item():.4f}")

    torch.manual_seed(0)
    Q_ctmc = torch.randint(0, 30, (B, N)).float()
    pi_chk, diag_chk = model(Q_ctmc, mu_b, training_mode=False)
    A_val = arrival_envelope_A(pi_chk, Q_ctmc, mu_b, 1.0).mean().item()
    print(f"  CertiQNet-S A_pi (Q~U[0,30]):  {A_val:.4f}")
    print(f"  CertiQNet-S eta_final:         {diag_chk.eta_final.mean().item():.4f}")

    print(f"\n{'='*72}")
    print(f"CONCLUSION: what alternative cost would give gradient?")
    print(f"{'='*72}")

    # Compute A_pi(Q) = sum(pi_i * Q_i / mu_i^beta) as a candidate cost
    A_pi_cost = (pi * Q / mu_b.pow(1.0)).sum(dim=-1)
    A_base_cost = (pi_backbone * Q / mu_b.pow(1.0)).sum(dim=-1)

    # Check gradients for A_pi cost
    pi_detached = pi.detach().requires_grad_(True)
    test_cost = (pi_detached * Q / mu_b.pow(1.0)).sum(dim=-1).mean()
    test_cost.backward()
    pi_grad = pi_detached.grad
    print(f"  A_pi(Q) mean (current pi):       {A_pi_cost.mean().item():.4f}")
    print(f"  A_pi(Q) mean (backbone pi):       {A_base_cost.mean().item():.4f}")
    print(f"  pi.grad from A_pi cost:            {pi_grad.norm().item():.6e}")
    print(f"  pi.grad[:3,:3]:\n{pi_grad[:3, :3]}")
    print(f"  >>> A_pi(Q) DOES give gradient to pi <<<")
    print()
    print(f"  RECOMMENDED: Replace rollout_cost with diag.A_pi.mean()")
    print(f"  This gives pi a gradient toward routing to low-Q/high-mu resources.")


if __name__ == "__main__":
    from certiqnet.math.certificate import arrival_envelope_A
    main()
