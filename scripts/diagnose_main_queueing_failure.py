"""Reproduce and diagnose why CertiQ-Net-S fails to beat JSWQ on queueing.

This script prints:
1. The learned backbone/gate/residual parameters from a saved checkpoint.
2. The train-data state distribution implied by ``runner.max_queue``.
3. The audit/state-bank distribution used during validation.
4. Surrogate ``A_pi`` versus true rollout cost for model, backbone, and JSWQ.
5. The gap between the learned policy and JSWQ on identical sampled states.

Usage:
    python scripts/diagnose_main_queueing_failure.py \
        --run-dir outputs/main-queueing-certiqnets-n10-fixed-lam8-0/..._seed42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from certiqnet.diagnostics.state_bank import generate_state_bank
from certiqnet.experiments.baseline_runner import RolloutConfig, evaluate_policy
from certiqnet.experiments.factory import build_model, build_mu
from certiqnet.math.certificate import arrival_envelope_A
from certiqnet.models.baselines import AnalyticBackbonePolicy, JoinShortestWeightedQueue


def _sample_uniform_states(n: int, N: int, max_queue: int, *, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, max_queue, (n, N), generator=gen).float()


def _describe_policy(name: str, model: torch.nn.Module, Q: torch.Tensor, mu: torch.Tensor) -> None:
    with torch.no_grad():
        pi, diag = model(Q, mu, training_mode=False)
    print(f"[{name}]")
    print(f"  A_pi mean               : {float(diag.A_pi.mean().item()):.6f}")
    print(f"  policy entropy mean     : {float(diag.policy_entropy.mean().item()):.6f}")
    print(f"  gate mean / open-rate   : {float(diag.eta_final.mean().item()):.6f} / {float((diag.eta_final > 0.1).float().mean().item()):.6f}")
    print(f"  fallback rate           : {float(diag.fallback_active.float().mean().item()):.6f}")
    print(f"  residual norm mean/max  : {float(diag.residual_norm.mean().item()):.6f} / {float(diag.residual_norm.max().item()):.6f}")
    print(f"  drift slack mean/min    : {float(diag.drift_slack.mean().item()):.6f} / {float(diag.drift_slack.min().item()):.6f}")
    return pi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to a single experiment run directory containing configs/resolved_config.yaml.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-n", type=int, default=1024)
    parser.add_argument("--rollout-steps", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=32)
    args = parser.parse_args()

    run_dir = args.run_dir
    cfg = OmegaConf.load(run_dir / "configs" / "resolved_config.yaml")
    mu, lam = build_mu(cfg)
    model = build_model(cfg, N=int(cfg.env.N), d_xi=0)
    ckpt = run_dir / "artifacts" / "final_model_state.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    N = int(cfg.env.N)
    max_queue = int(cfg.runner.max_queue)
    mu_b = mu.unsqueeze(0)

    print("=== Checkpoint parameters ===")
    bp = model.backbone_params
    print(f"alpha={float(bp.alpha.item()):.6f}")
    print(f"beta ={float(bp.beta.item()):.6f}")
    print(f"gamma={float(bp.gamma.item()):.6f}")
    print(f"c    ={float(bp.c.item()):.6e}")
    print(f"gate weight norm={float(model.gate.linear.weight.norm().item()):.6f}")
    print(f"gate bias       ={float(model.gate.linear.bias.item()):.6f}")
    print(f"residual weight norm={float(model.residual_head.proj.weight.norm().item()):.6f}")
    print(f"residual bias       ={float(model.residual_head.proj.bias.item()):.6f}")
    print()

    print("=== Distribution mismatch ===")
    train_mean_total = N * (max_queue - 1) / 2.0
    train_mean_per_resource = (max_queue - 1) / 2.0
    print(f"runner.max_queue                : {max_queue}")
    print(f"implied train mean / resource    : {train_mean_per_resource:.3f}")
    print(f"implied train mean total backlog  : {train_mean_total:.3f}")
    Q_train = _sample_uniform_states(args.sample_n, N, max_queue, seed=args.seed)
    print(f"sampled train-like mean total     : {float(Q_train.sum(dim=-1).mean().item()):.3f}")
    bank = generate_state_bank(N=N, mu=mu, beta=float(cfg.model.beta), R_cert=float(cfg.model.R_cert), n_random=128, n_grid=0, n_boundary=128)
    print(f"audit/state-bank mean total       : {float(bank.sum(dim=-1).mean().item()):.3f}")
    print(f"audit/state-bank p95 total        : {float(torch.quantile(bank.sum(dim=-1).float(), 0.95).item()):.3f}")
    print()

    print("=== Policy comparison on identical sampled states ===")
    Q_eval = _sample_uniform_states(args.sample_n, N, max_queue, seed=args.seed + 1)
    mu_eval = mu.unsqueeze(0).expand(Q_eval.shape[0], -1)
    jswq = JoinShortestWeightedQueue(N=N, beta=float(cfg.model.beta))
    with torch.no_grad():
        pi_model, diag_model = model(Q_eval, mu_eval, training_mode=False)
        pi_backbone, _ = model.backbone(Q_eval, mu_eval)
        pi_jswq, _ = jswq(Q_eval, mu_eval, training_mode=False)

    print(f"model vs backbone mean |dpi|     : {float((pi_model - pi_backbone).abs().mean().item()):.6f}")
    print(f"model vs JSWQ mean |dpi|         : {float((pi_model - pi_jswq).abs().mean().item()):.6f}")
    print(f"model A_pi mean                  : {float(arrival_envelope_A(pi_model, Q_eval, mu_eval, float(cfg.model.beta)).mean().item()):.6f}")
    print(f"backbone A_pi mean               : {float(arrival_envelope_A(pi_backbone, Q_eval, mu_eval, float(cfg.model.beta)).mean().item()):.6f}")
    print(f"JSWQ A_pi mean                   : {float(arrival_envelope_A(pi_jswq, Q_eval, mu_eval, float(cfg.model.beta)).mean().item()):.6f}")
    print(f"model entropy mean               : {float(diag_model.policy_entropy.mean().item()):.6f}")
    print(f"model gate mean                  : {float(diag_model.eta_final.mean().item()):.6f}")
    print(f"model fallback rate              : {float(diag_model.fallback_active.float().mean().item()):.6f}")
    print()

    print("=== True rollout cost ===")
    rollout = RolloutConfig(
        steps=args.rollout_steps,
        batch_size=args.rollout_batch_size,
        max_backlog=float(cfg.runner.max_backlog),
        show_progress=False,
    )
    backbone_policy = AnalyticBackbonePolicy(N=N, beta=float(cfg.model.beta))
    for name, pol in [("model", model), ("backbone", backbone_policy), ("jswq", jswq)]:
        metrics = evaluate_policy(
            name=name,
            model=pol,
            env_name=str(cfg.env.mu_mode),
            seed=args.seed,
            N=N,
            lam=lam,
            mu=mu,
            rollout=rollout,
        )
        print(
            f"{name:8s} avg_cost={metrics.performance.generic.avg_cost:.6f} "
            f"avg_queue={metrics.performance.queueing['avg_queue_length']:.6f} "
            f"p95={metrics.performance.queueing['p95_backlog']:.3f} "
            f"max={metrics.performance.queueing['max_backlog']:.3f}"
        )


if __name__ == "__main__":
    main()
