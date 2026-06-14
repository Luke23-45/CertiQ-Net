"""Diagnose a z3 CertiQ Index queueing run."""

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
from certiqnet.models.baselines import AnalyticBackbonePolicy, JoinShortestWeightedQueue


def _sample_uniform_states(n: int, N: int, max_queue: int, *, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, max_queue, (n, N), generator=gen).float()


def _describe_policy(name: str, model: torch.nn.Module, Q: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        pi, diag = model(Q, mu, training_mode=False)
    print(f"[{name}]")
    print(f"  A_final mean              : {diag.A_final.mean().item():.6f}")
    print(f"  policy entropy mean       : {diag.policy_entropy.mean().item():.6f}")
    print(f"  usage mean / open-rate    : {diag.usage_final.mean().item():.6f} / {(diag.usage_final > 0.1).float().mean().item():.6f}")
    print(f"  fallback rate             : {diag.fallback_active.float().mean().item():.6f}")
    print(f"  correction magnitude mean : {diag.correction_magnitude.mean().item():.6f}")
    print(f"  certificate slack mean/min: {diag.certificate_slack.mean().item():.6f} / {diag.certificate_slack.min().item():.6f}")
    return pi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-n", type=int, default=1024)
    parser.add_argument("--rollout-steps", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=32)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.run_dir / "configs" / "resolved_config.yaml")
    mu, lam = build_mu(cfg)
    model = build_model(cfg, N=int(cfg.env.N), d_xi=0)
    ckpt = args.run_dir / "artifacts" / "final_model_state.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    N = int(cfg.env.N)
    beta = float(model.beta)
    max_queue = int(cfg.runner.max_queue)
    Q_eval = _sample_uniform_states(args.sample_n, N, max_queue, seed=args.seed)
    mu_eval = mu.unsqueeze(0).expand(Q_eval.shape[0], -1)
    jswq = JoinShortestWeightedQueue(N=N, beta=beta)

    print("=== State Distributions ===")
    bank = generate_state_bank(
        N=N,
        mu=mu,
        beta=beta,
        R_cert=float(cfg.model.get("certificate", {}).get("fallback_radius", float("inf"))),
        n_random=128,
        n_grid=0,
        n_boundary=128,
    )
    print(f"train-like mean total backlog : {Q_eval.sum(dim=-1).mean().item():.3f}")
    print(f"state-bank mean total backlog : {bank.sum(dim=-1).mean().item():.3f}")
    print(f"state-bank p95 total backlog  : {torch.quantile(bank.sum(dim=-1).float(), 0.95).item():.3f}")

    print("\n=== Policy Comparison On Sampled States ===")
    pi_model = _describe_policy("model", model, Q_eval, mu_eval)
    with torch.no_grad():
        pi_cert = model.forward_full(Q_eval, mu_eval).p_cert
        pi_jswq, _ = jswq(Q_eval, mu_eval, training_mode=False)
    print(f"model vs certified mean |dpi|: {(pi_model - pi_cert).abs().mean().item():.6f}")
    print(f"model vs JSWQ mean |dpi|     : {(pi_model - pi_jswq).abs().mean().item():.6f}")

    print("\n=== True Rollout Cost ===")
    rollout = RolloutConfig(
        steps=args.rollout_steps,
        batch_size=args.rollout_batch_size,
        max_backlog=float(cfg.runner.max_backlog),
        show_progress=False,
    )
    base_policy = AnalyticBackbonePolicy(N=N, beta=beta)
    for name, policy in [("model", model), ("certified", base_policy), ("jswq", jswq)]:
        metrics = evaluate_policy(
            name=name,
            model=policy,
            env_name=str(cfg.env.mu_mode),
            seed=args.seed,
            N=N,
            lam=lam,
            mu=mu,
            rollout=rollout,
        )
        print(
            f"{name:10s} avg_cost={metrics.performance.generic.avg_cost:.6f} "
            f"avg_queue={metrics.performance.queueing['avg_queue_length']:.6f} "
            f"p95={metrics.performance.queueing['p95_backlog']:.3f}"
        )


if __name__ == "__main__":
    main()

