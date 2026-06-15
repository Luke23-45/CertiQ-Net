"""Diagnose exploration collapse and heuristic anchoring in CertiQ-Net.

This script compares certified vs. uncertified policies, checks whether
``training_mode`` changes anything, measures action entropy/diversity, and
estimates how fragile the policy is to small perturbations of the queue state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from certiqnet.experiments.factory import build_model, build_mu
from certiqnet.models.baselines import AnalyticBackbonePolicy, QuadraticMinDrift, ShortestExpectedDelay
from certiqnet.diagnostics.state_bank import generate_state_bank


def _load_model(run_dir: Path) -> tuple[dict, torch.nn.Module, torch.Tensor]:
    cfg = OmegaConf.load(run_dir / "configs" / "resolved_config.yaml")
    mu, _ = build_mu(cfg)
    model = build_model(cfg, N=int(cfg.env.N), d_xi=0)
    ckpt = run_dir / "artifacts" / "final_model_state.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return cfg, model, mu


def _sample_hard_states(
    *,
    n_states: int,
    N: int,
    mu: torch.Tensor,
    max_queue: int,
    seed: int,
) -> torch.Tensor:
    bank = generate_state_bank(
        N=N,
        mu=mu,
        beta=1.0,
        R_cert=float("inf"),
        n_random=max(4 * n_states, 1024),
        n_grid=0,
        n_boundary=max(128, 8 * N),
    )
    backlog = bank.sum(dim=-1)
    top_idx = torch.topk(backlog, k=min(n_states, bank.shape[0]), largest=True).indices
    hard = bank[top_idx]
    if hard.shape[0] < n_states:
        gen = torch.Generator().manual_seed(seed)
        fill = torch.randint(0, max_queue, (n_states - hard.shape[0], N), generator=gen).float()
        hard = torch.cat([hard, fill], dim=0)
    return hard[:n_states].float()


def _entropy(pi: torch.Tensor) -> torch.Tensor:
    return -(pi.clamp_min(1e-9) * pi.clamp_min(1e-9).log()).sum(dim=-1)


def _unique_action_count(pi: torch.Tensor, draws: int) -> torch.Tensor:
    samples = torch.multinomial(pi, num_samples=draws, replacement=True)
    return torch.tensor([torch.unique(row).numel() for row in samples], device=pi.device, dtype=pi.dtype)


def _flip_rate(pi_a: torch.Tensor, pi_b: torch.Tensor) -> float:
    return (pi_a.argmax(dim=-1) != pi_b.argmax(dim=-1)).float().mean().item()


def _l1_delta(pi_a: torch.Tensor, pi_b: torch.Tensor) -> float:
    return (pi_a - pi_b).abs().sum(dim=-1).mean().item()


def _agreement(pi_a: torch.Tensor, pi_b: torch.Tensor) -> float:
    return (pi_a.argmax(dim=-1) == pi_b.argmax(dim=-1)).float().mean().item()


def _stats(name: str, pi: torch.Tensor, *, draws: int) -> dict[str, float]:
    ent = _entropy(pi)
    uniq = _unique_action_count(pi, draws=draws)
    return {
        f"{name}.entropy_mean": ent.mean().item(),
        f"{name}.entropy_min": ent.min().item(),
        f"{name}.entropy_max": ent.max().item(),
        f"{name}.effective_support_mean": ent.exp().mean().item(),
        f"{name}.unique_actions_mean": uniq.mean().item(),
        f"{name}.top1_mass_mean": pi.max(dim=-1).values.mean().item(),
    }


def _perturbed_policy(
    model: torch.nn.Module,
    Q: torch.Tensor,
    mu: torch.Tensor,
    *,
    training_mode: bool,
    perturb_scale: float,
) -> torch.Tensor:
    Q_pert = Q.clone()
    row = torch.arange(Q.shape[0])
    col = torch.randint(0, Q.shape[1], (Q.shape[0],))
    Q_pert[row, col] = Q_pert[row, col] + perturb_scale
    with torch.no_grad():
        return model.forward_full(Q_pert, mu, training_mode=training_mode).pi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sample-n", type=int, default=512)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perturb-scale", type=float, default=1.0)
    parser.add_argument("--max-queue", type=int, default=15)
    args = parser.parse_args()

    cfg, model, mu = _load_model(args.run_dir)
    N = int(cfg.env.N)
    Q = _sample_hard_states(
        n_states=args.sample_n,
        N=N,
        mu=mu,
        max_queue=args.max_queue,
        seed=args.seed,
    )
    mu_b = mu.unsqueeze(0).expand(Q.shape[0], -1)

    with torch.no_grad():
        pi_train = model.forward_full(Q, mu_b, training_mode=True)
        pi_eval = model.forward_full(Q, mu_b, training_mode=False)

    backbone = AnalyticBackbonePolicy(N=N, beta=float(getattr(model, "beta", 1.0)), C=float(getattr(model, "C", float("inf"))))
    sed = ShortestExpectedDelay(N=N, beta=float(getattr(model, "beta", 1.0)), C=float(getattr(model, "C", float("inf"))))
    qmd = QuadraticMinDrift(N=N, beta=float(getattr(model, "beta", 1.0)), C=float(getattr(model, "C", float("inf"))))
    with torch.no_grad():
        pi_backbone, _ = backbone(Q, mu_b, training_mode=False)
        pi_sed, _ = sed(Q, mu_b, training_mode=False)
        pi_qmd, _ = qmd(Q, mu_b, training_mode=False)

    pert = _perturbed_policy(
        model,
        Q,
        mu_b,
        training_mode=False,
        perturb_scale=args.perturb_scale,
    )

    metrics: dict[str, float] = {}
    metrics.update(_stats("policy", pi_eval.pi, draws=args.draws))
    metrics["training_mode_l1_delta"] = _l1_delta(pi_train.pi, pi_eval.pi)
    metrics["training_mode_l1_delta_proposal"] = _l1_delta(raw_train.pi, raw_eval.pi)
    metrics["certified_vs_proposal_l1_delta"] = _l1_delta(cert_eval.pi, raw_eval.pi)
    metrics["projection_active_rate"] = cert_eval.diagnostics.projection_active.float().mean().item()
    metrics["projection_correction_mean"] = cert_eval.diagnostics.correction_magnitude.mean().item()
    metrics["certificate_slack_mean"] = cert_eval.diagnostics.certificate_slack.mean().item()
    metrics["certificate_slack_min"] = cert_eval.diagnostics.certificate_slack.min().item()
    metrics["agreement_backbone"] = _agreement(cert_eval.pi, pi_backbone)
    metrics["agreement_sed"] = _agreement(pi_eval.pi, pi_sed)
    metrics["agreement_qmd"] = _agreement(pi_eval.pi, pi_qmd)
    metrics["perturb_flip_rate"] = _flip_rate(pi_eval.pi, pert)

    red_flags: list[str] = []
    if metrics.get("policy.entropy_mean", 1.0) < 0.5:
        red_flags.append("low policy entropy")
    if metrics["agreement_qmd"] > 0.9 or metrics["agreement_sed"] > 0.9:
        red_flags.append("policy still tracks a heuristic")
    if metrics["training_mode_l1_delta"] < 1e-8:
        red_flags.append("training_mode does not alter the policy")
    if metrics["perturb_flip_rate"] < 0.05:
        red_flags.append("policy is state-insensitive under unit perturbations")

    print("=== Exploration Collapse Diagnostic ===")
    print(f"run_dir: {args.run_dir}")
    print(f"sample_n: {args.sample_n}")
    print(f"draws: {args.draws}")
    print(f"perturb_scale: {args.perturb_scale}")
    print()
    print("=== Key Metrics ===")
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]:.6f}")
    print()
    print("=== Red Flags ===")
    if red_flags:
        for flag in red_flags:
            print(f"- {flag}")
    else:
        print("- none")
    print()
    print("=== JSON ===")
    print(json.dumps({"metrics": metrics, "red_flags": red_flags}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
