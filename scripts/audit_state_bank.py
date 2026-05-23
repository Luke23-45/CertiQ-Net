"""Standalone state-bank certificate audit."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra
import torch
from omegaconf import DictConfig

from certiqnet.diagnostics.state_bank import generate_state_bank
from scripts.train import build_model, build_mu


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run an audit and print certificate metrics."""
    mu, _ = build_mu(cfg)
    model = build_model(cfg, N=int(cfg.env.N))
    model.eval()
    Q_bank = generate_state_bank(
        N=int(cfg.env.N),
        mu=mu,
        beta=float(cfg.model.beta),
        R_cert=float(cfg.model.get("R_cert", float("inf"))),
        n_random=512,
        n_grid=128,
        n_boundary=128,
    )
    mu_bank = mu.unsqueeze(0).expand(Q_bank.shape[0], -1)
    with torch.no_grad():
        _, diag = model(Q_bank, mu_bank, training_mode=False)
    violation = (diag.A_pi - diag.B_Q).clamp(min=0.0)
    print(f"states={Q_bank.shape[0]}")
    print(f"max_violation={violation.max().item():.6e}")
    print(f"violation_rate={(violation > 0).float().mean().item():.6e}")
    print(f"fallback_rate={diag.fallback_active.float().mean().item():.6e}")
    print(f"gate_rate={(diag.eta_final > 0).float().mean().item():.6e}")
    print(f"min_drift_slack={diag.drift_slack.min().item():.6e}")


if __name__ == "__main__":
    main()
