"""Standalone state-bank certificate audit with platform info."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from certiqnet.diagnostics.state_bank import generate_state_bank
from certiqnet.experiments.metrics import aggregate_metrics, save_metrics
from certiqnet.experiments.runner import prepare_run
from certiqnet.utils.platform import detect_platform
from certiqnet.utils.progress import configure_progress
from scripts.train import build_model, build_mu


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")  # type: ignore[untyped-decorator]
def main(cfg: DictConfig) -> None:
    platform_info = detect_platform()

    if "progress" in cfg:
        configure_progress(OmegaConf.to_container(cfg.progress, resolve=True))

    paths, run_logger = prepare_run(cfg, cwd=ROOT)
    run_logger.info("audit_platform_info",
                    os=platform_info.os_name,
                    torch=platform_info.torch_version,
                    gpu=platform_info.gpu_count)

    adapter = instantiate(cfg.adapter) if "adapter" in cfg else None
    d_xi = int(getattr(adapter, "context_dim", 0))
    mu, lam = build_mu(cfg)
    model = build_model(cfg, N=int(cfg.env.N), d_xi=d_xi)
    model.eval()

    run_logger.info("generating_state_bank")
    Q_bank = generate_state_bank(
        N=int(cfg.env.N),
        mu=mu,
        beta=float(cfg.model.beta),
        R_cert=float(cfg.model.get("R_cert", float("inf"))),
        n_random=512,
        n_grid=128,
        n_boundary=128,
    )
    run_logger.info("state_bank_generated", states=Q_bank.shape[0])

    mu_bank = mu.unsqueeze(0).expand(Q_bank.shape[0], -1)
    if adapter is not None:
        Q_bank, mu_bank, xi_bank = adapter.make_observation(Q_bank, mu_bank)
    else:
        xi_bank = None
    with torch.no_grad():
        _, diag = model(Q_bank, mu_bank, xi_bank, training_mode=False)

    violation = (diag.A_pi - diag.B_Q).clamp(min=0.0)
    audit_metrics = aggregate_metrics(
        model_name=str(cfg.model._target_).split(".")[-1],
        env_name=str(cfg.env.mu_mode),
        seed=int(cfg.project.seed),
        lam=lam,
        queue_trace=Q_bank,
        cost_trace=Q_bank.sum(dim=-1),
        dt_trace=torch.ones(Q_bank.shape[0]),
        diagnostics=[diag],
    )
    save_metrics([audit_metrics], paths.audits, filename="state_bank_audit")
    run_logger.metric(audit_metrics.flat())
    run_logger.flush()

    print(f"states={Q_bank.shape[0]}")
    print(f"max_violation={violation.max().item():.6e}")
    print(f"violation_rate={(violation > 0).float().mean().item():.6e}")
    print(f"fallback_rate={diag.fallback_active.float().mean().item():.6e}")
    print(f"gate_rate={(diag.eta_final > 0).float().mean().item():.6e}")
    print(f"min_drift_slack={diag.drift_slack.min().item():.6e}")


if __name__ == "__main__":
    main()
