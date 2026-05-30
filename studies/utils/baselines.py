"""Run baseline comparison rollouts and persist dual metric tables."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from certiqnet.experiments.baseline_runner import RolloutConfig, run_baseline_comparison
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
    run_logger.info("baseline_platform_info",
                    os=platform_info.os_name,
                    torch=platform_info.torch_version,
                    gpu=platform_info.gpu_count)

    adapter = instantiate(cfg.adapter) if "adapter" in cfg else None
    d_xi = int(getattr(adapter, "context_dim", 0))
    mu, lam = build_mu(cfg)
    model = build_model(cfg, N=int(cfg.env.N), d_xi=d_xi)
    rollout = RolloutConfig(
        steps=int(cfg.runner.rollout_steps),
        batch_size=int(cfg.runner.rollout_batch_size),
        max_backlog=float(cfg.runner.max_backlog),
        show_progress=bool(cfg.runner.show_progress),
    )
    metrics = run_baseline_comparison(
        env_name=str(cfg.env.mu_mode),
        N=int(cfg.env.N),
        lam=lam,
        mu=mu,
        seed=int(cfg.project.seed),
        output_dir=paths.metrics,
        rollout=rollout,
        extra_models={"configured_model": model},
        adapter=adapter,
    )
    for row in metrics:
        run_logger.metric(row.flat())
    run_logger.flush()
    run_logger.table("Baseline Comparison", [row.flat() for row in metrics])


if __name__ == "__main__":
    main()
