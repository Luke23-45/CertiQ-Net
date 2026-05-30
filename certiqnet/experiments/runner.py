"""High-level experiment runner utilities."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from certiqnet.experiments.logging import BufferedExperimentLogger as ExperimentLogger
from certiqnet.experiments.paths import (
    RunPaths,
    create_run_paths,
    make_run_id,
    save_manifest,
    save_resolved_config,
)


def experiment_name_from_cfg(cfg: DictConfig) -> str:
    """Build deterministic experiment name from composed Hydra config."""
    model_name = str(cfg.model._target_).split(".")[-1]
    env_name = str(cfg.env.mu_mode)
    n = int(cfg.env.N)
    rho = cfg.env.get("rho_target")
    rho_part = f"rho{rho}" if rho is not None else f"lam{cfg.env.lam}"
    family = str(cfg.get("experiment_family", "default"))
    return f"{family}_{model_name}_N{n}_{env_name}_{rho_part}"


def prepare_run(
    cfg: DictConfig,
    *,
    cwd: Path,
    output_root: Path | None = None,
) -> tuple[RunPaths, ExperimentLogger]:
    """Create run directories, resolved config, manifest, and logger."""
    experiment_name = str(cfg.project.get("experiment_name", "")) or experiment_name_from_cfg(cfg)
    run_id = str(cfg.project.get("run_id", "")) or make_run_id(
        experiment_name, int(cfg.project.seed)
    )
    root = output_root or Path(str(cfg.project.get("output_root", "outputs")))
    paths = create_run_paths(root, experiment_name, run_id)
    save_resolved_config(cfg, paths)
    save_manifest(cfg, paths, experiment_name, run_id, cwd)
    logger = ExperimentLogger(paths.logs)
    logger.info(
        "prepared_run",
        run_id=run_id,
        experiment_name=experiment_name,
        root=str(paths.root),
    )
    return paths, logger
