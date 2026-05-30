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


def _optional_text(value: object) -> str:
    """Return a usable string for optional config values."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "none" else text


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
    experiment_name = _optional_text(cfg.project.get("experiment_name", None))
    if not experiment_name:
        experiment_name = experiment_name_from_cfg(cfg)
    run_id = _optional_text(cfg.project.get("run_id", None))
    if not run_id:
        run_id = make_run_id(experiment_name, int(cfg.project.seed))
    output_root_value = _optional_text(cfg.project.get("output_root", "outputs")) or "outputs"
    root = output_root or Path(output_root_value)
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
