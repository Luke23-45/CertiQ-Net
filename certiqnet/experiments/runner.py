"""High-level experiment runner utilities."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from certiqnet.data.registry import DatasetRegistry
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
    family = str(cfg.get("experiment_family", "default"))

    if cfg.datatype == "qgym":
        profile = cfg.get("qgym")
        if profile is not None:
            ds_name = str(profile.data.init_args.get("dataset_name", "qgym"))
        else:
            ds_name = str(cfg.get("qgym_data", "qgym"))
        ds_name_clean = ds_name.replace("/", "_").replace("\\", "_")
        spec = DatasetRegistry().get(ds_name_clean) if ds_name_clean != "qgym" else None
        if spec is not None:
            n = int(spec.env_N)
            lam_str = str(spec.env_lam) if spec.env_lam is not None else "auto"
        else:
            n = int(cfg.env.N) if "env" in cfg else 6
            lam_str = "auto"
        return f"{family}_{model_name}_N{n}_{ds_name_clean}_lam{lam_str}"
    else:
        env_name = str(cfg.env.mu_mode)
        n = int(cfg.env.N)
        rho = cfg.env.get("rho_target")
        rho_part = f"rho{rho}" if rho is not None else f"lam{cfg.env.lam}"
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
    raw_root = output_root or Path(output_root_value)
    root = raw_root if raw_root.is_absolute() else (cwd / raw_root).resolve()
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
