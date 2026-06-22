from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from certiqnet.experiments.checkpoint_state import read_last_run
from certiqnet.experiments.logging import BufferedExperimentLogger as ExperimentLogger
from certiqnet.experiments.paths import RunPaths, slugify
from certiqnet.experiments.runner import experiment_name_from_cfg, prepare_run


def discover_and_prepare(
    cfg: DictConfig, *, cwd: Path
) -> tuple[RunPaths, ExperimentLogger]:
    output_root = Path(str(cfg.project.get("output_root", "outputs")))
    if not output_root.is_absolute():
        output_root = (cwd / output_root).resolve()
    experiment_path_name = str(cfg.project.get("experiment_name", None) or "")
    if not experiment_path_name.strip():
        experiment_path_name = experiment_name_from_cfg(cfg)
    experiment_root = output_root / slugify(experiment_path_name)

    last = read_last_run(experiment_root)
    if last is not None:
        if "project" not in cfg:
            cfg.project = OmegaConf.create()
        if not cfg.project.get("run_id"):
            cfg.project.run_id = last["run_id"]
        cfg.project.experiment_name = experiment_path_name

    paths, run_logger = prepare_run(cfg, cwd=cwd)
    return paths, run_logger
