"""Shared helpers for study runner entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from certiqnet.experiments.pipeline import (
    run_baseline_paper_comparison,
    run_state_bank_audit,
    run_training,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


@dataclass(frozen=True)
class StudyRunnerSpec:
    """Describe one dedicated study runner family."""

    config_name: str
    stages: tuple[str, ...]
    default_overrides: tuple[str, ...] = ()


def compose_study_config(spec: StudyRunnerSpec, cli_overrides: Sequence[str]) -> DictConfig:
    """Compose a full Hydra config for a study family plus CLI overrides."""
    overrides = [*spec.default_overrides, *cli_overrides]
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name=spec.config_name, overrides=list(overrides))


def run_stage(name: str, fn: Callable[[], None]) -> None:
    """Run one study stage with lightweight console framing."""
    print(f"\n>>> Starting Stage: {name}")
    try:
        fn()
    except Exception:
        print(f"\n>>> Stage Failed: {name}")
        raise
    print(f"\n>>> Stage Completed: {name}")


def run_study_family(spec: StudyRunnerSpec, cli_overrides: Sequence[str]) -> None:
    """Compose config and execute the declared study stages."""
    cfg = compose_study_config(spec, cli_overrides)

    stage_fns: dict[str, Callable[[], None]] = {
        "train": lambda: run_training(cfg, cwd=ROOT),
        "audit": lambda: run_state_bank_audit(cfg, cwd=ROOT),
        "baselines": lambda: run_baseline_paper_comparison(cfg, cwd=ROOT),
    }

    for stage in spec.stages:
        if stage not in stage_fns:
            raise ValueError(f"Unknown study stage: {stage}")
        run_stage(stage, stage_fns[stage])
