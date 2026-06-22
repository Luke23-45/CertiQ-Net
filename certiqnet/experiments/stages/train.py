from __future__ import annotations

import sys
from pathlib import Path

from certiqnet.experiments.catalog.runtime import Stage
from certiqnet.experiments.catalog.specs import OverrideSpec, StudySpec, TrainingVariant
from certiqnet.experiments.stages._base import ROOT


def build_training_stage(
    spec: StudySpec,
    variant: TrainingVariant,
    seed: int,
    overrides: tuple[OverrideSpec, ...],
    run_id: str,
) -> Stage:
    experiment_name = f"{spec.name}_{variant.label}"
    cmd = [
        sys.executable,
        "-m",
        "certiqnet.scripts.train",
        "--config-name",
        spec.config_name,
        f"project.experiment_name={experiment_name}",
        f"project.run_id={run_id}",
        f"project.seed={seed}",
    ]
    if variant.dataset:
        cmd.append(f"qgym.data.init_args.dataset_name={variant.dataset}")
    for extra in variant.extra_overrides:
        cmd.append(str(extra))
    for override in overrides:
        cmd.append(override.to_hydra_arg())

    return Stage(
        label=f"{spec.name} :: {variant.label} :: seed={seed}",
        command=cmd,
        cwd=ROOT,
    )
