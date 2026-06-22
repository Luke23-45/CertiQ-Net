from __future__ import annotations

import sys

from certiqnet.experiments.catalog.runtime import Stage
from certiqnet.experiments.catalog.specs import OverrideSpec, StudySpec, TrainingVariant
from certiqnet.experiments.stages._base import ROOT


def build_baselines_stage(
    spec: StudySpec,
    variant: TrainingVariant,
    seed: int,
    overrides: tuple[OverrideSpec, ...],
    run_dir: str,
) -> Stage:
    cmd = [
        sys.executable,
        "-m",
        "certiqnet.scripts.baselines",
        "--run-dir",
        run_dir,
    ]
    for override in overrides:
        cmd.append(override.to_hydra_arg())

    return Stage(
        label=f"{spec.name} :: {variant.label} :: seed={seed} :: baselines",
        command=cmd,
        cwd=ROOT,
    )
