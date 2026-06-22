"""Backward-compatibility shim — all functionality moved to certiqnet.experiments."""

from certiqnet.experiments.catalog.runtime import Stage
from certiqnet.experiments.catalog.specs import OverrideSpec, StudySpec
from certiqnet.experiments.runner import run_study
from certiqnet.experiments.stages._base import ROOT, CONFIG_DIR

# Preserve old name for StudySpec
StudyRunnerSpec = StudySpec

__all__ = [
    "ROOT",
    "CONFIG_DIR",
    "StudyRunnerSpec",
    "OverrideSpec",
    "Stage",
    "run_study",
]
