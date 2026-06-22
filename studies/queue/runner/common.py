"""Backwards-compatibility shim — see certiqnet.experiments for new code."""

from certiqnet.experiments.runner import run_study
from certiqnet.experiments.stages._base import CONFIG_DIR, ROOT
from certiqnet.experiments.catalog.specs import StudySpec

StudyRunnerSpec = StudySpec

__all__ = [
    "ROOT",
    "CONFIG_DIR",
    "StudyRunnerSpec",
    "run_study",
]
