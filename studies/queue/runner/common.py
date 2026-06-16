"""Backwards-compatibility shim — authoritative code has moved.

All symbols are re-exported from ``studies.runner.common``.
Import from there directly for new code.
"""

from __future__ import annotations

# Re-export everything from the new canonical location.
from studies.runner.common import (  # noqa: F401
    ROOT,
    CONFIG_DIR,
    StudyRunnerSpec,
    compose_study_config,
    run_stage,
    run_study_family,
)

__all__ = [
    "ROOT",
    "CONFIG_DIR",
    "StudyRunnerSpec",
    "compose_study_config",
    "run_stage",
    "run_study_family",
]
