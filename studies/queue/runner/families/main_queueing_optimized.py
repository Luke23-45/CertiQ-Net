"""Optimized main queueing study runner."""

from __future__ import annotations

from studies.runner.common import StudyRunnerSpec, run_study_family

SPEC = StudyRunnerSpec(
    config_name="experiments/main_queueing_optimized",
    stages=("train", "audit"),
)


def main(cli_overrides: list[str] | None = None) -> None:
    """Run the optimized certified queueing pipeline."""
    run_study_family(SPEC, cli_overrides or [])

