"""Main queueing study runner."""

from __future__ import annotations

from studies.runner.common import StudyRunnerSpec, run_study_family

SPEC = StudyRunnerSpec(
    config_name="experiments/main_queueing",
    stages=("train", "audit", "baselines"),
)


def main(cli_overrides: list[str] | None = None) -> None:
    """Run the exact-certified queueing pipeline."""
    run_study_family(SPEC, cli_overrides or [])

