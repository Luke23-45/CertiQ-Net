"""Certificate validation study runner."""

from __future__ import annotations

from studies.runner.common import StudyRunnerSpec, run_study_family

SPEC = StudyRunnerSpec(
    config_name="experiments/certificate_validation",
    stages=("audit",),
)


def main(cli_overrides: list[str] | None = None) -> None:
    """Run the standalone certificate audit pipeline."""
    run_study_family(SPEC, cli_overrides or [])

