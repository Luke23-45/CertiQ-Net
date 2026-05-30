"""Cross-domain study runner."""

from __future__ import annotations

from studies.runner.common import StudyRunnerSpec, run_study_family

SPEC = StudyRunnerSpec(
    config_name="experiments/cross_domain",
    stages=("train",),
)


def main(cli_overrides: list[str] | None = None) -> None:
    """Run the approximate cross-domain demonstration pipeline."""
    run_study_family(SPEC, cli_overrides or [])

