"""Optimized main queueing study runner family.

Entrypoint:
    python -m studies.runner.families.main_queueing_optimized [OPTIONS] [HYDRA_OVERRIDES...]

This family runs only the train + audit stages (no baselines comparison),
targeting the optimized architecture variant.

Examples:
    python -m studies.runner.families.main_queueing_optimized --dry-run
    python -m studies.runner.families.main_queueing_optimized --seeds 42,43
"""

from __future__ import annotations

import sys

from studies.runner.cli import parse_and_run
from studies.runner.common import StudyRunnerSpec

SPEC = StudyRunnerSpec(
    config_name="experiments/main_queueing_optimized",
    stages=("train", "audit"),
)


def main(cli_overrides: list[str] | None = None) -> None:
    """Run the optimized certified queueing pipeline."""
    parse_and_run(SPEC, argv=cli_overrides)


if __name__ == "__main__":
    main(sys.argv[1:])
