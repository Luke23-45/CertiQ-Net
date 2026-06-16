"""Main queueing study runner family.

Entrypoint:
    python -m studies.runner.families.main_queueing [OPTIONS] [HYDRA_OVERRIDES...]

Examples:
    # Dry-run: see what would execute
    python -m studies.runner.families.main_queueing --dry-run

    # Only SED and MaxWeight baselines on seed 42
    python -m studies.runner.families.main_queueing \\
        --seeds 42 --baselines-include sed,max_weight

    # Three seeds, all stages, skip failures
    python -m studies.runner.families.main_queueing \\
        --seeds 42,43,44 --failure-mode skip

    # Tag + Hydra override mixed
    python -m studies.runner.families.main_queueing \\
        --tag experiment-v3 studies.runner.verbose=false
"""

from __future__ import annotations

import sys

from studies.runner.cli import parse_and_run
from studies.runner.common import StudyRunnerSpec

SPEC = StudyRunnerSpec(
    config_name="experiments/main_queueing",
    stages=("train", "audit", "baselines"),
)


def main(cli_overrides: list[str] | None = None) -> None:
    """Run the exact-certified queueing pipeline."""
    parse_and_run(SPEC, argv=cli_overrides)


if __name__ == "__main__":
    main(sys.argv[1:])
