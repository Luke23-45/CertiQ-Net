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
import traceback

from studies.runner.cli import parse_and_run
from studies.runner.common import StudyRunnerSpec

SPEC = StudyRunnerSpec(
    config_name="experiments/main_queueing_optimized",
    stages=("train", "audit"),
)


def main(cli_overrides: list[str] | None = None) -> None:
    """Run the optimized certified queueing pipeline."""
    try:
        parse_and_run(SPEC, argv=cli_overrides)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except BaseException as e:
        print(f"\nFatal error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
