"""Runner for the Ablations experiment family."""

from __future__ import annotations

import sys

from studies.runner.families.ablations import main as _main


def main() -> None:
    """Run ablation pipeline."""
    _main(sys.argv[1:])


if __name__ == "__main__":
    main()
