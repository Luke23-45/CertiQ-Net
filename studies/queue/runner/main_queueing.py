"""Runner for the Main Queueing experiment family."""

from __future__ import annotations

import sys

from studies.runner.families.main_queueing import main as _main


def main() -> None:
    """Run the exact-certified queueing pipeline for one config."""
    _main(sys.argv[1:])


if __name__ == "__main__":
    main()
