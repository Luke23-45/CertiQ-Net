"""Runner for the Cross-Domain experiment family."""

from __future__ import annotations

import sys

from studies.runner.families.cross_domain import main as _main


def main() -> None:
    """Run cross-domain demonstration pipeline (Approximate/Empirical)."""
    _main(sys.argv[1:])


if __name__ == "__main__":
    main()
