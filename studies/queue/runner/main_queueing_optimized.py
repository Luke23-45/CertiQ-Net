"""Runner for the optimized Main Queueing experiment family.

Thin entrypoint — delegates to the canonical runner at
``studies.runner.families.main_queueing_optimized``.
"""

from __future__ import annotations

import sys

from studies.runner.families.main_queueing_optimized import main as _main


def main() -> None:
    """Run the optimized exact-certified queueing pipeline."""
    _main(sys.argv[1:])


if __name__ == "__main__":
    main()
