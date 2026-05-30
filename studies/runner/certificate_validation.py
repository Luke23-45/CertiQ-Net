"""Runner for the standalone Certificate Validation program."""

from __future__ import annotations

import sys

from studies.runner.families.certificate_validation import main as _main


def main() -> None:
    """Run standalone certification checks (audits)."""
    _main(sys.argv[1:])


if __name__ == "__main__":
    main()
