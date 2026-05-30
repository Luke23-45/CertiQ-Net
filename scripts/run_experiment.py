"""Robust experiment runner for training, audit, and baseline comparison."""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    overrides = sys.argv[1:]
    commands = [
        ["python", "scripts/train.py", *overrides],
        ["python", "scripts/audit_state_bank.py", *overrides],
        ["python", "scripts/compare_baselines.py", *overrides],
    ]
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
