"""Deprecated — use ``python run.py --study main_queueing`` instead."""

import sys

from certiqnet.experiments.runner import parse_and_run


def main() -> None:
    sys.exit(parse_and_run(["--study", "main_queueing", *sys.argv[1:]]))


if __name__ == "__main__":
    main()
