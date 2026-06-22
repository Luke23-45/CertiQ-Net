#!/usr/bin/env python3
"""
CertiQ-Net study runner — single entry point for all experiments.

Usage:
    python run.py --list
    python run.py --study main_queueing
    python run.py --study main_queueing --seed 42 --seed 43 --dry-run
    python run.py --study main_queueing --set model.C=5.0 --keep-going
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from certiqnet.experiments.runner import parse_and_run

if __name__ == "__main__":
    raise SystemExit(parse_and_run())
