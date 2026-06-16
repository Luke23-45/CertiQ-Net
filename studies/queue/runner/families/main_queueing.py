"""Backwards-compatibility shim — authoritative code has moved.

The canonical implementation is now at:
    studies.runner.families.main_queueing
"""

from __future__ import annotations

from studies.runner.families.main_queueing import SPEC, main  # noqa: F401

__all__ = ["SPEC", "main"]
