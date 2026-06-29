"""Retired CTMC comparison script.

The old fast comparison path was tied to the removed CTMC rollout
environment.  Use the QGym evaluation pipeline instead.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "scripts.compare_cost_fn_fast has been retired because CTMC was removed. "
        "Use the QGym benchmark/evaluator path instead."
    )


if __name__ == "__main__":
    main()

