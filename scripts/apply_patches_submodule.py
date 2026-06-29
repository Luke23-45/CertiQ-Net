"""Apply all local patch files to the bundled QGym submodule."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from certiqnet.utils.submodule_patches import apply_qgym_patches


def main() -> None:
    applied = apply_qgym_patches(reset=True)
    print(f"Applied {len(applied)} QGym patch file(s).")
    for patch in applied:
        print(f" - {patch.name}")


if __name__ == "__main__":
    main()
