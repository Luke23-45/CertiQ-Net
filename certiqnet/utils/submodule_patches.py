"""Apply local patch files to the bundled QGym submodule."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMODULE_ROOT = REPO_ROOT / "extern" / "QGym"
PATCH_ROOT = REPO_ROOT / "patches" / "qgym"


class PatchApplicationError(RuntimeError):
    """Raised when a submodule patch fails to apply."""


def _run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def apply_qgym_patches(*, reset: bool = True) -> list[Path]:
    """Reset the QGym submodule and apply all tracked patches in order."""
    if not SUBMODULE_ROOT.exists():
        raise FileNotFoundError(f"QGym submodule not found: {SUBMODULE_ROOT}")
    if not PATCH_ROOT.exists():
        raise FileNotFoundError(f"QGym patch directory not found: {PATCH_ROOT}")

    patch_files = sorted(PATCH_ROOT.glob("*.patch"))
    if not patch_files:
        raise FileNotFoundError(f"No patch files found in {PATCH_ROOT}")

    if reset:
        reset_result = _run_git(["reset", "--hard", "HEAD"], cwd=SUBMODULE_ROOT)
        if reset_result.returncode != 0:
            raise PatchApplicationError(
                "Failed to reset QGym submodule before applying patches.\n"
                f"stdout:\n{reset_result.stdout}\n"
                f"stderr:\n{reset_result.stderr}"
            )
        clean_result = _run_git(["clean", "-fd"], cwd=SUBMODULE_ROOT)
        if clean_result.returncode != 0:
            raise PatchApplicationError(
                "Failed to clean QGym submodule before applying patches.\n"
                f"stdout:\n{clean_result.stdout}\n"
                f"stderr:\n{clean_result.stderr}"
            )

    applied: list[Path] = []
    for patch in patch_files:
        result = _run_git(["apply", "--whitespace=nowarn", str(patch.resolve())], cwd=SUBMODULE_ROOT)
        if result.returncode != 0:
            raise PatchApplicationError(
                f"Failed to apply patch {patch.name}.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        applied.append(patch)
    return applied

