"""Standalone state-bank certificate audit with platform info."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra
from omegaconf import DictConfig

from certiqnet.experiments.pipeline import run_state_bank_audit


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")  # type: ignore[untyped-decorator]
def main(cfg: DictConfig) -> None:
    run_state_bank_audit(cfg, cwd=ROOT)


if __name__ == "__main__":
    main()
