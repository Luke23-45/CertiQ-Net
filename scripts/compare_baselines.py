"""Run baseline comparison rollouts and persist dual metric tables."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra
from omegaconf import DictConfig

from certiqnet.experiments.pipeline import run_baseline_paper_comparison


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_baseline_paper_comparison(cfg, cwd=ROOT)


if __name__ == "__main__":
    main()
