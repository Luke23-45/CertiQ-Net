"""Hydra training entry point with platform-aware config resolution."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from certiqnet.experiments.pipeline import run_training
from certiqnet.utils.config_schemas import RootConfig

cs = ConfigStore.instance()
cs.store(name="root_config", node=RootConfig)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")  # type: ignore[untyped-decorator]
def main(cfg: DictConfig) -> None:
    run_training(cfg, cwd=ROOT)


if __name__ == "__main__":
    main()
