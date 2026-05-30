"""In-process experiment runner: train, audit, and compare using a single Hydra context.

The training checkpoint is loaded by the audit and baseline comparison steps,
so all evaluations use the trained weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from certiqnet.experiments.pipeline import (
    run_baseline_paper_comparison,
    run_state_bank_audit,
    run_training,
)
from certiqnet.utils.config_schemas import RootConfig


def main() -> None:
    overrides = sys.argv[1:]

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    hydra.initialize(config_path="../configs", version_base="1.3")
    cs = ConfigStore.instance()
    cs.store(name="root_config", node=RootConfig)
    cfg = hydra.compose(config_name="config", overrides=overrides)
    OmegaConf.resolve(cfg)

    # Step 1: Train — saves checkpoint to paths.artifacts / "final_model_state.pt"
    run_training(cfg, cwd=ROOT)

    # Step 2: Audit with the trained model
    run_state_bank_audit(cfg, cwd=ROOT)

    # Step 3: Baseline comparison with the trained model
    run_baseline_paper_comparison(cfg, cwd=ROOT)


if __name__ == "__main__":
    main()
