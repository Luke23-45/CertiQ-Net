"""Training entrypoints.

Usage
-----
New (LightningCLI):
    python scripts/train.py --cli --config configs/experiments/queueing/certiq_index.yaml

Legacy (Hydra):
    python scripts/train.py model=certiq_index env=family_a adapter=queueing trainer=default
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── CLI mode (LightningCLI, BGSL-style) ─────────────────────────────


def _run_cli(config_path: str) -> None:
    sys.argv = ["certiqnet-train", "fit", "--config", config_path]
    from certiqnet.cli import main as cli_main
    cli_main()


# ── Legacy mode (Hydra, backward-compat) ────────────────────────────


def _run_legacy() -> None:
    import hydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig

    from certiqnet.experiments.pipeline import run_training
    from certiqnet.utils.config_schemas import RootConfig

    cs = ConfigStore.instance()
    cs.store(name="root_config", node=RootConfig)

    @hydra.main(version_base="1.3", config_path="../configs", config_name="config")
    def main(cfg: DictConfig) -> None:
        run_training(cfg, cwd=ROOT)

    main()


# ── Dispatch ────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CertiQ-Net training")
    parser.add_argument("--cli", action="store_true", help="Use LightningCLI mode")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    args, unknown = parser.parse_known_args()

    if args.cli:
        if args.config is None:
            print("ERROR: --cli requires --config <path>")
            sys.exit(1)
        _run_cli(args.config)
    else:
        sys.argv = [sys.argv[0]] + unknown
        _run_legacy()
