"""Hydra multirun sweep launcher driven by config/sweep.yaml."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf


def main() -> None:
    cfg = OmegaConf.load(ROOT / "configs/config.yaml")
    sweep = cfg.sweep

    seeds = ",".join(str(s) for s in sweep.seeds)
    models = ",".join(str(m) for m in sweep.models)
    envs = ",".join(str(e) for e in sweep.envs)

    cmd = [
        "python",
        "scripts/train.py",
        "--multirun",
        f"model={models}",
        f"env={envs}",
        f"project.seed={seeds}",
    ]
    print(f"[sweep] launching {len(sweep.seeds) * len(sweep.models) * len(sweep.envs)} jobs")
    print(f"[sweep] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
