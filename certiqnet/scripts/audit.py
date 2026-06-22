"""
Standalone state-bank certificate audit.

Usage:
    python -m certiqnet.scripts.audit --run-dir outputs/main_queueing/run_abc
    python -m certiqnet.scripts.audit --experiment outputs/main_queueing    (auto-discover)
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="State-bank certificate audit")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=str, help="Path to trained run output directory")
    group.add_argument("--experiment", type=str, help="Experiment root (auto-discovers latest run)")
    args, _ = parser.parse_known_args()

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        from certiqnet.experiments.checkpoint_state import read_last_run
        experiment_root = Path(args.experiment).resolve()
        last = read_last_run(experiment_root)
        if last is None:
            print(f"ERROR: No trained run found in {experiment_root}")
            sys.exit(1)
        run_dir = experiment_root / str(last["run_id"])

    config_path = run_dir / "configs" / "resolved_config.yaml"
    if not config_path.exists():
        print(f"ERROR: No resolved config found at {config_path}")
        sys.exit(1)

    cfg = OmegaConf.load(config_path)
    cfg.project.output_root = str(run_dir.parents[1])

    from certiqnet.eval.audit import run_state_bank_audit
    run_state_bank_audit(cfg, cwd=ROOT)


if __name__ == "__main__":
    main()
