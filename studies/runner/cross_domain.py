"""Runner for the Cross-Domain experiment family."""

import sys

from studies.utils.execution import run_stage


def main() -> None:
    """Run cross-domain demonstration pipeline (Approximate/Empirical)."""
    overrides: list[str] = sys.argv[1:]
    
    if "experiment_family=cross_domain" not in overrides:
        overrides.append("experiment_family=cross_domain")
    if "certificate_status=approximate" not in overrides:
        overrides.append("certificate_status=approximate")
        
    train_cmd = ["python", "scripts/train.py"] + overrides
    run_stage(train_cmd, "Training (Cross-Domain)")

if __name__ == "__main__":
    main()
