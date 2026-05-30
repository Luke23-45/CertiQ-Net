"""Runner for the Ablations experiment family."""

import sys

from studies.utils.execution import run_stage


def main() -> None:
    """Run ablation pipeline."""
    overrides: list[str] = sys.argv[1:]
    
    if "experiment_family=ablations" not in overrides:
        overrides.append("experiment_family=ablations")
        
    train_cmd = ["python", "scripts/train.py"] + overrides
    run_stage(train_cmd, "Training (Ablation)")

    audit_cmd = ["python", "studies/utils/audit.py"] + overrides
    run_stage(audit_cmd, "Certificate Audit (State Bank)")

if __name__ == "__main__":
    main()
