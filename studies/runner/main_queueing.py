"""Runner for the Main Queueing experiment family."""

import sys

from studies.utils.execution import run_stage


def main() -> None:
    """Run the exact-certified queueing pipeline for one config."""
    overrides: list[str] = sys.argv[1:]
    
    # Force the experiment family logic
    if "experiment_family=main_queueing" not in overrides:
        overrides.append("experiment_family=main_queueing")
        
    train_cmd = ["python", "scripts/train.py"] + overrides
    run_stage(train_cmd, "Training (Queueing CTMC Exact)")

    audit_cmd = ["python", "studies/utils/audit.py"] + overrides
    run_stage(audit_cmd, "Certificate Audit (State Bank)")

    baseline_cmd = ["python", "studies/utils/baselines.py"] + overrides
    run_stage(baseline_cmd, "Baseline Comparison")

if __name__ == "__main__":
    main()
