"""Runner for the standalone Certificate Validation program."""

import sys

from studies.utils.execution import run_stage


def main() -> None:
    """Run standalone certification checks (audits)."""
    overrides: list[str] = sys.argv[1:]
    
    if "experiment_family=certificate_validation" not in overrides:
        overrides.append("experiment_family=certificate_validation")
        
    audit_cmd = ["python", "studies/utils/audit.py"] + overrides
    run_stage(audit_cmd, "Certificate Audit (State Bank)")

if __name__ == "__main__":
    main()
