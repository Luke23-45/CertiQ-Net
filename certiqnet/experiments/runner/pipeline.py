"""
Compatibility shim — stage functions have moved to their canonical locations:
  - run_training               → certiqnet.train.runner
  - run_state_bank_audit        → certiqnet.eval.audit
  - run_baseline_paper_comparison → certiqnet.eval.baselines
"""

from certiqnet.eval.audit import run_state_bank_audit
from certiqnet.eval.baselines import run_baseline_paper_comparison
from certiqnet.train.runner import run_training

__all__ = [
    "run_training",
    "run_state_bank_audit",
    "run_baseline_paper_comparison",
]
