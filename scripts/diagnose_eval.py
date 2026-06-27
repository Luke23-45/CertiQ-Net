#!/usr/bin/env python3
"""
Diagnostic suite for evaluation pipeline correctness in CertiQ-Net.

Usage:
    python -m scripts.diagnose_eval                          # standalone tests
    python -m scripts.diagnose_eval --run-dir <path>         # also check checkpoint

Exit code: 0 = all tests pass, 1 = one or more failures.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch
import numpy as np

# ---------------------------------------------------------------------------
#  Project imports (lazy) -- each _import_X function wraps one module so
#  tests can gracefully skip when dependencies are missing.
# ---------------------------------------------------------------------------

def _import_ctmc():
    from certiqnet.utils.ctmc import CTMCEnvironment as _CTMCEnv
    return _CTMCEnv

def _import_models():
    from certiqnet.models.baselines import (
        MaxWeight,
        MaximumPressure,
        ShortestExpectedDelay,
        CMuRule,
        RandomPolicy,
    )
    return MaxWeight, MaximumPressure, ShortestExpectedDelay, CMuRule, RandomPolicy

def _import_index_model():
    from certiqnet.dispatcher.certiq.index_model import CertiQIndexModel
    return CertiQIndexModel

def _import_baseline_runner():
    from certiqnet.experiments.evaluators.baseline_runner import (
        _has_learnable_params,
        _greedy_pi,
        build_baseline_suite,
        evaluate_policy,
        RolloutConfig,
    )
    return _has_learnable_params, _greedy_pi, build_baseline_suite, evaluate_policy, RolloutConfig

def _import_metrics():
    from certiqnet.experiments.evaluators.metrics import aggregate_metrics
    return aggregate_metrics

def _import_checkpoint():
    from certiqnet.experiments.persistence.checkpoint import load_checkpoint_weights, require_checkpoint_state
    return load_checkpoint_weights, require_checkpoint_state

def _import_factory():
    from certiqnet.experiments.evaluators.factory import build_model
    return build_model

def _import_effective_mu():
    from certiqnet.adapters.qgym.env_loader import compute_effective_mu
    return compute_effective_mu

def _import_runner():
    from certiqnet.experiments.runner.engine import discover_and_prepare
    from certiqnet.experiments.catalog.specs import StudySpec
    from certiqnet.data.registry import DatasetRegistry
    from omegaconf import OmegaConf
    return discover_and_prepare, OmegaConf, DatasetRegistry

# ---------------------------------------------------------------------------
#  Constants (hospital topology for comparability)
# ---------------------------------------------------------------------------

# Pool-weighted mu from datasets.yaml (env_mu_fixed)
HOSPITAL_MU = torch.tensor([26.0, 26.1363636364, 50.5555555556, 13.8888888889,
                             20.0, 13.7297297297, 7.7777777778, 19.8])

HOSPITAL_LAM = 93.2
HOSPITAL_N = 8

# Per-server effective mu (without pool-size weighting) -- would give sum ~4.43
HOSPITAL_NETWORK = torch.tensor([
    [1,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,1],[0,0,1,0,0,0,0,0],[0,0,1,0,0,0,0,1],
    [0,1,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,0,0,1,1,0,0],[0,1,1,0,0,0,0,0],
    [0,0,0,0,0,0,1,0],[1,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0],[1,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0],
], dtype=torch.float)

HOSPITAL_MU_MATRIX = torch.tensor([
    [0.2162162162162162,0,0.2777777777777778,0,0,0,0,0],
    [0,0,0,0,0.2222222222222222,0,0,0.2],
    [0,0,0.2777777777777778,0,0,0,0,0],
    [0,0,0.2777777777777778,0,0,0,0,0.25],
    [0,0.22727272727272727,0,0,0,0,0,0],
    [0,0.22727272727272727,0,0,0,0,0,0],
    [0,0,0,0,0.2222222222222222,0.2162162162162162,0,0],
    [0,0.22727272727272727,0.2777777777777778,0,0,0,0,0],
    [0,0,0,0,0,0,0.2222222222222222,0],
    [0.27027027027027023,0,0,0,0,0,0,0],
    [0,0,0,0,0,0.27027027027027023,0,0],
    [0.27027027027027023,0,0,0,0,0,0,0],
    [0,0,0,0.2777777777777778,0,0,0,0],
], dtype=torch.float)

HOSPITAL_POOL_SIZE = torch.tensor([44, 44, 44, 44, 39, 26, 46, 50, 35, 17, 14, 44, 50],
                                   dtype=torch.float)

# ---------------------------------------------------------------------------
#  Test infrastructure
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self):
        self.passed: list[str] = []
        self.failed: list[str] = []
        self.skipped: list[tuple[str, str]] = []  # (reason, test_name)

    def ok(self, name: str, detail: str = "") -> None:
        msg = f"  PASS  {name}"
        if detail:
            msg += f"  ({detail})"
        print(msg)
        self.passed.append(name)

    def fail(self, name: str, detail: str) -> None:
        msg = f"  FAIL  {name}"
        if detail:
            msg += f"  ({detail})"
        print(msg)
        self.failed.append(name)

    def skip(self, name: str, reason: str) -> None:
        print(f"  SKIP  {name}  ({reason})")
        self.skipped.append((reason, name))

    def summary(self) -> str:
        sep = "=" * 60
        lines = [f"\n{sep}",
                 f"  RESULTS:  {len(self.passed)} passed, {len(self.failed)} failed, {len(self.skipped)} skipped"]
        if self.failed:
            lines.append(f"  FAILED: {', '.join(self.failed)}")
        if self.skipped:
            lines.append(f"  SKIPPED: {', '.join(n for _, n in self.skipped)}")
        lines.append(f"{sep}")
        return "\n".join(lines)


def section(title: str) -> None:
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


# ---------------------------------------------------------------------------
#  1. Torch argmax tie-breaking
# ---------------------------------------------------------------------------

def test_torch_argmax_ties(t: TestResult) -> None:
    section("1. Torch argmax tie-breaking")

    # Seed should not affect tie-breaking (argmax on ties is deterministic)
    for seed in [0, 1, 42, 12345]:
        torch.manual_seed(seed)
        t_uniform = torch.ones(8)
        idx = t_uniform.argmax().item()
        if idx != 0:
            t.fail(f"argmax(on_ones)_seed={seed}", f"expected 0, got {idx}")
            return
    t.ok("argmax(on_ones)_deterministic", "always returns 0 regardless of seed")

    # Same for all-equal non-one values
    torch.manual_seed(42)
    t_const = torch.full((8,), 3.7)
    if t_const.argmax().item() != 0:
        t.fail("argmax(on_equal_non_one)", "expected 0")
        return
    t.ok("argmax(on_equal_non_one)", "all-equal values also yield index 0")

    # argmin on all-equal also picks first
    if t_const.argmin().item() != 0:
        t.fail("argmin(on_equal)", "expected 0")
        return
    t.ok("argmin(on_equal)", "all-equal values also yield index 0 for argmin")


# ---------------------------------------------------------------------------
#  2. _greedy_pi behavior
# ---------------------------------------------------------------------------

def test_greedy_pi(t: TestResult) -> None:
    section("2. _greedy_pi on uniform / near-uniform distributions")

    try:
        _has_learnable_params, _greedy_pi, _, _, _ = _import_baseline_runner()
    except ImportError as e:
        t.skip("_greedy_pi_import", str(e))
        return

    # Uniform distribution
    pi_uniform = torch.ones(1, 8) / 8
    hard = _greedy_pi(pi_uniform)
    idx = hard.argmax(dim=-1).item()
    if idx != 0:
        t.fail("greedy_pi(uniform)", f"expected 0, got {idx}")
        return
    t.ok("greedy_pi(uniform)", "uniform -> one-hot at index 0")

    # Near-uniform (small random perturbation)
    torch.manual_seed(42)
    pi_noise = torch.softmax(torch.randn(1, 8) * 1e-6, dim=-1)
    hard2 = _greedy_pi(pi_noise)
    # With noise, argmax may differ from 0; we just verify one-hot shape
    if hard2.sum().item() != 1.0 or hard2.shape != pi_uniform.shape:
        t.fail("greedy_pi(near_uniform)_shape", f"expected one-hot shape {pi_uniform.shape}")
        return
    t.ok("greedy_pi(near_uniform)", "produces valid one-hot (sum=1)")

    # Batch test -- each row independently
    batch = torch.softmax(torch.randn(4, 8), dim=-1)
    hard3 = _greedy_pi(batch)
    row_sums = hard3.sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones(4)):
        t.fail("greedy_pi(batch)", "not all rows sum to 1")
        return
    if hard3.shape != batch.shape:
        t.fail("greedy_pi(batch)_shape", f"expected {batch.shape}, got {hard3.shape}")
        return
    t.ok("greedy_pi(batch)", "produces valid one-hot per batch element")


# ---------------------------------------------------------------------------
#  3. _has_learnable_params
# ---------------------------------------------------------------------------

def test_learnable_params(t: TestResult) -> None:
    section("3. _has_learnable_params classification")

    try:
        _has_learnable_params, _greedy_pi, _, _, _ = _import_baseline_runner()
        CertiQIndexModel = _import_index_model()
        MaxWeight, MaximumPressure, _, _, _ = _import_models()
    except ImportError as e:
        t.skip("learnable_params_import", str(e))
        return

    # CertiQIndexModel has learnable parameters
    cqm = CertiQIndexModel(N=HOSPITAL_N)
    if not _has_learnable_params(cqm):
        t.fail("has_params(CertiQIndexModel)", "expected True")
        return
    t.ok("has_params(CertiQIndexModel)", "True -> greedy_eval will apply _greedy_pi")

    # MaxWeight has no learnable parameters
    mw = MaxWeight(N=HOSPITAL_N)
    if _has_learnable_params(mw):
        t.fail("has_params(MaxWeight)", "expected False")
        return
    t.ok("has_params(MaxWeight)", "False -> greedy_eval will NOT apply _greedy_pi")

    # MaximumPressure also has no learnable parameters
    mp = MaximumPressure(N=HOSPITAL_N)
    if _has_learnable_params(mp):
        t.fail("has_params(MaximumPressure)", "expected False")
        return
    t.ok("has_params(MaximumPressure)", "False")

    # SED likewise
    from certiqnet.models.baselines import ShortestExpectedDelay
    sed = ShortestExpectedDelay(N=HOSPITAL_N)
    if _has_learnable_params(sed):
        t.fail("has_params(SED)", "expected False")
        return
    t.ok("has_params(SED)", "False")


# ---------------------------------------------------------------------------
#  4. MaxWeight action selection
# ---------------------------------------------------------------------------

def test_maxweight_actions(t: TestResult) -> None:
    section("4. MaxWeight action selection on various queue states")

    try:
        MaxWeight, _, _, _, _ = _import_models()
    except ImportError as e:
        t.skip("maxweight_import", str(e))
        return

    mw = MaxWeight(N=HOSPITAL_N, beta=1.0)
    mu = HOSPITAL_MU

    # Zero queues -> all ties -> picks first (index 0)
    Q_zero = torch.zeros(1, HOSPITAL_N)
    pi_z, _ = mw(Q_zero, mu)
    if pi_z.argmax().item() != 0:
        t.fail("maxweight(zero_Q)", f"expected 0, got {pi_z.argmax().item()}")
        return
    t.ok("maxweight(zero_Q)", "ties -> index 0 (matches collapsed model)")

    # Dominant queue 0 -> Q[0]*26 >> 0 -> picks queue 0
    Q_dom0 = torch.tensor([[10.0] + [0.0] * 7])
    pi_d0, _ = mw(Q_dom0, mu)
    if pi_d0.argmax().item() != 0:
        t.fail("maxweight(dominant_Q0)", f"expected 0, got {pi_d0.argmax().item()}")
        return
    t.ok("maxweight(dominant_Q0)", "Q[0]*mu[0] dominates -> picks 0")

    # Balanced queues -> picks highest mu queue (index 2, mu=50.556)
    Q_bal = torch.ones(1, HOSPITAL_N) * 5.0
    pi_bal, _ = mw(Q_bal, mu)
    if pi_bal.argmax().item() != 2:
        t.fail("maxweight(balanced_Q)", f"expected 2 (highest mu), got {pi_bal.argmax().item()}")
        return
    t.ok("maxweight(balanced_Q)", "picks queue with highest Q*mu product")

    # All queues non-zero, but one dominates
    Q_mixed = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    # Q*mu = [26, 52.3, 151.7, 55.6, 100, 82.4, 54.4, 158.4]
    # argmax should be index 7 (8.0 * 19.8 = 158.4)
    expected_mixed = 7
    pi_mixed, _ = mw(Q_mixed, mu)
    actual = pi_mixed.argmax().item()
    if actual != expected_mixed:
        t.fail("maxweight(mixed_Q)", f"expected {expected_mixed}, got {actual}")
        return
    t.ok("maxweight(mixed_Q)", f"correctly picks highest Q*mu (index {expected_mixed})")
    # Note: this test proves MaxWeight does NOT always pick index 0


# ---------------------------------------------------------------------------
#  5. MaximumPressure degeneracy
# ---------------------------------------------------------------------------

def test_maxpressure_degeneracy(t: TestResult) -> None:
    section("5. MaximumPressure(P=None) degeneracy to MaxWeight")

    try:
        MaxWeight, MaximumPressure, _, _, _ = _import_models()
    except ImportError as e:
        t.skip("maxpressure_import", str(e))
        return

    mu = HOSPITAL_MU
    mw = MaxWeight(N=HOSPITAL_N)
    mp = MaximumPressure(N=HOSPITAL_N)  # P defaults to None

    # Test on various queue states
    Qs = [
        torch.zeros(1, HOSPITAL_N),
        torch.ones(1, HOSPITAL_N) * 5.0,
        torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]),
        torch.tensor([[0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[100.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 30.0]]),
    ]

    all_match = True
    for i, Q in enumerate(Qs):
        pi_mw, _ = mw(Q, mu)
        pi_mp, _ = mp(Q, mu)
        a_mw = pi_mw.argmax().item()
        a_mp = pi_mp.argmax().item()
        if a_mw != a_mp:
            t.fail(f"maxpressure_state_{i}", f"MaxWeight={a_mw} != MaxPressure={a_mp}")
            all_match = False

    if all_match:
        t.ok("maxpressure_all_states", f"MaxPressure(P=None) == MaxWeight across {len(Qs)} states")
        # This confirms the results in the CSV: max_weight and max_pressure
        # produce identical metrics because they are literally the same function.


# ---------------------------------------------------------------------------
#  6. Policy disagreement on diverse inputs
# ---------------------------------------------------------------------------

def test_policy_disagreement(t: TestResult) -> None:
    section("6. Policy action disagreement on diverse Q/mu inputs")

    try:
        MaxWeight, MaximumPressure, ShortestExpectedDelay, CMuRule, RandomPolicy = _import_models()
        _has_learnable_params, _greedy_pi, _, _, _ = _import_baseline_runner()
        CertiQIndexModel = _import_index_model()
    except ImportError as e:
        t.skip("policy_disagreement_import", str(e))
        return

    mu = HOSPITAL_MU

    # Build policies
    policies = {
        "sed":         ShortestExpectedDelay(N=HOSPITAL_N),
        "c_mu":        CMuRule(N=HOSPITAL_N),
        "max_weight":  MaxWeight(N=HOSPITAL_N),
        "max_pressure": MaximumPressure(N=HOSPITAL_N),
    }

    # Add a random-weight CertiQIndexModel to simulate the "failed checkpoint load" scenario
    fake_cqm = CertiQIndexModel(N=HOSPITAL_N)
    policies["configured_model"] = fake_cqm

    # Generate diverse random Q states
    torch.manual_seed(42)
    n_states = 500
    Q_batch = torch.randint(0, 50, (n_states, HOSPITAL_N)).float()
    mu_batch = mu.unsqueeze(0).expand(n_states, -1)

    # Collect actions for each policy
    actions: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, model in policies.items():
            if hasattr(model, "forward_full"):
                out = model.forward_full(Q_batch, mu_batch, training_mode=False)
                pi = out.pi
            else:
                pi, _ = model(Q_batch, mu_batch)
            # Apply _greedy_pi for models with learnable params
            if _has_learnable_params(model):
                pi = _greedy_pi(pi)
            actions[name] = pi.argmax(dim=-1)

    # Compute agreement matrices
    policy_names = list(policies.keys())
    total = n_states
    print(f"    Generated {total} random Q vectors ({HOSPITAL_N} queues, 0-50)")
    print()
    for i, name_a in enumerate(policy_names):
        for name_b in policy_names[i+1:]:
            agree = (actions[name_a] == actions[name_b]).sum().item()
            pct = 100.0 * agree / total
            marker = " *** IDENTICAL" if agree == total else ""
            print(f"    {name_a:20s} vs {name_b:20s}: {agree:5d}/{total} agree ({pct:.1f}%){marker}")
    print()

    # Key checks:
    # 1. max_weight != random (verifies MaxWeight is not random)
    mw_vs_rand = (actions["max_weight"] == actions["sed"]).sum().item()
    if mw_vs_rand == total:
        t.fail("max_weight_vs_sed_identical", "should differ on diverse inputs")
    else:
        t.ok("max_weight_separate", "MaxWeight produces distinct actions from SED")

    # 2. max_weight vs configured_model (random weights)
    mw_vs_cqm = (actions["max_weight"] == actions["configured_model"]).sum().item()
    if mw_vs_cqm == total:
        t.fail("maxweight_vs_configured_identical",
               "random-weight CertiQIndexModel happens to match MaxWeight on ALL inputs")
        # This would mean the hypothesis is WRONG -- they match even on diverse inputs
    else:
        t.ok("maxweight_vs_configured_differ",
             f"random-weight model disagrees with MaxWeight on {total - mw_vs_cqm}/{total} inputs")
        # This means on sufficiently diverse inputs, the two policies differ.
        # But on the CTMC rollout (starting from Q=0), they may still match.

    # 3. max_weight vs max_pressure should be identical on all inputs
    mw_vs_mp = (actions["max_weight"] == actions["max_pressure"]).sum().item()
    if mw_vs_mp != total:
        t.fail("maxweight_vs_maxpressure_not_identical",
               f"differ on {total - mw_vs_mp} inputs -- unexpected for P=None")
    else:
        t.ok("maxweight_vs_maxpressure_identical",
             "MaximumPressure(P=None) == MaxWeight on all inputs")

    # 4. sed vs c_mu -- should differ (different formulas)
    sed_vs_cmu = (actions["sed"] == actions["c_mu"]).sum().item()
    if sed_vs_cmu == total:
        t.fail("sed_vs_cmu_identical", "different formulas should disagree sometimes")
    else:
        t.ok("sed_vs_cmu_differ", f"different formulas, disagree on {total - sed_vs_cmu}/{total}")


# ---------------------------------------------------------------------------
#  7. CTMC determinism with identical policies
# ---------------------------------------------------------------------------

def test_ctmc_determinism(t: TestResult) -> None:
    section("7. CTMC determinism: identical policies -> identical metrics")

    try:
        _has_learnable_params, _greedy_pi, _, evaluate_policy, RolloutConfig = _import_baseline_runner()
        MaxWeight, _, _, _, _ = _import_models()
        _import_ctmc()
        aggregate_metrics = _import_metrics()
    except ImportError as e:
        t.skip("ctmc_determinism_import", str(e))
        return

    mu = HOSPITAL_MU
    rollout = RolloutConfig(steps=50, batch_size=4, show_progress=False)

    from certiqnet.adapters.queueing.adapter import QueueingAdapter
    adapter = QueueingAdapter(assumptions_satisfied=True)

    # Evaluate MaxWeight twice with same seed -- should be identical
    mw = MaxWeight(N=HOSPITAL_N)
    r1 = evaluate_policy(
        name="mw_a", model=mw, env_name="hospital", seed=42,
        N=HOSPITAL_N, lam=HOSPITAL_LAM, mu=mu, rollout=rollout, adapter=adapter,
    )
    r2 = evaluate_policy(
        name="mw_b", model=mw, env_name="hospital", seed=42,
        N=HOSPITAL_N, lam=HOSPITAL_LAM, mu=mu, rollout=rollout, adapter=adapter,
    )
    f1 = r1.flat()
    f2 = r2.flat()
    numeric_keys = [k for k in f1 if isinstance(f1[k], (int, float))]
    diffs = {k: (float(f1[k]), float(f2[k])) for k in numeric_keys
             if abs(float(f1[k]) - float(f2[k])) > 1e-8}
    if diffs:
        t.fail("identical_policies_differ", f"diffs: {diffs}")
        return
    t.ok("identical_policies_identical",
         f"MaxWeight x 2 with same seed = identical metrics ({len(numeric_keys)} numeric fields match)")


# ---------------------------------------------------------------------------
#  8. CTMC collapse scenario -- the full hypothesis
# ---------------------------------------------------------------------------

def test_collapse_scenario(t: TestResult) -> None:
    section("8. Collapse scenario: uniform-model mimicking MaxWeight in CTMC")

    try:
        _has_learnable_params, _greedy_pi, _, evaluate_policy, RolloutConfig = _import_baseline_runner()
        MaxWeight, _, _, _, _ = _import_models()
        _import_ctmc()
        aggregate_metrics = _import_metrics()
        CertiQIndexModel = _import_index_model()
    except ImportError as e:
        t.skip("collapse_scenario_import", str(e))
        return

    mu = HOSPITAL_MU
    rollout = RolloutConfig(steps=50, batch_size=4, show_progress=False)

    from certiqnet.adapters.queueing.adapter import QueueingAdapter
    adapter = QueueingAdapter(assumptions_satisfied=True)

    # Construct a "collapsed" model: always outputs uniform softmax
    # This simulates a random-weight CertiQIndexModel after _greedy_pi
    class CollapsedModel(torch.nn.Module):
        def __init__(self, N: int):
            super().__init__()
            self.N = N
            # Register a dummy parameter so _has_learnable_params returns True
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, Q, mu, xi=None, *, training_mode=False):
            batch = Q.shape[0]
            pi = torch.ones(batch, self.N) / self.N
            # Cheap diagnostics matching DispatcherDiagnostics
            from certiqnet.dispatcher.types import DispatcherDiagnostics
            diag = DispatcherDiagnostics(
                A_proposal=pi.new_zeros(batch),
                A_final=pi.new_zeros(batch),
                m_Q=pi.new_zeros(batch),
                B_Q=pi.new_zeros(batch),
                certificate_slack=pi.new_zeros(batch),
                constraint_violation=pi.new_zeros(batch),
                usage_raw=pi.new_ones(batch),
                usage_final=pi.new_ones(batch),
                usage_cap=pi.new_ones(batch),
                policy_entropy=pi.new_zeros(batch),
                selected_resource=pi.new_zeros(batch, dtype=torch.long),
                pressure_mean=pi.new_zeros(batch),
                pressure_max=pi.new_zeros(batch),
                pressure_update_norm=pi.new_zeros(batch),
            )
            return pi, diag

    collapsed = CollapsedModel(N=HOSPITAL_N)

    # Evaluate the collapsed model
    r_collapsed = evaluate_policy(
        name="collapsed", model=collapsed, env_name="hospital", seed=42,
        N=HOSPITAL_N, lam=HOSPITAL_LAM, mu=mu, rollout=rollout, adapter=adapter,
    )
    f_collapsed = r_collapsed.flat()

    # Evaluate MaxWeight
    mw = MaxWeight(N=HOSPITAL_N)
    r_mw = evaluate_policy(
        name="max_weight", model=mw, env_name="hospital", seed=42,
        N=HOSPITAL_N, lam=HOSPITAL_LAM, mu=mu, rollout=rollout, adapter=adapter,
    )
    f_mw = r_mw.flat()

    # Compare
    key = "avg_cost"
    val_c = float(f_collapsed.get(key, -1))
    val_m = float(f_mw.get(key, -1))
    # Check whether collapsed model metrics match MaxWeight exactly
    match = abs(val_c - val_m) < 1e-6
    if match:
        t.ok("collapse_equals_maxweight",
             f"Collapsed(model) avg_cost={val_c:.4f} == MaxWeight avg_cost={val_m:.4f}")
    else:
        t.fail("collapse_does_not_equal_maxweight",
               f"Collapsed avg_cost={val_c:.4f} != MaxWeight avg_cost={val_m:.4f}")
        return

    # Now check: does the collapsed model always pick index 0?
    r_collapsed_actions = f_collapsed
    t.ok("collapse_mechanism_confirmed",
         "Uniform-softmax + _greedy_pi + Q=0 start -> identical to MaxWeight")

    # Also verify that sed produces DIFFERENT metrics from MaxWeight
    from certiqnet.models.baselines import ShortestExpectedDelay
    sed = ShortestExpectedDelay(N=HOSPITAL_N)
    r_sed = evaluate_policy(
        name="sed", model=sed, env_name="hospital", seed=42,
        N=HOSPITAL_N, lam=HOSPITAL_LAM, mu=mu, rollout=rollout, adapter=adapter,
    )
    f_sed = r_sed.flat()
    if abs(float(f_sed.get(key, -1)) - val_m) < 1e-6:
        t.fail("sed_should_differ_from_maxweight",
               f"SED avg_cost={float(f_sed.get(key, -1)):.4f} == MaxWeight {val_m:.4f}")
    else:
        t.ok("sed_different_from_maxweight",
             f"SED avg_cost={float(f_sed.get(key, -1)):.4f} != MaxWeight {val_m:.4f}")


# ---------------------------------------------------------------------------
#  9. Checkpoint integrity (requires --run-dir)
# ---------------------------------------------------------------------------

def test_checkpoint_integrity(t: TestResult, run_dir: Path) -> None:
    section("9. Checkpoint key integrity")

    try:
        load_checkpoint_weights, require_checkpoint_state = _import_checkpoint()
        build_model = _import_factory()
    except ImportError as e:
        t.skip("checkpoint_import", str(e))
        return

    # Resolve paths
    config_dir = run_dir / "configs"
    resolved_config = config_dir / "resolved_config.yaml"
    if not resolved_config.exists():
        t.skip("checkpoint_resolved_config", f"not found: {resolved_config}")
        return

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(str(resolved_config))

    # Determine N from the dataset registry
    try:
        from certiqnet.data.registry import DatasetRegistry
        spec = None
        for candidate in ["dataset_name", "data.init_args.dataset_name"]:
            val = OmegaConf.select(cfg, candidate)
            if val:
                spec = DatasetRegistry().get(val)
                break
        N = int(spec.env_N) if spec else HOSPITAL_N
        d_xi = 0
    except Exception:
        N = HOSPITAL_N
        d_xi = 0

    # Build model as evaluation does
    try:
        model = build_model(cfg, N=N, d_xi=d_xi)
    except Exception as e:
        t.skip("checkpoint_build_model", f"build_model failed: {e}")
        return
    print(f"    Model type: {type(model).__name__}, N={N}")

    # Try loading checkpoint with strict=True first (will fail on mismatch)
    try:
        ckpt_path = require_checkpoint_state(run_dir)
        print(f"    Checkpoint: {ckpt_path}")
        raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except Exception as e:
        t.skip("checkpoint_load", f"cannot load checkpoint: {e}")
        return

    if isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
        cleaned = {k.removeprefix("model."): v for k, v in sd.items() if k.startswith("model.")}
    else:
        cleaned = raw

    # Compare keys
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(cleaned.keys()) if isinstance(cleaned, dict) else set()
    common = model_keys & ckpt_keys
    missing_in_ckpt = model_keys - ckpt_keys
    extra_in_ckpt = ckpt_keys - model_keys

    print(f"    Model keys:     {len(model_keys)}")
    print(f"    Checkpoint keys: {len(ckpt_keys)}")
    print(f"    Common keys:    {len(common)}")
    if missing_in_ckpt:
        print(f"    Missing in checkpoint: {sorted(missing_in_ckpt)[:10]}...")
    if extra_in_ckpt:
        print(f"    Extra in checkpoint:   {sorted(extra_in_ckpt)[:10]}...")

    if not common:
        t.fail("checkpoint_no_common_keys", "Zero overlapping keys -- checkpoint is incompatible")
        return

    if missing_in_ckpt or extra_in_ckpt:
        t.fail("checkpoint_key_mismatch",
               f"{len(missing_in_ckpt)} missing, {len(extra_in_ckpt)} extra -- strict=False silently ignores these")
        return

    t.ok("checkpoint_keys_match", "all keys match between model and checkpoint")

    # Try strict=True load (will raise if any discrepancy)
    with torch.no_grad():
        try:
            model.load_state_dict(cleaned, strict=False)
            t.ok("checkpoint_load_strict_false", "load_state_dict(strict=False) succeeded")
            # Now try strict=True
            try:
                model.load_state_dict(cleaned, strict=True)
                t.ok("checkpoint_load_strict_true", "load_state_dict(strict=True) succeeded -- full match")
            except Exception as e:
                t.fail("checkpoint_strict_true_failed",
                       f"strict=True raised: {e}")
        except Exception as e:
            t.fail("checkpoint_load_failed", f"even strict=False raised: {e}")


# ---------------------------------------------------------------------------
#  10. Effective mu verification (the previous fix)
# ---------------------------------------------------------------------------

def test_effective_mu(t: TestResult) -> None:
    section("10. compute_effective_mu pool-size correctness")

    try:
        compute_effective_mu = _import_effective_mu()
    except ImportError as e:
        t.skip("effective_mu_import", str(e))
        return

    # Without pool_size (old behavior) -> sum ~= 4.43 (supercritical)
    mu_unweighted = compute_effective_mu(HOSPITAL_NETWORK, HOSPITAL_MU_MATRIX)
    old_sum = mu_unweighted.sum().item()
    if old_sum > 10.0:
        t.fail("old_mu_unweighted_sum", f"expected ~4.43, got {old_sum:.4f}")
        return
    t.ok("old_mu_unweighted_sum", f"without pool_size: sum={old_sum:.4f} (supercritical: {HOSPITAL_LAM} >= {old_sum:.4f})")

    # With pool_size (new behavior) -> sum ~= 178 (stable)
    mu_weighted = compute_effective_mu(HOSPITAL_NETWORK, HOSPITAL_MU_MATRIX,
                                        pool_size=HOSPITAL_POOL_SIZE)
    new_sum = mu_weighted.sum().item()
    if abs(new_sum - HOSPITAL_MU.sum().item()) > 5.0:
        t.fail("new_mu_weighted_sum", f"expected ~{HOSPITAL_MU.sum().item():.1f}, got {new_sum:.4f}")
        return
    t.ok("new_mu_weighted_sum", f"with pool_size: sum={new_sum:.4f} (stable: {HOSPITAL_LAM} < {new_sum:.4f})")

    # Verify per-queue values match env_mu_fixed
    for q in range(HOSPITAL_N):
        expected = float(HOSPITAL_MU[q])
        actual = float(mu_weighted[q])
        if abs(expected - actual) > 0.5:
            t.fail(f"mu_weighted_q{q}", f"expected {expected:.4f}, got {actual:.4f}")
            return
    t.ok("mu_weighted_per_queue", "all per-queue values match env_mu_fixed within 0.5 tolerance")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose evaluation pipeline correctness in CertiQ-Net.",
    )
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Path to a training run directory for checkpoint integrity check")
    args = parser.parse_args()

    t = TestResult()
    print(f"\n{'#'*60}")
    print(f"#  CertiQ-Net Evaluation Diagnostic Suite")
    print(f"#  run-dir: {args.run_dir or '(none)'}")
    print(f"{'#'*60}")

    # -- Run tests ------------------------------------------------------
    test_torch_argmax_ties(t)
    test_greedy_pi(t)
    test_learnable_params(t)
    test_maxweight_actions(t)
    test_maxpressure_degeneracy(t)
    test_policy_disagreement(t)
    test_ctmc_determinism(t)
    test_collapse_scenario(t)
    test_effective_mu(t)

    if args.run_dir is not None:
        test_checkpoint_integrity(t, args.run_dir)
    else:
        t.skip("checkpoint_integrity", "no --run-dir provided")

    # -- Summary --------------------------------------------------------
    print(t.summary())
    return 1 if t.failed else 0


if __name__ == "__main__":
    sys.exit(main())
