"""Audit a collected QGym dataset for integrity, quality, and completeness.

Usage:
    python -m certiqnet.data.qgym.audit_qgym <dataset_name> [--verbose] [--output report.json]
    python -m certiqnet.data.qgym.audit_qgym <dataset_name> --n-steps 50000

Checks performed:
  1. Schema integrity   — every shard has required keys, no unexpected absent keys
  2. Cross-shard consistency — mu/h/N are identical across all shards and splits
  3. Data range validity — no NaN/Inf, Q >= 0, mu > 0, cost >= 0
  4. Split integrity    — state counts match registry spec, shard sizes are sane
  5. Statistical profile — per-split Q/cost distribution, compared to metadata.yaml
  6. Hardness analysis  — fraction of states above backlog thresholds
  7. Spec alignment     — N, policy, seed match registry; env config is resolvable
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from certiqnet.adapters.qgym.config import QGymEnvConfig, resolve_env_config_path
from certiqnet.data.qgym.dataset import QGymDataset
from certiqnet.data.registry import DatasetRegistry

# ---------------------------------------------------------------------------
#  Thresholds
# ---------------------------------------------------------------------------

_BACKLOG_THRESHOLDS = [50, 100, 500, 1000, 2000, 5000]
"""Queue-length sum thresholds (inclusive) for hardness classification."""

_SHARD_KEY_REQUIRED = frozenset({"Q", "cost"})
"""Keys every shard must have (matches QGymDataset._REQUIRED_SHARD_KEYS)."""

_SHARD_KEY_OPTIONAL = frozenset({"mu", "h", "network", "mu_matrix", "env_config"})
"""Keys that may or may not be present."""

# ---------------------------------------------------------------------------
#  Audit data structures
# ---------------------------------------------------------------------------


@dataclass
class AuditCheck:
    """Result of one audit check sub-component."""

    passed: bool = True
    details: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class SplitAudit:
    """Per-split audit results."""

    name: str
    n_states: int = 0
    n_shards: int = 0
    N: int = 0
    schema: AuditCheck = field(default_factory=AuditCheck)
    cross_shard: AuditCheck = field(default_factory=AuditCheck)
    ranges: AuditCheck = field(default_factory=AuditCheck)
    stats: AuditCheck = field(default_factory=AuditCheck)
    hardness: AuditCheck = field(default_factory=AuditCheck)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _load_all_shards(dataset_dir: Path, split: str) -> list[dict]:
    """Load all shard dicts for a split, sorted by filename."""
    split_dir = dataset_dir / split
    if not split_dir.exists():
        return []
    shard_files = sorted(split_dir.glob("*.pt"))
    return [torch.load(f, weights_only=True) for f in shard_files]


def _read_metadata(dataset_dir: Path) -> dict | None:
    """Read metadata.yaml if it exists."""
    meta_path = dataset_dir / "metadata.yaml"
    if meta_path.exists():
        with open(meta_path) as f:
            return yaml.safe_load(f)
    return None


def _read_env_config(spec_env: str) -> QGymEnvConfig | None:
    """Resolve and load the QGym env config from a spec's env path."""
    try:
        resolved = resolve_env_config_path(spec_env)
        return QGymEnvConfig.from_yaml(resolved)
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  Core audit functions
# ---------------------------------------------------------------------------


def _audit_schema(shards: list[dict], split_name: str) -> AuditCheck:
    """Check that every shard has the required keys and optional keys are consistent."""
    check = AuditCheck()

    if not shards:
        check.passed = True
        check.details["n_shards"] = 0
        return check

    check.details["n_shards"] = len(shards)

    all_keys_list = [set(d.keys()) for d in shards]
    common_keys = set.intersection(*all_keys_list) if all_keys_list else set()

    missing_required = []
    for i, keys in enumerate(all_keys_list):
        missing = _SHARD_KEY_REQUIRED - keys
        if missing:
            missing_required.append(f"shard_{i}: missing {sorted(missing)}")

    if missing_required:
        check.passed = False
        check.errors.extend(missing_required)

    # Warn about optional keys that are present only in some shards
    for opt_key in _SHARD_KEY_OPTIONAL:
        present_in = [i for i, keys in enumerate(all_keys_list) if opt_key in keys]
        if 0 < len(present_in) < len(shards):
            check.warnings.append(
                f"Optional key '{opt_key}' present in {len(present_in)}/{len(shards)} "
                f"shards: {present_in}"
            )

    check.details["common_keys"] = sorted(common_keys)
    check.details["all_have_required"] = not missing_required

    # Validate Q and cost shapes in each shard
    for i, d in enumerate(shards):
        Q = d.get("Q")
        cost = d.get("cost")
        if Q is not None and Q.dim() != 2:
            check.passed = False
            check.errors.append(f"shard_{i}: Q has shape {tuple(Q.shape)}, expected 2-D (B, N)")
        if cost is not None and cost.dim() != 1:
            check.passed = False
            check.errors.append(f"shard_{i}: cost has shape {tuple(cost.shape)}, expected 1-D (B,)")
        if Q is not None and cost is not None and Q.shape[0] != cost.shape[0]:
            check.passed = False
            check.errors.append(
                f"shard_{i}: Q batch size {Q.shape[0]} != cost batch size {cost.shape[0]}"
            )

    return check


def _audit_cross_shard_consistency(shards: list[dict], split_name: str) -> AuditCheck:
    """Check mu, h, and N are consistent across all shards in this split."""
    check = AuditCheck()

    if len(shards) < 2:
        check.details["n_shards"] = len(shards)
        return check

    check.details["n_shards"] = len(shards)
    ref_N = shards[0]["Q"].shape[-1] if "Q" in shards[0] else None

    mu_ref = shards[0].get("mu")
    h_ref = shards[0].get("h")

    for i, d in enumerate(shards[1:], start=1):
        if ref_N is not None and "Q" in d:
            N_i = d["Q"].shape[-1]
            if N_i != ref_N:
                check.passed = False
                check.errors.append(f"shard_{i}: N={N_i} differs from shard_0 N={ref_N}")

        mu_i = d.get("mu")
        if mu_ref is not None and mu_i is not None and not torch.equal(mu_ref, mu_i):
            check.passed = False
            check.errors.append(f"shard_{i}: mu differs from shard_0")

        h_i = d.get("h")
        if h_ref is not None and h_i is not None and not torch.equal(h_ref, h_i):
            check.warnings.append(f"shard_{i}: h differs from shard_0 (may be intentional)")

    check.details["N"] = ref_N
    check.details["mu"] = mu_ref.tolist() if mu_ref is not None else None
    check.details["h"] = h_ref.tolist() if h_ref is not None else None

    return check


def _audit_ranges(shards: list[dict], split_name: str) -> AuditCheck:
    """Check data ranges: no NaN/Inf, Q >= 0, mu > 0, cost >= 0."""
    check = AuditCheck()

    has_nan_inf_Q = False
    has_nan_inf_cost = False
    has_negative_Q = False
    has_negative_cost = False
    has_non_positive_mu = False
    total_states = 0

    for i, d in enumerate(shards):
        Q = d.get("Q")
        cost = d.get("cost")
        mu = d.get("mu")

        if Q is not None:
            total_states += Q.shape[0]
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                has_nan_inf_Q = True
                check.errors.append(f"shard_{i}: Q contains NaN or Inf values")
            if (Q < 0).any():
                has_negative_Q = True
                check.errors.append(f"shard_{i}: Q contains negative values")

        if cost is not None:
            if torch.isnan(cost).any() or torch.isinf(cost).any():
                has_nan_inf_cost = True
                check.errors.append(f"shard_{i}: cost contains NaN or Inf values")
            if (cost < 0).any():
                has_negative_cost = True
                check.errors.append(f"shard_{i}: cost contains negative values")

        if mu is not None:
            if (mu <= 0).any():
                has_non_positive_mu = True
                check.errors.append(f"shard_{i}: mu contains non-positive values (mu <= 0)")

    check.passed = not (
        has_nan_inf_Q or has_nan_inf_cost or has_negative_Q or has_negative_cost or has_non_positive_mu
    )
    check.details["total_states"] = total_states
    check.details["has_nan_inf_Q"] = has_nan_inf_Q
    check.details["has_nan_inf_cost"] = has_nan_inf_cost
    check.details["has_negative_Q"] = has_negative_Q
    check.details["has_negative_cost"] = has_negative_cost
    check.details["has_non_positive_mu"] = has_non_positive_mu

    return check


def _audit_stats(shards: list[dict], split_name: str) -> AuditCheck:
    """Compute statistical profile of Q and cost for this split."""
    check = AuditCheck()

    Q_all = []
    cost_all = []

    for d in shards:
        if "Q" in d:
            Q_all.append(d["Q"])
        if "cost" in d:
            cost_all.append(d["cost"])

    if not Q_all:
        check.warnings.append("No Q data found in split")
        return check

    Q = torch.cat(Q_all, dim=0)
    cost = torch.cat(cost_all, dim=0) if cost_all else torch.zeros(Q.shape[0])

    Q_sum = Q.sum(dim=-1)

    # Compute percentiles robustly
    def _percentile(t: torch.Tensor, p: float) -> float:
        if t.numel() == 0:
            return 0.0
        k = int(math.ceil(p / 100.0 * t.numel())) - 1
        k = max(0, min(k, t.numel() - 1))
        return float(t.flatten().kthvalue(k + 1).values.item())

    percentiles = [1, 5, 25, 50, 75, 95, 99, 99.9]
    Q_percentiles = {f"P{p}": round(_percentile(Q, p), 2) for p in percentiles}
    Qsum_percentiles = {f"P{p}": round(_percentile(Q_sum, p), 2) for p in percentiles}
    cost_percentiles = {f"P{p}": round(_percentile(cost, p), 2) for p in percentiles}

    stats = {
        "N": Q.shape[-1],
        "n_states": Q.shape[0],
        "Q": {
            "min": round(float(Q.min().item()), 2),
            "max": round(float(Q.max().item()), 2),
            "mean": round(float(Q.mean().item()), 4),
            "std": round(float(Q.std().item()), 4),
            "percentiles": Q_percentiles,
        },
        "Q_sum": {
            "min": round(float(Q_sum.min().item()), 2),
            "max": round(float(Q_sum.max().item()), 2),
            "mean": round(float(Q_sum.mean().item()), 4),
            "std": round(float(Q_sum.std().item()), 4),
            "percentiles": Qsum_percentiles,
        },
        "cost": {
            "min": round(float(cost.min().item()), 2),
            "max": round(float(cost.max().item()), 2),
            "mean": round(float(cost.mean().item()), 4),
            "std": round(float(cost.std().item()), 4),
            "percentiles": cost_percentiles,
        },
    }
    check.details = stats
    return check


def _audit_hardness(shards: list[dict], split_name: str) -> AuditCheck:
    """Classify states by backlog thresholds."""
    check = AuditCheck()

    Q_all = []
    for d in shards:
        if "Q" in d:
            Q_all.append(d["Q"])

    if not Q_all:
        check.warnings.append("No Q data found for hardness analysis")
        return check

    Q = torch.cat(Q_all, dim=0)
    Q_sum = Q.sum(dim=-1)
    n = Q.shape[0]

    threshold_counts = {}
    for thresh in _BACKLOG_THRESHOLDS:
        count = int((Q_sum >= thresh).sum().item())
        threshold_counts[str(thresh)] = {
            "count": count,
            "fraction": round(count / n, 4) if n > 0 else 0.0,
        }

    # Per-queue max utilization analysis: fraction of states where any queue > threshold
    per_queue_high = {}
    for thresh in [10, 50, 100]:
        count = int((Q > thresh).any(dim=-1).sum().item())
        per_queue_high[str(thresh)] = {
            "count": count,
            "fraction": round(count / n, 4) if n > 0 else 0.0,
        }

    check.details = {
        "n_total_states": n,
        "backlog_thresholds": threshold_counts,
        "any_queue_above": per_queue_high,
    }
    return check


def _audit_split_integrity(
    dataset_dir: Path,
    registry_spec,
    split_names: list[str],
    shard_data: dict[str, list[dict]],
) -> AuditCheck:
    """Check that split structure matches registry expectations."""
    check = AuditCheck()

    expected_counts = {
        "train": registry_spec.collection.n_steps,
        "valid": registry_spec.collection.n_valid,
        "test": registry_spec.collection.n_test,
    }

    meta = _read_metadata(dataset_dir)
    check.details["has_metadata"] = meta is not None

    for split in split_names:
        shards = shard_data.get(split, [])
        n_shards = len(shards)
        total_states = sum(d["Q"].shape[0] for d in shards if "Q" in d)

        expected = expected_counts.get(split)
        entry = {
            "n_shards": n_shards,
            "n_states": total_states,
            "expected_states": expected,
        }

        if expected is not None and total_states != expected:
            entry["state_count_mismatch"] = True
            check.warnings.append(
                f"split '{split}': {total_states} states, expected {expected} "
                f"(diff={total_states - expected})"
            )

        # Check shard sizes are within shard_size bound
        shard_size = registry_spec.collection.shard_size
        for i, d in enumerate(shards):
            if "Q" in d and d["Q"].shape[0] > shard_size:
                entry.setdefault("oversized_shards", []).append(
                    f"shard_{i}: {d['Q'].shape[0]} > {shard_size}"
                )
                check.warnings.append(
                    f"split '{split}' shard_{i}: {d['Q'].shape[0]} states exceeds "
                    f"shard_size={shard_size}"
                )

        check.details[split] = entry

    # Check all expected splits exist
    for split in split_names:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            check.passed = False
            check.errors.append(f"Missing split directory: {split_dir}")

    return check


def _audit_spec_alignment(
    dataset_dir: Path,
    registry_spec,
    shard_data: dict[str, list[dict]],
) -> AuditCheck:
    """Verify dataset aligns with registry spec and env config."""
    check = AuditCheck()

    # Read first shard to get N and mu
    first_shard = None
    for split in ("train", "valid", "test"):
        if shard_data.get(split):
            first_shard = shard_data[split][0]
            break

    if first_shard is None:
        check.errors.append("No shards found in any split")
        check.passed = False
        return check

    data_N = first_shard["Q"].shape[-1] if "Q" in first_shard else None

    # Resolve and load the env config
    env_cfg = _read_env_config(registry_spec.env)
    if env_cfg is None:
        check.warnings.append(f"Could not resolve env config: {registry_spec.env}")
    else:
        check.details["env_config"] = env_cfg.name
        # Env config h length tells us the expected N
        if env_cfg.h is not None and data_N is not None:
            expected_N = len(env_cfg.h)
            if data_N != expected_N:
                check.passed = False
                check.errors.append(
                    f"N mismatch: data has {data_N} queues, env config '{env_cfg.name}' "
                    f"expects {expected_N} (from h length)"
                )

    # Check registry spec fields
    check.details["registry_spec"] = {
        "name": registry_spec.name,
        "policy": registry_spec.collection.policy,
        "seed": registry_spec.collection.seed,
        "n_steps": registry_spec.collection.n_steps,
        "n_valid": registry_spec.collection.n_valid,
        "n_test": registry_spec.collection.n_test,
    }

    check.details["data_N"] = data_N

    return check


# ---------------------------------------------------------------------------
#  Main audit orchestrator
# ---------------------------------------------------------------------------


def audit_dataset(name: str, verbose: bool = False) -> dict:
    """Run all audit checks on a named dataset.

    Parameters
    ----------
    name : str
        Registered dataset name.
    verbose : bool
        If True, print detailed per-shard info.

    Returns
    -------
    dict
        Complete audit report as a serialisable dictionary.
    """
    registry = DatasetRegistry()

    if name not in registry:
        print(f"Error: unknown dataset '{name}'.")
        print(f"Available: {', '.join(registry.list_datasets())}")
        sys.exit(1)

    spec = registry.get(name)
    dataset_dir = registry.resolve_path(name)
    all_splits = ["train", "valid", "test"]

    if not dataset_dir.exists():
        print(f"Error: dataset directory not found: {dataset_dir}")
        print(f"Run: python -m certiqnet.data.qgym.collect_qgym collect {name}")
        sys.exit(1)

    # Load all shards across splits
    shard_data: dict[str, list[dict]] = {}
    present_splits = []
    for split in all_splits:
        shards = _load_all_shards(dataset_dir, split)
        shard_data[split] = shards
        if shards:
            present_splits.append(split)

    if not present_splits:
        print(f"Error: no .pt files found in {dataset_dir}/{{train,valid,test}}")
        sys.exit(1)

    if verbose:
        for split in all_splits:
            shards = shard_data[split]
            if shards:
                sizes = [d["Q"].shape[0] for d in shards if "Q" in d]
                print(f"  {split}: {len(shards)} shard(s), {sum(sizes)} states, sizes={sizes}")
            else:
                print(f"  {split}: (empty)")

    # Run per-split checks
    split_results: list[dict] = []
    overall_passed = True
    all_errors: list[str] = []
    all_warnings: list[str] = []

    for split in all_splits:
        shards = shard_data[split]
        result = {
            "split": split,
            "n_shards": len(shards),
            "n_states": sum(d["Q"].shape[0] for d in shards if "Q" in d) if shards else 0,
        }

        if not shards:
            result["status"] = "empty"
            split_results.append(result)
            continue

        schema = _audit_schema(shards, split)
        cross_shard = _audit_cross_shard_consistency(shards, split)
        ranges = _audit_ranges(shards, split)
        stats = _audit_stats(shards, split)
        hardness = _audit_hardness(shards, split)

        result["schema"] = asdict(schema)
        result["cross_shard"] = asdict(cross_shard)
        result["ranges"] = asdict(ranges)
        result["stats"] = asdict(stats)
        result["hardness"] = asdict(hardness)

        for check in (schema, cross_shard, ranges, stats, hardness):
            if not check.passed:
                overall_passed = False
                all_errors.extend(check.errors)
            all_warnings.extend(check.warnings)

        split_results.append(result)

    # Cross-split checks
    split_integrity = _audit_split_integrity(dataset_dir, spec, all_splits, shard_data)
    spec_alignment = _audit_spec_alignment(dataset_dir, spec, shard_data)

    if not split_integrity.passed:
        overall_passed = False
        all_errors.extend(split_integrity.errors)
    all_warnings.extend(split_integrity.warnings)

    if not spec_alignment.passed:
        overall_passed = False
        all_errors.extend(spec_alignment.errors)
    all_warnings.extend(spec_alignment.warnings)

    # Metadata consistency check
    meta = _read_metadata(dataset_dir)
    meta_check = AuditCheck()
    if meta is not None:
        meta_Q_mean = meta.get("mean_Q")
        if meta_Q_mean is not None and split_results:
            train_stats = split_results[0].get("stats", {}).get("details", {})
            actual_mean = train_stats.get("Q", {}).get("mean")
            if actual_mean is not None and abs(float(meta_Q_mean) - actual_mean) > 1.0:
                meta_check.warnings.append(
                    f"metadata.yaml mean_Q={meta_Q_mean} differs from actual "
                    f"train mean_Q={actual_mean}"
                )

    if meta_check.warnings:
        all_warnings.extend(meta_check.warnings)

    # Build report
    report = {
        "dataset_name": name,
        "dataset_path": str(dataset_dir),
        "overall_passed": overall_passed,
        "n_splits_present": len(present_splits),
        "n_splits_total": len(all_splits),
        "splits": split_results,
        "split_integrity": asdict(split_integrity),
        "spec_alignment": asdict(spec_alignment),
        "metadata": meta,
        "errors": all_errors,
        "warnings": all_warnings,
    }

    return report


def _print_report(report: dict, verbose: bool) -> None:
    """Print a human-readable summary of the audit report."""
    name = report["dataset_name"]
    status = "PASS" if report["overall_passed"] else "FAIL"
    path = report["dataset_path"]

    path_str = str(path).encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
    print(f"dataset: {name}")
    print(f"path:    {path_str}")
    print(f"status:  {status}")
    print()

    for s in report["splits"]:
        split = s["split"]
        n_shards = s["n_shards"]
        n_states = s["n_states"]
        if n_shards == 0:
            print(f"  [{split:<6}] EMPTY")
            continue

        sc = s.get("schema", {}).get("passed", True)
        cc = s.get("cross_shard", {}).get("passed", True)
        rc = s.get("ranges", {}).get("passed", True)
        st = s.get("stats", {}).get("details", {})
        Q = st.get("Q", {})
        hs = s.get("hardness", {}).get("details", {})

        pct_over_1000 = "?"
        if hs.get("backlog_thresholds", {}).get("1000", {}).get("fraction") is not None:
            pct_over_1000 = f"{hs['backlog_thresholds']['1000']['fraction']*100:.1f}%"

        schema_str = "PASS" if sc else "FAIL"
        cross_str = "PASS" if cc else "FAIL"
        range_str = "PASS" if rc else "FAIL"

        q_mean = Q.get("mean", "?")
        q_p99 = Q.get("percentiles", {}).get("P99", "?")

        print(
            f"  [{split:<6}] schema={schema_str}  cross={cross_str}  ranges={range_str}  "
            f"|  {n_states:,} states ({n_shards} shard(s), N={st.get('N','?')})  "
            f"|  mean_Q={q_mean}  P99_Q={q_p99}  >1000={pct_over_1000}"
        )

    print()

    si = report.get("split_integrity", {})
    sa = report.get("spec_alignment", {})
    si_passed = si.get("passed", True)
    sa_passed = sa.get("passed", True)
    print(f"  splits:  {'PASS' if si_passed else 'FAIL'}  "
          f"spec: {'PASS' if sa_passed else 'FAIL'}")

    errors = report.get("errors", [])
    warnings = report.get("warnings", [])
    if errors:
        print(f"  errors:   {len(errors)}")
        if verbose:
            for e in errors:
                print(f"    - {e}")
    if warnings:
        print(f"  warnings: {len(warnings)}")
        if verbose:
            for w in warnings:
                print(f"    - {w}")

    if not errors and not warnings:
        print(f"  No errors or warnings.")

    print()
    print(f"result: {status}")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a collected QGym dataset for integrity and quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m certiqnet.data.qgym.audit_qgym reentrant_2_debug\n"
            "  python -m certiqnet.data.qgym.audit_qgym reentrant_2 --verbose\n"
            "  python -m certiqnet.data.qgym.audit_qgym reentrant_2 --output report.json\n"
        ),
    )
    parser.add_argument("dataset_name", type=str, help="Registered dataset name")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed per-shard info"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Path to write JSON report"
    )

    args = parser.parse_args()
    report = audit_dataset(args.dataset_name, verbose=args.verbose)

    _print_report(report, verbose=args.verbose)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        out_str = str(out_path.resolve()).encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
        print(f"\nReport saved to {out_str}")

    sys.exit(0 if report["overall_passed"] else 1)


if __name__ == "__main__":
    main()
