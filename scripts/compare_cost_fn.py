"""
Compare SED vs QMD base geometry for CertiQ-Net training.

Trains two identical models differing only in cost_fn,
compares validation metrics epoch-by-epoch, and produces
a table + summary at the end.

Usage:
    python -m scripts.compare_cost_fn                     # hospital, 20 epochs
    python -m scripts.compare_cost_fn 'max_epochs=5'      # override epochs
    python -m scripts.compare_cost_fn 'dataset=criss_cross_bh'
"""

from __future__ import annotations

import sys, json, itertools
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_DIR = ROOT / "configs"


def _parse_overrides(raw: list[str]) -> dict[str, str]:
    d = {}
    for arg in raw:
        if "=" in arg:
            k, v = arg.split("=", 1)
            d[k.strip()] = v.strip()
    return d


def train_with_cost_fn(
    cost_fn: str,
    max_epochs: int,
    dataset: str,
    base_overrides: list[str],
    output_dir: Path,
) -> dict:
    """Run a single training session and return aggregated validation metrics."""
    from hydra import compose, initialize_config_dir

    output_root_str = str(output_dir).replace("\\", "/").replace("\u2011", "-")
    overrides = list(base_overrides) + [
        f"model.cost_fn={cost_fn}",
        f"trainer.max_epochs={max_epochs}",
        f"qgym.data.init_args.dataset_name={dataset}",
        f"project.output_root={output_root_str}",
    ]

    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="experiments/main_queueing", overrides=overrides)

    # Inject CSV logger save_dir so we can read metrics after
    cfg.logger.save_dir = str(output_dir / cost_fn).replace("\\", "/")

    from certiqnet.train.runner import run_training
    run_training(cfg, cwd=ROOT)

    return cfg


def gather_metrics(run_root: Path) -> list[dict]:
    """Parse the training CSV log into a list of per-step records."""
    import csv

    csv_dir = run_root / "logs" / "csv" / "version_0"
    csv_path = csv_dir / "metrics.csv"
    if not csv_path.exists():
        return []

    records = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = {}
            for k, v in row.items():
                k = k.strip()
                v = v.strip()
                if v == "" or v == "nan" or v == "NaN":
                    continue
                try:
                    record[k] = float(v)
                except ValueError:
                    record[k] = v
            if record:
                records.append(record)
    return records


def extract_val_snapshot(records: list[dict]) -> list[dict]:
    """Extract one row per unique epoch from validation records."""
    seen: set[int] = set()
    snaps = []

    for ep_key in ("epoch",):
        for r in records:
            ep = r.get(ep_key)
            if ep is not None and int(ep) not in seen and int(ep) >= 0:
                ep_int = int(ep)
                seen.add(ep_int)
                snaps.append({
                    "epoch": ep_int,
                    "avg_cost": r.get("val/avg_cost"),
                    "violation": r.get("val/CONSTRAINT_VIOLATION"),
                    "violation_rate": r.get("val/CONSTRAINT_VIOLATION_RATE"),
                    "p95_backlog": r.get("val/p95_backlog"),
                    "selection_score": r.get("val/selection_score"),
                    "cert_slack_mean": r.get("val/certificate_slack_mean"),
                    "cert_slack_min": r.get("val/certificate_slack_min"),
                    "ds_avg_cost": r.get("val/dataset_start_avg_cost"),
                })
    return snaps


def find_latest_run(output_dir: Path, cost_fn: str) -> Path | None:
    """Find the most recent run subdirectory for a given cost_fn."""
    runs_dir = output_dir / "main-queueing-hospital"
    if not runs_dir.exists():
        return None
    subdirs = sorted(runs_dir.iterdir(), key=lambda p: p.name, reverse=True)
    # Filter for this cost_fn — runs are named by timestamp
    for sd in subdirs:
        if sd.is_dir():
            manifest = sd / "manifest.json"
            if manifest.exists():
                return sd
    return None


def main() -> None:
    args = _parse_overrides(sys.argv[1:] if len(sys.argv) > 1 else [])
    max_epochs = int(args.get("max_epochs", 20))
    dataset = args.get("dataset", "hospital")
    base_overrides = [
        "trainer=default",
        "trainer.val_check_interval=1.0",
        "trainer.log_every_n_steps=5",
        "trainer.precision=32-true",
        "trainer.accelerator=auto",
        "project.save_checkpoints=false",

    ]

    output_dir = ROOT / "outputs" / "cost_fn_compare"
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[dict]] = {}

    for cost_fn in ("sed", "qmd"):
        print(f"\n{'='*60}")
        print(f"Training with cost_fn={cost_fn!r}  (dataset={dataset}, {max_epochs} epochs)")
        print(f"{'='*60}")

        cfg = train_with_cost_fn(
            cost_fn=cost_fn,
            max_epochs=max_epochs,
            dataset=dataset,
            base_overrides=base_overrides,
            output_dir=output_dir,
        )

        run_root = find_latest_run(output_dir, cost_fn)
        if run_root is None:
            print(f"WARNING: no run output found for {cost_fn}")
            continue

        records = gather_metrics(run_root)
        snaps = extract_val_snapshot(records)
        results[cost_fn] = snaps

        print(f"  -> {len(snaps)} validation snapshots gathered from {run_root.name}")

    # ── Print comparison table ──
    if not results:
        print("No results to compare.")
        return

    print(f"\n{'='*80}")
    print("COMPARISON: SED vs QMD  (validation metrics from Q=0 rollout)")
    print(f"{'='*80}")

    epochs_sed = {s["epoch"] for s in results.get("sed", [])}
    epochs_qmd = {s["epoch"] for s in results.get("qmd", [])}
    common_epochs = sorted(epochs_sed & epochs_qmd)

    if not common_epochs:
        # Fallback: show each separately
        for cost_fn in ("sed", "qmd"):
            snaps = results.get(cost_fn, [])
            if not snaps:
                continue
            print(f"\n  {cost_fn.upper()} — {len(snaps)} val steps:")
            print(f"  {'Epoch':>6}  {'AvgCost':>10}  {'p95':>8}  {'Violation':>10}  {'SelScore':>10}  {'SlackMin':>9}")
            for s in snaps:
                print(f"  {s['epoch']:>6}  {s['avg_cost']:>10.3f}  {s['p95_backlog']:>8.1f}  {s['violation']:>10.4f}  {s['selection_score']:>10.1f}  {s['cert_slack_min']:>9.3f}")
    else:
        header = f"  {'Epoch':>6}  {'SED_cost':>10}  {'QMD_cost':>10}  {'SED_p95':>8}  {'QMD_p95':>8}  {'SED_viol':>8}  {'QMD_viol':>8}  {'SED_sel':>10}  {'QMD_sel':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for ep in common_epochs:
            s = {s["epoch"]: s for s in results["sed"]}
            q = {s["epoch"]: s for s in results["qmd"]}
            sr = s.get(ep, {})
            qr = q.get(ep, {})
            print(f"  {ep:>6}  {sr.get('avg_cost', -1):>10.3f}  {qr.get('avg_cost', -1):>10.3f}  "
                  f"{sr.get('p95_backlog', -1):>8.1f}  {qr.get('p95_backlog', -1):>8.1f}  "
                  f"{sr.get('violation', -1):>8.4f}  {qr.get('violation', -1):>8.4f}  "
                  f"{sr.get('selection_score', -1):>10.1f}  {qr.get('selection_score', -1):>10.1f}")

    # Print final (best) comparison
    print(f"\n{'='*80}")
    print("BEST (lowest selection_score) per cost_fn:")
    print(f"{'='*80}")
    for cost_fn in ("sed", "qmd"):
        snaps = results.get(cost_fn, [])
        if not snaps:
            continue
        best = min(snaps, key=lambda s: s["selection_score"] if s["selection_score"] is not None else float("inf"))
        worst_epoch = min(snaps[-1:], key=lambda s: s["selection_score"] if s["selection_score"] is not None else float("inf"))
        last = snaps[-1] if snaps else {}
        print(f"  {cost_fn.upper()}:")
        print(f"    Best  @ ep {best['epoch']}:  avg_cost={best['avg_cost']:.3f}  p95={best['p95_backlog']:.1f}  "
              f"violation={best['violation']:.4f}  score={best['selection_score']:.1f}")
        print(f"    Final @ ep {last['epoch']}:  avg_cost={last.get('avg_cost', -1):.3f}  p95={last.get('p95_backlog', -1):.1f}  "
              f"violation={last.get('violation', -1):.4f}  score={last.get('selection_score', -1):.1f}")

    print(f"\nResults saved to: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
