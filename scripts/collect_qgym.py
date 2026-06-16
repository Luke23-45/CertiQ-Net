"""Collect a QGym dataset to disk as ``.pt`` shards.

Usage
-----
    python scripts/collect_qgym.py --config configs/qgym/collection/reentrant_2.yaml
    python scripts/collect_qgym.py --config configs/qgym/collection/reentrant_2.yaml --force

The script reads a ``QGymCollectionConfig`` YAML, spins up a QGym
environment, collects states using the configured policy (or mixed-policy
blend), splits into train / valid / test, and saves ``.pt`` shard files
plus a ``metadata.yaml`` manifest.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from certiqnet.adapters.qgym.adapter import QGymAdapter
from certiqnet.adapters.qgym.config import QGymCollectionConfig
from certiqnet.adapters.qgym.env_loader import compute_effective_mu


def _progress_bar(current: int, total: int, width: int = 40) -> str:
    pct = current / max(total, 1)
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"\r  [{bar}] {current:>8,}/{total:,} ({pct:.1%})"


def collect(cfg: QGymCollectionConfig) -> Path:
    """Run the full collection pipeline and return the output directory."""
    output_root = Path(cfg.output_dir) / cfg.dataset_name
    if output_root.exists() and not cfg.force_collect:
        print(f"Dataset already exists at {output_root}.  Use --force to overwrite.")
        return output_root

    # ── Build adapter ─────────────────────────────────────────────────
    print(f"[collect] env config: {cfg.env_config_path}")
    adapter = QGymAdapter(
        env_config=str(cfg.env_config_path),
        mode="online",
        policy=cfg.policy,
        policy_weights=cfg.policy_weights,
        batch_size_env=1,
        seed=cfg.seed,
        device="cpu",
    )
    print(f"[collect] adapter: {adapter}")

    total = cfg.n_steps + cfg.n_valid + cfg.n_test
    print(f"[collect] collecting {total:,} states (train={cfg.n_steps:,}, "
          f"valid={cfg.n_valid:,}, test={cfg.n_test:,})")

    # ── Collect all states ────────────────────────────────────────────
    gen = torch.Generator().manual_seed(cfg.seed)
    t0 = time.perf_counter()
    batch = adapter.sample_batch(
        n_samples=total,
        N=adapter.N,
        mu=torch.ones(adapter.N),   # ignored by QGymAdapter
        generator=gen,
    )
    elapsed = time.perf_counter() - t0
    print(f"[collect] collected {total:,} states in {elapsed:.1f}s "
          f"({total / max(elapsed, 0.001):.0f} states/s)")

    Q_all = batch.Q.float()
    cost_all = batch.cost.float()
    mu_vec = batch.mu[0].float()     # shared across all rows

    # ── Get holding cost vector if available ──────────────────────────
    h_vec = adapter.env_h

    # ── Split ─────────────────────────────────────────────────────────
    perm = torch.randperm(total, generator=gen)
    Q_all = Q_all[perm]
    cost_all = cost_all[perm]

    splits = {
        "train": (0, cfg.n_steps),
        "valid": (cfg.n_steps, cfg.n_steps + cfg.n_valid),
        "test": (cfg.n_steps + cfg.n_valid, total),
    }

    # ── Save shards ──────────────────────────────────────────────────
    for split_name, (lo, hi) in splits.items():
        split_dir = output_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        # Remove old shards
        for old in split_dir.glob("shard_*.pt"):
            old.unlink()

        Q_split = Q_all[lo:hi]
        cost_split = cost_all[lo:hi]
        n = Q_split.shape[0]
        n_shards = max(1, math.ceil(n / cfg.shard_size))

        for shard_i in range(n_shards):
            s_lo = shard_i * cfg.shard_size
            s_hi = min(s_lo + cfg.shard_size, n)
            shard_data = {
                "Q": Q_split[s_lo:s_hi],
                "cost": cost_split[s_lo:s_hi],
                "mu": mu_vec,
            }
            if h_vec is not None:
                shard_data["h"] = h_vec
            shard_path = split_dir / f"shard_{shard_i:04d}.pt"
            torch.save(shard_data, shard_path)

        print(f"  {split_name}: {n:,} states → {n_shards} shard(s)")

    # ── Write metadata ───────────────────────────────────────────────
    metadata = {
        "dataset_name": cfg.dataset_name,
        "env_config": str(cfg.env_config_path),
        "n_train": cfg.n_steps,
        "n_valid": cfg.n_valid,
        "n_test": cfg.n_test,
        "N": int(mu_vec.shape[0]),
        "policy": cfg.policy,
        "policy_weights": cfg.policy_weights if cfg.policy == "mixed" else None,
        "seed": cfg.seed,
        "shard_size": cfg.shard_size,
        "collection_time_s": round(elapsed, 2),
    }
    meta_path = output_root / "metadata.yaml"
    with open(meta_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    print(f"[collect] metadata saved to {meta_path}")

    # ── Verify ───────────────────────────────────────────────────────
    _verify(output_root)
    return output_root


def _verify(dataset_dir: Path) -> None:
    """Quick integrity check on the saved dataset."""
    from certiqnet.data.qgym.dataset import QGymDataset

    for split in ("train", "valid", "test"):
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        ds = QGymDataset(str(dataset_dir), split=split)
        q, mu, cost = ds[0]
        assert q.dim() == 1, f"Expected 1-D Q, got {q.shape}"
        assert cost.dim() == 0, f"Expected scalar cost, got {cost.shape}"
        print(f"  ✓ {split}: {len(ds):,} states, N={ds.N}, Q[0]={q[:4].tolist()}...")
    print("[collect] verification passed ✓")


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect a QGym dataset to disk."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a QGymCollectionConfig YAML.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing dataset.",
    )
    args = parser.parse_args()

    cfg = QGymCollectionConfig.from_yaml(args.config)
    if args.force:
        cfg.force_collect = True

    output = collect(cfg)
    print(f"\n[collect] done → {output}")


if __name__ == "__main__":
    main()
