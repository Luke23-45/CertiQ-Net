"""Dataset collection manager — orchestrates QGym dataset collection.

Delegated to by both ``collect_qgym.py`` (standalone CLI) and
``QGymDataModule`` (auto-collect on setup).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from certiqnet.adapters.qgym.adapter import QGymAdapter
from certiqnet.adapters.qgym.config import QGymEnvConfig, resolve_env_config_path
from certiqnet.adapters.qgym.env_loader import compute_effective_mu
from certiqnet.data.registry import DatasetRegistry, DatasetSpec

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Git helpers
# ---------------------------------------------------------------------------


def _git_commit_hash(project_root: Path) -> str | None:
    """Return the current git commit hash, or ``None`` if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
#  Collection manager
# ---------------------------------------------------------------------------


class DatasetCollectionManager:
    """Orchestrates the collection of QGym datasets.

    Parameters
    ----------
    registry : DatasetRegistry, optional
        Registry instance used for name resolution.
    """

    def __init__(self, registry: DatasetRegistry | None = None) -> None:
        self._registry = registry or DatasetRegistry()

    # ── Adapter construction ──────────────────────────────────────────

    def build_adapter(self, spec: DatasetSpec) -> QGymAdapter:
        """Build a ``QGymAdapter`` from a dataset spec.

        The adapter is created in ``online`` mode using the environment
        config and policy specified in *spec*.
        """
        return QGymAdapter(
            env_config=str(spec.env),
            mode="online",
            policy=spec.collection.policy,
            policy_weights=spec.collection.policy_weights,
            batch_size_env=1,
            seed=spec.collection.seed,
            device="cpu",
        )

    # ── Core collection ───────────────────────────────────────────────

    def collect(
        self,
        spec: DatasetSpec,
        *,
        force: bool = False,
        skip_verify: bool = False,
    ) -> Path:
        """Run the full collection pipeline for a dataset spec.

        Parameters
        ----------
        spec : DatasetSpec
            The dataset specification to collect.
        force : bool
            If ``True``, overwrite an existing dataset directory.
        skip_verify : bool
            If ``True``, skip post-collection verification.

        Returns
        -------
        Path
            The output directory where the collected data was written.
        """
        output_dir = self._resolve_output(spec)

        # ── Check for existing data ───────────────────────────────────
        if output_dir.exists() and not force:
            already_complete = self._check_complete(output_dir)
            if already_complete:
                log.info(
                    "Dataset '%s' already exists at %s and appears complete. "
                    "Use force=True to re-collect.",
                    spec.name,
                    output_dir,
                )
                return output_dir
            log.warning(
                "Dataset '%s' at %s is incomplete — re-collecting.",
                spec.name,
                output_dir,
            )

        # ── Build adapter ─────────────────────────────────────────────
        log.info("Building QGymAdapter for '%s' (env: %s)...", spec.name, spec.env)
        adapter = self.build_adapter(spec)
        log.info("Adapter: %s", adapter)

        total = spec.collection.n_steps + spec.collection.n_valid + spec.collection.n_test
        log.info(
            "Collecting %,d states (train=%,d, valid=%,d, test=%,d)...",
            total,
            spec.collection.n_steps,
            spec.collection.n_valid,
            spec.collection.n_test,
        )

        # ── Run collection ────────────────────────────────────────────
        gen = torch.Generator().manual_seed(spec.collection.seed)
        t0 = time.perf_counter()

        batch = adapter.sample_batch(
            n_samples=total,
            N=adapter.N,
            mu=torch.ones(adapter.N),
            generator=gen,
            show_progress=True,
        )

        elapsed = time.perf_counter() - t0
        log.info(
            "Collected %,d states in %.1fs (%.0f states/s)",
            total,
            elapsed,
            total / max(elapsed, 0.001),
        )

        Q_all = batch.Q.float()
        cost_all = batch.cost.float()
        mu_vec = batch.mu[0].float()
        h_vec = adapter.env_h

        # ── Shuffle & split ───────────────────────────────────────────
        perm = torch.randperm(total, generator=gen)
        Q_all = Q_all[perm]
        cost_all = cost_all[perm]

        splits = {
            "train": (0, spec.collection.n_steps),
            "valid": (
                spec.collection.n_steps,
                spec.collection.n_steps + spec.collection.n_valid,
            ),
            "test": (
                spec.collection.n_steps + spec.collection.n_valid,
                total,
            ),
        }

        # ── Write shards ──────────────────────────────────────────────
        n_train = 0
        for split_name, (lo, hi) in splits.items():
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for old in split_dir.glob("shard_*.pt"):
                old.unlink()

            Q_split = Q_all[lo:hi]
            cost_split = cost_all[lo:hi]
            n = Q_split.shape[0]
            n_shards = max(1, math.ceil(n / spec.collection.shard_size))

            for shard_i in range(n_shards):
                s_lo = shard_i * spec.collection.shard_size
                s_hi = min(s_lo + spec.collection.shard_size, n)
                shard_data = {
                    "Q": Q_split[s_lo:s_hi],
                    "cost": cost_split[s_lo:s_hi],
                    "mu": mu_vec,
                }
                if h_vec is not None:
                    shard_data["h"] = h_vec
                shard_path = split_dir / f"shard_{shard_i:04d}.pt"
                torch.save(shard_data, shard_path)

            if split_name == "train":
                n_train = n
            log.info("  %s: %,d states -> %d shard(s)", split_name, n, n_shards)

        # ── Compute data statistics for metadata ──────────────────────
        mean_Q = float(Q_all.mean().item()) if Q_all.numel() > 0 else 0.0
        std_Q = float(Q_all.std().item()) if Q_all.numel() > 0 else 0.0
        min_Q = float(Q_all.min().item()) if Q_all.numel() > 0 else 0.0
        max_Q = float(Q_all.max().item()) if Q_all.numel() > 0 else 0.0
        mean_cost = float(cost_all.mean().item()) if cost_all.numel() > 0 else 0.0

        # ── Write metadata ────────────────────────────────────────────
        commit = _git_commit_hash(Path(__file__).resolve().parents[2])
        metadata = {
            "dataset_name": spec.name,
            "env_config": str(resolve_env_config_path(str(spec.env))),
            "n_train": n_train,
            "n_valid": spec.collection.n_valid,
            "n_test": spec.collection.n_test,
            "N": int(mu_vec.shape[0]),
            "s": int(getattr(adapter, "_env", adapter).s if hasattr(adapter, "_env") else 0),
            "q": adapter.N,
            "policy": spec.collection.policy,
            "policy_weights": (
                spec.collection.policy_weights
                if spec.collection.policy == "mixed"
                else None
            ),
            "seed": spec.collection.seed,
            "shard_size": spec.collection.shard_size,
            "collection_time": datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S %Z"
            ),
            "collection_duration_s": round(elapsed, 2),
            "git_commit": commit,
            "mean_Q": round(mean_Q, 4),
            "std_Q": round(std_Q, 4),
            "min_Q": round(min_Q, 4),
            "max_Q": round(max_Q, 4),
            "mean_cost": round(mean_cost, 4),
        }
        meta_path = output_dir / "metadata.yaml"
        with open(meta_path, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
        log.info("Metadata saved to %s", meta_path)

        # ── Save a copy of the input spec for reproducibility ─────────
        spec_path = output_dir / "dataset_spec.yaml"
        spec_dict = self._registry.spec_to_dict(spec)
        with open(spec_path, "w") as f:
            yaml.dump(spec_dict, f, default_flow_style=False, sort_keys=False)
        log.info("Dataset spec saved to %s", spec_path)

        # ── Save the resolved env config for reproducibility ──────────
        env_config_path = resolve_env_config_path(str(spec.env))
        env_spec_path = output_dir / "env_config.yaml"
        shutil.copy2(str(env_config_path), str(env_spec_path))
        log.info("Env config copied to %s", env_spec_path)

        # ── Verify ────────────────────────────────────────────────────
        if not skip_verify:
            self.verify(output_dir)

        return output_dir

    # ── Verification ──────────────────────────────────────────────────

    def verify(self, dataset_dir: str | Path) -> int:
        """Verify integrity of a collected dataset.

        Parameters
        ----------
        dataset_dir : str | Path
            Path to the dataset directory (containing train/valid/test).

        Returns
        -------
        int
            Number of shards verified (raises on failure).

        Raises
        ------
        ValueError
            If any integrity check fails.
        """
        from certiqnet.data.qgym.dataset import QGymDataset

        dataset_dir = Path(dataset_dir)
        n_verified = 0

        for split in ("train", "valid", "test"):
            split_dir = dataset_dir / split
            if not split_dir.exists():
                continue

            ds = QGymDataset(str(dataset_dir), split=split)
            # Verify first and last item
            q0, mu0, cost0 = ds[0]
            q_last, mu_last, cost_last = ds[len(ds) - 1]

            assert q0.dim() == 1, f"Expected 1-D Q, got {q0.shape}"
            assert cost0.dim() == 0, f"Expected scalar cost, got {cost0.shape}"
            assert q_last.dim() == 1
            assert cost_last.dim() == 0

            # Verify N consistency
            assert q0.shape[-1] == q_last.shape[-1], (
                f"N mismatch within {split}: {q0.shape[-1]} vs {q_last.shape[-1]}"
            )

            # Verify mu consistency
            if mu0 is not None and mu_last is not None:
                assert mu0.shape == mu_last.shape

            n_verified += 1
            log.info(
                "  [OK] %s: %,d states, N=%d, Q[0]=%s...",
                split,
                len(ds),
                ds.N,
                q0[:4].tolist(),
            )

        total_shards = sum(
            1 for _ in dataset_dir.glob("**/shard_*.pt")
        )
        log.info(
            "Verification passed: %d shard(s) across %d split(s)",
            total_shards,
            n_verified,
        )
        return n_verified

    # ── Ensure (auto-collect) ─────────────────────────────────────────

    def ensure(
        self,
        name: str,
        *,
        auto_collect: bool = True,
        force: bool = False,
        skip_verify: bool = False,
    ) -> Path:
        """Ensure a dataset exists, optionally collecting it if missing.

        This is the primary entry point used by ``QGymDataModule.setup()``.

        Parameters
        ----------
        name : str
            Dataset name (must be registered).
        auto_collect : bool
            If ``True`` and the dataset is missing, collect it.
            If ``False`` and the dataset is missing, raise ``FileNotFoundError``
            with detailed recovery instructions.
        force : bool
            If ``True``, re-collect even if the dataset exists.
        skip_verify : bool
            If ``True``, skip post-collection verification.

        Returns
        -------
        Path
            Canonical path to the collected dataset.

        Raises
        ------
        KeyError
            If *name* is not a registered dataset.
        FileNotFoundError
            If dataset is missing and *auto_collect* is ``False``.
        """
        spec = self._registry.get(name)
        output_dir = self._registry.resolve_path(name)

        if not force and self._registry.exists(name):
            complete = self._check_complete(output_dir)
            if complete:
                log.info(
                    "Dataset '%s' found at %s — skipping collection.",
                    name,
                    output_dir,
                )
                return output_dir
            log.warning(
                "Dataset '%s' at %s appears incomplete — will re-collect.",
                name,
                output_dir,
            )

        if not auto_collect:
            msg = (
                f"Dataset '{name}' not found at {output_dir}.\n\n"
                f"To collect it, run:\n"
                f"    python -m certiqnet.data.qgym.collect_qgym collect {name}\n\n"
                f"Or set auto_collect=True in your data module config."
            )
            raise FileNotFoundError(msg)

        log.info(
            "Dataset '%s' not found at %s — auto-collecting...",
            name,
            output_dir,
        )
        return self.collect(spec, force=force, skip_verify=skip_verify)

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _resolve_output(spec: DatasetSpec) -> Path:
        """Resolve the output directory for a spec."""
        output = Path(spec.output_dir)
        if not output.is_absolute():
            output = Path(__file__).resolve().parents[2] / output
        return (output / spec.name).resolve()

    @staticmethod
    def _check_complete(dataset_dir: Path) -> bool:
        """Check that all three split directories exist and have shards."""
        for split in ("train", "valid", "test"):
            split_dir = dataset_dir / split
            if not split_dir.exists():
                return False
            if not any(split_dir.glob("*.pt")):
                return False
        return True
