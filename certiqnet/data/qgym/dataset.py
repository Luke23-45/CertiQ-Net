"""PyTorch Dataset for pre-collected QGym ``.pt`` shards.

This module provides ``QGymDataset``, a ``torch.utils.data.Dataset`` that
lazily indexes into a directory of shard files produced by the collection
script (``scripts/collect_qgym.py``).

It is deliberately *data-only* — the Lightning ``DataModule`` that wires
it into training lives in ``certiqnet.data.qgym.datamodule``.
"""

from __future__ import annotations

import bisect
import logging
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

# Keys that every valid shard must contain.
_REQUIRED_SHARD_KEYS = frozenset({"Q", "cost"})


class QGymDataset(Dataset):
    """PyTorch Dataset over pre-collected QGym ``.pt`` shards.

    Shard files are expected to be dictionaries saved via ``torch.save``
    with at least the keys ``"Q"`` (queue lengths) and ``"cost"``
    (per-step holding cost).  An optional ``"mu"`` key stores the
    per-queue effective service rates shared across all states in the
    dataset.

    Parameters
    ----------
    dataset_dir : str | Path
        Path to a ``dataset/qgym/<name>/`` directory.
    split : str
        One of ``"train"``, ``"valid"``, ``"test"``.
    """

    def __init__(self, dataset_dir: str | Path, split: str = "train") -> None:
        self._dataset_dir = Path(dataset_dir)
        self._split = split
        self._shards: list[dict[str, Tensor]] = []
        self._cumulative_sizes: list[int] = []
        self._total_size: int = 0
        self._mu: Tensor | None = None
        self._h: Tensor | None = None

        split_dir = self._dataset_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"QGym split directory not found: {split_dir}"
            )

        shard_files = sorted(split_dir.glob("*.pt"))
        if not shard_files:
            raise FileNotFoundError(f"No .pt files found in {split_dir}")

        cumulative = 0
        for shard_path in shard_files:
            data = torch.load(shard_path, weights_only=True)
            self._validate_shard(data, shard_path)
            self._shards.append(data)
            shard_size = data["Q"].shape[0]
            cumulative += shard_size
            self._cumulative_sizes.append(cumulative)

        self._total_size = cumulative

        # mu and h are shared across all shards (per-queue effective rates)
        if self._shards:
            first = self._shards[0]
            if "mu" in first:
                self._mu = first["mu"]
            if "h" in first:
                self._h = first["h"]

        log.info(
            "QGymDataset loaded: split=%s, shards=%d, total=%d, N=%d",
            split,
            len(self._shards),
            self._total_size,
            self.N,
        )

    # ── Validation ────────────────────────────────────────────────────

    @staticmethod
    def _validate_shard(data: dict, path: Path) -> None:
        """Validate shard integrity at load time."""
        missing = _REQUIRED_SHARD_KEYS - set(data.keys())
        if missing:
            raise KeyError(
                f"Shard {path.name} is missing required keys: {sorted(missing)}.  "
                "Expected at least: Q, cost."
            )
        if data["Q"].dim() != 2:
            raise ValueError(
                f"Shard {path.name}: 'Q' must be 2-D (B, N), "
                f"got shape {tuple(data['Q'].shape)}."
            )
        if data["cost"].dim() != 1:
            raise ValueError(
                f"Shard {path.name}: 'cost' must be 1-D (B,), "
                f"got shape {tuple(data['cost'].shape)}."
            )
        if data["Q"].shape[0] != data["cost"].shape[0]:
            raise ValueError(
                f"Shard {path.name}: Q and cost have different batch sizes "
                f"({data['Q'].shape[0]} vs {data['cost'].shape[0]})."
            )

    # ── Dataset interface ─────────────────────────────────────────────

    def __len__(self) -> int:
        return self._total_size

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(Q, mu, cost)`` for the given global index.

        Notes
        -----
        Heuristic action labels (SED / QMD) are **not** stored in the
        shards — they are computed by the downstream ``QGymDataModule``
        after mixing all data sources.  This keeps the dataset layer
        clean and avoids stale labels.
        """
        if idx < 0:
            idx += self._total_size
        if idx < 0 or idx >= self._total_size:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {self._total_size}"
            )

        # O(log n) shard lookup via bisect
        shard_i = bisect.bisect_right(self._cumulative_sizes, idx)
        local_idx = (
            idx if shard_i == 0 else idx - self._cumulative_sizes[shard_i - 1]
        )

        data = self._shards[shard_i]
        Q = data["Q"][local_idx]
        cost = data["cost"][local_idx]
        mu = data["mu"] if "mu" in data else (self._mu if self._mu is not None else torch.ones(Q.shape[-1]))
        return Q, mu, cost

    # ── Properties ────────────────────────────────────────────────────

    @property
    def mu(self) -> Tensor | None:
        """Per-queue effective service rates (shared across the dataset)."""
        return self._mu

    @property
    def h(self) -> Tensor | None:
        """Per-queue holding-cost vector (shared across the dataset)."""
        return self._h

    @property
    def N(self) -> int:
        """Number of queues (from the first shard's Q shape)."""
        if self._shards:
            return self._shards[0]["Q"].shape[-1]
        return 0

    @property
    def split(self) -> str:
        """Which data split this dataset represents."""
        return self._split

    # ── Debugging ─────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"QGymDataset(dir='{self._dataset_dir}', split='{self._split}', "
            f"shards={len(self._shards)}, total={self._total_size}, N={self.N})"
        )
