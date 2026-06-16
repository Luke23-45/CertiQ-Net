"""Dataset registry — centralized management of pre-collected QGym datasets.

Provides a single source of truth for QGym dataset definitions, discovery,
integrity checking, and lazy auto-collection integration with the training
pipeline.

Usage
-----
    from certiqnet.data.registry import DatasetRegistry

    reg = DatasetRegistry()
    spec = reg.get("reentrant_2")
    print(spec.name, spec.collection.n_steps)

    if not reg.exists("reentrant_2"):
        reg.ensure("reentrant_2", auto_collect=True)
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

log = logging.getLogger(__name__)

_SUPPORTED_POLICIES = frozenset({"random", "sed", "qmd", "softmax", "mixed"})

# ---------------------------------------------------------------------------
#  Canonical paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
"""Absolute path to the project root (CertiQ‑Net/)."""

_REGISTRY_DIR = PROJECT_ROOT / "configs" / "dataset" / "qgym" / "registry"
"""Directory containing centralized dataset YAML definitions."""

_DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "final_dataset" / "qgym"
"""Canonical root for all collected QGym datasets."""

# ---------------------------------------------------------------------------
#  Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CollectionConfig:
    """Parameters controlling dataset collection via ``collect_qgym.py``.

    Attributes
    ----------
    n_steps : int
        Number of training-set simulation steps.
    n_valid : int
        Number of validation states.
    n_test : int
        Number of test states.
    shard_size : int
        Maximum states per ``.pt`` shard file.
    seed : int
        Base random seed.
    policy : str
        Collection policy (``"random"``, ``"sed"``, ``"qmd"``,
        ``"softmax"``, ``"mixed"``).
    policy_weights : dict[str, float]
        Per-policy sampling weights when ``policy="mixed"``.
    """

    n_steps: int = 500_000
    n_valid: int = 10_000
    n_test: int = 10_000
    shard_size: int = 50_000
    seed: int = 42
    policy: Literal["random", "sed", "qmd", "softmax", "mixed"] = "mixed"
    policy_weights: dict[str, float] = field(
        default_factory=lambda: {
            "random": 0.25,
            "sed": 0.25,
            "qmd": 0.25,
            "softmax": 0.25,
        }
    )

    def __post_init__(self) -> None:
        if self.policy not in _SUPPORTED_POLICIES:
            raise ValueError(
                f"Unknown policy '{self.policy}'. "
                f"Supported: {sorted(_SUPPORTED_POLICIES)}"
            )
        if self.policy == "mixed":
            if not self.policy_weights:
                raise ValueError(
                    "policy_weights must be non-empty when policy='mixed'."
                )
            bad = set(self.policy_weights) - _SUPPORTED_POLICIES
            if bad:
                raise ValueError(
                    f"Unknown policies in policy_weights: {sorted(bad)}. "
                    f"Supported: {sorted(_SUPPORTED_POLICIES)}"
                )
            total = sum(self.policy_weights.values())
            if not (0.99 < total < 1.01):
                raise ValueError(
                    f"policy_weights must sum to ~1.0, got {total:.4f}"
                )


@dataclass(frozen=True)
class DatasetSpec:
    """Complete specification for a single QGym dataset.

    This is the **single source of truth** — one YAML file per dataset
    defines both how to collect it and how to consume it during training.

    Attributes
    ----------
    name : str
        Unique dataset name (e.g. ``"reentrant_2"``).
    env : str
        Path (relative or absolute) to the QGym environment YAML.
    output_dir : str
        Parent directory under which the collected data will live
        (default ``final_dataset/qgym``).
    collection : CollectionConfig
        Parameters for the collection run.
    training_defaults : dict
        Default ``QGymDataModule`` parameters when this dataset is used
        in training.
    """

    name: str
    env: str
    output_dir: str = "final_dataset/qgym"
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    training_defaults: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
#  Registry
# ---------------------------------------------------------------------------


class DatasetRegistry:
    """Manages known QGym datasets: discovery, loading, existence checks.

    The registry scans ``configs/dataset/qgym/registry/*.yaml`` to discover
    all datasets that the project knows about.  Each YAML file is expected
    to conform to the ``DatasetSpec`` schema.
    """

    def __init__(
        self,
        registry_dir: str | Path | None = None,
        output_root: str | Path | None = None,
    ) -> None:
        self._registry_dir = Path(registry_dir or _REGISTRY_DIR)
        self._output_root = Path(output_root or _DEFAULT_OUTPUT_ROOT)
        self._specs: dict[str, DatasetSpec] | None = None

    # ── Discovery ─────────────────────────────────────────────────────

    def _load_all(self) -> dict[str, DatasetSpec]:
        """Load and index all registry YAML files."""
        specs: dict[str, DatasetSpec] = {}
        if not self._registry_dir.exists():
            log.warning("Registry directory not found: %s", self._registry_dir)
            return specs

        for yaml_path in sorted(self._registry_dir.glob("*.yaml")):
            try:
                spec = self._parse_yaml(yaml_path)
                if spec.name in specs:
                    log.warning(
                        "Duplicate dataset name '%s' in %s — overwriting.",
                        spec.name,
                        yaml_path,
                    )
                specs[spec.name] = spec
            except Exception as exc:
                log.warning("Failed to load registry YAML %s: %s", yaml_path, exc)
        return specs

    @staticmethod
    def _parse_yaml(path: Path) -> DatasetSpec:
        """Parse a single registry YAML into a ``DatasetSpec``."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Empty or invalid YAML: {path}")

        name = raw.get("name")
        if not name:
            raise ValueError(f"Registry YAML {path} is missing 'name'.")

        collection_raw = raw.get("collection", {})
        collection = CollectionConfig(
            n_steps=int(collection_raw.get("n_steps", 500_000)),
            n_valid=int(collection_raw.get("n_valid", 10_000)),
            n_test=int(collection_raw.get("n_test", 10_000)),
            shard_size=int(collection_raw.get("shard_size", 50_000)),
            seed=int(collection_raw.get("seed", 42)),
            policy=collection_raw.get("policy", "mixed"),
            policy_weights=collection_raw.get(
                "policy_weights",
                {"random": 0.25, "sed": 0.25, "qmd": 0.25, "softmax": 0.25},
            ),
        )

        return DatasetSpec(
            name=str(name),
            env=str(raw.get("env", "")),
            output_dir=str(raw.get("output_dir", "final_dataset/qgym")),
            collection=collection,
            training_defaults=raw.get("training_defaults", {}),
        )

    # ── Public API ────────────────────────────────────────────────────

    @property
    def specs(self) -> dict[str, DatasetSpec]:
        """All discovered dataset specs (lazy-loaded)."""
        if self._specs is None:
            self._specs = self._load_all()
        return self._specs

    def list_datasets(self) -> list[str]:
        """Return sorted names of all known datasets."""
        return sorted(self.specs.keys())

    def get(self, name: str) -> DatasetSpec:
        """Return the ``DatasetSpec`` for a given dataset name.

        Raises
        ------
        KeyError
            If ``name`` is not found in the registry.
        """
        if name not in self.specs:
            raise KeyError(
                f"Unknown dataset '{name}'. "
                f"Available: {', '.join(self.list_datasets())}"
            )
        return self.specs[name]

    def resolve_path(self, name: str) -> Path:
        """Return the canonical on-disk path for a dataset.

        The path is ``<output_root>/<name>/`` where ``output_root``
        is the registry's configured output root (defaults to
        ``PROJECT_ROOT / final_dataset/qgym``).
        """
        self.get(name)  # validates existence
        return (self._output_root / name).resolve()

    # ── Existence & integrity ─────────────────────────────────────────

    def exists(self, name: str) -> bool:
        """Check whether a dataset is fully collected on disk.

        Returns ``True`` only when *all three* split directories exist
        and contain at least one ``.pt`` shard each.
        """
        try:
            path = self.resolve_path(name)
        except KeyError:
            return False
        if not path.exists():
            return False
        for split in ("train", "valid", "test"):
            split_dir = path / split
            if not split_dir.exists():
                return False
            if not any(split_dir.glob("*.pt")):
                return False
        return True

    def status(self, name: str | None = None) -> dict[str, dict]:
        """Return collection status for one or all datasets.

        Parameters
        ----------
        name : str, optional
            If given, return status for a single dataset.
            If ``None``, return status for all known datasets.

        Returns
        -------
        dict
            Nested dict: ``{dataset_name: {"exists": bool, "path": str, ...}}``
        """
        names = [name] if name else self.list_datasets()
        result: dict[str, dict] = {}
        for n in names:
            try:
                path = self.resolve_path(n)
                exists = self.exists(n)
                info: dict = {
                    "exists": exists,
                    "path": str(path),
                }
                if exists:
                    meta_path = path / "metadata.yaml"
                    if meta_path.exists():
                        with open(meta_path) as f:
                            meta = yaml.safe_load(f) or {}
                        info["metadata"] = meta
                result[n] = info
            except KeyError:
                result[n] = {"exists": False, "error": f"Unknown dataset '{n}'"}
        return result

    def spec_to_dict(self, spec: DatasetSpec) -> dict:
        """Convert a ``DatasetSpec`` to a plain dict for serialisation."""
        return {
            "name": spec.name,
            "env": spec.env,
            "output_dir": spec.output_dir,
            "collection": dataclasses.asdict(spec.collection),
            "training_defaults": spec.training_defaults,
        }

    # ── Convenience ───────────────────────────────────────────────────

    def __contains__(self, name: str) -> bool:
        return name in self.specs

    def __repr__(self) -> str:
        names = self.list_datasets()
        return f"DatasetRegistry({len(names)} datasets: {', '.join(names)})"
