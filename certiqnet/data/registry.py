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
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

log = logging.getLogger(__name__)

_SUPPORTED_POLICIES = frozenset({"random", "sed", "qmd", "softmax", "mixed"})


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively deep-merge ``overrides`` into ``base``.

    Both *base* and *overrides* are left unchanged (pure function).
    Non-dict leaf values in *overrides* replace the corresponding value
    in *base*.  Dict values are merged recursively.
    """
    result = dict(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
#  Canonical paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
"""Absolute path to the project root (CertiQ‑Net/)."""

_REGISTRY_DIR = PROJECT_ROOT / "configs" / "dataset" / "qgym" / "registry"
"""Directory containing centralized dataset YAML definitions."""

_DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "final_dataset" / "qgym"
"""Canonical root for all collected QGym datasets."""


def _canonicalise_for_hash(value: object) -> object:
    """Normalise nested data into a JSON-serialisable structure."""
    if dataclasses.is_dataclass(value):
        return _canonicalise_for_hash(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _canonicalise_for_hash(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonicalise_for_hash(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def dataset_spec_hash(spec: "DatasetSpec") -> str:
    """Return a stable fingerprint for a dataset spec."""
    payload = _canonicalise_for_hash(DatasetRegistry.spec_to_dict(spec))
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _metadata_matches_spec(meta: dict, spec: "DatasetSpec") -> bool | None:
    """Check whether stored metadata still matches the registry spec."""
    stored_hash = meta.get("spec_hash")
    if stored_hash:
        return stored_hash == dataset_spec_hash(spec)

    env_path = Path(spec.env)
    if env_path.is_absolute():
        expected_env = str(env_path.resolve())
    else:
        project_env = (PROJECT_ROOT / env_path).resolve()
        expected_env = str(project_env if project_env.exists() else env_path)

    checks: list[bool] = []
    for key, expected in (
        ("n_train", spec.collection.n_steps),
        ("n_valid", spec.collection.n_valid),
        ("n_test", spec.collection.n_test),
        ("seed", spec.collection.seed),
        ("policy", spec.collection.policy),
        ("env_config", expected_env),
    ):
        if key in meta:
            checks.append(meta.get(key) == expected)

    if not checks:
        return None
    return all(checks)

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
    batch_size_env: int = 1
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
        if self.batch_size_env <= 0:
            raise ValueError(
                f"batch_size_env must be positive, got {self.batch_size_env}"
            )
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

    The authoritative source of truth for this dataset's collection
    parameters and training defaults.  Stored in ``datasets.yaml``
    (multi-dataset format) or, for legacy support, in its own YAML file.

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
    env_N: int | None = None
    env_mu_fixed: list[float] | None = None
    env_lam: float | None = None
    output_dir: str = "final_dataset/qgym"
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    training_defaults: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
#  Parsing helpers
# ---------------------------------------------------------------------------


def _parse_single_spec(raw: dict, source: str | Path) -> DatasetSpec:
    """Parse a raw *flat* dict (all keys at top level) into a ``DatasetSpec``."""
    name = raw.get("name")
    if not name:
        raise ValueError(f"Missing 'name' in dataset spec from {source}.")

    collection_raw = raw.get("collection") or {}
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

    env_N_raw = raw.get("env_N")
    env_N = int(env_N_raw) if env_N_raw is not None else None

    env_mu_fixed_raw = raw.get("env_mu_fixed")
    env_mu_fixed = (
        [float(v) for v in env_mu_fixed_raw] if env_mu_fixed_raw else None
    )

    env_lam_raw = raw.get("env_lam")
    env_lam = float(env_lam_raw) if env_lam_raw is not None else None

    return DatasetSpec(
        name=str(name),
        env=str(raw.get("env", "")),
        env_N=env_N,
        env_mu_fixed=env_mu_fixed,
        env_lam=env_lam,
        output_dir=str(raw.get("output_dir", "final_dataset/qgym")),
        collection=collection,
        training_defaults=raw.get("training_defaults") or {},
    )


# ---------------------------------------------------------------------------
#  Registry
# ---------------------------------------------------------------------------


class DatasetRegistry:
    """Manages known QGym datasets: discovery, loading, existence checks.

    The registry scans ``configs/dataset/qgym/registry/*.yaml`` to discover
    all datasets.  Each YAML file may contain either:

    - A **multi-dataset** file with ``defaults`` and ``datasets`` sections
      (preferred — see ``datasets.yaml``).
    - A **legacy** single-dataset file with a top-level ``name`` key.
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
        """Load and index all registry YAML files.

        Supports two YAML formats:
        1. **Multi-dataset** (preferred) — a single file with ``defaults``
           and ``datasets`` sections.
        2. **Legacy** — one ``DatasetSpec`` per YAML file, with a ``name``
           top-level key.
        """
        specs: dict[str, DatasetSpec] = {}
        if not self._registry_dir.exists():
            log.warning("Registry directory not found: %s", self._registry_dir)
            return specs

        for yaml_path in sorted(self._registry_dir.glob("*.yaml")):
            try:
                parsed = self._parse_yaml(yaml_path)
                for name, spec in parsed.items():
                    if name in specs:
                        log.warning(
                            "Duplicate dataset name '%s' in %s — overwriting.",
                            name,
                            yaml_path,
                        )
                    specs[name] = spec
            except Exception as exc:
                log.warning("Failed to load registry YAML %s: %s", yaml_path, exc)
        return specs

    @staticmethod
    def _parse_yaml(path: Path) -> dict[str, DatasetSpec]:
        """Parse a registry YAML into one or more ``DatasetSpec`` objects.

        Format detection
        ----------------
        - If the YAML has a top-level ``datasets`` key  → multi-dataset format.
          The ``defaults`` key (optional) provides shared parameters that are
          deep-merged into every entry under ``datasets``.
        - Otherwise → legacy single-dataset format with a top-level ``name`` key.
        """
        with open(path) as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Empty or invalid YAML: {path}")

        # ── Multi-dataset format ────────────────────────────────────────
        if "datasets" in raw:
            defaults = raw.get("defaults", {})
            datasets_raw = raw.get("datasets", {})
            if not isinstance(datasets_raw, dict):
                raise ValueError(
                    f"Registry YAML {path}: 'datasets' must be a mapping."
                )

            results: dict[str, DatasetSpec] = {}
            for ds_name, overrides in datasets_raw.items():
                if not isinstance(overrides, dict):
                    log.warning(
                        "Dataset '%s' in %s must be a mapping — skipping.",
                        ds_name, path,
                    )
                    continue
                try:
                    merged = _deep_merge(dict(defaults), overrides)
                    merged["name"] = ds_name
                    spec = _parse_single_spec(merged, path)
                    results[spec.name] = spec
                except Exception as exc:
                    log.warning(
                        "Failed to parse dataset '%s' in %s: %s — skipping.",
                        ds_name, path, exc,
                    )
            return results

        # ── Legacy single-dataset format ────────────────────────────────
        if "name" in raw:
            return {str(raw["name"]): _parse_single_spec(raw, path)}

        raise ValueError(
            f"Registry YAML {path}: must contain 'datasets' or 'name' key."
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
        meta_path = path / "metadata.yaml"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = yaml.safe_load(f) or {}
                matches = _metadata_matches_spec(meta, self.get(name))
                if matches is False:
                    return False
            except Exception:
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
                        matches = _metadata_matches_spec(meta, self.get(n))
                        if matches is not None:
                            info["spec_matches_registry"] = matches
                result[n] = info
            except KeyError:
                result[n] = {"exists": False, "error": f"Unknown dataset '{n}'"}
        return result

    @staticmethod
    def spec_to_dict(spec: DatasetSpec) -> dict:
        """Convert a ``DatasetSpec`` to a plain dict for serialisation."""
        return {
            "name": spec.name,
            "env": spec.env,
            "env_N": spec.env_N,
            "env_mu_fixed": spec.env_mu_fixed,
            "env_lam": spec.env_lam,
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
