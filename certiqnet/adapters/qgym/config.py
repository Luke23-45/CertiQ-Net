from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


# ---------------------------------------------------------------------------
#  QGym environment configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QGymEnvConfig:
    """Validated QGym environment configuration.

    Parameters
    ----------
    name : str
        Unique identifier for the environment topology (e.g. ``"reentrant_2"``).
    lam_type : str
        Arrival-rate mode (``"constant"``, ``"periodic"``, ``"trace"``).
    lam_params : dict
        Parameters for the arrival process (e.g. ``{"val": [1.0, 0.8]}``).
    network : list[list[float]] | None
        Server-to-queue connectivity matrix of shape ``(s, q)``.  ``None``
        defers to ``.npy`` files looked up by *name* at load time.
    mu : list[list[float]] | None
        Service-rate matrix of shape ``(s, q)``.  Same deferral rule as
        *network*.
    h : list[float] | None
        Per-queue holding-cost vector of length ``q``.
    init_queues : list[float] | None
        Initial queue lengths (length ``q``).
    queue_event_options : list[list[float]] | str | None
        Arrival-size distribution per queue, or ``"custom"`` to load from
        an ``.npy`` file.
    train_T : int
        Number of simulation steps for training episodes.
    test_T : int
        Number of simulation steps for evaluation episodes.
    num_pool : int
        Number of server pools.
    env_type : str | None
        Optional override for the environment-data subdirectory name.
    server_pool_size : list[int] | None
        Number of servers in each pool.
    """

    name: str
    lam_type: str = "constant"
    lam_params: dict = field(default_factory=lambda: {"val": None})
    service_type: str = "exponential"
    network: list[list[float]] | None = None
    mu: list[list[float]] | None = None
    h: list[float] | None = None
    init_queues: list[float] | None = None
    queue_event_options: list[list[float]] | str | None = None
    train_T: int = 20000
    test_T: int = 10000
    num_pool: int = 1
    env_type: str | None = None
    server_pool_size: list[int] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("QGymEnvConfig.name must be a non-empty string.")
        if self.train_T <= 0:
            raise ValueError(f"train_T must be positive, got {self.train_T}")
        if self.test_T <= 0:
            raise ValueError(f"test_T must be positive, got {self.test_T}")
        if self.h is not None and not all(
            isinstance(v, (int, float)) for v in self.h
        ):
            raise TypeError("h must be a list of numbers.")
        # Validate dimensional consistency when both network and mu are given
        if self.network is not None and self.mu is not None:
            n_rows = len(self.network)
            m_rows = len(self.mu)
            if n_rows != m_rows:
                raise ValueError(
                    f"network has {n_rows} rows but mu has {m_rows} rows "
                    "(server count mismatch)."
                )
            if n_rows > 0:
                n_cols = len(self.network[0])
                m_cols = len(self.mu[0])
                if n_cols != m_cols:
                    raise ValueError(
                        f"network has {n_cols} cols but mu has {m_cols} cols "
                        "(queue count mismatch)."
                    )
                if self.h is not None and len(self.h) != n_cols:
                    raise ValueError(
                        f"h has length {len(self.h)} but queue count is {n_cols}."
                    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> QGymEnvConfig:
        """Load from a YAML file, ignoring unknown keys."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        if raw is None:
            raise ValueError(f"Empty YAML file: {path}")
        known_fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known_fields})

    def to_dict(self) -> dict:
        """Serialise to a plain ``dict`` (safe for JSON/YAML)."""
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
#  QGym dataset-collection configuration
# ---------------------------------------------------------------------------

SUPPORTED_POLICIES = frozenset({"random", "sed", "qmd", "softmax", "mixed"})


@dataclass
class QGymCollectionConfig:
    """Configuration for QGym dataset collection.

    Parameters
    ----------
    dataset_name : str
        Name of the output dataset directory.
    env_config_path : str | Path
        Path to the environment YAML file.
    output_dir : str | Path
        Root directory for all collected datasets.
    n_steps : int
        Total training-set simulation steps.
    n_valid : int
        Number of validation states.
    n_test : int
        Number of test states.
    shard_size : int
        Maximum states per ``.pt`` shard file.
    seed : int
        Base random seed.
    force_collect : bool
        If *True*, overwrite an existing dataset.
    policy : str
        Collection policy (``"random"``, ``"sed"``, ``"qmd"``,
        ``"softmax"``, ``"mixed"``).
    policy_weights : dict[str, float]
        Per-policy sampling weights when ``policy="mixed"``.
    """

    dataset_name: str
    env_config_path: str | Path
    output_dir: str | Path = "dataset/qgym"
    n_steps: int = 500_000
    n_valid: int = 10_000
    n_test: int = 10_000
    shard_size: int = 50_000
    seed: int = 42
    force_collect: bool = False
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
        if not self.dataset_name:
            raise ValueError("dataset_name must be a non-empty string.")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")
        if self.n_valid < 0:
            raise ValueError(f"n_valid must be non-negative, got {self.n_valid}")
        if self.n_test < 0:
            raise ValueError(f"n_test must be non-negative, got {self.n_test}")
        if self.shard_size <= 0:
            raise ValueError(f"shard_size must be positive, got {self.shard_size}")
        if self.policy not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Unknown policy '{self.policy}'. "
                f"Supported: {sorted(SUPPORTED_POLICIES)}"
            )
        if self.policy == "mixed":
            if not self.policy_weights:
                raise ValueError(
                    "policy_weights must be non-empty when policy='mixed'."
                )
            bad = set(self.policy_weights) - (SUPPORTED_POLICIES - {"mixed"})
            if bad:
                raise ValueError(
                    f"Unknown policies in policy_weights: {sorted(bad)}. "
                    f"Supported: {sorted(SUPPORTED_POLICIES - {'mixed'})}"
                )
            total = sum(self.policy_weights.values())
            if not (0.99 < total < 1.01):
                raise ValueError(
                    f"policy_weights must sum to ~1.0, got {total:.4f}"
                )

    @classmethod
    def from_yaml(cls, path: str | Path) -> QGymCollectionConfig:
        """Load from a collection YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        if raw is None:
            raise ValueError(f"Empty YAML file: {path}")
        dataset_sec = raw.get("dataset", {})
        collection_sec = raw.get("collection", {})
        config = cls(
            dataset_name=dataset_sec["name"],
            env_config_path=dataset_sec["env"],
            output_dir=dataset_sec.get("output_dir", "dataset/qgym"),
            n_steps=collection_sec.get("n_steps", 500_000),
            n_valid=collection_sec.get("n_valid", 10_000),
            n_test=collection_sec.get("n_test", 10_000),
            shard_size=collection_sec.get("shard_size", 50_000),
            seed=collection_sec.get("seed", 42),
            force_collect=collection_sec.get("force_collect", False),
            policy=collection_sec.get("policy", "mixed"),
        )
        weights = collection_sec.get("policy_weights")
        if weights:
            config.policy_weights = weights
        return config

    def to_dict(self) -> dict:
        """Serialise to a plain ``dict``."""
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
#  Env-config path resolution
# ---------------------------------------------------------------------------


def resolve_env_config_path(env_name_or_path: str) -> Path:
    """Resolve an env config reference to an actual YAML file path.

    Checks (in order):
      1. ``configs/qgym/env/<name>.yaml``
      2. ``extern/QGym/configs/env/<name>.yaml``
      3. The path as-is
    """
    candidates = [
        Path("configs/qgym/env") / f"{env_name_or_path}.yaml",
        Path("extern/QGym/configs/env") / f"{env_name_or_path}.yaml",
        Path(env_name_or_path),
    ]
    # Also check with .yaml stripped if it already has an extension
    if "." in env_name_or_path:
        name_stem = env_name_or_path.rsplit(".", 1)[0]
        candidates.insert(2, Path(name_stem).with_suffix(".yaml"))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot resolve QGym env config '{env_name_or_path}'. "
        f"Searched: {[str(c) for c in candidates]}"
    )
