"""QGym adapter — interface between QGym environments and CertiQ-Net.

Provides ``QGymAdapter``, the bridge that maps a QGym ``(s, q)``
server-queue topology to the ``(N,)`` per-queue abstraction used
throughout CertiQ-Net.
"""

from __future__ import annotations

import dataclasses
import logging
import random as stdlib_random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from certiqnet.adapters.common.base import AdapterBatch, DispatchAdapter, DispatchTuple
from certiqnet.adapters.qgym.config import QGymEnvConfig, resolve_env_config_path
from certiqnet.adapters.qgym.env_loader import (
    compute_effective_mu,
    compute_queue_holding_cost,
    load_qgym_env,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Policy factories
# ---------------------------------------------------------------------------


def _make_random_policy(
    network_np: np.ndarray,
    s: int,
    q: int,
):
    """Return a closure that produces a random feasible action."""

    def _policy(_obs: np.ndarray) -> np.ndarray:
        action = np.random.uniform(0, 5, (s, q)).astype(np.float32)
        return action * network_np

    return _policy


def _make_sed_policy(
    network_np: np.ndarray,
    mu_effective: np.ndarray,
    s: int,
    q: int,
):
    """Return a closure implementing the SED (Shortest Expected Delay) policy."""

    def _policy(obs: np.ndarray) -> np.ndarray:
        Q = obs if isinstance(obs, np.ndarray) else obs.cpu().numpy()
        idx = int(((Q + 1.0) / (mu_effective + 1e-12)).argmin())
        action = np.zeros((s, q), dtype=np.float32)
        for server in range(s):
            if network_np[server, idx] > 0:
                action[server, idx] = max(1.0, float(Q[idx]))
        return action

    return _policy


def _make_qmd_policy(
    network_np: np.ndarray,
    mu_effective: np.ndarray,
    s: int,
    q: int,
):
    """Return a closure implementing the QMD (Quadratic Max-Drift) policy."""

    def _policy(obs: np.ndarray) -> np.ndarray:
        Q = obs if isinstance(obs, np.ndarray) else obs.cpu().numpy()
        idx = int(((2.0 * Q + 1.0) / (mu_effective + 1e-12)).argmin())
        action = np.zeros((s, q), dtype=np.float32)
        for server in range(s):
            if network_np[server, idx] > 0:
                action[server, idx] = max(1.0, float(Q[idx]))
        return action

    return _policy


def _make_softmax_policy(
    network_np: np.ndarray,
    mu_effective: np.ndarray,
    s: int,
    q: int,
    tau: float = 1.0,
):
    """Return a closure implementing a softmax (Boltzmann) policy over SED scores."""

    def _policy(obs: np.ndarray) -> np.ndarray:
        Q = obs if isinstance(obs, np.ndarray) else obs.cpu().numpy()
        scores = -((Q + 1.0) / (mu_effective + 1e-12))
        scores = scores / max(tau, 1e-8)
        exp_scores = np.exp(scores - scores.max())
        probs = exp_scores / (exp_scores.sum() + 1e-12)
        idx = int(np.random.choice(q, p=probs))
        action = np.zeros((s, q), dtype=np.float32)
        for server in range(s):
            if network_np[server, idx] > 0:
                action[server, idx] = max(1.0, float(Q[idx]))
        return action

    return _policy


def _make_mixed_policy(
    policies: dict[str, callable],
    weights: dict[str, float],
):
    """Return a closure that delegates to sub-policies by weighted sampling."""
    names = list(policies.keys())
    probs = np.array([weights.get(n, 0.0) for n in names], dtype=np.float64)
    probs = probs / (probs.sum() + 1e-12)

    def _policy(obs: np.ndarray) -> np.ndarray:
        chosen = names[int(np.random.choice(len(names), p=probs))]
        return policies[chosen](obs)

    return _policy


# ---------------------------------------------------------------------------
#  QGymAdapter
# ---------------------------------------------------------------------------


class QGymAdapter(DispatchAdapter):
    """Adapter that wraps a QGym environment as a data source.

    Supports two modes (set via ``mode``):

    * ``"online"`` — steps the QGym environment live during
      ``sample_batch()``, producing an infinite stream of states.
    * ``"static"`` — loads pre-collected ``.pt`` shards from a
      ``dataset/qgym/<dataset_name>/`` directory.

    The adapter maps QGym's ``(s, q)`` server-queue topology to the
    standard ``(N,)`` per-queue interface by computing **effective
    per-queue service rates** via ``(network * mu).sum(dim=0)``.

    Parameters
    ----------
    env_config : str | Path | dict | QGymEnvConfig | None
        QGym environment configuration.  Required when ``mode="online"``.
        Ignored when ``mode="static"``.
    dataset_path : str | Path | None
        Path to a ``dataset/qgym/<name>/`` directory.  Required when
        ``mode="static"``.
    mode : ``"online"`` | ``"static"``
        Data generation mode.
    policy : str
        Collection policy name for online generation (``"random"``,
        ``"sed"``, ``"qmd"``, ``"softmax"``, ``"mixed"``).
    policy_weights : dict[str, float] | None
        Per-policy sampling weights when ``policy="mixed"``.
    batch_size_env : int
        Number of parallel QGym environments for online generation.
    seed : int
        Random seed.
    device : str
        Torch device string.
    """

    CERTIFICATE_STATUS: str = "approximate"
    context_dim: int = 0

    def __init__(
        self,
        env_config: str | Path | dict | QGymEnvConfig | None = None,
        dataset_path: str | Path | None = None,
        mode: Literal["online", "static"] = "online",
        policy: str = "random",
        policy_weights: dict[str, float] | None = None,
        batch_size_env: int = 1,
        seed: int = 42,
        device: str = "cpu",
        assumptions_satisfied: bool = True,
    ) -> None:
        super().__init__(assumptions_satisfied=assumptions_satisfied)
        self.mode = mode
        self.policy = policy
        self.policy_weights = policy_weights or {
            "random": 0.25,
            "sed": 0.25,
            "qmd": 0.25,
            "softmax": 0.25,
        }
        self._device = device
        self._seed = seed
        self._batch_size_env = batch_size_env
        self._env = None
        self._env_config: QGymEnvConfig | dict | None = None
        self._dataset_shards: list[Path] = []
        self._shard_index = 0
        self._shard_data: dict | None = None
        self._metadata: dict = {}

        if mode == "online":
            if env_config is None:
                raise ValueError("env_config is required when mode='online'.")
            self._env_config = (
                QGymEnvConfig.from_yaml(resolve_env_config_path(str(env_config)))
                if isinstance(env_config, (str, Path))
                else env_config
            )
            self._env_config_source = (
                str(resolve_env_config_path(str(env_config)))
                if isinstance(env_config, (str, Path))
                else None
            )
            self._env = self._build_env()
            self._collect_policy = self._resolve_collect_policy()

        elif mode == "static":
            if dataset_path is None:
                raise ValueError("dataset_path is required when mode='static'.")
            self._dataset_path = Path(dataset_path)
            if not self._dataset_path.exists():
                raise FileNotFoundError(
                    f"QGym dataset directory not found: {self._dataset_path}"
                )
            self._load_dataset_metadata()

        else:
            raise ValueError(f"Unknown mode '{mode}'. Expected 'online' or 'static'.")

    # ── Environment construction ──────────────────────────────────────

    def _build_env(self):
        """Build the QGym environment from the stored config."""
        if isinstance(self._env_config, QGymEnvConfig):
            env_config_dict = dataclasses.asdict(self._env_config)
        elif isinstance(self._env_config, dict):
            env_config_dict = self._env_config.copy()
        else:
            env_config_dict = self._env_config.__dict__.copy()
        return load_qgym_env(
            env_config=env_config_dict,
            batch=self._batch_size_env,
            seed=self._seed,
            policy_name="WC",
            device=self._device,
        )

    # ── Policy resolution ─────────────────────────────────────────────

    def _extract_topology(self) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Extract (network_np, mu_effective_np, s, q) from the live env."""
        env = self._env
        network = getattr(env, "network", None)
        s = getattr(env, "s", 1)
        q = getattr(env, "q", 1)

        if network is not None:
            network_np = (
                network[0].cpu().numpy()
                if torch.is_tensor(network)
                else np.array(network)
            )
        else:
            network_np = np.ones((s, q), dtype=np.float32)

        mu_matrix = (
            env.mu[0].detach().cpu().numpy()
            if torch.is_tensor(env.mu)
            else np.array(env.mu)
        )
        mu_effective = compute_effective_mu(
            torch.tensor(network_np, dtype=torch.float),
            torch.tensor(mu_matrix, dtype=torch.float),
        ).numpy()

        return network_np, mu_effective, s, q

    def _resolve_collect_policy(self):
        """Return a callable ``obs -> action`` for data collection.

        Supports ``"random"``, ``"sed"``, ``"qmd"``, ``"softmax"``, and
        ``"mixed"`` (weighted combination of the others).
        """
        network_np, mu_effective, s, q = self._extract_topology()

        individual_policies = {
            "random": _make_random_policy(network_np, s, q),
            "sed": _make_sed_policy(network_np, mu_effective, s, q),
            "qmd": _make_qmd_policy(network_np, mu_effective, s, q),
            "softmax": _make_softmax_policy(network_np, mu_effective, s, q),
        }

        if self.policy == "mixed":
            return _make_mixed_policy(individual_policies, self.policy_weights)
        if self.policy in individual_policies:
            return individual_policies[self.policy]

        log.warning(
            "Unknown policy '%s', falling back to 'random'.", self.policy
        )
        return individual_policies["random"]

    # ── Static dataset loading ────────────────────────────────────────

    def _load_dataset_metadata(self) -> None:
        """Scan the dataset directory for shard files."""
        train_dir = self._dataset_path / "train"
        if not train_dir.exists():
            raise FileNotFoundError(f"No train/ directory in {self._dataset_path}")
        shard_files = sorted(train_dir.glob("shard_*.pt"))
        if not shard_files:
            raise FileNotFoundError(f"No shard files found in {train_dir}")
        self._dataset_shards = shard_files
        self._shard_index = 0

        meta_path = self._dataset_path / "metadata.yaml"
        if meta_path.exists():
            import yaml

            with open(meta_path) as f:
                self._metadata = yaml.safe_load(f) or {}

    def _load_next_shard(self) -> None:
        """Load the next dataset shard into memory (wraps at end)."""
        if self._shard_index >= len(self._dataset_shards):
            self._shard_index = 0
        shard_path = self._dataset_shards[self._shard_index]
        data = torch.load(shard_path, weights_only=True)
        # Validate shard integrity
        for required_key in ("Q", "cost"):
            if required_key not in data:
                raise KeyError(
                    f"Shard {shard_path.name} missing required key '{required_key}'."
                )
        self._shard_data = data
        self._shard_index += 1

    # ── Abstract method implementations ───────────────────────────────

    def to_dispatch_tuple(self, system_state: dict[str, object]) -> DispatchTuple:
        raise NotImplementedError("QGymAdapter does not support to_dispatch_tuple.")

    def apply_action(
        self, action: int, system_state: dict[str, object]
    ) -> dict[str, object]:
        raise NotImplementedError("QGymAdapter does not support apply_action.")

    def certificate_assumptions(self) -> list[str]:
        return [
            "Exact certificate inheritance requires the QGym environment's "
            "queueing dynamics and cost structure to be faithfully represented "
            "by the collected dataset.",
        ]

    # ── Properties ────────────────────────────────────────────────────

    @property
    def N(self) -> int:
        """Number of queues in the QGym environment."""
        if self._env is not None:
            return self._env.q
        if self._env_config is not None:
            if isinstance(self._env_config, dict):
                h = self._env_config.get("h", [])
                return len(h)
            return len(self._env_config.h) if self._env_config.h else 0
        return 0

    @property
    def env_h(self) -> Tensor | None:
        """Holding-cost vector from the QGym environment."""
        if self._env is not None:
            h_val = getattr(self._env, "h", None)
            if h_val is not None:
                if torch.is_tensor(h_val):
                    return h_val.detach().cpu().float()
                return torch.tensor(h_val, dtype=torch.float)
            return None
        if self._env_config is not None:
            if isinstance(self._env_config, dict):
                h = self._env_config.get("h")
                return torch.tensor(h, dtype=torch.float) if h else None
            return (
                torch.tensor(self._env_config.h, dtype=torch.float)
                if self._env_config.h
                else None
            )
        return None

    # ── Batch sampling ────────────────────────────────────────────────

    def sample_batch(
        self,
        *,
        n_samples: int,
        N: int,
        mu: Tensor,
        max_queue: int = 100,
        generator: torch.Generator | None = None,
    ) -> AdapterBatch:
        """Generate a batch of states from the QGym data source.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        N : int
            Expected number of queues (for consistency checking).
        mu : Tensor
            Service rates (ignored for QGym — uses environment's own rates).
        max_queue : int
            Ignored for QGym — environment determines state distribution.
        generator : torch.Generator | None
            Ignored for ``"static"`` mode (shard order is deterministic).

        Returns
        -------
        AdapterBatch with ``Q``, ``mu``, ``xi``, ``cost``.
        """
        del mu, max_queue
        if self.mode == "online":
            return self._sample_online(n_samples, N, generator)
        return self._sample_static(n_samples, N, generator)

    def _compute_cost(self, Q: Tensor) -> Tensor:
        """Compute per-sample cost, using holding-cost weights when available."""
        h = self.env_h
        if h is not None:
            return compute_queue_holding_cost(Q, h.to(Q.device))
        return Q.sum(dim=-1)

    def _sample_online(
        self,
        n_samples: int,
        N: int,
        generator: torch.Generator | None = None,
    ) -> AdapterBatch:
        """Step the live QGym environment to collect ``n_samples`` states."""
        env = self._env
        network_np, _, s, q = self._extract_topology()
        mu_effective = compute_effective_mu(
            torch.tensor(network_np, dtype=torch.float),
            torch.tensor(
                env.mu[0].detach().cpu().numpy()
                if torch.is_tensor(env.mu)
                else np.array(env.mu),
                dtype=torch.float,
            ),
        )

        # Seed numpy RNG from generator for reproducibility
        if generator is not None:
            seed = (generator.initial_seed() >> 32) ^ (
                generator.initial_seed() & 0xFFFFFFFF
            )
            np.random.seed(int(seed % (2**32)))

        Q_list: list[Tensor] = []
        cost_list: list[Tensor] = []

        obs, _ = env.reset()
        obs = obs.flatten() if hasattr(obs, "flatten") else obs
        collected = 0
        while collected < n_samples:
            action = self._collect_policy(obs)
            obs, reward, done, truncated, info = env.step(action)
            obs = obs.flatten() if hasattr(obs, "flatten") else obs
            Q_t = torch.tensor(obs, dtype=torch.float)
            Q_list.append(Q_t)
            cost_list.append(self._compute_cost(Q_t.unsqueeze(0)).squeeze(0))
            collected += 1
            if done or truncated:
                obs, _ = env.reset()
                obs = obs.flatten() if hasattr(obs, "flatten") else obs

        Q = torch.stack(Q_list)
        cost_t = torch.stack(cost_list)
        mu_b = mu_effective.unsqueeze(0).expand(Q.shape[0], -1)

        return AdapterBatch(Q=Q, mu=mu_b, xi=None, cost=cost_t)

    def _sample_static(
        self,
        n_samples: int,
        N: int,
        generator: torch.Generator | None = None,
    ) -> AdapterBatch:
        """Sample from pre-collected dataset shards.

        Aggregates across multiple shards if a single shard does not
        contain enough rows.
        """
        if self._shard_data is None:
            self._load_next_shard()

        Q_parts: list[Tensor] = []
        cost_parts: list[Tensor] = []
        mu_ref: Tensor | None = None
        remaining = n_samples

        while remaining > 0:
            data = self._shard_data
            total = data["Q"].shape[0]
            take = min(total, remaining)
            idx = torch.randperm(total, generator=generator)[:take]
            Q_parts.append(data["Q"][idx])
            cost_parts.append(data["cost"][idx])
            if mu_ref is None and "mu" in data:
                mu_ref = data["mu"]
            remaining -= take
            if remaining > 0:
                self._load_next_shard()

        Q = torch.cat(Q_parts, dim=0)
        cost = torch.cat(cost_parts, dim=0)

        if mu_ref is not None:
            mu_b = mu_ref.unsqueeze(0).expand(n_samples, -1)
        else:
            # Fallback: uniform unit rates
            mu_b = torch.ones(n_samples, Q.shape[-1])

        return AdapterBatch(Q=Q, mu=mu_b, xi=None, cost=cost)

    # ── Observation interface ─────────────────────────────────────────

    def make_observation(
        self, Q: Tensor, mu: Tensor
    ) -> tuple[Tensor, Tensor, None]:
        """Return tensors ready for a CertiQ-Net forward pass."""
        if mu.dim() == 1:
            mu = mu.unsqueeze(0).expand(Q.shape[0], -1)
        return Q, mu, None

    # ── Debugging ─────────────────────────────────────────────────────

    def __repr__(self) -> str:
        parts = [
            f"QGymAdapter(mode='{self.mode}'",
            f"policy='{self.policy}'",
            f"N={self.N}",
        ]
        if self.mode == "online":
            parts.append(f"batch_size_env={self._batch_size_env}")
        elif self.mode == "static":
            parts.append(f"n_shards={len(self._dataset_shards)}")
        parts.append(f"device='{self._device}'")
        return ", ".join(parts) + ")"
