"""Lightning data module backed by QGym-sampled queue states.

This is the **standalone** QGym data module — it owns the full training
data lifecycle and does *not* delegate to the synthetic data module.

Supports two modes:

* **static** — pre-collected states from a ``dataset/qgym/<name>/``
  structured directory, loaded via ``QGymDataset`` shards.
* **online** — a live ``QGymAdapter`` steps a QGym environment during
  training, producing an infinite stream of realistic states.

The module integrates with the **dataset registry** (``DatasetRegistry``):
when ``dataset_name`` is provided the module resolves the dataset spec,
verifies it exists (auto-collecting if necessary), and configures its
training defaults from the centralized registry YAML.

Data Composition
~~~~~~~~~~~~~~~~
QGym-sampled states are the *primary* source.  On top of them the module
layers:

* **Policy replay** — states visited by the model's own policy.
* **Hard / adversarial states** — high-backlog states from the
  state-bank generator, ensuring coverage of rarely-visited but
  safety-critical regions.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:  # pragma: no cover
    pl = None

from certiqnet.adapters.qgym.adapter import QGymAdapter
from certiqnet.adapters.qgym.env_loader import compute_queue_holding_cost
from certiqnet.data.collection_manager import DatasetCollectionManager
from certiqnet.data.qgym.context import build_qgym_context, QGYM_CONTEXT_DIM
from certiqnet.data.qgym.dataset import QGymDataset
from certiqnet.data.registry import DatasetRegistry
from certiqnet.data.common.state_bank import (
    generate_disagreement_states,
    generate_state_bank,
)
from certiqnet.utils.platform import resolve_num_workers

log = logging.getLogger(__name__)

# Maximum allowed sum of mix fractions before we emit a warning.
_MAX_MIX_SUM = 1.0 + 1e-6


class QGymDataModule(pl.LightningDataModule if pl is not None else object):
    """Standalone data module for QGym-backed CertiQ-Net training.

    Parameters
    ----------
    N : int
        Number of queues.
    mu : Tensor
        Per-queue effective service rates, shape ``(N,)``.
    batch_size : int
        Batch size for data loaders.
    n_samples : int
        Number of training samples per epoch / resample step.
    num_workers : int, optional
        DataLoader worker count.
    seed : int
        Random seed.
    max_queue : int
        Maximum queue length for synthetic / hard-state generation.
    resample_every_epoch : bool
        Rebuild the training dataset at the start of each epoch.
    policy_buffer_max : int
        Maximum number of policy-visited states in the replay buffer.
    policy_mix_fraction : float
        Fraction of states drawn from the policy replay buffer.
    hard_state_fraction : float
        Fraction of *QGym-sourced* states replaced by high-backlog
        states from the state bank.
    dataset_path : str | Path, optional
        Path to a ``dataset/qgym/<name>/`` directory (static mode).
        Superseded by *dataset_name* — prefer using the registry.
    dataset_name : str, optional
        Name of a registered dataset (see ``DatasetRegistry``).
        When provided, the module resolves the canonical path from
        the registry and auto-collects if the data is missing.
    auto_collect : bool
        If ``True`` (default) and the dataset is missing when
        ``dataset_name`` is used, automatically collect it.
        If ``False``, raise ``FileNotFoundError`` with recovery
        instructions when the dataset is absent.
    qgym_adapter : QGymAdapter, optional
        Live QGym adapter instance (online mode).
    h : Tensor | None
        Per-queue holding-cost vector.  If provided, cost is computed
        as ``(Q * h).sum(-1)`` instead of ``Q.sum(-1)``.
    """

    def __init__(
        self,
        N: int,
        mu: Tensor,
        batch_size: int = 64,
        n_samples: int = 512,
        num_workers: int | None = None,
        seed: int = 0,
        max_queue: int = 15,
        datatype: str = "qgym",
        resample_every_epoch: bool = True,
        policy_buffer_max: int = 4096,
        policy_mix_fraction: float = 0.25,
        hard_state_fraction: float = 0.5,
        disagreement_fraction: float = 0.25,
        adversarial_fraction: float = 0.25,
        dataset_path: str | Path | None = None,
        dataset_name: str | None = None,
        auto_collect: bool = True,
        qgym_adapter: QGymAdapter | None = None,
        h: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.N = int(N)
        self.mu = mu.float()
        self.datatype = str(datatype)
        self.batch_size = int(batch_size)
        self.n_samples = int(n_samples)
        self._num_workers = (
            0 if resample_every_epoch else resolve_num_workers(num_workers)
        )
        self.seed = int(seed)
        self.max_queue = int(max_queue)
        self.resample_every_epoch = bool(resample_every_epoch)
        self.policy_buffer_max = int(policy_buffer_max)
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.auto_collect = bool(auto_collect)
        self.qgym_adapter = qgym_adapter

        # ── Validate and store mix fractions ──────────────────────────
        self.policy_mix_fraction = float(policy_mix_fraction)
        self.hard_state_fraction = float(hard_state_fraction)
        self.disagreement_fraction = float(disagreement_fraction)
        self.adversarial_fraction = float(adversarial_fraction)

        if self.policy_mix_fraction > _MAX_MIX_SUM:
            raise ValueError(
                f"policy_mix_fraction={self.policy_mix_fraction} exceeds 1.0."
            )
        if self.hard_state_fraction + self.disagreement_fraction > 1.0 + 1e-6:
            raise ValueError(
                "hard_state_fraction + disagreement_fraction must not exceed 1.0."
            )

        # ── Mode detection ────────────────────────────────────────────
        self._online = (
            qgym_adapter is not None
            and getattr(qgym_adapter, "mode", "static") == "online"
        )
        self._static = dataset_path is not None or dataset_name is not None

        # ── Holding cost ──────────────────────────────────────────────
        if h is not None:
            self._h = h.float()
        elif qgym_adapter is not None:
            self._h = getattr(qgym_adapter, "env_h", None)
            if self._h is not None:
                self._h = self._h.float()
        else:
            self._h = None

        # ── Context dimension ─────────────────────────────────────────
        self.context_dim = 0
        if qgym_adapter is not None:
            self.context_dim = int(getattr(qgym_adapter, "context_dim", 0))

        # ── State ─────────────────────────────────────────────────────
        self._qgym_train_ds: QGymDataset | None = None
        self._qgym_test_ds: QGymDataset | None = None
        self._epoch = 0
        self.train_ds: TensorDataset | None = None
        self.val_ds: TensorDataset | None = None
        self.test_ds: TensorDataset | None = None
        self._policy_buffer: deque[Tensor] = deque(maxlen=self.policy_buffer_max)

    def _unpack_sample(self, sample: tuple[Tensor, ...]) -> tuple[Tensor, Tensor, Tensor | None, Tensor]:
        if len(sample) == 4:
            Q, mu, cost, xi = sample
            return Q, mu, xi, cost
        if len(sample) == 3:
            Q, mu, cost = sample
            return Q, mu, None, cost
        raise ValueError(f"Unexpected sample length: {len(sample)}")

    def _context_dim_from_data(self) -> int:
        if self.context_dim > 0:
            return self.context_dim
        if self._qgym_train_ds is not None and getattr(self._qgym_train_ds, "context_dim", 0) > 0:
            return int(self._qgym_train_ds.context_dim)
        if self._qgym_test_ds is not None and getattr(self._qgym_test_ds, "context_dim", 0) > 0:
            return int(self._qgym_test_ds.context_dim)
        return 0

    def _ensure_context(
        self,
        Q: Tensor,
        mu: Tensor,
        xi: Tensor | None = None,
        *,
        network: Tensor | None = None,
        history_Q: list[Tensor] | None = None,
        history_action: list[Tensor] | None = None,
        history_dt: list[Tensor] | None = None,
    ) -> Tensor | None:
        ctx_dim = self._context_dim_from_data()
        if ctx_dim <= 0:
            return None
        if xi is not None:
            if xi.dim() == 2 and xi.shape[-1] == ctx_dim:
                return xi.float()
            if xi.dim() == 3 and xi.shape[-1] == ctx_dim:
                return xi.float()
        return build_qgym_context(
            Q,
            mu,
            history_Q=history_Q,
            history_action=history_action,
            history_dt=history_dt,
            network=network,
            context_dim=ctx_dim if ctx_dim > 0 else QGYM_CONTEXT_DIM,
        )

    # ── Compatibility properties ──────────────────────────────────────

    @property
    def adapter(self) -> QGymAdapter | None:
        """Expose the QGym adapter as ``dm.adapter`` for training-module compatibility.

        Training modules (``QueueingLightningModule`` et al.) access
        ``dm.adapter.make_observation(Q, mu)`` during rollouts.  The
        adapter is stored as ``self.qgym_adapter``.
        """
        return self.qgym_adapter

    @property
    def h(self) -> Tensor | None:
        """Per-queue holding-cost vector (or ``None`` for uniform cost)."""
        return self._h

    # ── Cost helper ───────────────────────────────────────────────────

    def _compute_cost(self, Q: Tensor) -> Tensor:
        """Compute per-sample cost.  Uses *h*-weighted cost when available."""
        if self._h is not None:
            return compute_queue_holding_cost(Q, self._h.to(Q.device))
        return Q.sum(dim=-1)

    # ── Registry resolution ───────────────────────────────────────────

    def _resolve_dataset_from_registry(self) -> None:
        """Resolve ``dataset_name`` through the registry.

        Populates ``dataset_path`` (static mode) or ``qgym_adapter``
        (online mode).  Auto-collects if the data is missing when
        ``auto_collect=True``.
        """
        registry = DatasetRegistry()

        if self.dataset_name not in registry:
            raise KeyError(
                f"Unknown dataset '{self.dataset_name}'. "
                f"Available: {', '.join(registry.list_datasets())}"
            )

        spec = registry.get(self.dataset_name)
        log.info(
            "Resolved dataset '%s' from registry: env=%s, policy=%s",
            self.dataset_name,
            spec.env,
            spec.collection.policy,
        )

        # ── Determine mode ────────────────────────────────────────────
        mode = spec.training_defaults.get("mode", "static")
        self._mode = mode
        self._online = mode == "online"
        self._static = mode == "static"

        if mode == "static":
            canonical_path = registry.resolve_path(self.dataset_name)
            if not registry.exists(self.dataset_name):
                manager = DatasetCollectionManager(registry)
                try:
                    manager.ensure(
                        self.dataset_name,
                        auto_collect=self.auto_collect,
                    )
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"Dataset '{self.dataset_name}' not found at "
                        f"{canonical_path}. "
                        f"Set auto_collect=True or run:\n"
                        f"    python -m certiqnet.data.qgym.collect_qgym "
                        f"collect {self.dataset_name}"
                    )
            self.dataset_path = str(canonical_path)
            log.info(
                "Using static dataset '%s' from %s",
                self.dataset_name,
                self.dataset_path,
            )

        elif mode == "online":
            if self.qgym_adapter is not None:
                log.info(
                    "Using provided QGymAdapter for online dataset '%s'",
                    self.dataset_name,
                )
                return
            log.info(
                "Building online QGymAdapter for dataset '%s' (env: %s)...",
                self.dataset_name,
                spec.env,
            )
            self.qgym_adapter = QGymAdapter(
                env_config=str(spec.env),
                mode="online",
                policy=spec.collection.policy,
                policy_weights=spec.collection.policy_weights,
                batch_size_env=spec.collection.batch_size_env,
                seed=spec.collection.seed,
                device="cpu",
            )
            log.info("Online QGymAdapter created: %s", self.qgym_adapter)
        else:
            raise ValueError(f"Unknown mode '{mode}' for dataset '{self.dataset_name}'.")

        # ── Apply training defaults from spec (low-priority) ──────────
        defaults = spec.training_defaults
        if defaults:
            if (
                self.policy_mix_fraction == 0.25
                and "policy_mix_fraction" in defaults
            ):
                self.policy_mix_fraction = float(defaults["policy_mix_fraction"])
            if (
                self.hard_state_fraction == 0.5
                and "hard_state_fraction" in defaults
            ):
                self.hard_state_fraction = float(defaults["hard_state_fraction"])
            if (
                self.disagreement_fraction == 0.25
                and "disagreement_fraction" in defaults
            ):
                self.disagreement_fraction = float(defaults["disagreement_fraction"])
            if (
                self.adversarial_fraction == 0.25
                and "adversarial_fraction" in defaults
            ):
                self.adversarial_fraction = float(
                    defaults["adversarial_fraction"]
                )

    # ── Setup ─────────────────────────────────────────────────────────

    def setup(self, stage: str | None = None) -> None:
        self._epoch = 0

        # ── Resolve dataset_name through the registry ─────────────────
        if self.dataset_name is not None and self.dataset_path is None:
            self._resolve_dataset_from_registry()

        if stage in (None, "fit", "validate"):
            self._setup_validation()
            self._setup_train_source()
            self._build_train()

        if stage in (None, "test"):
            self._setup_test()

    def _setup_validation(self) -> None:
        """Build the validation dataset."""
        n_val = max(128, self.n_samples // 8)

        if self._static:
            val_ds = QGymDataset(str(self.dataset_path), split="valid")
            self.context_dim = max(self.context_dim, int(getattr(val_ds, "context_dim", 0)))
            n_val = min(len(val_ds), n_val)
            rows = [self._unpack_sample(val_ds[i]) for i in range(n_val)]
            val_Q = torch.stack([row[0] for row in rows])
            val_mu = torch.stack([row[1] for row in rows])
            val_xi = torch.stack([row[2] for row in rows]) if rows and rows[0][2] is not None else None
            val_xi = self._ensure_context(val_Q, val_mu, val_xi, network=getattr(val_ds, "network", None))
            val_cost = self._compute_cost(val_Q)
            self.val_ds = TensorDataset(val_Q, val_mu, val_xi, val_cost) if val_xi is not None else TensorDataset(val_Q, val_mu, val_cost)

        elif self._online:
            val_gen = torch.Generator().manual_seed(self.seed + 1)
            val = self.qgym_adapter.sample_batch(
                n_samples=n_val,
                N=self.N,
                mu=self.mu,
                generator=val_gen,
            )
            val_cost = self._compute_cost(val.Q.float())
            self.val_ds = TensorDataset(val.Q.float(), val.mu.float(), val.xi.float(), val_cost) if val.xi is not None else TensorDataset(val.Q.float(), val.mu.float(), val_cost)

        else:
            # Pure-synthetic fallback (should not normally be reached for QGym)
            val_gen = torch.Generator().manual_seed(self.seed + 1)
            Q = torch.randint(
                0, max(1, self.max_queue), (n_val, self.N), generator=val_gen
            ).float()
            mu = self.mu.unsqueeze(0).expand(Q.shape[0], -1)
            cost = self._compute_cost(Q)
            xi = self._ensure_context(Q, mu)
            self.val_ds = TensorDataset(Q, mu, xi, cost) if xi is not None else TensorDataset(Q, mu, cost)

    def _setup_train_source(self) -> None:
        """Load the static training dataset (if applicable)."""
        if self._static:
            self._qgym_train_ds = QGymDataset(
                str(self.dataset_path), split="train"
            )
            # Use the dataset's own mu and N so they match the shard data,
            # regardless of what the experiment env config specifies.
            if self._qgym_train_ds.mu is not None:
                self.mu = self._qgym_train_ds.mu
            self.context_dim = max(self.context_dim, int(getattr(self._qgym_train_ds, "context_dim", 0)))
            self.N = self._qgym_train_ds.N

    def _setup_test(self) -> None:
        """Build the test dataset from the ``test`` split when available."""
        if self._static:
            test_dir = Path(str(self.dataset_path)) / "test"
            if test_dir.exists() and any(test_dir.glob("*.pt")):
                test_ds = QGymDataset(str(self.dataset_path), split="test")
                self.context_dim = max(self.context_dim, int(getattr(test_ds, "context_dim", 0)))
                n_test = len(test_ds)
                rows = [self._unpack_sample(test_ds[i]) for i in range(n_test)]
                test_Q = torch.stack([row[0] for row in rows])
                test_mu = torch.stack([row[1] for row in rows])
                test_xi = torch.stack([row[2] for row in rows]) if rows and rows[0][2] is not None else None
                test_xi = self._ensure_context(test_Q, test_mu, test_xi, network=getattr(test_ds, "network", None))
                test_cost = self._compute_cost(test_Q)
                self.test_ds = TensorDataset(test_Q, test_mu, test_xi, test_cost) if test_xi is not None else TensorDataset(test_Q, test_mu, test_cost)
                self._qgym_test_ds = test_ds
                return

        # Fallback: reuse validation data for test
        if self.val_ds is not None:
            self.test_ds = self.val_ds

    # ── Replay buffers ────────────────────────────────────────────────

    def record_policy_states(self, Q: Tensor) -> None:
        """Append model-visited states to the policy replay buffer."""
        self._policy_buffer.append(Q.detach().cpu())

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _sample_rows(
        tensor: Tensor, count: int, generator: torch.Generator
    ) -> Tensor:
        """Sub-sample or over-sample *tensor* to exactly *count* rows."""
        if count <= 0 or tensor.numel() == 0:
            return tensor[:0]
        if tensor.shape[0] >= count:
            idx = torch.randperm(tensor.shape[0], generator=generator)[:count]
            return tensor[idx]
        # Over-sample with replacement
        idx = torch.randint(0, tensor.shape[0], (count,), generator=generator)
        return tensor[idx]

    def _buffer_tensor(self, buffer: deque[Tensor]) -> Tensor:
        """Concatenate all entries in a replay buffer into one tensor."""
        if not buffer:
            return torch.empty(0, self.N)
        return torch.cat(list(buffer), dim=0)

    # ── Main training-set builder ─────────────────────────────────────

    def _build_train(self) -> None:
        """Assemble the composite training set for the current epoch."""
        gen = torch.Generator().manual_seed(self.seed + self._epoch * 9973)

        # ── Compute counts ────────────────────────────────────────────
        qgym_fraction = max(
            0.0, 1.0 - self.policy_mix_fraction,
        )
        qgym_count = max(1, int(self.n_samples * qgym_fraction))
        hard_count = max(0, int(qgym_count * self.hard_state_fraction))
        disagreement_count = max(0, int(qgym_count * self.disagreement_fraction))
        easy_qgym_count = max(0, qgym_count - hard_count - disagreement_count)

        policy_count = max(0, int(self.n_samples * self.policy_mix_fraction))
        adversarial_count = max(1, int(self.n_samples * self.adversarial_fraction))

        # ── QGym source ──────────────────────────────────────────────
        if self._online and self.qgym_adapter is not None and easy_qgym_count > 0:
            batch = self.qgym_adapter.sample_batch(
                n_samples=easy_qgym_count,
                N=self.N,
                mu=self.mu,
                generator=gen,
            )
            qgym_q = batch.Q.float()
            qgym_mu = batch.mu.float() if batch.mu is not None else self.mu.unsqueeze(0).expand(easy_qgym_count, -1)
            qgym_xi = batch.xi.float() if batch.xi is not None else None
        elif (
            self._static
            and self._qgym_train_ds is not None
            and easy_qgym_count > 0
        ):
            n = min(easy_qgym_count, len(self._qgym_train_ds))
            idx = torch.randperm(len(self._qgym_train_ds), generator=gen)[:n]
            rows = [self._qgym_train_ds[int(i)] for i in idx]
            unpacked = [self._unpack_sample(r) for r in rows]
            qgym_q = torch.stack([r[0] for r in unpacked])
            qgym_mu = torch.stack([r[1] for r in unpacked])
            qgym_xi = torch.stack([r[2] for r in unpacked]) if unpacked and unpacked[0][2] is not None else None
        else:
            qgym_q = torch.empty(0, self.N)
            qgym_mu = torch.empty(0, self.N)
            qgym_xi = None

        # ── Replay buffers ───────────────────────────────────────────
        policy_states = self._sample_rows(
            self._buffer_tensor(self._policy_buffer), policy_count, gen
        )

        # Bootstrap from QGym source if policy buffer is empty
        if policy_states.numel() == 0 and policy_count > 0:
            policy_states = self._bootstrap_states(policy_count, gen)

        # ── Hard / adversarial states ────────────────────────────────
        bank = generate_state_bank(
            N=self.N,
            mu=self.mu,
            beta=1.0,
            R_cert=float("inf"),
            n_random=max(adversarial_count * 4, hard_count * 4, disagreement_count * 4),
            n_grid=0,
            n_boundary=max(64, self.N * 8),
        )
        backlog = bank.sum(dim=-1)
        if hard_count > 0:
            k = min(hard_count, bank.shape[0])
            hard_rank = torch.topk(backlog, k=max(k, 1), largest=True).indices
            hard_states = bank[hard_rank]
            if hard_states.shape[0] < hard_count:
                hard_states = self._sample_rows(hard_states, hard_count, gen)
        else:
            hard_states = torch.empty(0, self.N)

        if disagreement_count > 0:
            disagreement_states = generate_disagreement_states(
                mu=self.mu,
                N=self.N,
                n_states=disagreement_count,
                bank_size=max(1024, disagreement_count * 16),
                beta=1.0,
                bank=bank,
            )
            if disagreement_states.shape[0] < disagreement_count:
                disagreement_states = self._sample_rows(disagreement_states, disagreement_count, gen)
        else:
            disagreement_states = torch.empty(0, self.N)
        adversarial_states = self._sample_rows(bank, adversarial_count, gen)

        # ── Concatenate all sources ──────────────────────────────────
        Q = torch.cat(
            [
                qgym_q,
                hard_states.float(),
                disagreement_states.float(),
                policy_states.float(),
                adversarial_states.float(),
            ],
            dim=0,
        )
        mu = torch.cat(
            [
                qgym_mu,
                self.mu.unsqueeze(0).expand(hard_states.shape[0], -1),
                self.mu.unsqueeze(0).expand(disagreement_states.shape[0], -1),
                self.mu.unsqueeze(0).expand(policy_states.shape[0], -1),
                self.mu.unsqueeze(0).expand(adversarial_states.shape[0], -1),
            ],
            dim=0,
        )
        cost = self._compute_cost(Q)
        network = None
        if self.qgym_adapter is not None:
            network = self.qgym_adapter.env_network
        elif self._qgym_train_ds is not None:
            network = getattr(self._qgym_train_ds, "network", None)

        xi_parts: list[Tensor] = []
        offset = 0
        source_specs = [
            (qgym_q, qgym_mu, qgym_xi),
            (hard_states.float(), self.mu.unsqueeze(0).expand(hard_states.shape[0], -1), None),
            (disagreement_states.float(), self.mu.unsqueeze(0).expand(disagreement_states.shape[0], -1), None),
            (policy_states.float(), self.mu.unsqueeze(0).expand(policy_states.shape[0], -1), None),
            (adversarial_states.float(), self.mu.unsqueeze(0).expand(adversarial_states.shape[0], -1), None),
        ]
        for q_part, mu_part, xi_part in source_specs:
            if q_part.numel() == 0:
                continue
            if xi_part is not None:
                xi_parts.append(xi_part.float())
            else:
                xi_parts.append(
                    self._ensure_context(
                        q_part.float(),
                        mu_part.float(),
                        network=network,
                    )
                )
            offset += q_part.shape[0]
        xi = torch.cat(xi_parts, dim=0) if xi_parts else None
        tensors = (Q, mu, xi, cost) if xi is not None else (Q, mu, cost)

        if self.train_ds is None:
            self.train_ds = TensorDataset(*tensors)
        else:
            self.train_ds.tensors = tensors

    def _bootstrap_states(
        self, count: int, gen: torch.Generator
    ) -> Tensor:
        """Draw bootstrap states from the primary QGym source."""
        if self._online and self.qgym_adapter is not None:
            batch = self.qgym_adapter.sample_batch(
                n_samples=count, N=self.N, mu=self.mu, generator=gen
            )
            return batch.Q.float()
        if self._static and self._qgym_train_ds is not None:
            n = min(count, len(self._qgym_train_ds))
            idx = torch.randint(
                0, len(self._qgym_train_ds), (n,), generator=gen
            )
            return torch.stack(
                [self._qgym_train_ds[int(i)][0] for i in idx]
            )
        # Synthetic fallback
        return torch.randint(
            0, max(1, self.max_queue), (max(1, count), self.N), generator=gen
        ).float()

    # ── Epoch management ──────────────────────────────────────────────

    def resample_train_data(self) -> None:
        """Rebuild the training set (called at the start of each epoch)."""
        if not self.resample_every_epoch:
            return
        self._epoch += 1
        self._build_train()

    # ── Collation ─────────────────────────────────────────────────────

    @staticmethod
    def _collate(batch: list[tuple[Tensor, ...]]) -> dict[str, Tensor]:
        """Convert a list of tuples into a named dictionary batch.

        Handles the following tuple lengths:

        * 4: ``(Q, mu, xi, cost)``
        * 3: ``(Q, mu, cost)``
        """
        first = batch[0]
        n = len(first)

        if n == 4:
            Q, mu, xi, cost = zip(*batch, strict=True)
            return {
                "Q": torch.stack(list(Q)),
                "mu": torch.stack(list(mu)),
                "xi": torch.stack(list(xi)),
                "cost": torch.stack(list(cost)),
            }
        if n == 3:
            Q, mu, cost = zip(*batch, strict=True)
            return {
                "Q": torch.stack(list(Q)),
                "mu": torch.stack(list(mu)),
                "cost": torch.stack(list(cost)),
            }
        raise ValueError(f"Unexpected batch element length: {n}")

    # ── Data loaders ──────────────────────────────────────────────────

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None, (
            "train_ds is None — call setup('fit') first."
        )
        pin = self._num_workers > 0 and torch.cuda.is_available()
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate,
            num_workers=self._num_workers,
            pin_memory=pin,
            persistent_workers=self._num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None, (
            "val_ds is None — call setup('fit') first."
        )
        pin = self._num_workers > 0 and torch.cuda.is_available()
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            collate_fn=self._collate,
            num_workers=self._num_workers,
            pin_memory=pin,
            persistent_workers=self._num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_ds is not None, (
            "test_ds is None — call setup('test') first."
        )
        pin = self._num_workers > 0 and torch.cuda.is_available()
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            collate_fn=self._collate,
            num_workers=self._num_workers,
            pin_memory=pin,
            persistent_workers=self._num_workers > 0,
        )

    # ── Debugging ─────────────────────────────────────────────────────

    def __repr__(self) -> str:
        mode = "online" if self._online else ("static" if self._static else "fallback")
        return (
            f"QGymDataModule(N={self.N}, mode='{mode}', "
            f"n_samples={self.n_samples}, batch_size={self.batch_size}, "
            f"mix=[qgym={1 - self.policy_mix_fraction:.2f}, "
            f"policy={self.policy_mix_fraction:.2f}], "
            f"h={'yes' if self._h is not None else 'no'})"
        )
