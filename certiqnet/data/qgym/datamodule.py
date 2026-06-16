"""Lightning data module backed by QGym-sampled queue states.

This is the **standalone** QGym data module — it owns the full training
data lifecycle and does *not* delegate to the synthetic data module.

Supports two modes:

* **static** — pre-collected states from a ``dataset/qgym/<name>/``
  structured directory, loaded via ``QGymDataset`` shards.
* **online** — a live ``QGymAdapter`` steps a QGym environment during
  training, producing an infinite stream of realistic states.

Data Composition
~~~~~~~~~~~~~~~~
QGym-sampled states are the *primary* source.  On top of them the module
layers:

* **Teacher replay** — states visited by the SED/QMD expert during
  previous rollouts.
* **Policy replay** — states visited by the model's own policy.
* **Hard / adversarial states** — high-backlog states from the
  state-bank generator, ensuring coverage of rarely-visited but
  safety-critical regions.
* *(optional)* **Synthetic padding** — a small fraction of i.i.d.
  uniform-random states for regularisation.
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
from certiqnet.data.qgym.dataset import QGymDataset
from certiqnet.diagnostics.state_bank import generate_state_bank
from certiqnet.train.common.supervision import heuristic_actions
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
    synthetic_mix_fraction : float
        Fraction of i.i.d. uniform synthetic states (default 0.0).
    teacher_mix_fraction : float
        Fraction of states drawn from the teacher replay buffer.
    policy_mix_fraction : float
        Fraction of states drawn from the policy replay buffer.
    hard_state_fraction : float
        Fraction of *QGym-sourced* states replaced by high-backlog
        states from the state bank.
    dataset_path : str | Path, optional
        Path to a ``dataset/qgym/<name>/`` directory (static mode).
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
        resample_every_epoch: bool = True,
        policy_buffer_max: int = 4096,
        synthetic_mix_fraction: float = 0.0,
        teacher_mix_fraction: float = 0.25,
        policy_mix_fraction: float = 0.25,
        hard_state_fraction: float = 0.5,
        dataset_path: str | Path | None = None,
        qgym_adapter: QGymAdapter | None = None,
        h: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.N = int(N)
        self.mu = mu.float()
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
        self.qgym_adapter = qgym_adapter

        # ── Validate and store mix fractions ──────────────────────────
        self.synthetic_mix_fraction = float(synthetic_mix_fraction)
        self.teacher_mix_fraction = float(teacher_mix_fraction)
        self.policy_mix_fraction = float(policy_mix_fraction)
        self.hard_state_fraction = float(hard_state_fraction)

        frac_sum = (
            self.synthetic_mix_fraction
            + self.teacher_mix_fraction
            + self.policy_mix_fraction
        )
        if frac_sum > _MAX_MIX_SUM:
            raise ValueError(
                f"Mix fractions (synthetic={self.synthetic_mix_fraction}, "
                f"teacher={self.teacher_mix_fraction}, "
                f"policy={self.policy_mix_fraction}) sum to {frac_sum:.3f}, "
                "which exceeds 1.0.  Reduce one or more fractions."
            )

        # ── Mode detection ────────────────────────────────────────────
        self._online = (
            qgym_adapter is not None
            and getattr(qgym_adapter, "mode", "static") == "online"
        )
        self._static = dataset_path is not None

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
        self._teacher_buffer: deque[Tensor] = deque(maxlen=self.policy_buffer_max)

    # ── Compatibility properties ──────────────────────────────────────

    @property
    def adapter(self) -> QGymAdapter | None:
        """Expose the QGym adapter as ``dm.adapter`` for training-module compatibility.

        Training modules (``QueueingLightningModule`` et al.) access
        ``dm.adapter.make_observation(Q, mu)`` during rollouts.  This
        property ensures QGymDataModule works as a drop-in replacement
        for ``CertiQNetDataModule`` which stores the adapter as
        ``self.adapter``.
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

    # ── Setup ─────────────────────────────────────────────────────────

    def setup(self, stage: str | None = None) -> None:
        self._epoch = 0

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
            n_val = min(len(val_ds), n_val)
            val_Q = torch.stack([val_ds[i][0] for i in range(n_val)])
            val_mu = (
                val_ds.mu.unsqueeze(0).expand(n_val, -1)
                if val_ds.mu is not None
                else self.mu.unsqueeze(0).expand(n_val, -1)
            )
            val_cost = self._compute_cost(val_Q)
            self.val_ds = TensorDataset(val_Q, val_mu, val_cost)

        elif self._online:
            val_gen = torch.Generator().manual_seed(self.seed + 1)
            val = self.qgym_adapter.sample_batch(
                n_samples=n_val,
                N=self.N,
                mu=self.mu,
                generator=val_gen,
            )
            val_cost = self._compute_cost(val.Q.float())
            self.val_ds = TensorDataset(val.Q.float(), val.mu.float(), val_cost)

        else:
            # Pure-synthetic fallback (should not normally be reached for QGym)
            val_gen = torch.Generator().manual_seed(self.seed + 1)
            Q = torch.randint(
                0, max(1, self.max_queue), (n_val, self.N), generator=val_gen
            ).float()
            mu = self.mu.unsqueeze(0).expand(Q.shape[0], -1)
            cost = self._compute_cost(Q)
            self.val_ds = TensorDataset(Q, mu, cost)

    def _setup_train_source(self) -> None:
        """Load the static training dataset (if applicable)."""
        if self._static:
            self._qgym_train_ds = QGymDataset(
                str(self.dataset_path), split="train"
            )

    def _setup_test(self) -> None:
        """Build the test dataset from the ``test`` split when available."""
        if self._static:
            test_dir = Path(str(self.dataset_path)) / "test"
            if test_dir.exists() and any(test_dir.glob("*.pt")):
                test_ds = QGymDataset(str(self.dataset_path), split="test")
                n_test = len(test_ds)
                test_Q = torch.stack([test_ds[i][0] for i in range(n_test)])
                test_mu = (
                    test_ds.mu.unsqueeze(0).expand(n_test, -1)
                    if test_ds.mu is not None
                    else self.mu.unsqueeze(0).expand(n_test, -1)
                )
                test_cost = self._compute_cost(test_Q)
                self.test_ds = TensorDataset(test_Q, test_mu, test_cost)
                self._qgym_test_ds = test_ds
                return

        # Fallback: reuse validation data for test
        if self.val_ds is not None:
            self.test_ds = self.val_ds

    # ── Replay buffers ────────────────────────────────────────────────

    def record_policy_states(self, Q: Tensor) -> None:
        """Append model-visited states to the policy replay buffer."""
        self._policy_buffer.append(Q.detach().cpu())

    def record_teacher_states(self, Q: Tensor) -> None:
        """Append teacher-visited states to the teacher replay buffer."""
        self._teacher_buffer.append(Q.detach().cpu())

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
            0.0,
            1.0
            - self.synthetic_mix_fraction
            - self.teacher_mix_fraction
            - self.policy_mix_fraction,
        )
        qgym_count = max(1, int(self.n_samples * qgym_fraction))
        hard_count = int(qgym_count * self.hard_state_fraction)
        easy_qgym_count = qgym_count - hard_count

        teacher_count = max(0, int(self.n_samples * self.teacher_mix_fraction))
        policy_count = max(0, int(self.n_samples * self.policy_mix_fraction))
        synthetic_count = max(
            0, int(self.n_samples * self.synthetic_mix_fraction)
        )
        adversarial_count = max(32, self.n_samples // 4)

        # ── QGym source ──────────────────────────────────────────────
        if self._online and self.qgym_adapter is not None and easy_qgym_count > 0:
            batch = self.qgym_adapter.sample_batch(
                n_samples=easy_qgym_count,
                N=self.N,
                mu=self.mu,
                generator=gen,
            )
            qgym_q = batch.Q.float()
            qgym_mu = (
                batch.mu.float()
                if batch.mu is not None
                else self.mu.unsqueeze(0).expand(easy_qgym_count, -1)
            )
        elif (
            self._static
            and self._qgym_train_ds is not None
            and easy_qgym_count > 0
        ):
            n = min(easy_qgym_count, len(self._qgym_train_ds))
            idx = torch.randperm(len(self._qgym_train_ds), generator=gen)[:n]
            rows = [self._qgym_train_ds[int(i)] for i in idx]
            qgym_q = torch.stack([r[0] for r in rows])
            mu_val = rows[0][1]
            qgym_mu = mu_val.reshape(1, -1).expand(n, -1)
        else:
            qgym_q = torch.empty(0, self.N)
            qgym_mu = torch.empty(0, self.N)

        # ── Synthetic padding ────────────────────────────────────────
        if synthetic_count > 0:
            synthetic_q = torch.randint(
                0, max(1, self.max_queue), (synthetic_count, self.N), generator=gen
            ).float()
            synthetic_mu = self.mu.unsqueeze(0).expand(synthetic_count, -1)
        else:
            synthetic_q = torch.empty(0, self.N)
            synthetic_mu = torch.empty(0, self.N)

        # ── Replay buffers ───────────────────────────────────────────
        teacher_states = self._sample_rows(
            self._buffer_tensor(self._teacher_buffer), teacher_count, gen
        )
        policy_states = self._sample_rows(
            self._buffer_tensor(self._policy_buffer), policy_count, gen
        )

        # Bootstrap from QGym source if replay buffers are empty
        if teacher_states.numel() == 0 and teacher_count > 0:
            teacher_states = self._bootstrap_states(teacher_count, gen)
        if policy_states.numel() == 0 and policy_count > 0:
            policy_states = self._bootstrap_states(policy_count, gen)

        # ── Hard / adversarial states ────────────────────────────────
        bank = generate_state_bank(
            N=self.N,
            mu=self.mu,
            beta=1.0,
            R_cert=float("inf"),
            n_random=max(adversarial_count * 4, hard_count * 4),
            n_grid=0,
            n_boundary=max(64, self.N * 8),
        )
        backlog = bank.sum(dim=-1)
        k = min(hard_count, bank.shape[0])
        hard_rank = torch.topk(backlog, k=max(k, 1), largest=True).indices
        hard_states = bank[hard_rank]
        if hard_states.shape[0] < hard_count:
            hard_states = self._sample_rows(hard_states, hard_count, gen)
        adversarial_states = self._sample_rows(bank, adversarial_count, gen)

        # ── Concatenate all sources ──────────────────────────────────
        Q = torch.cat(
            [
                qgym_q,
                synthetic_q,
                hard_states.float(),
                teacher_states.float(),
                policy_states.float(),
                adversarial_states.float(),
            ],
            dim=0,
        )
        mu = torch.cat(
            [
                qgym_mu,
                synthetic_mu,
                self.mu.unsqueeze(0).expand(hard_states.shape[0], -1),
                self.mu.unsqueeze(0).expand(teacher_states.shape[0], -1),
                self.mu.unsqueeze(0).expand(policy_states.shape[0], -1),
                self.mu.unsqueeze(0).expand(adversarial_states.shape[0], -1),
            ],
            dim=0,
        )
        sed_action, qmd_action = heuristic_actions(Q, mu)
        cost = self._compute_cost(Q)

        if self.context_dim > 0:
            xi = torch.zeros(Q.shape[0], self.N, self.context_dim)
            tensors = (Q, mu, xi, cost, sed_action, qmd_action)
        else:
            tensors = (Q, mu, cost, sed_action, qmd_action)

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

        * 6: ``(Q, mu, xi, cost, sed_action, qmd_action)``
        * 5: ``(Q, mu, cost, sed_action, qmd_action)``
        * 4: ``(Q, mu, xi, cost)``
        * 3: ``(Q, mu, cost)``
        """
        first = batch[0]
        n = len(first)

        if n == 6:
            Q, mu, xi, cost, sed_action, qmd_action = zip(*batch, strict=True)
            return {
                "Q": torch.stack(list(Q)),
                "mu": torch.stack(list(mu)),
                "xi": torch.stack(list(xi)),
                "cost": torch.stack(list(cost)),
                "sed_action": torch.stack(list(sed_action)),
                "qmd_action": torch.stack(list(qmd_action)),
            }
        if n == 5:
            Q, mu, cost, sed_action, qmd_action = zip(*batch, strict=True)
            return {
                "Q": torch.stack(list(Q)),
                "mu": torch.stack(list(mu)),
                "cost": torch.stack(list(cost)),
                "sed_action": torch.stack(list(sed_action)),
                "qmd_action": torch.stack(list(qmd_action)),
            }
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
            f"mix=[qgym={1 - self.synthetic_mix_fraction - self.teacher_mix_fraction - self.policy_mix_fraction:.2f}, "
            f"synth={self.synthetic_mix_fraction:.2f}, "
            f"teacher={self.teacher_mix_fraction:.2f}, "
            f"policy={self.policy_mix_fraction:.2f}], "
            f"h={'yes' if self._h is not None else 'no'})"
        )
