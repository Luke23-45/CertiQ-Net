"""Lightning data module backed by synthetic and aggregated queue states."""

from __future__ import annotations

from collections import deque

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    pl = None

from certiqnet.adapters.base import DispatchAdapter
from certiqnet.adapters.queueing import QueueingAdapter
from certiqnet.diagnostics.state_bank import generate_state_bank
from certiqnet.utils.platform import resolve_num_workers


class CertiQNetDataModule(pl.LightningDataModule if pl is not None else object):
    """Data module with on-policy, teacher, and adversarial queue coverage."""

    def __init__(
        self,
        N: int,
        mu: Tensor,
        batch_size: int = 64,
        n_samples: int = 2048,
        num_workers: int | None = None,
        adapter: DispatchAdapter | None = None,
        seed: int = 0,
        max_queue: int = 15,
        resample_every_epoch: bool = False,
        policy_buffer_max: int = 4096,
        synthetic_mix_fraction: float = 0.5,
        teacher_mix_fraction: float = 0.25,
        policy_mix_fraction: float = 0.25,
    ) -> None:
        super().__init__()
        self.N = N
        self.mu = mu.float()
        self.batch_size = batch_size
        self.n_samples = n_samples
        self._num_workers = 0 if resample_every_epoch else resolve_num_workers(num_workers)
        self.adapter = (
            adapter if adapter is not None else QueueingAdapter(assumptions_satisfied=True)
        )
        self.context_dim = int(getattr(self.adapter, "context_dim", 0))
        self.seed = int(seed)
        self.max_queue = int(max_queue)
        self.resample_every_epoch = bool(resample_every_epoch)
        self.synthetic_mix_fraction = float(synthetic_mix_fraction)
        self.teacher_mix_fraction = float(teacher_mix_fraction)
        self.policy_mix_fraction = float(policy_mix_fraction)
        self.policy_buffer_max = int(policy_buffer_max)
        self._epoch = 0
        self.train_ds: TensorDataset | None = None
        self.val_ds: TensorDataset | None = None
        self._policy_buffer: deque[Tensor] = deque(maxlen=self.policy_buffer_max)
        self._teacher_buffer: deque[Tensor] = deque(maxlen=self.policy_buffer_max)

    def setup(self, stage: str | None = None) -> None:
        del stage
        self._epoch = 0
        val_gen = torch.Generator().manual_seed(self.seed + 1)
        val = self.adapter.sample_batch(
            n_samples=max(128, self.n_samples // 8),
            N=self.N,
            mu=self.mu,
            max_queue=self.max_queue,
            generator=val_gen,
        )
        if val.xi is None and self.context_dim > 0:
            xi = torch.zeros(val.Q.shape[0], self.N, self.context_dim)
            self.val_ds = TensorDataset(val.Q, val.mu, xi, val.cost)
        elif val.xi is None:
            self.val_ds = TensorDataset(val.Q, val.mu, val.cost)
        else:
            self.val_ds = TensorDataset(val.Q, val.mu, val.xi, val.cost)
        self._build_train()

    @staticmethod
    def _sample_rows(tensor: Tensor, count: int, generator: torch.Generator) -> Tensor:
        if count <= 0 or tensor.numel() == 0:
            return tensor[:0]
        if tensor.shape[0] >= count:
            idx = torch.randperm(tensor.shape[0], generator=generator)[:count]
            return tensor[idx]
        idx = torch.randint(0, tensor.shape[0], (count,), generator=generator)
        return tensor[idx]

    def _buffer_tensor(self, buffer: deque[Tensor]) -> Tensor:
        if not buffer:
            return torch.empty(0, self.N)
        return torch.cat(list(buffer), dim=0)

    def record_policy_states(self, Q: Tensor) -> None:
        """Store visited states from current-policy rollouts."""
        self._policy_buffer.append(Q.detach().cpu())

    def record_teacher_states(self, Q: Tensor) -> None:
        """Store states used for JSWQ warm-start labels."""
        self._teacher_buffer.append(Q.detach().cpu())

    def _build_train(self) -> None:
        """Create or refresh the mixed training dataset."""
        gen = torch.Generator().manual_seed(self.seed + self._epoch * 9973)

        synthetic_count = max(1, int(self.n_samples * self.synthetic_mix_fraction))
        teacher_count = max(0, int(self.n_samples * self.teacher_mix_fraction))
        policy_count = max(0, int(self.n_samples * self.policy_mix_fraction))
        adversarial_count = max(32, self.n_samples // 8)

        synthetic = self.adapter.sample_batch(
            n_samples=synthetic_count,
            N=self.N,
            mu=self.mu,
            max_queue=self.max_queue,
            generator=gen,
        )

        teacher_states = self._sample_rows(self._buffer_tensor(self._teacher_buffer), teacher_count, gen)
        policy_states = self._sample_rows(self._buffer_tensor(self._policy_buffer), policy_count, gen)

        if policy_states.numel() == 0:
            policy_states = self.adapter.sample_batch(
                n_samples=max(1, policy_count),
                N=self.N,
                mu=self.mu,
                max_queue=self.max_queue,
                generator=torch.Generator().manual_seed(self.seed + 11 + self._epoch),
            ).Q
        if teacher_states.numel() == 0:
            teacher_states = self.adapter.sample_batch(
                n_samples=max(1, teacher_count),
                N=self.N,
                mu=self.mu,
                max_queue=self.max_queue,
                generator=torch.Generator().manual_seed(self.seed + 17 + self._epoch),
            ).Q

        bank = generate_state_bank(
            N=self.N,
            mu=self.mu,
            beta=1.0,
            R_cert=float("inf"),
            n_random=adversarial_count,
            n_grid=0,
            n_boundary=max(64, self.N * 8),
        )
        adversarial_states = self._sample_rows(bank, adversarial_count, gen)

        Q = torch.cat(
            [
                synthetic.Q,
                teacher_states.float(),
                policy_states.float(),
                adversarial_states.float(),
            ],
            dim=0,
        )
        mu = self.mu.unsqueeze(0).expand(Q.shape[0], -1).clone()
        cost = Q.sum(dim=-1)
        if self.context_dim > 0:
            xi = torch.zeros(Q.shape[0], self.N, self.context_dim)
            if self.train_ds is None:
                self.train_ds = TensorDataset(Q, mu, xi, cost)
            else:
                self.train_ds.tensors = (Q, mu, xi, cost)
        else:
            if self.train_ds is None:
                self.train_ds = TensorDataset(Q, mu, cost)
            else:
                self.train_ds.tensors = (Q, mu, cost)

    def resample_train_data(self) -> None:
        """Refresh the training set each epoch when configured to do so."""
        if not self.resample_every_epoch:
            return
        self._epoch += 1
        self._build_train()

    @staticmethod
    def _collate(batch: list[tuple[Tensor, ...]]) -> dict[str, Tensor]:
        first = batch[0]
        if len(first) == 4:
            Q, mu, xi, cost = zip(*batch, strict=True)
            return {
                "Q": torch.stack(list(Q)),
                "mu": torch.stack(list(mu)),
                "xi": torch.stack(list(xi)),
                "cost": torch.stack(list(cost)),
            }
        Q, mu, cost = zip(*batch, strict=True)
        return {
            "Q": torch.stack(list(Q)),
            "mu": torch.stack(list(mu)),
            "cost": torch.stack(list(cost)),
        }

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        pin_memory = self._num_workers > 0 and torch.cuda.is_available()
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate,
            num_workers=self._num_workers,
            pin_memory=pin_memory,
            persistent_workers=self._num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        pin_memory = self._num_workers > 0 and torch.cuda.is_available()
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            collate_fn=self._collate,
            num_workers=self._num_workers,
            pin_memory=pin_memory,
            persistent_workers=self._num_workers > 0,
        )
