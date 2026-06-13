"""Lightning data module backed by synthetic and aggregated queue states."""

from __future__ import annotations

from collections import deque
from pathlib import Path

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
from certiqnet.training.oracle import heuristic_actions, load_queueing_oracle_archive
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
        oracle_mix_fraction: float = 0.25,
        oracle_data_path: str | None = None,
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
        self.oracle_mix_fraction = float(oracle_mix_fraction)
        self.oracle_data_path = oracle_data_path
        self.policy_buffer_max = int(policy_buffer_max)
        self._epoch = 0
        self.train_ds: TensorDataset | None = None
        self.val_ds: TensorDataset | None = None
        self._policy_buffer: deque[Tensor] = deque(maxlen=self.policy_buffer_max)
        self._teacher_buffer: deque[Tensor] = deque(maxlen=self.policy_buffer_max)
        self._oracle_Q: Tensor | None = None
        self._oracle_mu: Tensor | None = None
        self._oracle_action: Tensor | None = None
        self._oracle_delta_v: Tensor | None = None
        self._oracle_qmd: Tensor | None = None
        self._oracle_sed: Tensor | None = None
        self._oracle_jswq: Tensor | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        self._epoch = 0
        self._load_oracle_archive()
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

    def _load_oracle_archive(self) -> None:
        if self.oracle_data_path is None:
            return
        path = Path(self.oracle_data_path)
        if not path.exists():
            return
        bundles = [bundle for bundle in load_queueing_oracle_archive(path) if bundle.N == self.N]
        if not bundles:
            return
        self._oracle_Q = torch.cat([bundle.states for bundle in bundles], dim=0).float()
        self._oracle_mu = torch.cat([bundle.mu for bundle in bundles], dim=0).float()
        self._oracle_action = torch.cat([bundle.pi_oracle for bundle in bundles], dim=0).long()
        self._oracle_delta_v = torch.cat([bundle.delta_V for bundle in bundles], dim=0).float()
        self._oracle_qmd = torch.cat([bundle.pi_qmd for bundle in bundles], dim=0).long()
        self._oracle_sed = torch.cat([bundle.pi_sed for bundle in bundles], dim=0).long()
        self._oracle_jswq = torch.cat([bundle.pi_jswq for bundle in bundles], dim=0).long()

    def record_policy_states(self, Q: Tensor) -> None:
        """Store visited states from current-policy rollouts."""
        self._policy_buffer.append(Q.detach().cpu())

    def record_teacher_states(self, Q: Tensor) -> None:
        """Store states used for SED warm-start labels."""
        self._teacher_buffer.append(Q.detach().cpu())

    def _sample_aligned(self, tensor: Tensor | None, count: int, generator: torch.Generator) -> Tensor | None:
        if tensor is None or tensor.numel() == 0 or count <= 0:
            return None
        return self._sample_rows(tensor, count, generator)

    def _build_train(self) -> None:
        """Create or refresh the mixed training dataset."""
        gen = torch.Generator().manual_seed(self.seed + self._epoch * 9973)

        synthetic_count = max(1, int(self.n_samples * self.synthetic_mix_fraction))
        teacher_count = max(0, int(self.n_samples * self.teacher_mix_fraction))
        policy_count = max(0, int(self.n_samples * self.policy_mix_fraction))
        oracle_count = max(0, int(self.n_samples * self.oracle_mix_fraction))
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
        oracle_states = self._sample_aligned(self._oracle_Q, oracle_count, gen)
        oracle_mu = self._sample_aligned(self._oracle_mu, oracle_count, gen)
        oracle_action = self._sample_aligned(self._oracle_action, oracle_count, gen)
        oracle_delta_v = self._sample_aligned(self._oracle_delta_v, oracle_count, gen)

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

        groups: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]] = []

        def _build_group(
            Q_group: Tensor,
            mu_group: Tensor,
            *,
            oracle_action_group: Tensor | None = None,
            oracle_delta_v_group: Tensor | None = None,
        ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
            sed_action, qmd_action = heuristic_actions(Q_group, mu_group)
            if oracle_action_group is None:
                oracle_action_group = qmd_action
                has_oracle = torch.zeros(Q_group.shape[0], dtype=torch.bool, device=Q_group.device)
            else:
                oracle_action_group = oracle_action_group.to(device=Q_group.device, dtype=torch.long)
                has_oracle = torch.ones(Q_group.shape[0], dtype=torch.bool, device=Q_group.device)
            if oracle_delta_v_group is None:
                oracle_delta_v_group = torch.zeros(
                    Q_group.shape[0], Q_group.shape[1], device=Q_group.device, dtype=Q_group.dtype
                )
            else:
                oracle_delta_v_group = oracle_delta_v_group.to(device=Q_group.device, dtype=Q_group.dtype)
            return Q_group.float(), mu_group.float(), sed_action, qmd_action, oracle_action_group, oracle_delta_v_group, has_oracle

        groups.append(_build_group(synthetic.Q, synthetic.mu))
        groups.append(
            _build_group(
                teacher_states.float(),
                self.mu.unsqueeze(0).expand(teacher_states.shape[0], -1).clone(),
            )
        )
        groups.append(
            _build_group(
                policy_states.float(),
                self.mu.unsqueeze(0).expand(policy_states.shape[0], -1).clone(),
            )
        )
        groups.append(
            _build_group(
                adversarial_states.float(),
                self.mu.unsqueeze(0).expand(adversarial_states.shape[0], -1).clone(),
            )
        )
        if oracle_states is not None and oracle_mu is not None:
            groups.append(
                _build_group(
                    oracle_states.float(),
                    oracle_mu.float(),
                    oracle_action_group=oracle_action,
                    oracle_delta_v_group=oracle_delta_v,
                )
            )

        Q = torch.cat([group[0] for group in groups], dim=0)
        mu = torch.cat([group[1] for group in groups], dim=0)
        sed_action = torch.cat([group[2] for group in groups], dim=0)
        qmd_action = torch.cat([group[3] for group in groups], dim=0)
        oracle_action_full = torch.cat([group[4] for group in groups], dim=0)
        oracle_delta_v_full = torch.cat([group[5] for group in groups], dim=0)
        has_oracle = torch.cat([group[6] for group in groups], dim=0)
        cost = Q.sum(dim=-1)

        if self.context_dim > 0:
            xi = torch.zeros(Q.shape[0], self.N, self.context_dim)
            tensors = (Q, mu, xi, cost, sed_action, qmd_action, oracle_action_full, oracle_delta_v_full, has_oracle)
            if self.train_ds is None:
                self.train_ds = TensorDataset(*tensors)
            else:
                self.train_ds.tensors = tensors
        else:
            tensors = (Q, mu, cost, sed_action, qmd_action, oracle_action_full, oracle_delta_v_full, has_oracle)
            if self.train_ds is None:
                self.train_ds = TensorDataset(*tensors)
            else:
                self.train_ds.tensors = tensors

    def resample_train_data(self) -> None:
        """Refresh the training set each epoch when configured to do so."""
        if not self.resample_every_epoch:
            return
        self._epoch += 1
        self._build_train()

    @staticmethod
    def _collate(batch: list[tuple[Tensor, ...]]) -> dict[str, Tensor]:
        first = batch[0]
        if len(first) == 9:
            Q, mu, xi, cost, sed_action, qmd_action, oracle_action, oracle_delta_v, has_oracle = zip(
                *batch,
                strict=True,
            )
            return {
                "Q": torch.stack(list(Q)),
                "mu": torch.stack(list(mu)),
                "xi": torch.stack(list(xi)),
                "cost": torch.stack(list(cost)),
                "sed_action": torch.stack(list(sed_action)),
                "qmd_action": torch.stack(list(qmd_action)),
                "oracle_action": torch.stack(list(oracle_action)),
                "oracle_delta_v": torch.stack(list(oracle_delta_v)),
                "has_oracle": torch.stack(list(has_oracle)),
            }
        Q, mu, cost, sed_action, qmd_action, oracle_action, oracle_delta_v, has_oracle = zip(
            *batch,
            strict=True,
        )
        return {
            "Q": torch.stack(list(Q)),
            "mu": torch.stack(list(mu)),
            "cost": torch.stack(list(cost)),
            "sed_action": torch.stack(list(sed_action)),
            "qmd_action": torch.stack(list(qmd_action)),
            "oracle_action": torch.stack(list(oracle_action)),
            "oracle_delta_v": torch.stack(list(oracle_delta_v)),
            "has_oracle": torch.stack(list(has_oracle)),
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
