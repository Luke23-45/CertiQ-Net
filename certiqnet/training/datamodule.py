"""Lightning data module backed by synthetic state banks."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    pl = None

from certiqnet.adapters.base import DispatchAdapter
from certiqnet.adapters.queueing import QueueingAdapter
from certiqnet.utils.platform import resolve_num_workers


class CertiQNetDataModule(pl.LightningDataModule if pl is not None else object):
    """Data module with auto-detected num_workers and batch size."""

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
    ) -> None:
        super().__init__()
        self.N = N
        self.mu = mu.float()
        self.batch_size = batch_size
        self.n_samples = n_samples
        self._num_workers = resolve_num_workers(num_workers)
        self.adapter = (
            adapter if adapter is not None else QueueingAdapter(assumptions_satisfied=True)
        )
        self.seed = int(seed)
        self.max_queue = int(max_queue)
        self.train_ds: TensorDataset | None = None
        self.val_ds: TensorDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        train_gen = torch.Generator().manual_seed(self.seed)
        val_gen = torch.Generator().manual_seed(self.seed + 1)
        train = self.adapter.sample_batch(
            n_samples=self.n_samples,
            N=self.N,
            mu=self.mu,
            max_queue=self.max_queue,
            generator=train_gen,
        )
        val = self.adapter.sample_batch(
            n_samples=max(128, self.n_samples // 8),
            N=self.N,
            mu=self.mu,
            max_queue=self.max_queue,
            generator=val_gen,
        )
        if train.xi is None:
            self.train_ds = TensorDataset(train.Q, train.mu, train.cost)
            self.val_ds = TensorDataset(val.Q, val.mu, val.cost)
        else:
            assert val.xi is not None
            self.train_ds = TensorDataset(train.Q, train.mu, train.xi, train.cost)
            self.val_ds = TensorDataset(val.Q, val.mu, val.xi, val.cost)

    @staticmethod
    def _collate(batch: list[tuple[Tensor, ...]]) -> dict[str, Tensor]:
        first = batch[0]
        if len(first) == 3:
            Q, mu, cost = zip(*batch, strict=True)
            return {
                "Q": torch.stack(list(Q)),
                "mu": torch.stack(list(mu)),
                "cost": torch.stack(list(cost)),
            }
        Q, mu, xi, cost = zip(*batch, strict=True)
        return {
            "Q": torch.stack(list(Q)),
            "mu": torch.stack(list(mu)),
            "xi": torch.stack(list(xi)),
            "cost": torch.stack(list(cost)),
        }

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate,
            num_workers=self._num_workers,
            pin_memory=self._num_workers > 0,
            persistent_workers=self._num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            collate_fn=self._collate,
            num_workers=self._num_workers,
            pin_memory=self._num_workers > 0,
            persistent_workers=self._num_workers > 0,
        )
