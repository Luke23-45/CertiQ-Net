"""Lightning data module backed by synthetic state banks."""

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:  # pragma: no cover
    pl = None


class CertiQNetDataModule(pl.LightningDataModule if pl is not None else object):
    """Simple local data module for smoke tests and cloud training bootstrap."""

    def __init__(self, N: int, mu: Tensor, batch_size: int = 64, n_samples: int = 2048) -> None:
        super().__init__()
        self.N = N
        self.mu = mu.float()
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.train_ds: TensorDataset | None = None
        self.val_ds: TensorDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Create random queue state datasets."""
        del stage
        Q_train = torch.randint(0, 100, (self.n_samples, self.N)).float()
        mu_train = self.mu.unsqueeze(0).expand(self.n_samples, -1)
        Q_val = torch.randint(0, 100, (max(128, self.n_samples // 8), self.N)).float()
        mu_val = self.mu.unsqueeze(0).expand(Q_val.shape[0], -1)
        self.train_ds = TensorDataset(Q_train, mu_train)
        self.val_ds = TensorDataset(Q_val, mu_val)

    @staticmethod
    def _collate(batch: list[tuple[Tensor, Tensor]]) -> dict[str, Tensor]:
        Q, mu = zip(*batch, strict=True)
        return {"Q": torch.stack(list(Q)), "mu": torch.stack(list(mu))}

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(self.val_ds, batch_size=self.batch_size, collate_fn=self._collate)
