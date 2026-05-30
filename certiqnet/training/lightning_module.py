"""PyTorch Lightning module for CertiQ-Net."""

import torch
from omegaconf import DictConfig
from torch import Tensor

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:  # pragma: no cover
    pl = None

from certiqnet.math.certificate import CertificateDiagnostics
from certiqnet.training.loss import CertiQNetLoss


class CertiQNetLightningModule(pl.LightningModule if pl is not None else torch.nn.Module):
    """Lightning wrapper that logs all certificate diagnostics."""

    def __init__(self, model: torch.nn.Module, cfg: DictConfig) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.loss_fn = CertiQNetLoss(cfg.loss)
        if hasattr(self, "save_hyperparameters"):
            self.save_hyperparameters(ignore=["model"])

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Run a training step using the smooth/projection training mode."""
        del batch_idx
        Q, mu, xi = batch["Q"], batch["mu"], batch.get("xi")
        pi, diag = self.model(Q, mu, xi, training_mode=True)
        losses = self.loss_fn(Q, mu, pi, diag, self.cfg.loss, batch.get("cost"))
        for key, value in losses.items():
            self.log(f"train/{key}", value, on_step=True, on_epoch=True, prog_bar=(key == "total"))
        self._log_diagnostics(diag, "train")
        return losses["total"]

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """Run exact-gate validation."""
        del batch_idx
        Q, mu, xi = batch["Q"], batch["mu"], batch.get("xi")
        with torch.no_grad():
            _, diag = self.model(Q, mu, xi, training_mode=False)
        self.log("val/CERTIFICATE_VIOLATION", (diag.drift_slack.min() < -1e-4).float())
        self._log_diagnostics(diag, "val")

    def _log_diagnostics(self, diag: CertificateDiagnostics, stage: str) -> None:
        """Log all diagnostic fields in aggregate form."""
        self.log(f"{stage}/A_base", diag.A_base.mean())
        self.log(f"{stage}/A_nn", diag.A_nn.mean())
        self.log(f"{stage}/A_pi", diag.A_pi.mean())
        self.log(f"{stage}/m_Q", diag.m_Q.mean())
        self.log(f"{stage}/B_Q", diag.B_Q.mean())
        self.log(f"{stage}/drift_slack_min", diag.drift_slack.min())
        self.log(f"{stage}/drift_slack_mean", diag.drift_slack.mean())
        self.log(f"{stage}/eta_raw", diag.eta_raw.nanmean())
        self.log(f"{stage}/eta_final", diag.eta_final.nanmean())
        self.log(f"{stage}/eta_safe", diag.eta_safe.nanmean())
        self.log(f"{stage}/fallback_rate", diag.fallback_active.float().mean())
        self.log(f"{stage}/residual_norm", diag.residual_norm.mean())
        self.log(f"{stage}/policy_entropy", diag.policy_entropy.mean())

    def configure_optimizers(self) -> dict[str, object]:
        """Configure AdamW with cosine annealing."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.trainer.lr,
            weight_decay=self.cfg.trainer.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
