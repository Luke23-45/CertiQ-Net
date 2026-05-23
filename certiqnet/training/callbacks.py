"""Training callbacks for certificate audits."""

from typing import Protocol

import torch
from torch import Tensor

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:  # pragma: no cover
    pl = None

from certiqnet.diagnostics.state_bank import generate_state_bank
from certiqnet.math.certificate import CertificateDiagnostics


class _DataModuleLike(Protocol):
    mu: Tensor


class _TrainerLike(Protocol):
    datamodule: _DataModuleLike


class _ModelLike(Protocol):
    N: int
    beta: float

    def eval(self) -> object: ...

    def __call__(
        self, Q: Tensor, mu: Tensor, xi: Tensor | None = None, training_mode: bool = False
    ) -> tuple[Tensor, CertificateDiagnostics]: ...


class _LightningLike(Protocol):
    model: _ModelLike

    def log(self, name: str, value: object, prog_bar: bool = False) -> None: ...


class CertificateAuditCallback(pl.Callback if pl is not None else object):
    """Run full state-bank audit after each validation epoch."""

    def on_validation_epoch_end(self, trainer: _TrainerLike, pl_module: _LightningLike) -> None:
        """Fail training if projected models violate the certificate."""
        model = pl_module.model
        mu = trainer.datamodule.mu
        Q_bank = generate_state_bank(
            N=model.N,
            mu=mu,
            beta=model.beta,
            R_cert=getattr(model, "R_cert", float("inf")),
            n_random=256,
            n_grid=128,
            n_boundary=64,
        )
        model.eval()
        with torch.no_grad():
            _, diag = model(Q_bank, mu.unsqueeze(0).expand(len(Q_bank), -1), training_mode=False)
        max_violation = (diag.A_pi - diag.B_Q).clamp(min=0).max().item()
        pl_module.log("audit/max_violation", max_violation, prog_bar=True)
        pl_module.log("audit/violation_rate", (diag.A_pi > diag.B_Q).float().mean(), prog_bar=True)
        if getattr(model, "C_B", float("inf")) < float("inf"):
            assert max_violation < 1e-5, (
                f"CERTIFICATE AUDIT FAILED. Max projection violation: {max_violation:.2e}."
            )
