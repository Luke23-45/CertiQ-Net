"""Training callbacks for certificate audits."""

import warnings
from typing import Protocol

import torch
from torch import Tensor

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:  # pragma: no cover
    pl = None

from certiqnet.diagnostics.state_bank import generate_state_bank
from certiqnet.dispatcher.types import DispatcherDiagnostics


class _DataModuleLike(Protocol):
    mu: Tensor


class _TrainerLike(Protocol):
    datamodule: _DataModuleLike
    current_epoch: int


class _ModelLike(Protocol):
    N: int
    beta: float

    def eval(self) -> object: ...

    def __call__(
        self, Q: Tensor, mu: Tensor, xi: Tensor | None = None, training_mode: bool = False
    ) -> tuple[Tensor, DispatcherDiagnostics]: ...


class _LightningLike(Protocol):
    model: _ModelLike

    def log(self, name: str, value: object, prog_bar: bool = False) -> None: ...


class CertificateAuditCallback(pl.Callback if pl is not None else object):
    """Run full state-bank audit after each validation epoch.

    The audit checks whether the dispatch certificate is satisfied on a
    held-out state bank.
    """

    def __init__(self, assert_after_epoch: int = 10, violation_tol: float = 0.5) -> None:
        super().__init__()
        self.assert_after_epoch = int(assert_after_epoch)
        self.violation_tol = float(violation_tol)

    def on_validation_epoch_end(self, trainer: _TrainerLike, pl_module: _LightningLike) -> None:
        """Fail training if projected models violate the certificate."""
        model = pl_module.model
        mu = trainer.datamodule.mu
        Q_bank = generate_state_bank(
            N=model.N,
            mu=mu,
            beta=model.beta,
            R_cert=float(getattr(getattr(model, "cfg", object()), "certificate", object()).fallback_radius)
            if hasattr(getattr(model, "cfg", object()), "certificate")
            else float("inf"),
            n_random=256,
            n_grid=128,
            n_boundary=64,
        )
        model.eval()
        with torch.no_grad():
            _, diag = model(Q_bank, mu.unsqueeze(0).expand(len(Q_bank), -1), training_mode=False)
        max_violation = (diag.A_final - diag.B_Q).clamp(min=0).max().item()
        pl_module.log("audit/max_violation", max_violation, prog_bar=True)
        pl_module.log("audit/violation_rate", (diag.A_final > diag.B_Q).float().mean(), prog_bar=True)
        fin_cb = getattr(model, "C", float("inf")) < float("inf")
        epoch = trainer.current_epoch
        if fin_cb and max_violation > self.violation_tol and epoch >= self.assert_after_epoch:
            raise AssertionError(
                f"CERTIFICATE AUDIT FAILED after epoch {epoch}. "
                f"Max projection violation: {max_violation:.2e} "
                f"(tolerance: {self.violation_tol}, C={getattr(model,'C',float('inf')):.2e})."
            )
        if fin_cb and max_violation > self.violation_tol:
            warnings.warn(
                f"Early certificate violation (epoch {epoch}): {max_violation:.2e} "
                f"(tolerance kicks in at epoch {self.assert_after_epoch}).",
                stacklevel=2,
            )
