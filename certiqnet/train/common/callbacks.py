"""Training callbacks for certificate audits."""

import warnings
from typing import Protocol

import torch
from torch import Tensor

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    pl = None

from certiqnet.diagnostics.state_bank import generate_state_bank
from certiqnet.dispatcher.types import DispatcherDiagnostics


class _DataModuleLike(Protocol):
    mu: Tensor


class _TrainerLike(Protocol):
    datamodule: _DataModuleLike
    current_epoch: int
    max_epochs: int | None


class _ModelLike(Protocol):
    N: int
    beta: float

    def eval(self) -> object: ...

    def __call__(
        self, Q: Tensor, mu: Tensor, xi: Tensor | None = None, training_mode: bool = False
    ) -> tuple[Tensor, DispatcherDiagnostics]: ...

    def parameters(self): ...


def _model_device(model: _ModelLike) -> torch.device:
    try:
        param = next(model.parameters())
    except (StopIteration, TypeError):
        return torch.device("cpu")
    return param.device


class _LightningLike(Protocol):
    model: _ModelLike
    dual_lambda: Tensor

    def log(self, name: str, value: object, prog_bar: bool = False) -> None: ...


class CertificateAuditCallback(pl.Callback if pl is not None else object):
    """Run full state-bank audit after each validation epoch."""

    def __init__(
        self,
        assert_after_epoch: int = 10,
        violation_tol: float = 0.5,
        lagrangian_decay: float = 0.95,
        lagrangian_check_fraction: float = 0.5,
    ) -> None:
        super().__init__()
        self.assert_after_epoch = int(assert_after_epoch)
        self.violation_tol = float(violation_tol)
        self.lagrangian_decay = float(lagrangian_decay)
        self.lagrangian_check_fraction = float(lagrangian_check_fraction)

    def on_validation_epoch_end(self, trainer: _TrainerLike, pl_module: _LightningLike) -> None:
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
        if hasattr(model, "reset_dispatch_state"):
            model.reset_dispatch_state()
        device = _model_device(model)
        Q_bank = Q_bank.to(device=device)
        mu_dev = mu.to(device=device)
        with torch.no_grad():
            _, diag = model(
                Q_bank,
                mu_dev.unsqueeze(0).expand(len(Q_bank), -1),
                training_mode=False,
            )
        max_violation = (diag.A_final - diag.B_Q).clamp(min=0).max().item()
        pl_module.log("audit/max_violation", max_violation, prog_bar=True)
        pl_module.log("audit/violation_rate", (diag.A_final > diag.B_Q).float().mean(), prog_bar=True)
        pl_module.log("audit/dual_lambda", pl_module.dual_lambda, prog_bar=True)
        fin_cb = getattr(model, "C", float("inf")) < float("inf")
        epoch = trainer.current_epoch
        constraint_mode = getattr(model, "constraint_mode", "projection")
        max_epochs = float(getattr(trainer, "max_epochs", 200))

        if not fin_cb:
            return

        # Warn about early violations (before assertion window opens)
        if max_violation > self.violation_tol and epoch < self.assert_after_epoch:
            warnings.warn(
                f"Early certificate violation (epoch {epoch}): {max_violation:.2e} "
                f"(tolerance kicks in at epoch {self.assert_after_epoch}).",
                stacklevel=2,
            )

        if epoch < self.assert_after_epoch:
            return

        if constraint_mode == "lagrangian":
            training_progress = epoch / max_epochs
            if training_progress >= self.lagrangian_check_fraction:
                # Late training: enforce relaxed exponentially decaying tolerance
                C_val = float(getattr(model, "C", 20.0))
                progress = epoch - self.assert_after_epoch
                effective_tol = max(self.violation_tol, C_val * (self.lagrangian_decay ** progress))
                if max_violation > effective_tol:
                    raise AssertionError(
                        f"CERTIFICATE AUDIT FAILED after epoch {epoch}. "
                        f"Max violation: {max_violation:.2e} "
                        f"(Lagrangian relaxed tolerance: {effective_tol:.2e}, "
                        f"base: {self.violation_tol}, C={C_val:.2e})."
                    )
        else:
            if max_violation > self.violation_tol:
                raise AssertionError(
                    f"CERTIFICATE AUDIT FAILED after epoch {epoch}. "
                    f"Max projection violation: {max_violation:.2e} "
                    f"(tolerance: {self.violation_tol}, C={getattr(model,'C',float('inf')):.2e})."
                )
